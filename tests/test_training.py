import json

import numpy as np
import pytest
from stable_baselines3 import PPO

from vertex.learning.evaluate import evaluate_checkpoint
from vertex.learning.train import TrainingConfig, _build_parser, train_ppo


def test_training_cli_defaults_are_numeric_values():
    args = _build_parser().parse_args(["--output-dir", "runs/cli-default-test"])
    assert args.timesteps == 10_000
    assert args.seed == 7


def test_short_ppo_run_saves_reloadable_checkpoint_and_metadata(tmp_path):
    output_dir = tmp_path / "ppo-smoke"
    metadata = train_ppo(
        output_dir,
        TrainingConfig(total_timesteps=128, seed=17, n_steps=64, batch_size=32, n_epochs=1),
    )

    checkpoint = output_dir / "ppo_vertex.zip"
    assert checkpoint.is_file()
    assert (output_dir / "logs" / "progress.csv").is_file()
    saved_metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert saved_metadata["algorithm"] == "PPO"
    assert saved_metadata["device"] == "cpu"
    assert saved_metadata["actual_timesteps"] == 128
    assert metadata["seed"] == 17

    model = PPO.load(checkpoint, device="cpu")
    observation = np.zeros(8, dtype=np.float32)
    action, _ = model.predict(observation, deterministic=True)
    assert model.action_space.contains(action)

    evaluation_dir = tmp_path / "evaluation"
    report = evaluate_checkpoint(checkpoint, evaluation_dir, seeds=[101, 202])
    assert report["seeds"] == [101, 202]
    assert report["episode_count"] == 2
    assert 0.0 <= report["success_rate"] <= 1.0
    assert sum(report["terminal_category_counts"].values()) == 2
    assert report["training_reward_trend"]["updates_with_episode_data"] >= 1
    assert (evaluation_dir / "episodes.csv").is_file()
    assert not (evaluation_dir / "step_diagnostics.jsonl").exists()
    traces = (evaluation_dir / "episode_traces.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(traces) == 2

    diagnostic_a = tmp_path / "diagnostic-a"
    diagnostic_b = tmp_path / "diagnostic-b"
    report_a = evaluate_checkpoint(checkpoint, diagnostic_a, seeds=[101, 202], diagnostics=True)
    report_b = evaluate_checkpoint(checkpoint, diagnostic_b, seeds=[101, 202], diagnostics=True)
    assert report_a["terminal_category_counts"] == report_b["terminal_category_counts"]
    assert report_a["mean_episode_return"] == report_b["mean_episode_return"]
    diagnostic_lines = (diagnostic_a / "step_diagnostics.jsonl").read_text(encoding="utf-8").splitlines()
    assert diagnostic_lines == (diagnostic_b / "step_diagnostics.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(diagnostic_lines[0])
    assert first["schema_version"] == 1
    assert first["episode_seed"] == 101
    assert first["step"] == 1
    assert set(first["state_before"]) == {"x_m", "z_m", "vx_mps", "vz_mps", "theta_rad", "omega_radps"}
    assert len(first["observation"]) == 8
    assert len(first["raw_deterministic_policy_action"]) == 3
    assert len(first["action_applied"]) == 3
    assert set(first["applied_wrench"]) == {"force_x_N", "force_z_N", "moment_Nm"}
    assert "horizontal_position_penalty" in first["reward_components"]
    assert "total_reward" in first["reward_components"]
    assert first["total_reward"] == first["reward_components"]["total_reward"]
    assert first["state_after"] == json.loads(diagnostic_lines[0])["state_after"]
    assert first["terminated"] is False
    assert first["truncated"] is False
    assert first["termination_reason"] is None
    with pytest.raises(FileExistsError, match="existing evaluation data was preserved"):
        evaluate_checkpoint(checkpoint, evaluation_dir, seeds=[101, 202])

