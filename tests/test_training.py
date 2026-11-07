import json

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from vertex.learning.evaluate import evaluate_checkpoint
from vertex.learning.train import TrainingConfig, _build_parser, train_ppo
from vertex.environment import LandingEnv


def test_training_cli_defaults_are_numeric_values():
    args = _build_parser().parse_args(["--output-dir", "runs/cli-default-test"])
    assert args.timesteps == 10_000
    assert args.seed == 7
    assert args.gamma == 0.99
    assert args.diagnostics is False
    assert args.diagnostic_steps == 2_048


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


def test_training_diagnostics_preserve_seeded_ppo_updates(tmp_path):
    training = TrainingConfig(
        total_timesteps=128,
        seed=37,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
    )
    plain_dir = tmp_path / "plain"
    diagnostic_dir = tmp_path / "diagnostic"
    train_ppo(plain_dir, training)
    metadata = train_ppo(
        diagnostic_dir,
        training,
        diagnostics=True,
        diagnostic_steps=128,
    )

    plain_model = PPO.load(plain_dir / "ppo_vertex.zip", device="cpu")
    diagnostic_model = PPO.load(diagnostic_dir / "ppo_vertex.zip", device="cpu")
    assert plain_model.policy.state_dict().keys() == diagnostic_model.policy.state_dict().keys()
    for name, plain_parameter in plain_model.policy.state_dict().items():
        torch.testing.assert_close(
            plain_parameter,
            diagnostic_model.policy.state_dict()[name],
            rtol=0,
            atol=0,
        )

    assert metadata["training_diagnostics"]["recorded_environment_steps"] == 128
    lines = (diagnostic_dir / "training_diagnostics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 128
    records = [json.loads(line) for line in lines]
    assert [record["global_timestep"] for record in records] == list(range(1, 129))
    assert all(len(record["observation"]) == 8 for record in records)
    assert all(len(record["state_before"]) == 6 for record in records)
    assert all(len(record["sampled_action_before_clipping"]) == 3 for record in records)
    assert all(len(record["action_passed_to_environment"]) == 3 for record in records)
    assert all(len(record["policy_mean"]) == 3 for record in records)
    assert all(len(record["policy_standard_deviation"]) == 3 for record in records)
    for record in records:
        raw = np.asarray(record["sampled_action_before_clipping"])
        applied = np.asarray(record["action_passed_to_environment"])
        assert np.array_equal(record["action_clipped_by_channel"], (raw < -1.0) | (raw > 1.0))
        np.testing.assert_array_equal(applied, np.clip(raw, -1.0, 1.0))
    assert all(record["episode_id"] >= 1 for record in records)

    replay_env = LandingEnv()
    expected_observation, _ = replay_env.reset(seed=training.seed)
    previous_episode_id = records[0]["episode_id"]
    previous_done = False
    for record in records:
        expected_state = {
            "x_m": replay_env.state.x,
            "z_m": replay_env.state.z,
            "vx_mps": replay_env.state.vx,
            "vz_mps": replay_env.state.vz,
            "theta_rad": replay_env.state.theta,
            "omega_radps": replay_env.state.omega,
        }
        assert record["state_before"] == expected_state
        np.testing.assert_array_equal(record["observation"], expected_observation)
        if previous_done:
            assert record["episode_id"] == previous_episode_id + 1
        else:
            assert record["episode_id"] == previous_episode_id
        expected_observation, reward, terminated, truncated, info = replay_env.step(
            np.asarray(record["action_passed_to_environment"], dtype=np.float32)
        )
        assert record["reward"] == pytest.approx(reward, abs=1e-6)
        assert record["terminated"] is terminated
        assert record["truncated"] is truncated
        assert record["termination_reason"] == info["terminal_reason"]
        previous_episode_id = record["episode_id"]
        previous_done = terminated or truncated
        if previous_done:
            expected_observation, _ = replay_env.reset()
    replay_env.close()

