"""Deterministic headless evaluation of a saved PPO checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

import numpy as np
import torch
from stable_baselines3 import PPO

from vertex.environment import LandingEnv, LandingEnvConfig


DEFAULT_EVAL_SEEDS = tuple(range(1000, 1010))
_TERMINAL_REASONS = ("success", "unstable_touchdown", "out_of_bounds", "time_limit")


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _training_reward_trend(checkpoint_path: Path) -> dict[str, Any] | None:
    """Summarize SB3's recorded training episode-return trend when available."""
    log_path = checkpoint_path.parent / "logs" / "progress.csv"
    if not log_path.is_file():
        return None
    with log_path.open(newline="", encoding="utf-8") as stream:
        values = [
            float(row["rollout/ep_rew_mean"])
            for row in csv.DictReader(stream)
            if row.get("rollout/ep_rew_mean")
        ]
    if not values:
        return {"source": str(log_path), "updates_with_episode_data": 0}
    return {
        "source": str(log_path),
        "updates_with_episode_data": len(values),
        "first_recorded_mean_episode_return": values[0],
        "last_recorded_mean_episode_return": values[-1],
        "change": values[-1] - values[0],
    }


def _source_state() -> tuple[str | None, bool | None]:
    repository = Path(__file__).resolve().parents[3]
    safe_directory = repository.as_posix()
    revision = subprocess.run(
        ["git", "-c", f"safe.directory={safe_directory}", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "-c", f"safe.directory={safe_directory}", "status", "--porcelain"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if revision.returncode or status.returncode:
        return None, None
    return revision.stdout.strip(), bool(status.stdout.strip())


def _state_record(state: Any) -> dict[str, float]:
    """Serialize a planar state with units in each field name."""
    return {
        "x_m": state.x,
        "z_m": state.z,
        "vx_mps": state.vx,
        "vz_mps": state.vz,
        "theta_rad": state.theta,
        "omega_radps": state.omega,
    }


def _raw_deterministic_action(model: PPO, observation: np.ndarray) -> np.ndarray:
    """Get the policy distribution's deterministic output before Box clipping."""
    observation_tensor, _ = model.policy.obs_to_tensor(observation)
    with torch.no_grad():
        action = model.policy.get_distribution(observation_tensor).get_actions(
            deterministic=True
        )
    return action.detach().cpu().numpy().reshape(-1)


def evaluate_checkpoint(
    checkpoint: str | Path,
    output_dir: str | Path,
    seeds: Sequence[int] = DEFAULT_EVAL_SEEDS,
    environment: LandingEnvConfig | None = None,
    diagnostics: bool = False,
) -> dict[str, Any]:
    """Run one deterministic episode per seed and save compact reports/traces.

    The output directory must not already exist. This routine loads a policy on
    CPU and reports task terminal outcomes separately from episodic return.
    """
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"PPO checkpoint not found: {checkpoint_path}")
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must contain at least one non-negative integer")
    if len(set(seeds)) != len(seeds):
        raise ValueError("evaluation seeds must be unique")
    output_path = Path(output_dir)
    if output_path.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_path}. Choose a new --output-dir; existing evaluation data was preserved."
        )

    env_config = environment or LandingEnvConfig()
    env = LandingEnv(env_config)
    model = PPO.load(checkpoint_path, device="cpu")
    if (
        model.observation_space.shape != env.observation_space.shape
        or model.observation_space.dtype != env.observation_space.dtype
        or not np.array_equal(model.observation_space.low, env.observation_space.low)
        or not np.array_equal(model.observation_space.high, env.observation_space.high)
    ):
        raise ValueError("checkpoint observation space does not match LandingEnv")
    if (
        model.action_space.shape != env.action_space.shape
        or model.action_space.dtype != env.action_space.dtype
        or not np.array_equal(model.action_space.low, env.action_space.low)
        or not np.array_equal(model.action_space.high, env.action_space.high)
    ):
        raise ValueError("checkpoint action space does not match LandingEnv")

    output_path.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    diagnostic_path = output_path / "step_diagnostics.jsonl"
    diagnostic_stream = diagnostic_path.open("w", encoding="utf-8") if diagnostics else None
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            episode_return = 0.0
            trace: list[dict[str, Any]] = []
            while True:
                observation_before = observation.copy() if diagnostics else None
                state_before = env.state if diagnostics else None
                raw_action = (
                    _raw_deterministic_action(model, observation)
                    if diagnostics
                    else None
                )
                policy_action, _ = model.predict(observation, deterministic=True)
                applied_action, wrench = (
                    env.decode_action(policy_action) if diagnostics else (None, None)
                )
                observation, reward, terminated, truncated, info = env.step(policy_action)
                episode_return += float(reward)
                state = env.state
                step_number = int(info["steps"])
                trace.append(
                    {
                        "step": step_number,
                        "time_s": step_number * env_config.dt,
                        **_state_record(state),
                        "reward": float(reward),
                    }
                )
                if diagnostic_stream is not None:
                    components = env.reward_components(terminated, info["terminal_reason"])
                    diagnostic_record = {
                        "schema_version": 1,
                        "episode_seed": seed,
                        "step": step_number,
                        "time_s": step_number * env_config.dt,
                        "state_before": _state_record(state_before),
                        "observation": observation_before.tolist(),
                        "raw_deterministic_policy_action": raw_action.tolist(),
                        "action_applied": applied_action.tolist(),
                        "applied_wrench": {
                            "force_x_N": wrench.force_x,
                            "force_z_N": wrench.force_z,
                            "moment_Nm": wrench.moment,
                        },
                        "state_after": _state_record(state),
                        "reward_components": components,
                        "total_reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "termination_reason": info["terminal_reason"],
                    }
                    diagnostic_stream.write(
                        json.dumps(diagnostic_record, separators=(",", ":")) + "\n"
                    )
                if terminated or truncated:
                    break

            rows.append(
                {
                    "seed": seed,
                    "episode_return": episode_return,
                    "steps": int(info["steps"]),
                    "terminal_reason": info["terminal_reason"],
                    "success": bool(info["success"]),
                    **_state_record(env.state),
                }
            )
            traces.append(
                {
                    "seed": seed,
                    "target_x": env_config.target_x,
                    "terminal_reason": info["terminal_reason"],
                    "samples": trace,
                }
            )
    finally:
        env.close()
        if diagnostic_stream is not None:
            diagnostic_stream.close()

    source_revision, source_tree_dirty = _source_state()
    category_counts = {reason: 0 for reason in _TERMINAL_REASONS}
    for row in rows:
        reason = row["terminal_reason"]
        if reason not in category_counts:
            raise RuntimeError(f"unrecognized terminal reason: {reason}")
        category_counts[reason] += 1
    touchdowns = [row for row in rows if row["terminal_reason"] in ("success", "unstable_touchdown")]
    touchdown_metrics = None
    if touchdowns:
        touchdown_metrics = {
            "mean_abs_lateral_error_m": fmean(abs(row["x_m"] - env_config.target_x) for row in touchdowns),
            "mean_abs_horizontal_speed_mps": fmean(abs(row["vx_mps"]) for row in touchdowns),
            "mean_abs_vertical_speed_mps": fmean(abs(row["vz_mps"]) for row in touchdowns),
            "mean_abs_pitch_rad": fmean(abs(row["theta_rad"]) for row in touchdowns),
            "mean_abs_pitch_rate_radps": fmean(abs(row["omega_radps"]) for row in touchdowns),
            "sample_count": len(touchdowns),
        }

    report: dict[str, Any] = {
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path.resolve()),
        "algorithm": "PPO",
        "device": "cpu",
        "deterministic_actions": True,
        "python": platform.python_version(),
        "source_revision": source_revision,
        "source_tree_dirty": source_tree_dirty,
        "packages": {
            name: _package_version(name)
            for name in ("stable-baselines3", "torch", "gymnasium", "numpy")
        },
        "environment_config": {
            "dt": env_config.dt,
            "max_episode_steps": env_config.max_episode_steps,
            "target_x": env_config.target_x,
        },
        "seeds": list(seeds),
        "episode_count": len(rows),
        "success_count": category_counts["success"],
        "success_rate": category_counts["success"] / len(rows),
        "failure_categories": {
            reason: count for reason, count in category_counts.items() if reason != "success"
        },
        "terminal_category_counts": category_counts,
        "mean_episode_return": fmean(row["episode_return"] for row in rows),
        "mean_episode_steps": fmean(row["steps"] for row in rows),
        "training_reward_trend": _training_reward_trend(checkpoint_path),
        "touchdown_metrics": touchdown_metrics,
        "episodes_csv": "episodes.csv",
        "traces_jsonl": "episode_traces.jsonl",
        "interpretation": "Evaluation of this educational toy model only; no physical safety or flight-readiness claim.",
    }
    if diagnostics:
        report["step_diagnostics_jsonl"] = diagnostic_path.name
    (output_path / "evaluation.json").write_text(
        json.dumps({"summary": report, "episodes": rows}, indent=2) + "\n", encoding="utf-8"
    )
    with (output_path / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_path / "episode_traces.jsonl").open("w", encoding="utf-8") as stream:
        for trace in traces:
            stream.write(json.dumps(trace, separators=(",", ":")) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_EVAL_SEEDS))
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="write per-step state, observation, action, wrench, reward, and terminal diagnostics",
    )
    args = parser.parse_args()
    try:
        report = evaluate_checkpoint(
            args.checkpoint, args.output_dir, args.seeds, diagnostics=args.diagnostics
        )
    except FileExistsError as error:
        parser.error(str(error))
    print(f"Evaluated {report['episode_count']} episodes; success rate {report['success_rate']:.1%}")
    print(f"Failure categories: {report['failure_categories']}")
    print(f"Wrote evaluation artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
