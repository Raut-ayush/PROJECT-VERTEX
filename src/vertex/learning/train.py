"""Headless PPO training entry point for the educational landing task."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from vertex.environment import LandingEnv, LandingEnvConfig

DEFAULT_TOTAL_TIMESTEPS = 10_000
DEFAULT_SEED = 7
DEFAULT_DIAGNOSTIC_STEPS = 2_048


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Conservative single-environment PPO settings for M3 integration."""

    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS
    seed: int = DEFAULT_SEED
    n_steps: int = 128
    batch_size: int = 64
    n_epochs: int = 5
    hidden_layers: tuple[int, ...] = (64, 64)
    gamma: float = 0.99
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if self.n_steps <= 1 or self.batch_size <= 0 or self.n_epochs <= 0:
            raise ValueError("n_steps, batch_size, and n_epochs must be positive (n_steps > 1)")
        if self.batch_size > self.n_steps:
            raise ValueError("batch_size must not exceed n_steps")
        if any(width <= 0 for width in self.hidden_layers):
            raise ValueError("hidden layer widths must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1 inclusive")
        if self.device != "cpu":
            raise ValueError("M3 baseline is CPU-only; benchmark before enabling GPU")


class TrainingDiagnosticsCallback(BaseCallback):
    """Optionally record sampled PPO transitions without changing collection."""

    def __init__(self, output_path: Path, diagnostic_steps: int) -> None:
        super().__init__(verbose=0)
        if diagnostic_steps <= 0:
            raise ValueError("diagnostic_steps must be positive")
        self.output_path = output_path
        self.diagnostic_steps = diagnostic_steps
        self._stream: Any = None
        self._states_before: list[Any] = []
        self._episode_ids: list[int] = []
        self._next_episode_id = 1

    def _on_training_start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.output_path.open("w", encoding="utf-8")
        env_count = self.training_env.num_envs
        self._episode_ids = list(range(self._next_episode_id, self._next_episode_id + env_count))
        self._next_episode_id += env_count
        self._states_before = self.training_env.get_attr("state")

    @staticmethod
    def _state_record(state: Any) -> dict[str, float]:
        return {
            "x_m": float(state.x),
            "z_m": float(state.z),
            "vx_mps": float(state.vx),
            "vz_mps": float(state.vz),
            "theta_rad": float(state.theta),
            "omega_radps": float(state.omega),
        }

    def _on_step(self) -> bool:
        # SB3 2.9.2 calls callbacks after env.step() but before rollout_buffer.add().
        # Read the collector locals without replacing or mutating its sampled actions.
        if self.num_timesteps > self.diagnostic_steps:
            return True

        local = self.locals
        sampled_actions = np.asarray(local["actions"])
        applied_actions = np.asarray(local["clipped_actions"])
        observations = local["obs_tensor"]
        with torch.no_grad():
            action_distribution = self.model.policy.get_distribution(observations).distribution
            policy_means = action_distribution.mean.detach().cpu().numpy()
            policy_stds = action_distribution.stddev.detach().cpu().numpy()

        rewards = np.asarray(local["rewards"]).reshape(-1)
        dones = np.asarray(local["dones"]).reshape(-1)
        infos = local["infos"]
        bounds_low = np.asarray(self.training_env.action_space.low)
        bounds_high = np.asarray(self.training_env.action_space.high)
        observation_values = observations.detach().cpu().numpy()

        for env_index, (sampled, applied, reward, done, info) in enumerate(
            zip(sampled_actions, applied_actions, rewards, dones, infos, strict=True)
        ):
            clipped = np.logical_or(sampled < bounds_low, sampled > bounds_high)
            truncated = bool(done and info.get("TimeLimit.truncated", False))
            terminated = bool(done and not truncated)
            record = {
                "schema_version": 1,
                "global_timestep": int(self.num_timesteps - len(dones) + env_index + 1),
                "rollout_index": int((self.num_timesteps - len(dones)) // self.model.n_steps + 1),
                "env_index": env_index,
                "episode_id": self._episode_ids[env_index],
                "observation": observation_values[env_index].tolist(),
                "state_before": self._state_record(self._states_before[env_index]),
                "sampled_action_before_clipping": sampled.tolist(),
                "action_passed_to_environment": applied.tolist(),
                "action_clipped_by_channel": clipped.tolist(),
                "policy_mean": policy_means[env_index].tolist(),
                "policy_standard_deviation": policy_stds[env_index].tolist(),
                "reward": float(reward),
                "terminated": terminated,
                "truncated": truncated,
                "termination_reason": info.get("terminal_reason"),
            }
            self._stream.write(json.dumps(record, separators=(",", ":")) + "\n")

        # DummyVecEnv has already reset completed episodes. This read only seeds
        # the next transition's pre-action physical state and consumes no RNG.
        self._states_before = self.training_env.get_attr("state")
        for env_index, done in enumerate(dones):
            if done:
                self._episode_ids[env_index] = self._next_episode_id
                self._next_episode_id += 1
        return True

    def _on_training_end(self) -> None:
        self.close()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _source_state() -> tuple[str | None, bool | None]:
    repository = Path(__file__).resolve().parents[3]
    git_safe_directory = repository.as_posix()
    revision = subprocess.run(
        ["git", "-c", f"safe.directory={git_safe_directory}", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "-c", f"safe.directory={git_safe_directory}", "status", "--porcelain"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if revision.returncode or status.returncode:
        return None, None
    return revision.stdout.strip(), bool(status.stdout.strip())


def train_ppo(
    output_dir: str | Path,
    training: TrainingConfig | None = None,
    environment: LandingEnvConfig | None = None,
    *,
    diagnostics: bool = False,
    diagnostic_steps: int = DEFAULT_DIAGNOSTIC_STEPS,
) -> dict[str, Any]:
    """Train PPO once, writing a checkpoint, CSV log, and run metadata.

    The output directory must not already exist so previous run artifacts are
    never silently overwritten. GPU use is deliberately disabled for M3.
    """
    training = training or TrainingConfig()
    environment = environment or LandingEnvConfig()
    if diagnostic_steps <= 0:
        raise ValueError("diagnostic_steps must be positive")
    output_path = Path(output_dir)
    if output_path.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_path}. Choose a new --output-dir; existing run data was preserved."
        )
    output_path.mkdir(parents=True, exist_ok=False)
    log_dir = output_path / "logs"
    log_dir.mkdir()

    env = LandingEnv(environment)
    model = PPO(
        "MlpPolicy",
        env,
        seed=training.seed,
        device="cpu",
        n_steps=training.n_steps,
        batch_size=training.batch_size,
        n_epochs=training.n_epochs,
        gamma=training.gamma,
        policy_kwargs={"net_arch": list(training.hidden_layers)},
        verbose=1,
    )
    model.set_logger(configure(str(log_dir), ["stdout", "csv"]))
    diagnostics_callback = (
        TrainingDiagnosticsCallback(output_path / "training_diagnostics.jsonl", diagnostic_steps)
        if diagnostics
        else None
    )
    run_started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    try:
        model.learn(
            total_timesteps=training.total_timesteps,
            progress_bar=False,
            callback=diagnostics_callback,
        )
        elapsed_seconds = time.perf_counter() - started
        checkpoint_base = output_path / "ppo_vertex"
        model.save(str(checkpoint_base))
    finally:
        if diagnostics_callback is not None:
            diagnostics_callback.close()
        env.close()

    actual_timesteps = int(model.num_timesteps)
    source_revision, source_tree_dirty = _source_state()
    metadata: dict[str, Any] = {
        "run_started_utc": run_started_utc,
        "source_revision": source_revision,
        "source_tree_dirty": source_tree_dirty,
        "python": platform.python_version(),
        "packages": {
            "project-vertex": _package_version("project-vertex"),
            "stable-baselines3": _package_version("stable-baselines3"),
            "torch": _package_version("torch"),
            "gymnasium": _package_version("gymnasium"),
            "numpy": _package_version("numpy"),
        },
        "algorithm": "PPO",
        "device": "cpu",
        "seed": training.seed,
        "training_config": asdict(training),
        "environment_config": asdict(environment),
        "requested_timesteps": training.total_timesteps,
        "actual_timesteps": actual_timesteps,
        "elapsed_seconds": elapsed_seconds,
        "steps_per_second": actual_timesteps / elapsed_seconds if elapsed_seconds else None,
        "checkpoint": "ppo_vertex.zip",
        "training_log": "logs/progress.csv",
        "learning_result_claim": "integration smoke only; no learning-success claim",
    }
    if diagnostics:
        metadata["training_diagnostics"] = {
            "enabled": True,
            "path": "training_diagnostics.jsonl",
            "recorded_environment_steps": min(actual_timesteps, diagnostic_steps),
            "requested_window_environment_steps": diagnostic_steps,
            "method": "SB3 callback reads collector locals after env.step and before rollout_buffer.add",
        }
    (output_path / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--diagnostic-steps", type=int, default=DEFAULT_DIAGNOSTIC_STEPS)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or Path("runs") / run_id
    try:
        metadata = train_ppo(
            output_dir,
            TrainingConfig(total_timesteps=args.timesteps, seed=args.seed, gamma=args.gamma),
            diagnostics=args.diagnostics,
            diagnostic_steps=args.diagnostic_steps,
        )
    except FileExistsError as error:
        parser.error(str(error))
    print(f"Saved {metadata['checkpoint']} and metadata to {output_dir}")
    print(f"Throughput: {metadata['steps_per_second']:.1f} steps/s on CPU")


if __name__ == "__main__":
    main()
