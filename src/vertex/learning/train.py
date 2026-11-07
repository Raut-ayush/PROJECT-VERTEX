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

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from vertex.environment import LandingEnv, LandingEnvConfig


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Conservative single-environment PPO settings for M3 integration."""

    total_timesteps: int = 10_000
    seed: int = 7
    n_steps: int = 128
    batch_size: int = 64
    n_epochs: int = 5
    hidden_layers: tuple[int, ...] = (64, 64)
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
        if self.device != "cpu":
            raise ValueError("M3 baseline is CPU-only; benchmark before enabling GPU")


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
) -> dict[str, Any]:
    """Train PPO once, writing a checkpoint, CSV log, and run metadata.

    The output directory must not already exist so previous run artifacts are
    never silently overwritten. GPU use is deliberately disabled for M3.
    """
    training = training or TrainingConfig()
    environment = environment or LandingEnvConfig()
    output_path = Path(output_dir)
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
        policy_kwargs={"net_arch": list(training.hidden_layers)},
        verbose=1,
    )
    model.set_logger(configure(str(log_dir), ["stdout", "csv"]))
    run_started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    try:
        model.learn(total_timesteps=training.total_timesteps, progress_bar=False)
        elapsed_seconds = time.perf_counter() - started
        checkpoint_base = output_path / "ppo_vertex"
        model.save(str(checkpoint_base))
    finally:
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
    (output_path / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=TrainingConfig.total_timesteps)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or Path("runs") / run_id
    metadata = train_ppo(
        output_dir,
        TrainingConfig(total_timesteps=args.timesteps, seed=args.seed),
    )
    print(f"Saved {metadata['checkpoint']} and metadata to {output_dir}")
    print(f"Throughput: {metadata['steps_per_second']:.1f} steps/s on CPU")


if __name__ == "__main__":
    main()
