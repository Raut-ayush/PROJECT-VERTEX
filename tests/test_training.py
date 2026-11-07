import json

import numpy as np
from stable_baselines3 import PPO

from vertex.learning.train import TrainingConfig, train_ppo


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

