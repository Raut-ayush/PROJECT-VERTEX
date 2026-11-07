import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from vertex.environment import LandingEnv, LandingEnvConfig
from vertex.simulation import PlanarState


def test_gymnasium_checker_accepts_environment():
    check_env(LandingEnv(), skip_render_check=True)


def test_reset_is_repeatable_for_same_seed_and_observation_is_valid():
    env = LandingEnv()
    observation_a, info_a = env.reset(seed=123)
    observation_b, info_b = env.reset(seed=123)
    np.testing.assert_array_equal(observation_a, observation_b)
    assert info_a["steps"] == info_b["steps"] == 0
    assert observation_a.dtype == np.float32
    assert env.observation_space.contains(observation_a)


def test_step_returns_gymnasium_values_and_uses_bounded_abstract_actions():
    env = LandingEnv()
    env.reset(seed=4)
    observation, reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
    assert env.observation_space.contains(observation)
    assert isinstance(reward, float)
    assert not terminated
    assert not truncated
    assert info["terminal_reason"] is None
    with pytest.raises(ValueError, match="action must be"):
        env.step(np.array([2.0, 0.0, 0.0], dtype=np.float32))


def test_controlled_touchdown_is_terminal_success():
    env = LandingEnv()
    env.reset(seed=1, options={"initial_state": PlanarState(z=0.01, vz=-0.1)})
    _, _, terminated, _, _ = env.step(np.zeros(3, dtype=np.float32))
    assert not terminated
    _, reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
    assert terminated and not truncated
    assert reward > 0.0
    assert info["terminal_reason"] == "success"
    assert info["success"] is True


def test_unstable_touchdown_is_failure():
    env = LandingEnv()
    env.reset(seed=1, options={"initial_state": PlanarState(z=0.01, vz=-2.0)})
    _, reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
    assert terminated and not truncated
    assert reward < 0.0
    assert info["terminal_reason"] == "unstable_touchdown"


def test_episode_limit_truncates_without_claiming_success():
    env = LandingEnv(LandingEnvConfig(max_episode_steps=1))
    env.reset(seed=2)
    _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
    assert not terminated and truncated
    assert info["terminal_reason"] == "time_limit"

