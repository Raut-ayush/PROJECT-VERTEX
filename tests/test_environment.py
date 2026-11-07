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


def test_observation_indices_scaling_and_float32_for_known_state():
    env = LandingEnv(LandingEnvConfig(target_x=2.5))
    state = PlanarState(
        x=3.5, z=5.0, vx=-2.5, vz=5.0, theta=np.pi / 6, omega=-1.0
    )
    observation, _ = env.reset(seed=3, options={"initial_state": state})

    assert observation.dtype == np.float32
    np.testing.assert_allclose(
        observation,
        [0.1, 0.25, -0.5, 0.5, 0.5, np.sqrt(3) / 2, -0.5, 1.0],
        rtol=1e-6,
        atol=1e-7,
    )


def test_observation_clips_each_scaled_state_component():
    env = LandingEnv()
    state = PlanarState(x=20.0, z=30.0, vx=-6.0, vz=11.0, theta=np.pi, omega=-3.0)
    observation, _ = env.reset(seed=3, options={"initial_state": state})
    np.testing.assert_allclose(observation, [1.0, 1.0, -1.0, 1.0, 0.0, -1.0, -1.0, 1.0], atol=1e-6)

    env.state = PlanarState(x=-20.0, z=-1.0, vx=6.0, vz=-11.0, theta=0.0, omega=3.0)
    clipped = env._observation()
    np.testing.assert_allclose(clipped[:7], [-1.0, -0.05, 1.0, -1.0, 0.0, 1.0, 1.0], atol=1e-6)


@pytest.mark.parametrize(
    "action",
    [
        np.array([0.0, 0.0], dtype=np.float32),
        np.zeros((3, 1), dtype=np.float32),
        np.array([1.01, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, np.nan, 0.0], dtype=np.float32),
    ],
)
def test_action_interface_rejects_invalid_shape_range_and_nan(action):
    env = LandingEnv()
    assert env.action_space.shape == (3,)
    np.testing.assert_array_equal(env.action_space.low, [-1.0, -1.0, -1.0])
    np.testing.assert_array_equal(env.action_space.high, [1.0, 1.0, 1.0])
    env.reset(seed=5)
    with pytest.raises(ValueError, match="finite length-3 vector"):
        env.step(action)


@pytest.mark.parametrize(
    "action, expected",
    [
        ([1, 0, 0], (20.0, 0.0, 0.0)),
        ([-1, 0, 0], (-20.0, 0.0, 0.0)),
        ([0, 1, 0], (0.0, 20.0, 0.0)),
        ([0, -1, 0], (0.0, -20.0, 0.0)),
        ([0, 0, 1], (0.0, 0.0, 2.0)),
        ([0, 0, -1], (0.0, 0.0, -2.0)),
    ],
)
def test_action_channels_decode_to_documented_simulation_wrench(action, expected):
    env = LandingEnv()
    applied, wrench = env.decode_action(np.asarray(action, dtype=np.float64))
    assert applied.dtype == np.float32
    np.testing.assert_allclose(
        [wrench.force_x, wrench.force_z, wrench.moment], expected
    )


@pytest.mark.parametrize(
    "action, expected_state",
    [
        ([1, 0, 0], (0.1, 5.0, 1.0, 0.0, 0.0, 0.0)),
        ([-1, 0, 0], (-0.1, 5.0, -1.0, 0.0, 0.0, 0.0)),
        ([0, 1, 0], (0.0, 5.1, 0.0, 1.0, 0.0, 0.0)),
        ([0, -1, 0], (0.0, 4.9, 0.0, -1.0, 0.0, 0.0)),
        ([0, 0, 1], (0.0, 5.0, 0.0, 0.0, 0.005, 0.05)),
        ([0, 0, -1], (0.0, 5.0, 0.0, 0.0, -0.005, -0.05)),
    ],
)
def test_action_channels_produce_documented_state_response(action, expected_state):
    config = LandingEnvConfig(
        dt=0.1, mass=2.0, moment_of_inertia=4.0, gravity=0.0
    )
    env = LandingEnv(config)
    env.reset(seed=5, options={"initial_state": PlanarState(z=5.0)})
    env.step(np.asarray(action, dtype=np.float32))
    state = env.state
    np.testing.assert_allclose(
        [state.x, state.z, state.vx, state.vz, state.theta, state.omega],
        expected_state,
        atol=1e-7,
    )


def test_reward_components_match_each_documented_state_penalty():
    env = LandingEnv()
    env.reset(
        seed=1,
        options={
            "initial_state": PlanarState(
                x=2.0, z=4.0, vx=-1.0, vz=-2.0, theta=-0.5, omega=0.8
            )
        },
    )
    components = env.reward_components(False, None)
    expected = {
        "horizontal_position_penalty": -0.002,
        "height_penalty": -0.002,
        "horizontal_velocity_penalty": -0.002,
        "vertical_velocity_penalty": -0.002,
        "pitch_penalty": -0.005,
        "pitch_rate_penalty": -0.004,
        "terminal_adjustment": 0.0,
        "state_shaping_total": -0.017,
        "total_reward": -0.017,
    }
    for name, value in expected.items():
        assert components[name] == pytest.approx(value)


def test_success_bonus_and_failure_penalty_are_reported_without_changing_reward():
    env = LandingEnv()
    env.reset(seed=1, options={"initial_state": PlanarState(z=0.001, vz=-0.1)})
    _, success_reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
    assert terminated and not truncated
    success_components = env.reward_components(terminated, info["terminal_reason"])
    assert success_components["terminal_adjustment"] == 10.0
    assert success_components["total_reward"] == pytest.approx(success_reward)

    env.reset(seed=1, options={"initial_state": PlanarState(z=0.01, vz=-2.0)})
    _, failure_reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
    assert terminated and not truncated
    failure_components = env.reward_components(terminated, info["terminal_reason"])
    assert failure_components["terminal_adjustment"] == -10.0
    assert failure_components["total_reward"] == pytest.approx(failure_reward)


def test_success_tolerances_accept_exact_threshold_values():
    env = LandingEnv()
    env.reset(
        seed=1,
        options={
            "initial_state": PlanarState(
                x=0.5, z=0.0, vx=0.5, vz=0.5, theta=0.15, omega=0.25
            )
        },
    )
    assert env._terminal_status() == (True, "success")


@pytest.mark.parametrize(
    "state",
    [
        PlanarState(x=0.5001, z=0.0),
        PlanarState(z=0.0, vx=0.5001),
        PlanarState(z=0.0, vz=0.5001),
        PlanarState(z=0.0, theta=0.1501),
        PlanarState(z=0.0, omega=0.2501),
    ],
)
def test_touchdown_just_beyond_any_success_tolerance_is_unstable(state):
    env = LandingEnv()
    env.reset(seed=1, options={"initial_state": state})
    assert env._terminal_status() == (True, "unstable_touchdown")


@pytest.mark.parametrize(
    "state, expected_reason",
    [
        (PlanarState(x=10.0, z=5.0), None),
        (PlanarState(x=10.0001, z=5.0), "out_of_bounds"),
        (PlanarState(x=-10.0, z=5.0), None),
        (PlanarState(x=-10.0001, z=5.0), "out_of_bounds"),
        (PlanarState(z=20.0), None),
        (PlanarState(z=20.0001), "out_of_bounds"),
    ],
)
def test_horizontal_and_upper_boundaries_are_strict(state, expected_reason):
    env = LandingEnv()
    env.reset(seed=9, options={"initial_state": state})
    assert env._terminal_status() == (expected_reason is not None, expected_reason)
