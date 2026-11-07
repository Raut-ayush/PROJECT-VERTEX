"""Fixed scripted physics cases; these do not exercise or score an RL policy."""

import math

import pytest

from vertex.simulation import AppliedWrench, DynamicsConfig, PlanarState, step


def test_scripted_free_fall_scenario():
    state = PlanarState(z=2.0, vz=-1.0)
    result = step(state, AppliedWrench(), DynamicsConfig(gravity=9.81), 0.1)
    assert result.z == pytest.approx(1.8019)
    assert result.vz == pytest.approx(-1.981)


def test_scripted_body_x_force_rotates_to_positive_world_z():
    state = PlanarState(z=3.0, theta=math.pi / 2)
    result = step(
        state,
        AppliedWrench(force_x=2.0),
        DynamicsConfig(mass=2.0, gravity=0.0),
        0.5,
    )
    assert result.vx == pytest.approx(0.0, abs=1e-12)
    assert result.vz == pytest.approx(0.5)
    assert result.z == pytest.approx(3.25)


def test_scripted_positive_moment_increases_pitch_and_rate():
    result = step(
        PlanarState(),
        AppliedWrench(moment=2.0),
        DynamicsConfig(moment_of_inertia=2.0, gravity=0.0),
        0.25,
    )
    assert result.omega == pytest.approx(0.25)
    assert result.theta == pytest.approx(0.0625)
