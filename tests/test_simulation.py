import pytest

from vertex.simulation import AppliedWrench, DynamicsConfig, PlanarState, step


def test_free_fall_accelerates_downward_and_is_deterministic():
    initial = PlanarState(z=10.0)
    expected = PlanarState(z=9.9019, vz=-0.981)
    first = step(initial, AppliedWrench(), DynamicsConfig(), 0.1)
    second = step(initial, AppliedWrench(), DynamicsConfig(), 0.1)
    assert first == second
    assert first.z == pytest.approx(expected.z)
    assert first.vz == pytest.approx(expected.vz)


def test_body_x_force_rotates_into_vertical_at_quarter_turn():
    result = step(
        PlanarState(theta=1.5707963267948966),
        AppliedWrench(force_x=2.0),
        DynamicsConfig(mass=2.0, gravity=0.0),
        0.5,
    )
    assert result.vx == pytest.approx(0.0, abs=1e-12)
    assert result.vz == pytest.approx(0.5)


def test_positive_moment_increases_positive_pitch():
    result = step(PlanarState(), AppliedWrench(moment=2.0), DynamicsConfig(moment_of_inertia=2.0, gravity=0.0), 0.25)
    assert result.omega == pytest.approx(0.25)
    assert result.theta == pytest.approx(0.0625)


def test_one_step_uses_updated_rates_for_position_and_angle():
    initial = PlanarState(x=1.0, z=5.0, vx=2.0, vz=-1.0, theta=0.2, omega=0.3)
    result = step(
        initial,
        AppliedWrench(moment=2.0),
        DynamicsConfig(mass=1.0, moment_of_inertia=2.0, gravity=10.0),
        0.1,
    )
    assert result.x == pytest.approx(1.2)
    assert result.z == pytest.approx(4.8)
    assert result.vx == pytest.approx(2.0)
    assert result.vz == pytest.approx(-2.0)
    assert result.theta == pytest.approx(0.24)
    assert result.omega == pytest.approx(0.4)


@pytest.mark.parametrize("dt", [0.0, -0.1])
def test_non_positive_time_step_is_rejected(dt):
    with pytest.raises(ValueError, match="dt must be positive"):
        step(PlanarState(), AppliedWrench(), DynamicsConfig(), dt)


def test_invalid_mass_is_rejected():
    with pytest.raises(ValueError, match="mass must be positive"):
        DynamicsConfig(mass=0.0)

