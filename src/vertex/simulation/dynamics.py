"""Small deterministic planar rigid-body model using forward Euler integration."""

from dataclasses import dataclass
from math import cos, sin

from vertex.simulation.state import PlanarState


@dataclass(frozen=True, slots=True)
class DynamicsConfig:
    """Educational model parameters: kg, kg m², m/s², and seconds."""

    mass: float = 1.0
    moment_of_inertia: float = 1.0
    gravity: float = 9.81

    def __post_init__(self) -> None:
        if self.mass <= 0.0:
            raise ValueError("mass must be positive")
        if self.moment_of_inertia <= 0.0:
            raise ValueError("moment_of_inertia must be positive")
        if self.gravity < 0.0:
            raise ValueError("gravity must be non-negative")


@dataclass(frozen=True, slots=True)
class AppliedWrench:
    """Generic simulation-only force (N) and moment (N m), expressed in body axes."""

    force_x: float = 0.0
    force_z: float = 0.0
    moment: float = 0.0


def step(
    state: PlanarState,
    wrench: AppliedWrench,
    config: DynamicsConfig,
    dt: float,
) -> PlanarState:
    """Advance one step; positive z is up and positive pitch rotates +x toward +z.

    Uses explicit forward Euler: update rates from current accelerations, then
    positions and angles from the updated rates. Forces are resolved in body axes.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    c, s = cos(state.theta), sin(state.theta)
    ax = (c * wrench.force_x - s * wrench.force_z) / config.mass
    az = (s * wrench.force_x + c * wrench.force_z) / config.mass - config.gravity
    alpha = wrench.moment / config.moment_of_inertia

    vx = state.vx + ax * dt
    vz = state.vz + az * dt
    omega = state.omega + alpha * dt
    return PlanarState(
        x=state.x + vx * dt,
        z=state.z + vz * dt,
        vx=vx,
        vz=vz,
        theta=state.theta + omega * dt,
        omega=omega,
    )

