"""State for a two-dimensional planar rigid body, in SI units."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlanarState:
    """Position (m), velocity (m/s), pitch (rad), and pitch rate (rad/s)."""

    x: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vz: float = 0.0
    theta: float = 0.0
    omega: float = 0.0

