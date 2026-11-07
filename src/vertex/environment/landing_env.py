"""A small headless landing task around the independent physics model."""

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from vertex.simulation import AppliedWrench, DynamicsConfig, PlanarState, step


@dataclass(frozen=True, slots=True)
class LandingEnvConfig:
    """Educational task parameters, not specifications for a real vehicle."""

    dt: float = 0.02
    max_episode_steps: int = 500
    mass: float = 1.0
    moment_of_inertia: float = 1.0
    gravity: float = 9.81
    force_scale: float = 20.0
    moment_scale: float = 2.0
    target_x: float = 0.0
    initial_x_range: tuple[float, float] = (-1.0, 1.0)
    initial_z_range: tuple[float, float] = (8.0, 10.0)
    initial_vx_range: tuple[float, float] = (-0.5, 0.5)
    initial_vz_range: tuple[float, float] = (-0.5, 0.0)
    initial_theta_range: tuple[float, float] = (-0.1, 0.1)
    initial_omega_range: tuple[float, float] = (-0.05, 0.05)
    x_limit: float = 10.0
    z_limit: float = 20.0

    def __post_init__(self) -> None:
        if self.dt <= 0.0 or self.max_episode_steps <= 0:
            raise ValueError("dt and max_episode_steps must be positive")
        DynamicsConfig(self.mass, self.moment_of_inertia, self.gravity)
        if self.force_scale <= 0.0 or self.moment_scale <= 0.0:
            raise ValueError("force_scale and moment_scale must be positive")
        if self.x_limit <= 0.0 or self.z_limit <= 0.0:
            raise ValueError("position limits must be positive")
        for name in (
            "initial_x_range", "initial_z_range", "initial_vx_range",
            "initial_vz_range", "initial_theta_range", "initial_omega_range",
        ):
            low, high = getattr(self, name)
            if low > high:
                raise ValueError(f"{name} lower bound must not exceed upper bound")


class LandingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Abstract-control landing task; no rendering or hardware interface."""

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, config: LandingEnvConfig | None = None) -> None:
        super().__init__()
        self.config = config or LandingEnvConfig()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self._dynamics = DynamicsConfig(
            mass=self.config.mass,
            moment_of_inertia=self.config.moment_of_inertia,
            gravity=self.config.gravity,
        )
        self.state = PlanarState(z=9.0)
        self._steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options:
            unknown = set(options) - {"initial_state"}
            if unknown:
                raise ValueError(f"unsupported reset options: {sorted(unknown)}")
        initial_state = (options or {}).get("initial_state")
        if initial_state is not None:
            if not isinstance(initial_state, PlanarState):
                raise TypeError("initial_state option must be a PlanarState")
            self.state = initial_state
        else:
            rng = self.np_random
            cfg = self.config
            self.state = PlanarState(
                x=float(rng.uniform(*cfg.initial_x_range)),
                z=float(rng.uniform(*cfg.initial_z_range)),
                vx=float(rng.uniform(*cfg.initial_vx_range)),
                vz=float(rng.uniform(*cfg.initial_vz_range)),
                theta=float(rng.uniform(*cfg.initial_theta_range)),
                omega=float(rng.uniform(*cfg.initial_omega_range)),
            )
        self._steps = 0
        return self._observation(), self._info(terminal_reason=None)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_array = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action_array):
            raise ValueError("action must be a finite length-3 vector in [-1, 1]")

        cfg = self.config
        self.state = step(
            self.state,
            AppliedWrench(
                force_x=float(action_array[0]) * cfg.force_scale,
                force_z=float(action_array[1]) * cfg.force_scale,
                moment=float(action_array[2]) * cfg.moment_scale,
            ),
            self._dynamics,
            cfg.dt,
        )
        self._steps += 1
        terminated, reason = self._terminal_status()
        truncated = not terminated and self._steps >= cfg.max_episode_steps
        if truncated:
            reason = "time_limit"
        reward = self._reward(terminated, reason)
        return self._observation(), reward, terminated, truncated, self._info(reason)

    def decode_action(self, action: np.ndarray) -> tuple[np.ndarray, AppliedWrench]:
        """Validate and convert a normalized simulation action to its body wrench.

        The returned float32 action is the exact bounded vector consumed by the
        environment. The wrench is a generic simulation input, not a hardware
        or propulsion command.
        """
        action_array = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action_array):
            raise ValueError("action must be a finite length-3 vector in [-1, 1]")
        cfg = self.config
        wrench = AppliedWrench(
            force_x=float(action_array[0]) * cfg.force_scale,
            force_z=float(action_array[1]) * cfg.force_scale,
            moment=float(action_array[2]) * cfg.moment_scale,
        )
        return action_array, wrench

    def _observation(self) -> np.ndarray:
        """Return clipped normalized target-relative state; angles use sin/cos."""
        s, c = self.state, self.config
        values = np.array(
            [
                (s.x - c.target_x) / c.x_limit,
                s.z / c.z_limit,
                s.vx / 5.0,
                s.vz / 10.0,
                np.sin(s.theta),
                np.cos(s.theta),
                s.omega / 2.0,
                1.0 - self._steps / c.max_episode_steps,
            ],
            dtype=np.float32,
        )
        return np.clip(values, -1.0, 1.0).astype(np.float32, copy=False)

    def _terminal_status(self) -> tuple[bool, str | None]:
        s, c = self.state, self.config
        if abs(s.x - c.target_x) > c.x_limit or s.z > c.z_limit:
            return True, "out_of_bounds"
        if s.z <= 0.0:
            success = (
                abs(s.x - c.target_x) <= 0.5
                and abs(s.vx) <= 0.5
                and abs(s.vz) <= 0.5
                and abs(s.theta) <= 0.15
                and abs(s.omega) <= 0.25
            )
            return True, "success" if success else "unstable_touchdown"
        return False, None

    def _reward(self, terminated: bool, reason: str | None) -> float:
        s, c = self.state, self.config
        shaping = -0.01 * (
            abs(s.x - c.target_x) / c.x_limit
            + s.z / c.z_limit
            + abs(s.vx) / 5.0
            + abs(s.vz) / 10.0
            + abs(s.theta)
            + abs(s.omega) / 2.0
        )
        if terminated:
            return shaping + (10.0 if reason == "success" else -10.0)
        return shaping

    def reward_components(self, terminated: bool, reason: str | None) -> dict[str, float]:
        """Return individually weighted reward terms for evaluation diagnostics."""
        s, c = self.state, self.config
        terminal_adjustment = (
            10.0 if reason == "success" else -10.0 if terminated else 0.0
        )
        components = {
            "horizontal_position_penalty": -0.01 * abs(s.x - c.target_x) / c.x_limit,
            "height_penalty": -0.01 * s.z / c.z_limit,
            "horizontal_velocity_penalty": -0.01 * abs(s.vx) / 5.0,
            "vertical_velocity_penalty": -0.01 * abs(s.vz) / 10.0,
            "pitch_penalty": -0.01 * abs(s.theta),
            "pitch_rate_penalty": -0.01 * abs(s.omega) / 2.0,
            "terminal_adjustment": terminal_adjustment,
        }
        components["state_shaping_total"] = sum(
            value for key, value in components.items() if key != "terminal_adjustment"
        )
        components["total_reward"] = self._reward(terminated, reason)
        return components

    def _info(self, terminal_reason: str | None) -> dict[str, Any]:
        return {
            "terminal_reason": terminal_reason,
            "success": terminal_reason == "success",
            "steps": self._steps,
        }
