"""Optional animated or static viewer for saved M4 episode traces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_trace(path: str | Path, seed: int | None = None) -> dict[str, Any]:
    """Load the first episode trace, or the trace matching ``seed``."""
    traces = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON on line {line_number}: {error}") from error
    for trace in traces:
        if seed is None or trace.get("seed") == seed:
            if not trace.get("samples"):
                raise ValueError(f"episode seed {trace.get('seed')} has no samples")
            _add_report_metadata(Path(path), trace)
            return trace
    if seed is None:
        raise ValueError(f"no episode traces found in {path}")
    raise ValueError(f"no episode with seed {seed} found in {path}")


def _add_report_metadata(trace_path: Path, trace: dict[str, Any]) -> None:
    """Backfill outcome/target metadata for traces created before M5."""
    if trace.get("terminal_reason"):
        return
    report_path = trace_path.with_name("evaluation.json")
    if not report_path.is_file():
        return
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    summary = report.get("summary", {})
    trace.setdefault("target_x", summary.get("environment_config", {}).get("target_x", 0.0))
    for episode in report.get("episodes", []):
        if episode.get("seed") == trace.get("seed"):
            trace["terminal_reason"] = episode.get("terminal_reason", "trace")
            return


def _make_figure(trace: dict[str, Any]):
    """Create a trajectory plot; imported lazily to keep the core headless."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Matplotlib is optional; install it with `python -m pip install -e \".[visualization]\"`."
        ) from error

    samples = trace["samples"]
    target_x = float(trace.get("target_x", 0.0))
    x_values = [float(row["x_m"]) for row in samples]
    z_values = [float(row["z_m"]) for row in samples]
    half_width = max((max(x_values) - min(x_values)) * 0.08, 1.0)
    x_min = min(min(x_values), target_x - 0.5) - half_width
    x_max = max(max(x_values), target_x + 0.5) + half_width
    z_max = max(max(z_values) * 1.08, 1.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhspan(-0.15, 0.0, color="saddlebrown", alpha=0.25)
    ax.axhline(0.0, color="black", linewidth=1.2)
    ax.axvspan(target_x - 0.5, target_x + 0.5, color="green", alpha=0.15, label="toy target band")
    trajectory, = ax.plot([], [], color="tab:blue", linewidth=1.4, label="trajectory")
    body, = ax.plot([], [], color="tab:red", linewidth=5, solid_capstyle="round", label="vehicle glyph")
    point, = ax.plot([], [], marker="o", color="tab:red", markersize=5)
    status = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", family="monospace")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.4, z_max)
    ax.set_xlabel("horizontal position x (m)")
    ax.set_ylabel("height z (m)")
    ax.set_title(
        f"Toy landing episode · seed {trace.get('seed')} · "
        f"{trace.get('terminal_reason', 'trace')}\n"
        "Body length is a display glyph only; it is not modeled geometry."
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    return fig, ax, (trajectory, body, point, status)


def save_trace_image(trace: dict[str, Any], output_path: str | Path) -> Path:
    """Save a static whole-episode trajectory image (no display required)."""
    fig, ax, artists = _make_figure(trace)
    import matplotlib.pyplot as plt

    trajectory, body, point, status = artists
    samples = trace["samples"]
    x_values = [float(row["x_m"]) for row in samples]
    z_values = [float(row["z_m"]) for row in samples]
    trajectory.set_data(x_values, z_values)
    _place_vehicle(body, point, samples[-1])
    status.set_text(_state_label(samples[-1]))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return destination


def _place_vehicle(body: Any, point: Any, sample: dict[str, Any]) -> None:
    """Draw a unit-length glyph along body +z; glyph dimensions are illustrative."""
    x, z = float(sample["x_m"]), float(sample["z_m"])
    theta = float(sample["theta_rad"])
    dx, dz = -math.sin(theta) * 0.5, math.cos(theta) * 0.5
    body.set_data([x - dx, x + dx], [z - dz, z + dz])
    point.set_data([x], [z])


def _state_label(sample: dict[str, Any]) -> str:
    return (
        f"t = {float(sample['time_s']):5.2f} s\n"
        f"x = {float(sample['x_m']):6.2f} m   z = {float(sample['z_m']):6.2f} m\n"
        f"vx = {float(sample['vx_mps']):6.2f} m/s  vz = {float(sample['vz_mps']):6.2f} m/s\n"
        f"pitch = {float(sample['theta_rad']):6.3f} rad  rate = {float(sample['omega_radps']):6.3f} rad/s"
    )


def show_trace(trace: dict[str, Any], interval_ms: int = 40) -> None:
    """Animate one saved episode in a desktop Matplotlib window."""
    if interval_ms <= 0:
        raise ValueError("interval_ms must be positive")
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError as error:
        raise RuntimeError(
            "Matplotlib is optional; install it with `python -m pip install -e \".[visualization]\"`."
        ) from error

    fig, _, artists = _make_figure(trace)
    trajectory, body, point, status = artists
    samples = trace["samples"]

    def update(index: int):
        current = samples[: index + 1]
        trajectory.set_data(
            [float(row["x_m"]) for row in current],
            [float(row["z_m"]) for row in current],
        )
        _place_vehicle(body, point, samples[index])
        status.set_text(_state_label(samples[index]))
        return trajectory, body, point, status

    animation = FuncAnimation(
        fig, update, frames=len(samples), interval=interval_ms, blit=False, repeat=False
    )
    # Keep a live reference until show() returns so the animation is not collected.
    fig._vertex_animation = animation
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_file", type=Path, help="M4 episode_traces.jsonl file")
    parser.add_argument("--seed", type=int, default=None, help="episode seed; defaults to the first trace")
    parser.add_argument("--save-image", type=Path, default=None, help="save a static PNG instead of opening a window")
    parser.add_argument("--interval-ms", type=int, default=40, help="animation frame interval")
    args = parser.parse_args()
    trace = load_trace(args.trace_file, args.seed)
    if args.save_image:
        result = save_trace_image(trace, args.save_image)
        print(f"Saved episode {trace['seed']} image to {result}")
    else:
        show_trace(trace, args.interval_ms)


if __name__ == "__main__":
    main()
