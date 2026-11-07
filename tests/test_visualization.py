import json

import pytest

from vertex.visualization.viewer import load_trace, save_trace_image


def test_load_trace_selects_episode_by_seed(tmp_path):
    trace_file = tmp_path / "episode_traces.jsonl"
    records = [
        {"seed": 11, "target_x": 0.0, "terminal_reason": "unstable_touchdown", "samples": [{"step": 1}]},
        {"seed": 22, "target_x": 0.0, "terminal_reason": "success", "samples": [{"step": 1}]},
    ]
    trace_file.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    assert load_trace(trace_file, seed=22)["terminal_reason"] == "success"
    assert load_trace(trace_file)["seed"] == 11


def test_load_trace_reports_missing_seed(tmp_path):
    trace_file = tmp_path / "episode_traces.jsonl"
    trace_file.write_text('{"seed": 11, "samples": [{"step": 1}]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="no episode with seed 22"):
        load_trace(trace_file, seed=22)


def test_load_trace_backfills_outcome_from_evaluation_summary(tmp_path):
    trace_file = tmp_path / "episode_traces.jsonl"
    trace_file.write_text('{"seed": 11, "samples": [{"step": 1}]}\n', encoding="utf-8")
    (tmp_path / "evaluation.json").write_text(
        json.dumps(
            {
                "summary": {"environment_config": {"target_x": 1.0}},
                "episodes": [{"seed": 11, "terminal_reason": "out_of_bounds"}],
            }
        ),
        encoding="utf-8",
    )
    trace = load_trace(trace_file)
    assert trace["terminal_reason"] == "out_of_bounds"
    assert trace["target_x"] == 1.0


def test_save_static_episode_image_when_visualization_extra_is_installed(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    trace = {
        "seed": 4,
        "terminal_reason": "unstable_touchdown",
        "target_x": 0.0,
        "samples": [
            {"time_s": 0.0, "x_m": 0.0, "z_m": 2.0, "vx_mps": 0.0, "vz_mps": -1.0, "theta_rad": 0.0, "omega_radps": 0.0},
            {"time_s": 0.1, "x_m": 0.1, "z_m": 1.8, "vx_mps": 1.0, "vz_mps": -2.0, "theta_rad": 0.1, "omega_radps": 0.2},
        ],
    }
    image = save_trace_image(trace, tmp_path / "episode.png")
    assert image.is_file()
    assert image.stat().st_size > 0
