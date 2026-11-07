# Baseline Instrumentation and Interface Validation Report

**Project:** PROJECT VERTEX  
**Date:** 2026-11-07  
**Purpose:** Add evaluation observability and verify the current interfaces without changing the baseline task or retraining PPO.  
**Model status:** Simplified educational research simulation; not a validated physical flight controller.

## 1. Executive summary

Optional step-level JSONL diagnostics are now available from evaluation. Each record captures the observation and physical state before a step, the policy action before clipping, the bounded action applied by the environment, the corresponding simulation-only wrench, post-step state, reward breakdown, and terminal flags/reason. The default evaluator remains compatible, and the training module does not collect diagnostics or import visualization.

The original checkpoint was evaluated twice into new output directories. Both runs reproduced identical episode rows and byte-identical step diagnostics (958 records); episode state/reward samples match the existing M4 trace. The preserved baseline remains 0/10 successes, all 10 episodes `out_of_bounds`.

The added tests cover the eight observation values, action dimensions/scales/signs, reward terms, success and failure thresholds, strict upper/horizontal boundary behavior, one-step integration, and fixed scripted physics scenarios. The full suite passes: **55 passed in 6.31 s**.

One useful interface detail is now measurable: the raw deterministic policy output exceeded the action bound in 17 of 958 evaluation steps, all on action channel 1 (body-z force). The largest raw value was 1.02057; the action applied by Stable-Baselines3 was clipped to 1.0. This is a real clipping event, but its low frequency alone does not explain the repeated climb-out failures.

## 2. Files modified and why

- `src/vertex/learning/evaluate.py` — added an opt-in `diagnostics` argument and `--diagnostics` CLI switch. Diagnostics are streamed to JSONL only when requested. Existing summary, CSV, and state trace outputs remain available.
- `src/vertex/environment/landing_env.py` — added `decode_action()` to expose the existing action-to-wrench mapping for diagnostics and tests, and `reward_components()` to report the existing reward terms. Neither method changes the ordinary `step()` action or reward calculation.
- `src/vertex/simulation/dynamics.py` — corrected the integration method description to semi-implicit (symplectic) Euler. No dynamics equations or update order changed.
- `tests/test_environment.py` — added observation, action scaling/response, invalid input, reward term, touchdown tolerance, and boundary tests.
- `tests/test_simulation.py` — added a hand-calculated one-step integration regression.
- `tests/test_scenarios.py` — added deterministic free-fall, rotated body-force, and applied-moment cases using fixed states and generic simulation-only wrenches.
- `tests/test_training.py` — extended the existing short-checkpoint integration test to verify optional diagnostics, repeated deterministic output, and no diagnostic file in ordinary evaluation mode. The test's temporary smoke model is not the preserved baseline checkpoint.
- `Docs/ENGINEERING_DECISIONS.md` — corrected EDL-009's terminology while retaining its numerical integration choice; no new engineering decision was made.
- `Docs/BASELINE_FAILURE_REPORT.md` — clarified that its Euler terminology finding was corrected here.
- `Docs/PROJECT_CONTEXT.md` and `CHANGES.md` — recorded the completed instrumentation work and current evidence.
- `Docs/BASELINE_INSTRUMENTATION_REPORT.md` — this report.

## 3. Diagnostic artifact format

Run diagnostics with a new output directory:

```powershell
.\.venv\Scripts\python.exe -m vertex.learning.evaluate `
  --checkpoint runs\my-training-run\ppo_vertex.zip `
  --output-dir runs\m6-instrumented-eval-20261107 `
  --diagnostics
```

The optional file is `step_diagnostics.jsonl`: one JSON object per simulated step. The report metadata names it as `step_diagnostics_jsonl` when present. Fields are:

- `episode_seed`, `step`, `time_s`, `schema_version`
- `state_before`, `observation`
- `raw_deterministic_policy_action` — deterministic policy distribution output before Box clipping
- `action_applied` — float32 bounded action consumed by the environment
- `applied_wrench` — generic simulation force/moment in N and N·m, expressed in body axes
- `state_after`
- `reward_components`, `total_reward`
- `terminated`, `truncated`, `termination_reason`

Example record (seed 1000, step 1; abbreviated only by formatting):

```json
{"schema_version":1,"episode_seed":1000,"step":1,"time_s":0.02,
 "state_before":{"x_m":0.04277,"z_m":9.20768,"vx_mps":-0.02906,"vz_mps":-0.39838,"theta_rad":0.00575,"omega_radps":-0.03090},
 "observation":[0.004277,0.460384,-0.005812,-0.039838,0.005752,0.999983,-0.015448,1.0],
 "raw_deterministic_policy_action":[-0.142643,0.979413,-0.605510],
 "action_applied":[-0.142643,0.979413,-0.605510],
 "applied_wrench":{"force_x_N":-2.852868,"force_z_N":19.588255,"moment_Nm":-1.211019},
 "state_after":{"x_m":0.041004,"z_m":9.203621,"vx_mps":-0.088368,"vz_mps":-0.203146,"theta_rad":0.004649,"omega_radps":-0.055117},
 "reward_components":{"horizontal_position_penalty":-0.000041,"height_penalty":-0.004602,"horizontal_velocity_penalty":-0.000177,"vertical_velocity_penalty":-0.000203,"pitch_penalty":-0.000046,"pitch_rate_penalty":-0.000276,"terminal_adjustment":0.0,"state_shaping_total":-0.005345,"total_reward":-0.005345},
 "total_reward":-0.005345,"terminated":false,"truncated":false,"termination_reason":null}
```

These are abstract simulation inputs only. They are not physical propulsion or hardware commands.

## 4. Observation and action contract findings

Observation is float32, clipped to [-1, 1], in this fixed order:

| Index | Value | Test evidence |
|---:|---|---|
| 0 | `(x - target_x) / x_limit` | Known-state scale and both clipping signs |
| 1 | `z / z_limit` | Known-state scale and upper/lower clipping |
| 2 | `vx / 5` | Known-state scale and both clipping signs |
| 3 | `vz / 10` | Known-state scale and both clipping signs |
| 4 | `sin(theta)` | Known angle and clipping range |
| 5 | `cos(theta)` | Known angle and clipping range |
| 6 | `omega / 2` | Known-state scale and both clipping signs |
| 7 | `1 - steps / max_episode_steps` | Reset value and range behavior |

The action Box has shape `(3,)`, float32, and bounds [-1, 1]. Channel 0 scales to body-x force at 20 N per normalized unit; channel 1 to body-z force at 20 N per unit; channel 2 to pitch moment at 2 N·m per unit. Positive and negative values on all three channels were tested both at the decoded wrench and resulting next state. Wrong shape, out-of-range values, and NaN are rejected with `ValueError`.

On the 958-step baseline replay, raw versus applied actions differed 17 times, only on channel 1. The raw maximum was +1.020571 and the environment saw +1.0. This confirms the report's prior “second channel” interpretation as body-z force and makes the SB3 clipping boundary visible.

## 5. Physics validation results

The convention remains x right, z up, SI units, radians, body-axis applied wrench, and the existing force rotation. The implementation updates velocity/rate first and then position/angle from updated rates: **semi-implicit (symplectic) Euler**. The method wording in the source and EDL-009 is corrected; the integration method itself is unchanged.

The one-step regression checks initial state `(x=1, z=5, vx=2, vz=-1, theta=0.2, omega=0.3)`, moment 2 N·m, inertia 2 kg·m², gravity 10 m/s², and dt 0.1 s. Expected output is `(x=1.2, z=4.8, vx=2, vz=-2, theta=0.24, omega=0.4)`, confirming updated-rate integration.

Deterministic scripted scenarios passed:

| Scenario | Initial condition and input | Result |
|---|---|---|
| Free fall | z=2 m, vz=-1 m/s, zero wrench, dt=0.1 s, g=9.81 m/s² | z=1.8019 m, vz=-1.981 m/s |
| Rotated force | z=3 m, theta=π/2, body Fx=2 N, mass=2 kg, zero gravity, dt=0.5 s | vx≈0, vz=+0.5 m/s, z=3.25 m |
| Positive moment | zero state, moment=2 N·m, inertia=2 kg·m², zero gravity, dt=0.25 s | omega=+0.25 rad/s, theta=+0.0625 rad |

These cases verify the implementation against its stated mathematical conventions. They are not evidence of RL performance or physical rocket performance.

## 6. Reward and termination test results

Each of the six existing state-shaping terms is checked independently: horizontal position error, height, horizontal velocity, vertical velocity, absolute pitch, and pitch rate. The terminal adjustment is verified as +10 for success, -10 for failure, and zero for a continuing step. Diagnostic `total_reward` is the environment's actual returned reward.

Tests verify exact success threshold values are accepted, just-beyond each touchdown tolerance becomes `unstable_touchdown`, exact x/z outer bounds remain in-bounds, and just-beyond horizontal/upper bounds is `out_of_bounds`. Existing tests verify controlled and unstable touchdown plus the distinction between `terminated` and time-limit `truncated`.

## 7. Deterministic replay and baseline preservation

Commands run against the original saved checkpoint:

```powershell
.\.venv\Scripts\python.exe -m vertex.learning.evaluate --checkpoint runs\my-training-run\ppo_vertex.zip --output-dir runs\m6-instrumented-eval-20261107 --diagnostics
.\.venv\Scripts\python.exe -m vertex.learning.evaluate --checkpoint runs\my-training-run\ppo_vertex.zip --output-dir runs\m6-instrumented-eval-repeat-20261107 --diagnostics
```

Results:

- Both runs: 10 episodes, 0 successes, 10 `out_of_bounds`, 958 total steps.
- The episode rows from the new evaluation match the existing M4 `evaluation.json` episode rows exactly.
- Every state/reward sample matches the existing M4 trace. Its older top-level records only had `seed` and `samples`; new traces also include `target_x` and `terminal_reason`.
- The two `step_diagnostics.jsonl` files are byte-identical.
- Reward-component `total_reward` matches the environment reward exactly for all records; there are 10 terminal records.
- Existing checkpoint, metadata, progress log, evaluation JSON, CSV, and JSONL trace SHA-256 values were checked before and after and remained unchanged.

The automated integration test also evaluates the same temporary checkpoint twice on the same environment and seeds, compares diagnostic JSONL byte-for-byte, and confirms ordinary evaluation does not create the diagnostics file.

## 8. Defects and inconsistencies discovered

1. **Integration terminology was incorrect.** The source and EDL-009 called the update explicit forward Euler even though it uses updated rates to advance position/angle. Corrected to semi-implicit (symplectic) Euler; no numerical code changed and no new engineering decision was made.
2. **The baseline policy occasionally requests an action beyond the Box limit.** In deterministic evaluation, 17/958 raw outputs exceeded +1 on body-z force; SB3 clipped these to +1 before the environment step. This should be considered when studying policy output versus applied behavior, but the frequency and magnitude do not by themselves account for the baseline's repeated upper-bound exits.
3. **Old M4 traces lack terminal metadata at the JSONL episode level.** They remain readable by the existing viewer, which can backfill from the adjacent evaluation report. New evaluation traces include both `target_x` and `terminal_reason`.

No physics sign, action scale, observation order, reward, or terminal-condition defect was demonstrated by these tests. This is validation of a simplified research simulation only.

## 9. Test results and training isolation

Focused command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_environment.py tests\test_simulation.py tests\test_scenarios.py -q
```

Result: **48 passed in 0.33 s**.

Evaluation/checkpoint instrumentation test:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_training.py -q
```

Result: **2 passed in 4.84 s**. That test uses its existing short-lived smoke checkpoint under pytest's temporary directory; it does not retrain or replace the baseline checkpoint.

Complete suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Result on the final source version: **55 passed in 6.13 s**.

Diagnostics execute only in the separate evaluation module when `--diagnostics` is supplied. Training code, PPO settings, environment reward, observation definition, action scaling, physics dynamics, rendering, and baseline checkpoint are unchanged. No new training benchmark was run; there is no diagnostic recording work in the training loop.

## 10. Recommended next engineering actions

1. Use the new seed-1000 step records to examine when the policy starts increasing body-z force and applying negative moment relative to the reward terms and normalized observation.
2. Treat clipping as a measurable but likely secondary effect: it affected 17 of 958 deterministic evaluation steps, all by at most about 0.0206 above the channel bound. Do not alter action scaling based on this alone.
3. Form one evidence-backed hypothesis about why training favors the observed trajectory, define a test that can distinguish it, and propose one isolated change for review. Preserve this checkpoint as the control. Do not retrain automatically.

## 11. Limitations

The episode seed fixes the environment's reset state. Evaluation uses deterministic policy inference; identical inputs and software versions yielded byte-identical traces here. This does not establish reproducibility across different library versions, platforms, or devices. The results do not establish real-world safety, vehicle controllability, or flight readiness.
