# Initial PPO Baseline Failure Report

**Project:** PROJECT VERTEX  
**Date:** 2026-11-07  
**Scope:** Read-only diagnosis of the existing 10,112-step PPO baseline. No environment, training configuration, checkpoint, or run log was changed.  
**Model status:** Simplified educational research simulation; not a validated physical flight controller.

## Executive summary

The 10,112-step PPO checkpoint failed on all ten fixed evaluation seeds; all episodes ended `out_of_bounds` at the upper boundary. Seed 1000 ended at z = 20.0836 m with upward velocity 7.3384 m/s, while the upper boundary is z > 20 m.

The strongest direct evidence is the policy's action sequence. The second action dimension maps to body-frame +z force, scaled by 20 N. On seed 1000 the deterministic policy's first value was 0.9794, which the environment converted to +19.588 N. Given the seeded initial state (z = 9.2077 m, vz = -0.3984 m/s, theta = 0.00575 rad), this produces approximately +9.76 m/sÂ² vertical acceleration under the implemented model. Across the 99-step episode this action averaged +0.549 (about +10.98 N), and the pitch-moment action averaged -0.555 (about -1.11 NÂ·m). The resulting trajectory climbs, rotates, and exits the upper boundary.

This identifies the policy behavior that directly precedes failure, but not why PPO learned it. The state/action signs and force scaling inspected here are internally consistent with the documented convention. The current reward, training volume, and lack of action telemetry leave the learning cause unresolved. No single environment correction is justified from this evidence alone; do not retrain or alter task parameters based on this report by itself.

**Correction to an earlier diagnostic:** a prior quick print labeled observation elements 0â€“2 as z, vz, and pitch. That label was wrong. They are normalized target-relative x, normalized z, and normalized vx. In particular, the initial vertical velocity on seed 1000 was **negative**, not the positive value implied by that print. This report uses the actual state and observation mapping below.

## Observations and evidence

### Baseline artifacts inspected

- Checkpoint: `runs/my-training-run/ppo_vertex.zip` (preserved).
- Training metadata and log: `runs/my-training-run/run_metadata.json` and `runs/my-training-run/logs/progress.csv` (preserved).
- Evaluation: `runs/m4-evaluation-my-training-run/evaluation.json` and `episode_traces.jsonl` (preserved).
- Evaluation used deterministic prediction on seeds 1000â€“1009: 0/10 successes, 10/10 `out_of_bounds`, mean episode length 95.8 steps, mean return -12.6150.
- Seed 1000: 99 steps; final x = -0.0876 m, z = 20.0836 m, vx = 2.3659 m/s, vz = 7.3384 m/s, theta = -2.2749 rad, omega = -2.2268 rad/s.
- Training requested 10,000 steps and collected 10,112 due to 128-step rollout batches. The logged mean episode return changed from -10.876 to -12.243 over 79 records (change -1.367); the mean episode length at the final record was 97.91 steps.
- Final recorded PPO diagnostics include action-distribution `std` near 1.006, explained variance 0.680, approximate KL 0.00180, and clip fraction 0.0. The checkpoint's learned Gaussian standard deviations are approximately [1.011, 0.976, 1.034]. These indicate broad stochastic exploration at the end; they do not by themselves explain the deterministic mean policy.

The M4 saved trace contains state and reward values but not observations or actions. To inspect the action sequence, the existing checkpoint was replayed deterministically in memory on the same ten seeds; no output files were written. Across 958 replayed steps, mean deterministic actions were [-0.268, +0.596, -0.548]. The second channel was at or above +0.95 for 10.0% of steps. On seed 1000, the first ten second-channel values ranged from +0.979 to +0.935. These are checkpoint replay measurements, not additional evaluation artifacts.

At seed 1000 reset, the state was:

```text
x=+0.04277 m, z=9.20768 m, vx=-0.02906 m/s, vz=-0.39838 m/s,
theta=+0.005752 rad, omega=-0.03090 rad/s
```

The observation was `[+0.004277, +0.460384, -0.005812, -0.039838, +0.005752, +0.999983, -0.015448, 1.0]`. The first deterministic action was `[-0.142643, +0.979413, -0.605510]`, converted to the generic simulated wrench `[-2.853 N, +19.588 N, -1.211 NÂ·m]`. Applying the inspected force transform gives initial vertical acceleration about +9.76 m/sÂ² (up). This is consistent with the observed climb.

## Observation/action interface specification

### Observation pipeline

`LandingEnv.step()` advances `PlanarState` through `vertex.simulation.step()`. The environment then creates a float32 observation from the updated state, clips each component to [-1, 1], and returns it to Gymnasium/SB3. `reset()` uses Gymnasium's seeded RNG unless a test supplies an explicit `PlanarState`.

| Index | Value | Unit/scale | Meaning |
|---:|---|---|---|
| 0 | `(x - target_x) / x_limit` | dimensionless; x_limit = 10 m | Target-relative horizontal position |
| 1 | `z / z_limit` | dimensionless; z_limit = 20 m | Vertical position |
| 2 | `vx / 5` | dimensionless; 5 m/s scale | Horizontal velocity |
| 3 | `vz / 10` | dimensionless; 10 m/s scale | Vertical velocity |
| 4 | `sin(theta)` | dimensionless | Pitch encoded periodically |
| 5 | `cos(theta)` | dimensionless | Pitch encoded periodically |
| 6 | `omega / 2` | dimensionless; 2 rad/s scale | Pitch rate |
| 7 | `1 - steps / max_episode_steps` | dimensionless | Fraction of 500-step budget remaining |

Every component is clipped to [-1, 1]. Clipping means the policy loses magnitude information beyond each scale. In this failure, z and omega reach their observation limits near the terminal boundary, but the episode is already close to termination; this is a lower-ranked contributor than the commanded trajectory. The baseline uses a direct, perfect-state observation with no noise or estimator.

### Action pipeline

The action is a float32 Gymnasium `Box` of shape `(3,)`, each value in [-1, 1]. The environment rejects values outside the box, then scales channels into the simulation-only body-axis wrench:

| Index | Input range | Conversion | Current interpretation |
|---:|---:|---|---|
| 0 | [-1, 1] | `action[0] * 20 N` | Body-frame x force |
| 1 | [-1, 1] | `action[1] * 20 N` | Body-frame z force |
| 2 | [-1, 1] | `action[2] * 2 NÂ·m` | Applied pitch moment |

The three values are applied once per 0.02 s environment step, held constant over that step. These are generic simulation inputs, not a physical propulsion or hardware command interface. SB3's continuous policy samples a Gaussian during training; environment actions are bounded, and evaluation requests the deterministic policy output.

## Physics and coordinate convention findings

- State order is `x, z, vx, vz, theta, omega`; positions are metres, velocities m/s, angle radians, and rate rad/s.
- Inertial x is positive right and z positive up. `theta = 0` is upright in the convention where body +z points up; positive theta rotates body +x toward inertial +z. Positive moment increases positive theta.
- Wrench forces are in body coordinates. The code rotates them to inertial axes using `ax = (cos(theta)*Fx - sin(theta)*Fz)/mass` and `az = (sin(theta)*Fx + cos(theta)*Fz)/mass - gravity`.
- Mass and moment of inertia are constant. Gravity is constant. No contact model, aerodynamics, mass change, disturbances, actuator dynamics, or sensor error are included.
- The implementation updates velocity and angular rate from current acceleration, then advances position and angle using those updated rates. This is commonly called semi-implicit (symplectic) Euler. At report time, the code and EDL-009 called it â€œexplicit forward Eulerâ€; that name is technically inconsistent with the update order. The implemented update is deterministic and its existing free-fall test matches it. That nomenclature issue was not evidence that the observed climb comes from an integration-sign defect.
- Existing sign tests cover free fall, rotated body-x force, and positive moment/pitch. There is no dedicated environment-level invariant asserting each action channel's scaling/sign, nor an observation-index/scaling test. The current tests cover Gymnasium compatibility, seeded reset, action bounds, touchdown categories, and timeout.

## Reward and termination findings

Each nonterminal step returns a negative shaping term: 0.01 times the sum of normalized absolute target-relative x, z, vx, vz, absolute theta, and normalized absolute omega, with a minus sign. Terminal success adds +10; every other terminal event adds -10, in addition to the state shaping. There is no action-magnitude, action-change, or energy-use penalty. Reward is not the success definition.

Termination checks are applied after the physics update:

- `out_of_bounds` if `abs(x - target_x) > 10 m` or `z > 20 m`.
- At `z <= 0`, `success` requires target error <= 0.5 m, |vx| <= 0.5 m/s, |vz| <= 0.5 m/s, |theta| <= 0.15 rad, and |omega| <= 0.25 rad/s; otherwise it is `unstable_touchdown`.
- If neither termination applies and step count reaches 500, Gymnasium reports truncation with `time_limit`.

Seed 1000's terminal state satisfies the upper-bound failure condition, and the evaluation report records `out_of_bounds`. The interface appears internally consistent: reward magnitude remains modest per step and is not substituted for the terminal success metric.

## PPO configuration and progress

The saved run used Stable-Baselines3 2.9.2, PyTorch 2.15.0+cpu, Gymnasium 1.4.0, and NumPy 2.5.4 on CPython 3.12.3. PPO was single-environment and CPU-only, with a [64, 64] MLP, 128 steps per rollout, batch size 64, and 5 epochs. The loaded checkpoint records gamma 0.99, GAE lambda 0.95, entropy coefficient 0, value coefficient 0.5, learning rate 0.0003, and clip range 0.2. Requested 10,000 steps became 10,112 actual steps.

The training reward trend declined and evaluation success remained zero. This is evidence that this run did not learn the task successfully. It is not enough to distinguish insufficient training, reward/task signal issues, policy optimization behavior, or a task-interface design problem. The run metadata itself labels the result an integration smoke run, not a learning-success claim.

## Root-cause hypotheses, ranked by evidence

1. **The learned mean policy commands sustained upward force and negative pitch moment; the resulting simulated motion exits through the upper boundary. High confidence as the immediate trajectory cause.** This is directly measured by replaying the saved policy and matches the observed upward terminal velocity, altitude, and pitch. It does not explain why PPO learned these commands.
2. **The current learning signal/configuration did not produce a successful policy in this short run. Medium confidence that this is the broader learning failure; specific cause unresolved.** All ten evaluation episodes failed, recorded mean return worsened, and no success examples provide a positive landing signal. Ten thousand steps represent only roughly 100 episodes at the observed duration. More steps alone are not yet justified as a fix.
3. **Reward scale/structure may not guide the policy toward the terminal objective strongly enough. Plausible, unproven.** Dense state shaping is small per step, terminal reward is sparse, and control effort is unpenalized. But the recorded returns and policy actions do not isolate reward as the cause; changing reward now would confound diagnosis.
4. **Observation clipping may reduce useful state magnitude information near the boundary. Plausible but low confidence as the initiating cause.** It affects z and omega after the policy has already driven the system far from the initial region.
5. **Physics sign or integration defect caused the climb. Low confidence based on current evidence.** The force-to-world transform and sign convention agree with EDL-009, existing tests cover key signs, and the measured upward acceleration follows the positive body-z action. The Euler method label does need correction in documentation/code comments, but it does not explain this failure.

## Recommended minimal correction

No change to reward, action scale, PPO hyperparameters, or physics is justified yet. The minimal next diagnostic is to extend a **new** evaluation artifact (preserving the existing baseline files) to record the raw observation and bounded action for each step, plus the already-computable reward components. This adds evidence without changing environment dynamics or retraining. Then add focused interface tests for observation order/scaling/clipping and for each action dimension's signed force/moment scaling and resulting state response.

After those invariants pass, use a scripted, non-learning scenario test to establish expected descent/attitude response under fixed actions. Only then choose one evidence-backed task or learning change, record it separately, and compare against this preserved checkpoint on the same ten seeds. Do not change multiple parameters together.

## Tests needed before accepting any correction

1. Observation contract test: set a known `PlanarState` and assert all eight values, float32 dtype, order, normalization, and clipping.
2. Action contract tests: assert positive/negative body-x, body-z, and moment actions produce the documented wrench scale and expected signs in the next state; test action boundaries and invalid shape/NaN handling.
3. Dynamics regression: retain current free-fall/sign tests and add a hand-calculated one-step case that makes the rate-first, updated-rate position integration explicit. Rename â€œexplicit forward Eulerâ€ to the correct method description consistently, or change the integration only through a separate decision and regression baseline.
4. Reward/termination tests: exercise every shaping term and boundary equality/just-over-boundary case, including success tolerances, unstable touchdown, out-of-bounds, and time-limit `terminated`/`truncated` semantics.
5. Deterministic replay test: the same checkpoint, environment version, and seed must reproduce the same observation/action/state sequence. Record actions in a fresh diagnostic artifact so this can be inspected later.
6. If a later correction changes reward or task parameters, add a focused test for that intended reward/termination behavior, then compare a newly trained checkpoint against the preserved baseline on seeds 1000â€“1009. Report both success categories and reward; neither replaces the other.

## Limits

This is a source-and-artifact inspection plus deterministic replay of the saved checkpoint; no environment or training code was changed and no new training run was performed. The replay measurements are limited to the current software/model and saved checkpoint. They do not validate real vehicle behavior, physical safety, or flight readiness.

