# Pretraining Engineering Audit

**Date:** 2026-11-07  
**Scope:** Headless investigation of the existing educational simulation and PPO artifacts. No training, environment, physics, reward, observation, action, or checkpoint changes were made. This is not a validated physical flight controller or model.

## 1. Executive summary

- The current mathematical model has enough force authority to oppose gravity near upright: 20 N maximum body-Z force versus 9.81 N weight for the configured 1 kg mass. A transparent full-state feedback controller using the existing action bounds landed successfully from all ten fixed reset seeds (1000–1009). This establishes sampled task feasibility in this model, not robustness or physical feasibility.
- The actual reward implementation ranked the successful landing above the four tested undesirable behaviors at both gamma=0.99 and gamma=1.0. At gamma=0.99, however, success only narrowly beat a constant-altitude time-limit episode (-0.316 versus -0.447). Sustained upward force had a slightly better discounted return than free fall (-5.510 versus -5.555) because its terminal penalty came later. The time-limit truncation receives no terminal penalty.
- The untrained deterministic policy was near zero in every tested state and did not clip. Its initial diagonal Gaussian standard deviation was 1.0 in each normalized action channel. With mean near zero, the expected initial stochastic clipping rate is about 31.7% per scalar action, or about 68.2% of three-channel action vectors having at least one clipped component. These are analytic initial-policy probabilities, not measured rollout frequencies.
- Existing PPO logs do not record training action samples or action-bound clipping counts. SB3's rollout collector samples from an unbounded Gaussian, clips the action passed to the environment, and stores the sampled action in the rollout buffer. Deterministic evaluation clipping therefore cannot stand in for training-time clipping.

## 2. Vehicle parameters and control authority

All values below are configured educational model values, not specifications for a real vehicle.

| Parameter | Current value |
|---|---:|
| Mass | 1.0 kg |
| Moment of inertia | 1.0 kg·m² |
| Gravity | 9.81 m/s² downward |
| Simulation time step | 0.02 s |
| Force action limits | body-X and body-Z each independently [-20, +20] N |
| Pitch moment action limits | [-2, +2] N·m |
| Maximum episode duration | 500 steps = 10 s |
| Initial x | [-1, +1] m |
| Initial z | [8, 10] m |
| Initial vx | [-0.5, +0.5] m/s |
| Initial vz | [-0.5, 0] m/s |
| Initial pitch | [-0.1, +0.1] rad |
| Initial angular rate | [-0.05, +0.05] rad/s |

At upright attitude (`theta=0`), body-Z force is vertical. With no body-X force, the maximum upward net acceleration is `(20 - 9.81) / 1 = +10.19 m/s²`. The minimum net vertical acceleration at -20 N is `(-20 - 9.81) / 1 = -29.81 m/s²`. Thus maximum body-Z force is about 2.04 times the model weight and can counter gravity with substantial remaining authority. At this attitude, the force component range is symmetric; the independently bounded body-X force controls horizontal acceleration.

The moment bound gives angular acceleration `alpha = M/I = +/-2 rad/s²`, changing angular velocity by at most 0.04 rad/s in one 0.02 s step. The combined force box permits a force-vector magnitude up to `sqrt(20² + 20²) = 28.28 N`; the two component maxima cannot both be interpreted as a 28.28 N vertical force when upright.

Touchdown is success when `z <= 0`, `abs(x-target_x) <= 0.5 m`, `abs(vx) <= 0.5 m/s`, `abs(vz) <= 0.5 m/s`, `abs(theta) <= 0.15 rad`, and `abs(omega) <= 0.25 rad/s`. The upper boundary is `z > 20 m`; the horizontal boundary is `abs(x-target_x) > 10 m`.

## 3. L1 task-feasibility results

### Reference controller and assumptions

The controller below was evaluated only inside the existing `LandingEnv`. It observes the exact simulated state, knows the configured mass/gravity/action scales, has instantaneous action application, and uses the existing simulator to rotate desired inertial force into body coordinates. It has no sensors, estimation error, actuator dynamics, wind, contact model, or hardware mapping.

At each step it:

1. Selects a desired descent rate `vz* = -min(1.5, max(0.15, 0.6*z))` m/s. The 0.15 m/s floor prevents the earlier proportional-to-height version from approaching zero descent speed asymptotically and timing out just above the ground.
2. Requests `az* = clip(2*(vz* - vz), -15, +10)` m/s².
3. Selects `vx* = clip(-0.8*x, -1, +1)` m/s and `ax* = clip(2*(vx* - vx), -5, +5)` m/s².
4. Computes inertial force `(m*ax*, m*(g+az*))`, rotates it into body coordinates with the inverse of the documented state rotation, and clips each normalized force action to [-1, 1].
5. Commands the generic simulation moment `M = clip(-4*theta - 3*omega, -2, +2)` N·m, then normalizes it by the configured 2 N·m action scale.

The initial, overly cautious version used `vz*=-min(1.5, 0.6*z)`. It landed successfully on four seeds but timed out on six because the desired speed approached zero as height approached zero. Adding the explicit 0.15 m/s approach-speed floor produced success on all ten. This is evidence that the first controller form was deficient; it is not evidence of a model limitation. No model parameter or environment code changed between these controller probes.

### Fixed-seed outcomes

Each result uses the existing seeded reset distribution and exactly the current action bounds and physics. Terminal states are shown at the step where success was declared.

| Seed | Result | Steps | x (m) | z (m) | vx (m/s) | vz (m/s) | pitch (rad) | omega (rad/s) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1000 | success | 431 | 0.0000 | -0.0015 | -0.0000 | -0.1610 | 0.0000 | -0.0000 |
| 1001 | success | 397 | 0.0001 | -0.0014 | -0.0001 | -0.1610 | 0.0000 | -0.0000 |
| 1002 | success | 416 | -0.0000 | -0.0014 | 0.0001 | -0.1610 | -0.0000 | 0.0000 |
| 1003 | success | 412 | -0.0001 | -0.0031 | 0.0000 | -0.1607 | -0.0000 | 0.0000 |
| 1004 | success | 401 | -0.0002 | -0.0020 | -0.0002 | -0.1609 | -0.0000 | 0.0 |
| 1005 | success | 456 | -0.0001 | -0.0007 | 0.0001 | -0.1611 | 0.0000 | -0.0000 |
| 1006 | success | 431 | -0.0001 | -0.0013 | 0.0000 | -0.1610 | 0.0000 | -0.0000 |
| 1007 | success | 445 | -0.0000 | -0.0012 | 0.0001 | -0.1610 | 0.0000 | -0.0000 |
| 1008 | success | 451 | -0.0001 | -0.0009 | 0.0001 | -0.1610 | 0.0000 | -0.0000 |
| 1009 | success | 413 | -0.0002 | -0.0010 | -0.0001 | -0.1610 | -0.0000 | 0.0 |

The repeated small negative z values are the simulator's one-step penetration below `z=0` before terminal detection; no ground-contact response is modeled. All ten sampled initial states meet the task's success criteria. This does not establish performance over all possible states, unseen disturbances, model uncertainty, or real systems.

## 4. L2 reward-ranking results

Each behavior started from the same controlled upright state `(x=0, z=9 m, vx=vz=theta=omega=0)`. The successful row uses the reference controller above. “Free fall” applies a zero physical wrench; “zero action” applies the environment's `[0,0,0]` action. In this simulator those two behaviors are exactly identical because there are no other forces besides gravity. Constant altitude uses body-Z force `mg = 9.81 N` (normalized action 0.4905) with zero other channels. Sustained upward force uses `[0,1,0]`, i.e. +20 N body-Z force.

All totals below were collected from calls to the actual environment `step()` and existing reward-component implementation. Discounted return weights the first step's reward by 1, the next by gamma, and so on. A truncation is not a termination: time-limit episodes receive no terminal adjustment.

| Behavior | Steps / duration | Outcome | Undiscounted return (= gamma 1.0) | Discounted return, gamma 0.99 |
|---|---:|---|---:|---:|
| Reference-controller landing | 430 / 8.60 s | terminated: success | +8.7533 | -0.3159 |
| Free fall | 68 / 1.36 s | terminated: unstable_touchdown | -10.6589 | -5.5551 |
| Constant altitude | 500 / 10.00 s | truncated: time_limit | -2.2500 | -0.4470 |
| Sustained upward force | 73 / 1.46 s | terminated: out_of_bounds | -11.0166 | -5.5098 |
| Zero action | 68 / 1.36 s | terminated: unstable_touchdown | -10.6589 | -5.5551 |

| Reward-component total | Successful landing | Free fall | Constant altitude | Sustained upward force | Zero action |
|---|---:|---:|---:|---:|---:|
| Horizontal-position penalty | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Height penalty | -0.7967 | -0.1986 | -2.2500 | -0.4661 | -0.1986 |
| Horizontal-velocity penalty | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Vertical-velocity penalty | -0.4500 | -0.4603 | -0.0000 | -0.5505 | -0.4603 |
| Pitch penalty | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Pitch-rate penalty | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Terminal adjustment | +10.0000 | -10.0000 | 0.0000 | -10.0000 | -10.0000 |
| State-shaping total | -1.2467 | -0.6589 | -2.2500 | -1.0166 | -0.6589 |

**Answer:** For these representative behaviors, the reward does assign a better return to successful landing than to each undesirable behavior at both gamma values. Under gamma=0.99, the margin over constant altitude is small: success is -0.3159 versus -0.4470. The main structure is a small negative state-shaping term on every step and a large terminal bonus/penalty of +/-10. Because time-limit truncation has no terminal penalty, holding altitude avoids the large failure penalty. Sustained upward force also delays the terminal penalty: its gamma=0.99 return (-5.5098) is slightly better than free fall (-5.5551), despite a worse undiscounted return (-11.0166 versus -10.6589). That is a measured discount-timing effect, not proof of why a trained policy selected upward force.

## 5. L3 initial-policy results

A fresh SB3 PPO model was constructed with seed 7, CPU, `MlpPolicy`, `[64,64]` actor and critic networks, `n_steps=128`, `batch_size=64`, `n_epochs=5`, learning rate 0.0003, gamma 0.99, GAE lambda 0.95, entropy coefficient 0, value coefficient 0.5, and clip range 0.2. SB3's default Tanh hidden activations and orthogonal initialization were retained. It was not trained: `learn()` was never called. The initial log standard deviation was 0.0 per channel, so the policy standard deviation was **1.0 in each of the three normalized action dimensions**.

Action order is `(body-X force, body-Z force, pitch moment)`, normalized to [-1,1]. `raw` is the Gaussian mean used by deterministic inference. Every value below is inside the action bounds, so bounded output equals raw output and deterministic clipping frequency is zero for all three channels. Values are rounded to six decimal places.

### Original reset observations

| Seed | Raw action = bounded action |
|---:|---|
| 1000 | [-0.004722, 0.001676, 0.002747] |
| 1001 | [-0.004988, 0.001874, 0.002488] |
| 1002 | [-0.004775, 0.001880, 0.002765] |
| 1003 | [-0.004560, 0.001312, 0.002532] |
| 1004 | [-0.004495, 0.001359, 0.002266] |
| 1005 | [-0.004406, 0.001398, 0.002582] |
| 1006 | [-0.004536, 0.001339, 0.002590] |
| 1007 | [-0.004484, 0.001735, 0.002729] |
| 1008 | [-0.004418, 0.001148, 0.002540] |
| 1009 | [-0.004412, 0.001170, 0.002295] |

### Controlled one-variable probes

The nominal state is `(x=0, z=9, vx=0, vz=0, theta=0, omega=0)`. Each other row changes only the named state variable. The `raw` output is also the bounded output because no component clips.

| State probe | Raw action = bounded action |
|---|---|
| Nominal | [-0.004786, 0.001630, 0.002714] |
| Lower altitude, z=4 m | [-0.005055, 0.000919, 0.001991] |
| Higher altitude, z=16 m | [-0.004341, 0.002570, 0.003690] |
| Upward velocity, vz=+2 m/s | [-0.004890, 0.001126, 0.002703] |
| Downward velocity, vz=-2 m/s | [-0.004646, 0.002130, 0.002746] |
| Positive displacement, x=+1 m | [-0.005079, 0.001717, 0.002876] |
| Negative displacement, x=-1 m | [-0.004486, 0.001535, 0.002553] |
| Positive horizontal velocity, vx=+1 m/s | [-0.004820, 0.001816, 0.003209] |
| Negative horizontal velocity, vx=-1 m/s | [-0.004757, 0.001414, 0.002218] |
| Positive pitch, theta=+0.2 rad | [-0.004448, 0.000586, 0.002714] |
| Negative pitch, theta=-0.2 rad | [-0.005047, 0.002720, 0.002785] |
| Positive angular rate, omega=+0.5 rad/s | [-0.005256, 0.002411, 0.002606] |
| Negative angular rate, omega=-0.5 rad/s | [-0.004318, 0.000804, 0.002819] |

The deterministic initial mean is effectively zero and small changes in state alter its sign/magnitude without producing a meaningful corrective command. At the nominal probe, the body-Z mean is 0.001630, equivalent to only about +0.033 N from the 20 N action scale; at upright attitude that leaves approximately -9.777 m/s² net vertical acceleration after gravity. This is confirmed for the tested inputs only; it does not explain later training updates by itself.

### Initial stochastic clipping

SB3's PPO policy uses an unbounded diagonal Gaussian for these Box actions. With standard deviation 1.0 and means near zero, the analytical probability that one scalar draw lies outside [-1,1] is approximately **31.73%** per channel. Assuming the policy's independent diagonal components, the probability that at least one of the three channels clips on a sampled action vector is approximately **68.2%**. The channel-by-channel estimates across the tested reset and controlled states were all about 31.73%.

This is a distribution-based estimate at initialization, not an observed training-rollout count. SB3's collector code was inspected: it samples `actions`, clips them before `env.step`, and stores the original sampled `actions` in the rollout buffer for PPO updates. The environment therefore sees bounded actions while PPO retains likelihood data for its unbounded sample. Existing logs do not count how often this happens during training.

For comparison only, saved-checkpoint deterministic evaluation diagnostics show raw out-of-range action rates of 0% / 1.77% / 0% by channel for the gamma=0.99 baseline, and 100% / 99.18% / 65.24% for EXP-001. Those are deterministic evaluation observations after training and are not training-time stochastic clipping rates.

## 6. Baseline versus EXP-001 training diagnostics

Values below come from the existing `runs/my-training-run/logs/progress.csv` and `runs/exp-001-gamma-1.0/logs/progress.csv`. Each run has 79 rollout records; the first PPO update metrics appear at 256 steps, so there are 78 records for training-only metrics. Both requested 10,000 steps and collected 10,112.

| Metric | Baseline first → final | EXP-001 first → final | Interpretation / coverage |
|---|---:|---:|---|
| Explained variance | 0.0147 → 0.6795 | 0.0125 → 0.9472 | Varied widely; minimum was -3.657 baseline and -14.030 EXP-001. High final value alone is not task success. |
| Approximate KL | 0.000585 → 0.001797 | 0.000579 → 0.007889 | Largest recorded values: 0.02626 and 0.01888. |
| PPO clip fraction | 0 → 0 | 0 → 0.02656 | Baseline maximum 0.14688; EXP-001 maximum 0.125. This is PPO probability-ratio clipping, not action-bound clipping. |
| Policy standard deviation | 0.9996 → 1.0064 | 0.9998 → 0.9688 | Baseline range 0.99895–1.00842; EXP-001 range 0.96794–1.00494. |
| Rollout mean episode length | 65.0 → 97.91 | 65.0 → 84.45 | Moving rollout statistic; not fixed-seed evaluation length. |
| Rollout mean episode reward | -10.876 → -12.243 | -10.876 → -13.082 | Both deteriorated from the first recorded value; these are moving rollout means. |

The logs are insufficient to reconstruct per-step policy means, sampled actions, environment-clipped actions, action saturation rates, policy entropy by action/state, state occupancy, or reward-component totals during training. The M6/EXP-001 evaluation diagnostics are deterministic after training and cannot fill those training-time gaps.

## 7. Confirmed findings

1. In the current equations and bounds, upright vertical control can both counter gravity and generate upward acceleration.
2. A single transparent, full-state feedback rule achieved the task's defined success on all ten fixed reset seeds. The initial failed variant exposed an asymptotic approach-speed flaw in that controller, not an established simulator limitation.
3. Under the actual reward implementation, success ranked above the four scripted undesirable behaviors for both discount factors tested. At gamma=0.99 the discounted margin over constant altitude was small, and delayed upward-force failure slightly outscored free fall.
4. The fresh deterministic initial policy was near-zero and did not saturate on any tested input. Its Gaussian standard deviation was 1.0, implying frequent initial stochastic action-bound clipping.
5. Training-time action clipping is separate from PPO's `train/clip_fraction`; existing run logs do not report action-bound clipping counts.
6. Baseline and EXP-001 training returns became more negative, even though explained variance ended higher (especially EXP-001). These aggregate metrics do not show which states/actions drove those changes.

## 8. Unresolved hypotheses

- Initial stochastic clipping may reduce effective exploration resolution near action bounds, but the logs do not reveal the realized clipped samples during either 10,112-step training run.
- The gamma=0.99 reward may favor postponing terminal failure in some state/action sequences; the scripted sustained-upward/free-fall comparison demonstrates one small discounted ranking reversal, not that this caused the learned policy.
- The near-zero initial mean does not reveal why policy optimization later moved toward large actions. Reward timing, sparse terminal outcomes, observation scaling, and PPO update dynamics remain possible contributors; this audit does not isolate them.
- The reference controller relies on exact full state, known parameters, and ideal force/moment application. Its success does not establish robustness to uncertainty, disturbances, partial observations, or more realistic dynamics.

## 9. Recommended next experiment

The recommended action-clipping diagnostic was implemented and run as EXP-002. It recorded the first 2,048 transitions and matched the original baseline policy parameters and PPO metrics exactly. See `Docs/experiments/EXP-002_EXPLORATION_ANALYSIS.md` for measured clipping rates and update-level policy statistics.

The next engineering decision should use the measured policy means, standard deviations, sampled actions and applied actions together to state one falsifiable learning hypothesis. No further experiment was started.

## 10. Limitations

- The feasibility check covers only ten seeded initial states and one manually specified feedback controller; it is not a proof for the entire initial-state distribution.
- The model has no ground-contact response, aerodynamic forces, actuator lag, force limits varying with attitude beyond the documented body-axis box, uncertainty, or sensor noise.
- Reward results depend on the stated starting state and scripted actions/controller. They do not establish a universal ranking over all possible trajectories.
- Initial stochastic clipping figures are analytical probabilities from the untrained policy distribution. No training was performed to count realized initial actions.
- PPO log fields are aggregate summaries; they cannot identify causal effects or reconstruct action/state trajectories from training.
- All results concern the existing simplified educational simulation. They are not evidence of physical rocket performance, controllability, or safety.
