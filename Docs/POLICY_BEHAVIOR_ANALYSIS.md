# Baseline PPO Policy Behavior Analysis

**Project:** PROJECT VERTEX  
**Date:** 2026-11-07  
**Checkpoint:** `runs/my-training-run/ppo_vertex.zip`  
**Evaluation data:** `runs/m6-instrumented-eval-20261107/step_diagnostics.jsonl`  
**Status:** Analysis only. No training, environment, reward, physics, observation, or action configuration was changed.

## 1. Executive summary

The saved policy has learned a consistent “push upward” response over the tested initial states. Across the ten evaluated episodes, the first body-z force action was +0.958 to +1.000 normalized (about +19.2 to +20 N in the toy model). The action remained positive for 82.6% to 100% of each episode. All ten episodes climbed to the z=20 m upper boundary and terminated out of bounds.

The policy also applied negative pitch moment on every step in every episode. As angular velocity became negative, this moment accelerated the rotation further in the negative direction. The vehicle ended each episode with a large negative pitch and negative pitch rate.

**Primary testable hypothesis:** PPO learned to delay negative terminal outcomes by sustaining upward force. With gamma = 0.99, a -10 terminal failure penalty is increasingly discounted when failure is delayed, while dense per-step state penalties are small. The policy’s upward-biased actions, increased episode lengths during training, and discounted return decomposition are consistent with this mechanism. They do not prove causation.

The single recommended experiment is to change only PPO gamma from 0.99 to 1.0 in a new run, keeping the environment, reward, action interface, training seed, timestep budget, and all other settings fixed. This is a recommendation only; it was not run.

## 2. Seed 1000 chronological analysis

At reset, seed 1000 was at z=9.2077 m, vz=-0.3984 m/s, theta=+0.00575 rad, and omega=-0.03090 rad/s. The first policy observation was:

```text
[0.00428, 0.46038, -0.00581, -0.03984, 0.00575, 0.99998, -0.01545, 1.0]
```

That represents a small horizontal offset, altitude 9.21 m, mild leftward motion, downward velocity, nearly upright attitude, slight negative rotation, and the full episode time budget. The first action was `[-0.14264, +0.97941, -0.60551]`. Its body-z request was +19.59 N, and its moment was -1.21 N·m. Under the current force transform, the initial vertical acceleration was about +9.76 m/s². Vertical velocity changed from downward to upward by step 3.

For this analysis, “large upward request” means normalized body-z action `u_z >= 0.8` (at least +16 N in the educational force scale). That threshold is descriptive, not a task limit. It occurred on steps 1–30. The action stayed positive through step 94, then first became slightly negative on step 95. Its mean over the episode was +0.549; its positive fraction was 94.9%. Thus the policy did change its output as altitude and upward velocity rose, but the reduction came late.

| Step | z after (m) | vz after (m/s) | theta after (rad) | omega after (rad/s) | applied `u_z` | applied `u_M` | reward |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.204 | -0.203 | +0.005 | -0.055 | +0.979 | -0.606 | -0.0053 |
| 10 | 9.337 | +1.473 | -0.027 | -0.267 | +0.935 | -0.576 | -0.0089 |
| 25 | 10.180 | +3.902 | -0.161 | -0.602 | +0.841 | -0.547 | -0.0159 |
| 50 | 12.912 | +6.606 | -0.604 | -1.150 | +0.607 | -0.557 | -0.0268 |
| 75 | 16.508 | +7.477 | -1.324 | -1.710 | +0.270 | -0.553 | -0.0393 |
| 90 | 18.752 | +7.453 | -1.890 | -2.034 | +0.056 | -0.531 | -0.0493 |
| 95 | 19.495 | +7.402 | -2.099 | -2.140 | -0.001 | -0.534 | -0.0530 |
| 98 | 19.937 | +7.357 | -2.230 | -2.205 | -0.033 | -0.541 | -0.0553 |
| 99 | 20.084 | +7.338 | -2.275 | -2.227 | -0.044 | -0.544 | -10.0561 |

### Pitch-moment effect

The moment action was negative on all 99 steps (mean -0.555, corresponding to roughly -1.11 N·m). At the start, omega was already negative; negative moment therefore increased negative angular acceleration and reinforced the existing rotation. As theta and omega became more negative, moment remained negative rather than reversing to brake the angular rate or restore the attitude. For seed 1000, moment and omega had the same sign on all steps. The observed moment response was destabilizing for this trajectory.

The negative pitch also changes the direction of the body-axis forces. As the body rotates, body-z force contributes a horizontal component, while body-x force contributes vertical acceleration. The action channels cannot be interpreted as world-axis force once the attitude departs from zero.

### Reward evolution

All six state-shaping penalties became more negative in aggregate as the vehicle climbed and rotated. Across seed 1000, their undiscounted episode totals were:

| Reward component | Episode sum |
|---|---:|
| Horizontal position penalty | -0.0463 |
| Height penalty | -0.6677 |
| Horizontal velocity penalty | -0.1563 |
| Vertical velocity penalty | -0.5442 |
| Pitch penalty | -0.7770 |
| Pitch-rate penalty | -0.5702 |
| Terminal adjustment | -10.0000 |
| Total episode return | -12.7617 |

The per-step state shaping grew from -0.0053 on step 1 to about -0.0553 by step 98. The terminal -10 penalty dominated the final reward. The height penalty alone was only about -0.0046 initially and -0.0100 near the boundary; pitch became the largest single dense penalty near the end. This shows the current reward values penalize climbing and rotation, but much less per step than the terminal adjustment.

### First provably unrecoverable logged state

The trace alone cannot establish when every possible multi-step recovery plan became impossible. A narrower one-step bound can be calculated from the existing force limits and dynamics without changing the environment. Immediately before step 99, the state was z=19.93688 m, vz=7.35661 m/s, theta=-2.23039 rad. At that attitude, the most downward world-vertical acceleration available from the independent body-force bounds Fx,Fz ∈ [-20,20] N is approximately -37.8707 m/s² (gravity included). With dt=0.02 s and the existing updated-velocity position step, even that best-case force gives next z≈20.06886 m, above the strict z>20 m terminal boundary. Before step 98, the corresponding one-step minimum next z was about 19.92214 m, still inside.

Therefore, **the first state on the recorded trajectory proven unable to avoid upper-bound termination in the very next step is the pre-step-99 state** (after step 98). This is a one-step reachability bound, not a complete proof of the earliest point at which all longer recovery plans became impossible.

## 3. Cross-seed comparison

Every episode shows the same qualitative behavior: a near-maximum positive body-z action immediately after reset, negative moment throughout, upward velocity growth, negative pitch-rate growth, and termination at the upper altitude boundary. Initial conditions change the exact action magnitudes and episode duration, but do not change the overall policy pattern.

| Seed | Steps | Initial z / vz | Initial theta / omega | First `u_z` | Mean `u_z` (% positive) | Mean `u_M` | Final z / vz | Final theta / omega | Return |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 99 | 9.208 / -0.398 | +0.006 / -0.031 | +0.979 | +0.549 (94.9%) | -0.555 | 20.084 / +7.338 | -2.275 / -2.227 | -12.762 |
| 1001 | 109 | 8.031 / -0.071 | -0.085 / -0.030 | +0.958 | +0.444 (82.6%) | -0.558 | 20.007 / +6.691 | -2.848 / -2.464 | -13.375 |
| 1002 | 103 | 8.714 / -0.305 | -0.033 / +0.005 | +0.963 | +0.509 (90.3%) | -0.564 | 20.011 / +7.174 | -2.477 / -2.316 | -12.956 |
| 1003 | 97 | 8.513 / -0.170 | +0.050 / +0.022 | +0.988 | +0.606 (100%) | -0.543 | 20.027 / +8.026 | -1.974 / -2.084 | -12.610 |
| 1004 | 99 | 8.260 / -0.310 | +0.024 / +0.034 | +1.000 | +0.598 (100%) | -0.541 | 20.086 / +8.052 | -2.037 / -2.110 | -12.721 |
| 1005 | 85 | 9.861 / -0.181 | +0.067 / +0.048 | +1.000 | +0.707 (100%) | -0.545 | 20.059 / +8.800 | -1.434 / -1.807 | -12.124 |
| 1006 | 94 | 9.202 / -0.381 | +0.060 / -0.016 | +1.000 | +0.623 (100%) | -0.542 | 20.024 / +7.953 | -1.903 / -2.054 | -12.506 |
| 1007 | 92 | 9.632 / -0.402 | +0.018 / +0.040 | +1.000 | +0.627 (100%) | -0.564 | 20.105 / +8.038 | -1.846 / -2.034 | -12.411 |
| 1008 | 85 | 9.626 / -0.008 | +0.097 / +0.034 | +1.000 | +0.715 (100%) | -0.534 | 20.128 / +9.005 | -1.390 / -1.780 | -12.137 |
| 1009 | 95 | 8.654 / -0.359 | +0.065 / +0.011 | +1.000 | +0.640 (98.9%) | -0.533 | 20.060 / +8.298 | -1.832 / -2.015 | -12.548 |

Negative moment was applied for 100% of recorded steps in all ten episodes. The initial horizontal position range was -0.998 to +0.225 m and initial horizontal velocity range was -0.450 to +0.248 m/s; these differences did not prevent the common upward/negative-moment outcome. The first applied upward force was at least +19.16 N in every episode.

## 4. Policy input-output behavior

The preserved checkpoint was queried deterministically on valid, nonterminal `PlanarState` values. The seed-1000 reset state was the reference; one variable was changed at a time, while all others—including the reset time-budget observation—were held fixed. Values were inside observation bounds and task position limits. Some probes exceed the narrow reset distribution (for example, vx=±3 m/s and pitch=±0.5 rad); they are valid environment states but are local sensitivity probes, not typical initial samples.

The table shows the raw deterministic policy output `[u_x, u_z, u_M]` before clipping. The environment/SB3 applies each value clipped to [-1,1].

| Variable varied | Low value → raw `[u_x,u_z,u_M]` | High value → raw `[u_x,u_z,u_M]` | Observed response |
|---|---|---|---|
| Altitude z | 8 m → [-0.142, +0.968, -0.597] | 16 m → [-0.143, +1.033, -0.645] | Higher altitude increases the upward request; high output is clipped. |
| Vertical velocity vz | -2 m/s → [-0.125, +1.020, -0.592] | +2 m/s → [-0.169, +0.914, -0.623] | Faster upward motion reduces `u_z`, but it remains strongly positive. Faster downward motion raises it to the bound. |
| Horizontal position x | -4 m → [-0.005, +1.051, -0.529] | +4 m → [-0.277, +0.884, -0.670] | Positive x displacement produces more negative body-x force, directionally corrective; negative displacement produces almost no positive body-x correction in this probe. |
| Horizontal velocity vx | -3 m/s → [+0.105, +1.086, -0.373] | +3 m/s → [-0.382, +0.804, -0.793] | Body-x force changes sign against horizontal velocity; body-z remains positive. |
| Pitch theta | -0.5 rad → [-0.226, +0.885, -0.600] | +0.5 rad → [-0.045, +0.998, -0.573] | Moment remains negative for both signs; it is corrective for positive pitch but worsens negative pitch. |
| Angular velocity omega | -0.5 rad/s → [-0.168, +0.943, -0.596] | +0.5 rad/s → [-0.113, +1.013, -0.612] | Moment remains negative; it reinforces negative rotation and opposes positive rotation. The change across this rate range is small relative to the negative bias. |

The probes support several narrow observations about this checkpoint’s mapping. It reacts to horizontal velocity with a directional force response. It reacts to faster upward velocity by reducing, but not reversing, body-z force. It does not reverse its moment when pitch or angular rate changes sign. These are local deterministic input-output results; they do not establish why PPO learned the mapping, and they do not establish behavior for untested states.

## 5. Learning problem review

- **Reward:** The per-step penalties are small, while every failure ends with -10 and success receives +10. PPO’s gamma is 0.99. For seed 1000, the observed path’s undiscounted return is -12.7617, but its gamma-discounted return is about -5.2426: the -10 terminal component contributes -3.7346 at step 99 and discounted dense shaping contributes about -1.508. This decomposition is calculated from the saved path; it does not compare against a controlled alternative.
- **Observation:** The eight inputs include the relevant state signs, but normalized values clip at [-1,1]. On this episode, pitch-rate magnitude reached the clipped observation limit near step 60 and stayed saturated; z approached its scale limit near termination. Clipping therefore removes magnitude information late in an already abnormal trajectory, but not at reset.
- **Initialization:** All seeds start at 8–10 m with vertical velocity in [-0.5,0] m/s, nearly upright, and small pitch rate. Each gets essentially the same high upward action immediately. The one-variable probes also show high upward force for descending states.
- **Termination:** Success is a narrow controlled-touchdown condition; out-of-bounds occurs above 20 m or 10 m horizontal error. The observed episodes reach the upper boundary in 85–109 steps, well before the 500-step timeout.
- **PPO setup and progress:** One environment, seed 7, [64,64] policy, n_steps=128, batch size=64, five epochs, learning rate 0.0003, gamma=0.99, GAE lambda=0.95, entropy coefficient=0, and 10,112 actual steps. The logged mean episode length rose from 65.0 at the first record to 97.91 at the final record, while mean undiscounted return worsened from -10.876 to -12.243. Final action standard deviation remained about 1.006. The run contains roughly a hundred episodes and no successful landing examples.

## 6. Root-cause hypotheses ranked by evidence

1. **Discounted failure delay is rewarding “stay aloft” behavior — strongest testable learning hypothesis, not proven.** Directly demonstrated facts are gamma=0.99, terminal failures receiving -10, dense penalties being much smaller, upward-biased policy output, and training episode lengths increasing while return worsened. The inference is that postponing a negative terminal event can look favorable to the discounted PPO objective even while the undiscounted episode return gets worse. The seed-1000 discounted decomposition is consistent with that mechanism.
2. **The policy failed to learn angular-rate control — strong behavior evidence, underlying cause unknown.** Negative moment was constant across all ten episodes, and one-variable probes show it does not reverse for positive or negative angular velocity. The resulting seed-1000 motion increases negative pitch and rate. Why optimization settled on this output remains unproven.
3. **Training exposure was limited and had no successful examples — plausible contributing factor.** About 10k steps produced roughly 100 episodes; there is no positive touchdown trajectory to reinforce. This is not proof that more steps alone would solve the task.
4. **Observation clipping weakens late failure-state feedback — demonstrated interface behavior, weak as the initiating cause.** omega clips before the terminal event, but initial observations are within scale and the large upward action appears at reset.

## 7. Recommended next experiment

Run one isolated PPO comparison that changes **only gamma from 0.99 to 1.0**. Keep the baseline reward, observation, action bounds/scales, physics, training seed (7), requested budget (10,000; expected 10,112 rollout steps), network, rollout length, batch size, epochs, optimizer settings, and evaluation seeds 1000–1009 unchanged. Save to a new run directory; do not replace the preserved baseline.

Predefine comparison measures: mean applied `u_z` over the first 30 steps, fraction of steps with positive `u_z`, terminal-category counts, episode length, and success rate. The hypothesis predicts less incentive to postpone terminal failure, expressed as a lower early upward-force request and/or fewer upper-bound exits, possibly with more touchdown attempts. If the policy mapping and outcome categories do not move in that direction, the discount-delay hypothesis is weakened. A change in success rate is not guaranteed.

This is a proposal only. No gamma change, training run, or environment modification was made here.

## 8. Limitations

The ten episodes are a small fixed evaluation set. Deterministic inference describes the policy mean at selected states, not stochastic training actions or all possible observations. Controlled probes are local and do not establish global policy behavior. The one-step unrecoverability check uses the current simplified force bounds and integrator; it is not a general reachability proof. Discounted-return decomposition of the realized trace is observational and cannot alone prove the policy’s learning motive. Nothing here validates physical safety or flight readiness.
