# EXP-002: Training-time exploration analysis

**Date:** 2026-11-07  
**Status:** Complete; one instrumented reproduction of the gamma=0.99 baseline  
**Run:** `runs/exp-002-exploration-diagnostics/`  
**Scope:** Educational simulation only; no physical-controller interpretation.

## 1. Purpose and hypothesis

The pretraining audit estimated that an untrained unit-standard-deviation Gaussian policy would generate actions outside the environment's [-1,1] bounds frequently, but existing baseline logs did not measure sampled actions. EXP-002 records the sampled action, the action delivered to the environment, and the policy distribution during only the first 2,048 environment steps. The remaining steps train normally without JSONL writes or the extra distribution query.

The instrumentation question was whether actual PPO samples clipped at a rate near the initial estimate, and whether policy updates quickly reduced clipping or developed useful state-dependent means.

## 2. Configuration and artifacts

Training used gamma 0.99, seed 7, 10,000 requested steps, 10,112 collected steps, CPU, network [64,64], rollout 128, batch 64, five epochs, learning rate 0.0003, GAE lambda 0.95, entropy coefficient 0, value coefficient 0.5, clip range 0.2, and the unchanged environment/reward/action/observation/physics configuration.

Command:

```powershell
.\.venv\Scripts\python.exe -m vertex.learning.train --timesteps 10000 --seed 7 --gamma 0.99 --diagnostics --diagnostic-steps 2048 --output-dir runs/exp-002-exploration-diagnostics
```

Training took 7.401 seconds according to run metadata (about 1,366 steps/s). The checkpoint, training CSV and metadata are in the run directory. The 2,048 transition records are in `training_diagnostics.jsonl`. This was the only full training run for EXP-002; no evaluation retraining or hyperparameter search was performed.

### Diagnostic schema and non-interference

Each JSONL record contains schema version, global timestep, rollout index, environment and episode identifiers, observation before the action, physical `state_before`, sampled action before clipping, action passed to the environment, per-channel clipping booleans, policy mean and standard deviation, reward, terminated/truncated flags, and terminal reason. The window contains exactly timesteps 1–2,048 (16 rollouts of 128 steps).

Before implementation, the installed Stable-Baselines3 2.9.2 `OnPolicyAlgorithm.collect_rollouts()` source was inspected. It calls the callback after `env.step()` and before `rollout_buffer.add()`. The callback reads those collector locals without replacing or mutating them. It makes an additional deterministic `get_distribution()` query on the same observation to record the policy mean/std; no samples are drawn by that query. Physical state is read from the existing environment attribute before the action and cached across steps; the callback does not reset or step the environment itself.

A seeded parity test trained the same short run with diagnostics disabled and enabled. The resulting policy parameters were exactly equal. In the full reproduction, the final policy parameters were also exactly equal to the original baseline checkpoint (maximum absolute parameter difference 0.0). All non-timing training-metric CSV values and updates matched the original baseline exactly; elapsed/FPS timing fields differed as expected. This confirms that the diagnostic callback did not change the sampled-action/RNG sequence or PPO updates in these reproductions. Recording adds runtime overhead during the first 2,048 steps only.

## 3. Measured clipping frequency

Clipping is measured by whether each sampled value was outside its Box bound before collection and whether any channel in the vector clipped. There were 2,048 sampled action vectors and 6,144 scalar action values.

| Channel | Clipped samples | Measured clipping rate |
|---|---:|---:|
| Body-X force | 646 / 2,048 | 31.54% |
| Body-Z force | 637 / 2,048 | 31.10% |
| Pitch moment | 635 / 2,048 | 31.01% |
| Any channel in complete action vector | 1,405 / 2,048 | 68.60% |

The measured per-channel rates are close to the pretraining audit's analytical initial estimate of 31.7% per scalar and 68.2% for at least one channel in a three-channel vector. EXP-002 rates are empirical for this actual early training trajectory; they are not the analytical estimate.

## 4. Sampled and applied action distributions

Actions are normalized in [-1,1] and ordered `(body-X, body-Z, pitch moment)`. Quantiles below pool all first-window samples across the three PPO rollouts/states and should not be read as independent identically distributed samples.

| Channel | Stage | Mean | Std dev | 5th percentile | 95th percentile | Minimum | Maximum |
|---|---|---:|---:|---:|---:|---:|---:|
| Body-X | sampled | -0.0691 | 1.0015 | -1.7267 | +1.5800 | -3.1197 | +3.0962 |
| Body-X | applied | -0.0488 | 0.7211 | -1.0000 | +1.0000 | -1.0000 | +1.0000 |
| Body-Z | sampled | +0.0846 | 0.9983 | -1.5011 | +1.7313 | -3.3131 | +3.6598 |
| Body-Z | applied | +0.0490 | 0.7259 | -1.0000 | +1.0000 | -1.0000 | +1.0000 |
| Pitch moment | sampled | +0.0128 | 0.9913 | -1.6141 | +1.6947 | -3.7903 | +2.9825 |
| Pitch moment | applied | +0.0048 | 0.7158 | -1.0000 | +1.0000 | -1.0000 | +1.0000 |

Clipping reduced the applied-action standard deviation from about 1.0 to about 0.72 and created probability mass at both action bounds. It also shifted the pooled means toward zero. The raw distributions are the samples PPO stores for likelihood updates; the bounded distributions are what the environment receives.

## 5. Policy means and standard deviation across PPO rollouts

Each row summarizes the 128 transitions collected by the policy version in that rollout. `mean μ` is the per-channel policy mean averaged across the observations in that rollout; `σ` is the learned diagonal Gaussian standard deviation. Clip rates are empirical sampled-action clipping rates in that rollout.

| Rollout | Mean μ (body-X, body-Z, moment) | σ (body-X, body-Z, moment) | Clipped by channel (%) | Any channel clipped (%) |
|---:|---|---|---|---:|
| 1 | [-0.0043, +0.0030, +0.0030] | [1.0000, 1.0000, 1.0000] | [28.91, 32.03, 28.91] | 65.63 |
| 2 | [-0.0280, +0.0335, +0.0098] | [0.9985, 0.9981, 1.0022] | [33.59, 27.34, 28.91] | 67.19 |
| 3 | [-0.0606, +0.0704, +0.0283] | [0.9979, 0.9978, 1.0042] | [24.22, 29.69, 32.03] | 67.19 |
| 4 | [-0.0832, +0.1126, +0.0325] | [0.9984, 0.9973, 1.0051] | [28.13, 33.59, 31.25] | 69.53 |
| 5 | [-0.1075, +0.1498, +0.0272] | [1.0001, 0.9969, 1.0062] | [34.38, 30.47, 32.81] | 71.09 |
| 6 | [-0.0975, +0.1351, +0.0237] | [1.0017, 0.9952, 1.0064] | [22.66, 25.78, 29.69] | 56.25 |
| 7 | [-0.0607, +0.1036, +0.0485] | [1.0031, 0.9940, 1.0058] | [32.03, 34.38, 39.06] | 77.34 |
| 8 | [-0.0261, +0.0986, +0.0558] | [1.0043, 0.9937, 1.0042] | [29.69, 39.84, 26.56] | 67.19 |
| 9 | [-0.0072, +0.0508, +0.0294] | [1.0066, 0.9942, 1.0014] | [39.06, 33.59, 25.78] | 68.75 |
| 10 | [+0.0071, -0.0074, -0.0286] | [1.0070, 0.9948, 0.9997] | [36.72, 28.13, 22.66] | 68.75 |
| 11 | [+0.0003, -0.0206, -0.0061] | [1.0055, 0.9949, 0.9990] | [36.72, 29.69, 32.81] | 71.88 |
| 12 | [-0.0225, -0.0173, +0.0034] | [1.0037, 0.9947, 0.9989] | [32.81, 27.34, 30.47] | 67.97 |
| 13 | [-0.0506, -0.0154, +0.0157] | [1.0032, 0.9942, 0.9994] | [34.38, 29.69, 32.81] | 75.00 |
| 14 | [-0.0714, -0.0193, +0.0425] | [1.0032, 0.9936, 1.0006] | [27.34, 33.59, 38.28] | 71.88 |
| 15 | [-0.0675, -0.0075, +0.0561] | [1.0028, 0.9923, 1.0028] | [35.94, 33.59, 26.56] | 67.19 |
| 16 | [-0.0567, +0.0228, +0.0724] | [1.0065, 0.9927, 1.0080] | [28.13, 28.91, 37.50] | 64.84 |

Across the whole window, mean policy standard deviations were `[1.0027, 0.9953, 1.0027]`; their measured ranges were `[0.9979, 0.9923, 0.9989]` to `[1.0070, 1.0000, 1.0080]`. Exploration scale therefore stayed close to one. Clipping did not steadily decline: the action-vector rate ranged from 56.25% to 77.34% by rollout, and the final rollout was 64.84%.

The policy means did change across PPO updates. For example, the average body-Z mean moved from +0.0030 in rollout 1 to +0.1498 in rollout 5, then to -0.0206 in rollout 11 and +0.0228 in rollout 16. The shifts are nonmonotonic. Across all records the policy-mean ranges were [-0.1252,+0.0143] for body-X, [-0.0453,+0.1705] for body-Z, and [-0.0476,+0.1320] for moment. These magnitudes are small compared with standard deviation near 1.0.

Some within-rollout state association is visible late in the window. In rollout 16, the body-Z policy mean correlated 0.927 with normalized height and 0.991 with normalized vertical velocity; the body-X mean correlated -0.980 with normalized horizontal velocity. These are correlations along one short, policy-generated trajectory where state variables co-vary. They do not establish causal response or controller quality. The weak mean magnitude relative to the stochastic spread means exploration still dominates the action selected on most individual steps.

## 6. State, reward, and episode coverage

The file contains one record per environment step from global timestep 1 through 2,048. Each record's observation and physical state are the pre-action values. The first record exactly matched a fresh environment reset at training seed 7. Within the window, 29 episode identifiers appeared and 28 episodes terminated as `unstable_touchdown`; episode 29 was still in progress at the window boundary. Records include each transition's scalar reward, termination/truncation flags and reason. No image or rendering data was collected.

## 7. Comparison with the original baseline

The baseline was `runs/my-training-run/`; EXP-002 used a new directory. The original baseline checkpoint and `logs/progress.csv` SHA-256 hashes were unchanged by the experiment. Checkpoint hash: `6675eee26c6d64b05512a47a4874e32f1ab9f531d725f784c3af770c05e2725a`; progress CSV hash: `ddf9adcd957e9a5d19fa76868bc699c1064d03490f704042c068ba14da0514da`. The experiment checkpoint's policy parameters match the original baseline checkpoint exactly (maximum absolute parameter difference 0.0). Every common non-timing training-metric CSV value and update matched exactly; elapsed/FPS timing fields differed due to runtime, as expected. This is strong reproduction evidence that the opt-in instrumentation preserved actions, RNG sequence, stored training experiences and PPO updates in this run.

| Training log metric | Original baseline final | EXP-002 final |
|---|---:|---:|
| Collected steps | 10,112 | 10,112 |
| Mean rollout episode return | -12.242646 | -12.242646 |
| Mean rollout episode length | 97.91 | 97.91 |
| Explained variance | 0.679527 | 0.679527 |
| Approximate KL | 0.001797 | 0.001797 |
| PPO `train/clip_fraction` | 0.000000 | 0.000000 |
| Policy standard deviation | 1.006372 | 1.006372 |

`train/clip_fraction` is PPO's probability-ratio clipping statistic during policy optimization. It is not the 68.60% measured frequency of sampled action vectors with at least one action-bound clip.

## 8. Findings and interpretation

1. Measured initial-window action-bound clipping closely matched the unit-Gaussian estimate: about 31% per channel and 68.6% per vector.
2. Clipping remained frequent throughout all 16 rollout updates. There is no evidence in this window that PPO rapidly reduced action-bound saturation.
3. PPO updates moved the policy means, but means remained small relative to standard deviation near one. The resulting per-step actions remained largely noise-driven.
4. A weak state-dependent policy mean emerged in some rollout slices, but the state trajectory is small and confounded; this is not evidence of a useful learned landing response.
5. The instrumented full run exactly reproduced baseline policy weights and logged PPO metrics, while the original artifacts remained unchanged.

These findings confirm that initial stochastic clipping is real and that early policy updates do not quickly eliminate it. They do not establish that clipping caused the baseline failure. The run still follows the original unsuccessful training trajectory.

## 9. Recommended next experiment

Do not change the environment or PPO parameters based on clipping frequency alone. The next decision should use the measured `policy_mean`, `policy_standard_deviation`, sampled action and applied action together. If a new experiment is later approved, select one falsifiable explanation for why PPO updates move the mean in this trajectory and change only one factor. EXP-002 does not itself justify a reward, action-space, or algorithm change. No further experiment was started.

## 10. Limitations

- The JSONL window covers the first 2,048 of 10,112 environment steps, not the full training trajectory.
- It is one seed and one run. The deterministic match to the original checkpoint proves this instrumentation did not perturb this reproduction; it does not establish run-to-run learning robustness.
- State correlations use one trajectory per rollout and cannot separate state response from time/trajectory covariance.
- Action clipping percentages depend on the policy's evolving means/std and visited states. They are not universal properties of PPO.
- This remains a simplified research simulation and says nothing about physical rocket controllability, safety, or flight readiness.
