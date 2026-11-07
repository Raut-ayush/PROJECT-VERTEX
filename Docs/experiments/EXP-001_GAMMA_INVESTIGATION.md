# EXP-001: Gamma investigation

**Date:** 2026-11-07  
**Status:** Complete; one training run and one fixed-seed evaluation  
**Scope:** Simplified research simulation only. This experiment says nothing about physical rocket controllability or flight readiness.

## 1. Hypothesis

The prior analysis proposed that PPO's `gamma=0.99` may favor delaying terminal failure, helping explain the baseline's sustained upward body-Z force. The test prediction was that changing only `gamma` to `1.0` could reduce early upward action and promote descent or touchdown attempts. An unchanged or worse policy would reduce support for this explanation.

## 2. Experimental configuration

Training used the existing headless PPO pipeline and environment. The new run directory was `runs/exp-001-gamma-1.0/`; evaluation artifacts are in `runs/exp-001-gamma-1.0/evaluation/`.

| Setting | Baseline | EXP-001 |
|---|---:|---:|
| Algorithm / network | PPO / [64, 64] | Same |
| Training seed | 7 | 7 |
| Requested / collected steps | 10,000 / 10,112 | 10,000 / 10,112 |
| Rollout / batch / epochs | 128 / 64 / 5 | Same |
| Learning rate | 0.0003 | Same |
| Gamma | 0.99 | **1.0** |
| GAE lambda | 0.95 | 0.95 |
| Entropy / value coefficient | 0 / 0.5 | Same |
| Clip range | 0.2 | 0.2 |
| Device | CPU | CPU |
| Environment, reward, observation, action, vehicle and physics | Existing baseline | Same |

The original checkpoint reports `gamma=0.99` when loaded; the new checkpoint reports `gamma=1.0`. Other listed PPO settings were checked from the loaded models and run metadata. `src/vertex/learning/train.py` now accepts `--gamma` and records it in `training_config`; its default remains 0.99 so existing default runs retain their previous setting. This was the only training-code change.

## 3. Training results

Exact command:

```powershell
.\.venv\Scripts\python.exe -m vertex.learning.train --timesteps 10000 --seed 7 --gamma 1.0 --output-dir runs/exp-001-gamma-1.0
```

The run collected 10,112 steps over 79 logged rollout records. Training elapsed time was 7.169 seconds (metadata timing), or about 1,410.6 steps/s. The final saved checkpoint is `runs/exp-001-gamma-1.0/ppo_vertex.zip`; exact configuration and package/source metadata are in `run_metadata.json`, with SB3 metrics in `logs/progress.csv`.

| Training log metric | First recorded (128 steps) | Final (10,112 steps) |
|---|---:|---:|
| Mean episode return | -10.876 | -13.082 |
| Mean episode length | 65.00 | 84.45 |
| Approximate KL (final) | — | 0.00789 |
| Explained variance (final) | — | 0.9472 |
| Value loss (final) | — | 0.03005 |
| Policy gradient loss (final) | — | -0.00604 |
| Clip fraction (final) | — | 0.02656 |
| Policy standard deviation (final) | — | 0.9688 |

For comparison, the baseline's final training log reports mean episode return -12.243 and length 97.91. These training statistics are moving rollout-window values, not independent evaluation results.

## 4. Evaluation results

Exact command (the evaluator's default fixed seeds are 1000–1009):

```powershell
.\.venv\Scripts\python.exe -m vertex.learning.evaluate --checkpoint runs/exp-001-gamma-1.0/ppo_vertex.zip --output-dir runs/exp-001-gamma-1.0/evaluation --diagnostics
```

The command evaluated ten deterministic episodes and reported 0% success; all ten terminated out of bounds. Per-episode summaries are in `evaluation/evaluation.json` and `evaluation/episodes.csv`; per-step state, normalized observation, raw policy output, clipped action, applied wrench and rewards are in `evaluation/step_diagnostics.jsonl`.

| Metric | Baseline (`gamma=.99`) | EXP-001 (`gamma=1.0`) |
|---|---:|---:|
| Successes | 0/10 | 0/10 |
| Out of bounds | 10/10 | 10/10 |
| Unstable touchdowns | 0/10 | 0/10 |
| Episodes reaching touchdown | 0/10 | 0/10 |
| Mean episode length (steps) | 95.8 | 60.7 |
| Mean episode return | -12.6150 | -12.7642 |
| Mean applied normalized body-Z action, first 30 steps pooled across seeds | 0.9154 | 1.0000 |
| Positive body-Z actions, first 30 steps pooled | 100% | 100% |
| Positive body-Z actions, all evaluated steps | 96.45% | 100% |
| Mean applied pitch moment over all steps (N·m) | -1.0964 | +1.9218 |
| Mean final pitch (rad) | -2.0015 | +1.5317 |
| Mean final angular velocity (rad/s) | -2.0891 | +2.3448 |

`action_applied[1]` is the normalized body-Z channel in `[-1, 1]`; the corresponding applied force is recorded separately in newtons. The mean is computed across diagnostic steps (first-30 metrics include each episode's first 30 steps). Both policies requested positive body-Z force immediately. In EXP-001 all applied channels were positive on every step; the horizontal channel was at +1.0 on all 607 steps, body-Z was at +1.0 on 602/607 steps, and moment was at +2.0 N-m on 396/607 steps. On its first step, the raw deterministic action was `[2.3565, 1.7432, 1.0980]` and the environment-applied action was clipped to `[1, 1, 1]`.

### Per-seed results

Each seed started from the same reset state as in the baseline. Final pitch and rate are shown at the out-of-bounds terminal state.

| Seed | Baseline steps / return | EXP-001 steps / return | EXP-001 final pitch (rad) | EXP-001 final rate (rad/s) |
|---:|---:|---:|---:|---:|
| 1000 | 99 / -12.762 | 58 / -12.609 | 1.332 | 2.234 |
| 1001 | 109 / -13.375 | 54 / -12.303 | 1.069 | 2.104 |
| 1002 | 103 / -12.956 | 56 / -12.493 | 1.246 | 2.202 |
| 1003 | 97 / -12.610 | 64 / -13.019 | 1.721 | 2.448 |
| 1004 | 99 / -12.721 | 66 / -13.113 | 1.811 | 2.514 |
| 1005 | 85 / -12.124 | 60 / -12.687 | 1.576 | 2.358 |
| 1006 | 94 / -12.506 | 63 / -12.920 | 1.635 | 2.393 |
| 1007 | 92 / -12.411 | 63 / -13.004 | 1.663 | 2.444 |
| 1008 | 85 / -12.137 | 59 / -12.584 | 1.542 | 2.311 |
| 1009 | 95 / -12.548 | 64 / -12.910 | 1.722 | 2.439 |

### Seed 1000, first 30 steps

The reset state is identical for both runs (`z=9.2077 m`, `vz=-0.3984 m/s`, `theta=0.00575 rad`, `omega=-0.03090 rad/s`). At the displayed checkpoints, `u_z` is the applied normalized body-Z action, `M` is applied simulation moment, and state values are immediately before that step.

| Step | Baseline z / vz | Baseline pitch / rate | Baseline u_z / M | EXP-001 z / vz | EXP-001 pitch / rate | EXP-001 u_z / M |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.208 / -0.398 | 0.006 / -0.031 | 0.979 / -1.211 | 9.208 / -0.398 | 0.006 / -0.031 | 1.000 / +2.000 |
| 5 | 9.215 / +0.372 | -0.002 / -0.127 | 0.960 / -1.184 | 9.217 / +0.428 | 0.011 / +0.129 | 1.000 / +2.000 |
| 10 | 9.307 / +1.294 | -0.021 / -0.244 | 0.935 / -1.152 | 9.323 / +1.486 | 0.036 / +0.329 | 1.000 / +2.000 |
| 15 | 9.490 / +2.168 | -0.053 / -0.358 | 0.907 / -1.126 | 9.538 / +2.607 | 0.081 / +0.529 | 1.000 / +2.000 |
| 20 | 9.756 / +2.991 | -0.095 / -0.470 | 0.875 / -1.106 | 9.871 / +3.825 | 0.146 / +0.729 | 1.000 / +2.000 |
| 25 | 10.102 / +3.756 | -0.149 / -0.580 | 0.841 / -1.093 | 10.333 / +5.166 | 0.231 / +0.929 | 1.000 / +2.000 |
| 30 | 10.520 / +4.458 | -0.213 / -0.689 | 0.803 / -1.088 | 10.937 / +6.647 | 0.336 / +1.129 | 1.000 / +2.000 |

By step 30, EXP-001 is rising 2.19 m/s faster than baseline at a 0.42 m higher altitude. It also rotates in the opposite direction because the applied pitch moment changed sign. Neither run starts a transition toward descent within these first 30 steps.

## 5. Hypothesis assessment

The prediction that `gamma=1.0` would reduce early upward force is contradicted in this run: first-30 mean normalized body-Z action rose from 0.9154 to 1.0, and mean vertical velocity after step 30 across seeds rose from 4.879 to 7.412 m/s. Episodes reached a boundary sooner (60.7 versus 95.8 steps); seven ended above the upper height limit and three exceeded the horizontal limit, while all ten baseline episodes exceeded the upper height limit. Success and touchdown counts remained zero. Mean return was also slightly worse (-12.764 versus -12.615).

This result **reduces support** for the specific claim that gamma=0.99 is the cause of the sustained upward action. It does not establish why PPO chose that behavior. A discount factor of 1.0 substantially changed the learned mapping in this single controlled comparison, including reversing moment sign and driving all three action channels to their positive bounds; it did not produce a viable descent or touchdown attempt.

## 6. Limitations

- This is one training seed and one training run per gamma. The fixed training seed and single-parameter change make the comparison controlled, but do not estimate run-to-run variability.
- Ten evaluation seeds are a small deterministic sample from this toy environment's reset distribution.
- Evaluation is deterministic policy inference; it does not measure the stochastic action distribution used during training.
- Raw policy outputs can exceed the environment's action bounds and are clipped before application. The large raw outputs explain saturation but do not by themselves explain the optimization outcome.
- The simulation is simplified research software. These results are not evidence of physical rocket performance, controllability, or safety.

## 7. Recommended next experiment

Do not treat `gamma=1.0` as a correction. The next informative experiment is a paired-seed replication: train both gamma=0.99 and gamma=1.0 with seeds 8 and 9, keeping every other setting identical within each pair, then evaluate each checkpoint on seeds 1000?1009. Predeclare first-30 mean applied body-Z action and terminal categories as primary measures. This tests whether EXP-001?s saturation and shorter episodes follow gamma or this one training run. It is a recommendation only; no further run was started. Preserve both existing checkpoints and logs.
