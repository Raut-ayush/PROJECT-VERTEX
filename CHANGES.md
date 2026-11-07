# Change log

## 2026-11-07 — EXP-002 training-time exploration audit

- Added opt-in headless training JSONL diagnostics for sampled pre-clip actions, applied actions, clipping flags, policy means/stds, pre-action observation/state, rewards, terminal state, global timestep, rollout, and episode identifier. Default training remains uninstrumented.
- Inspected SB3 2.9.2 collector order and used a callback after `env.step()` but before rollout-buffer insertion. A parity test confirmed identical seeded policy parameters with diagnostics enabled and disabled; the 10,112-step EXP-002 checkpoint also exactly matches the original baseline policy parameters.
- Measured 31.54%, 31.10%, and 31.01% clipping by channel and 68.60% of action vectors with at least one clipped channel in the first 2,048 steps. Policy standard deviation stayed near 1.0 and clipping did not decline steadily over 16 rollouts.
- Full pytest suite: 56 passed. See `Docs/experiments/EXP-002_EXPLORATION_ANALYSIS.md`; no subsequent experiment was started.

## 2026-11-07 — Pretraining engineering audit

- Audited model control authority, actual environment reward rankings, the untrained PPO policy, stochastic action clipping behavior, and baseline/EXP-001 training metrics without retraining or changing simulator code.
- A documented full-state controller using existing physics/actions achieved the current success criteria on all ten fixed seeds; its first asymptotic descent variant exposed a controller issue and was corrected without changing the model.
- Used the real reward path on success, free fall/zero action, constant-altitude truncation, and sustained upward-force scenarios. Found a narrow discounted-return margin between success and constant altitude and a small gamma=0.99 ranking reversal between delayed upward failure and free fall.
- Fresh untrained policy means were near zero and never clipped deterministically on tested states; initial Gaussian standard deviation was 1.0, giving about 31.7% expected clipping per sampled scalar channel. Existing run logs do not contain training-time action-bound clipping counts.
- See `Docs/PRETRAINING_ENGINEERING_AUDIT.md` for equations, controller assumptions, per-seed results, reward-component totals, policy probes, log summaries, and limitations. No PPO training or EXP-002 was started.

## 2026-11-07 — EXP-001 gamma investigation

- Added an explicit `--gamma` training option with the baseline-preserving default of 0.99; saved run metadata now records gamma.
- Ran one new CPU PPO experiment at gamma=1.0 with seed 7 and 10,000 requested steps (10,112 collected), then deterministic diagnostics on seeds 1000–1009. No reward, environment, physics, observation, action or architecture changes were made.
- The new policy had 0/10 successes, 10/10 out-of-bounds episodes, and mean episode length 60.7 steps versus the baseline's 95.8. Its first-30 applied normalized body-Z action averaged 1.0 versus 0.9154, opposing the predicted reduction in upward force.
- See `Docs/experiments/EXP-001_GAMMA_INVESTIGATION.md` for settings, per-seed results, seed-1000 action/state comparison, limitations and recommendation. Existing baseline artifacts remain preserved.

## 2026-11-07 — PPO baseline behavior analysis

- Analyzed all 958 saved M6 diagnostic steps across seeds 1000–1009 and queried the preserved checkpoint on controlled one-variable state probes; no training or simulation files were changed.
- Recorded per-seed state/action outcomes, reward trends, moment/rotation behavior, raw versus clipped policy responses, and a one-step upper-bound unrecoverability estimate for seed 1000.
- Identified discounted terminal-failure delay as one testable learning hypothesis and recommended a gamma-only comparison. No experiment or retraining was performed.
- See `Docs/POLICY_BEHAVIOR_ANALYSIS.md` for detailed evidence, caveats, and the proposed next experiment.

## 2026-11-07 — Baseline instrumentation and interface validation

- Added opt-in per-step JSONL evaluation diagnostics for pre/post state, policy observation, pre-clipping deterministic policy action, applied bounded action, simulation-only wrench, reward components, and terminal status/reason.
- Added environment action decoding and reward-component reporting for diagnostics; the ordinary `step()` reward/action path and PPO training path remain unchanged.
- Added observation/action interface, reward, touchdown/boundary, integration, deterministic diagnostic replay, and scripted physics scenario tests.
- Corrected the existing integration-method wording to semi-implicit (symplectic) Euler without changing the update equations; no new engineering decision was introduced.
- Evaluated the preserved baseline checkpoint into new run directories. Repeated diagnostic JSONL was byte-identical, its episode results match the saved baseline, and existing checkpoint/log/evaluation files were hash-checked unchanged.
- See `Docs/BASELINE_INSTRUMENTATION_REPORT.md` for commands, results, detected action clipping, and limitations.

## 2026-11-07 — Initial PPO baseline failure diagnosis

- Added `Docs/BASELINE_FAILURE_REPORT.md` documenting observation/action data flow, physics signs/integration, reward and termination, checkpoint/training evidence, ranked hypotheses, and follow-up tests.
- Replayed the preserved checkpoint deterministically in memory on the existing ten evaluation seeds to inspect actions; did not change the environment, checkpoint, logs, or train a new policy.
- Corrected a previous diagnostic's mislabeled observation columns in the report. The seed-1000 starting vertical velocity is -0.398 m/s; the initial positive second action is body-z force and yields approximately +9.76 m/s² vertical acceleration in the toy model.
- No correction or retraining is claimed. Further action/reward telemetry and interface invariants are recommended before selecting a single task or learning change.

## 2026-11-07 — M5 optional episode viewer

- Added a lazy-import Matplotlib viewer for animated playback or static PNG export of M4 JSONL episode traces.
- The viewer plots trajectory, toy target band, ground line, state readout, and a display-only body glyph; it does not change physics or training.
- Added optional `visualization` dependency extra and tests for trace selection, older report metadata, and static image rendering.
- Installed Matplotlib 3.11.2 locally and rendered `runs/m4-evaluation-my-training-run/episode-1000.png` for visual inspection.

## 2026-11-07 — CLI defaults and duplicate-output handling

- Fixed training CLI numeric defaults: slotted dataclass fields were descriptors when referenced on the class, which caused the reported `TypeError`. CLI defaults now use scalar constants, with a regression test.
- Made train/evaluation commands report an existing output directory as a concise actionable CLI error and preserve existing artifacts. Evaluation checks for collisions before loading the policy.
- Verified the training CLI with a 128-step run and the full test suite after the fix.
- Ran the user's default 10,000-step command successfully: SB3 collected 10,112 steps, producing a checkpoint and logs. Evaluation on the default ten seeds found 0 successes (10 out-of-bounds episodes).

## 2026-11-07 — M4 deterministic evaluation

- Added CPU-only deterministic PPO checkpoint evaluation over a fixed default seed set (1000–1009), with explicit seed override.
- Added per-episode success/failure reason, return, length, final state, terminal category counts, touchdown metrics, and compact JSONL state traces.
- Evaluation summarizes recorded SB3 training return trends when the checkpoint's training CSV is present and checks policy/environment spaces before running.
- Added an end-to-end training, checkpoint reload, and evaluation artifact test.
- Evaluated the 256-step M3 smoke checkpoint on default seeds 1000–1009: 0/10 successes; all ten ended in unstable touchdown. Mean return was -10.908 and mean vertical touchdown speed was 12.505 m/s. The training CSV's recorded mean episode return changed from -10.783 to -10.918. This short run does not establish learning quality.

## 2026-11-07 — M3 PPO integration

- Added an optional Stable-Baselines3 PPO dependency extra and a headless training module using a small MLP policy and explicit CPU execution.
- Training creates a new run folder, CSV progress log, PPO checkpoint, and JSON metadata with source revision/dirty state, seed, configuration, versions, elapsed time, and throughput.
- Added a short training-and-reload integration test and documented the official CPU-only PyTorch setup.
- Environment: CPython 3.12.3, Stable-Baselines3 2.9.2, PyTorch 2.15.0+cpu, Gymnasium 1.4.0, NumPy 2.5.4.
- Smoke run: 256 timesteps, seed 123, CPU, approximately 1,711.8 steps/s. This is integration evidence only; no landing-learning claim.

## 2026-11-07 — M2 landing environment

- Added a headless Gymnasium environment around the independent physics step function.
- Defined a three-channel normalized simulation action, eight-value normalized observation, seeded initial-condition distribution, toy-task touchdown/out-of-bounds/time-limit criteria, and readable reward.
- Added optional initial-state reset injection for deterministic scenario verification.
- Added environment API, seeding, bounds, terminal-condition, and episode-truncation tests; included Gymnasium's environment checker.
- Added Gymnasium as an environment-layer dependency and recorded resolved versions and M2 task choices in `Docs/ENGINEERING_DECISIONS.md` EDL-011.

## 2026-11-07 — Initial M0/M1 implementation

- Added setuptools project metadata, Python 3.12 virtual-environment setup guidance, pytest configuration, and `.gitignore`.
- Added the `vertex` source package and headless `vertex.simulation` module.
- Implemented fixed-mass planar state advancement with constant gravity, generic simulation-only applied force/moment, and explicit forward Euler integration.
- Added focused tests for package imports, deterministic free fall, body/world force direction, pitch/moment signs, and invalid inputs.
- Added a root README describing setup, subsystem interface, and simulation/safety limitations.
- Recorded coordinate/integration conventions and inspected host environment in `Docs/ENGINEERING_DECISIONS.md`.

### Environment and evidence

- OS: Windows 10 Home Single Language.
- Python: CPython 3.12 and 3.14 are installed; `.venv` uses CPython 3.12.
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU, driver 616.56, CUDA UMD 13.4 as shown by `nvidia-smi`. No GPU use or speed claim was made.
- Package versions installed in `.venv`: pytest 9.2.0 and the package. Runtime dependencies: none.
- WMI access was denied; GPU inventory came from `nvidia-smi`. No prior project virtual environment was present. PyTorch/CUDA compatibility remains unchecked because M0/M1 do not require it.
