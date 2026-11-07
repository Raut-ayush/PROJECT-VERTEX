# PROJECT VERTEX — Project Context

**Version:** 0.1  
**Date:** 2026-11-07  
**Status:** Initial baseline

## 1. Mission and long-term intent

PROJECT VERTEX is a systems-engineering and learning project whose long-term aspiration is to develop a real, reusable vertical rocket demonstrator capable of autonomous powered vertical landing.

The project owner wants to learn how simulation, reinforcement learning, vehicle dynamics, avionics, and system integration fit together. The project is not intended to turn the owner into an ML specialist. Explanations should stay at an engineering-overview level: what a subsystem does, why it is needed, its inputs and outputs, what can fail, and how we will verify it.

The initial project must be simple enough for a first reinforcement-learning project while preserving a credible path toward the long-term objective.

## 2. First project objective

Build a lightweight, headless 2D planar rigid-body landing simulation and train one PPO agent to learn a controlled landing task. The simulation must run without visualization and expose a separate optional evaluation/viewer path.

The first achievement is not a real powered flight. It is:

1. A documented, testable 2D vehicle model.
2. A working Gymnasium-compatible environment.
3. A reproducible PPO training run.
4. Quantitative evaluation of the trained policy.
5. A separate optional visualization for inspection and communication.

After the simulation has passed defined validation gates, the project may investigate software/hardware interfaces and a safe non-propulsive physical demonstrator. That demonstrator can validate selected electronics, sensor interfaces, inference, command handling, and passive landing mechanics. It does not prove powered-rocket flight capability.

## 3. Frozen user decisions

| Topic | Decision |
|---|---|
| Overall outcome | Simulation plus eventual physical demonstrator |
| Long-term vehicle aspiration | Real vertical rocket with powered landing |
| Initial model | 2D rigid-body |
| Model state | Position, velocity, pitch, angular velocity |
| Initial RL algorithm | PPO only |
| Algorithm comparison | Not in v0.1 |
| Initial rendering | Off during training; optional outside training |
| First physical validation | Start with whichever safe non-propulsive rig is simpler; eventually consider both a captive/guided test rig and passive drop-test lander |
| Initial spending | ₹0; software-only |
| Learning style | Beginner-friendly, system-engineering overview; avoid deep derivations unless requested |

## 4. User's working context

The owner works in VS Code and plans to use Codex for implementation. The owner is comfortable learning Python and has prior exposure to electronics, embedded systems, drones, and simulation-related projects, but is new to reinforcement learning.

The last reported development machine configuration was:

- Intel Core i5-11800HX
- NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM
- 16 GB DDR4 RAM
- 1 TB SSD
- Windows development environment

Earlier software troubleshooting showed a PyTorch CUDA-enabled installation working with CUDA 12.8, but this is historical information. Codex must inspect the actual current machine, Python installations, driver, CUDA availability, and package compatibility rather than assuming that old environment details are still valid.

The owner previously had multiple Python versions installed, including Python 3.12 and newer versions. Avoid blindly using the system default Python. Select a version supported by the current dependency matrix and create a project-local virtual environment.

### Current workspace inspection (2026-11-07)

The current machine reports Windows 10 Home Single Language, CPython 3.12 and 3.14, and an NVIDIA GeForce RTX 3050 Laptop GPU with driver 616.56 / CUDA UMD 13.4 (`nvidia-smi`). WMI inventory access was denied. A project `.venv` now uses CPython 3.12. M0/M1 require no GPU or runtime package; PyTorch/CUDA package compatibility remains to be checked if a later milestone needs it. See `Docs/ENGINEERING_DECISIONS.md` EDL-010.

M2 now adds Gymnasium 1.4.0 and NumPy 2.5.4 to the CPython 3.12 environment. The standalone physics layer remains independent from those packages. The landing task interface and all educational thresholds are recorded in EDL-011; none are real vehicle limits.

M3 adds Stable-Baselines3 PPO 2.9.2 with PyTorch 2.15.0+cpu. Training is explicitly CPU-only pending any future benchmark. The corrected default CLI completed 10,112 actual steps (whole 128-step rollouts) and saved a checkpoint. M4 evaluation on the default ten seeds found zero successes, all out-of-bounds. See EDL-012 and EDL-013.

M4 adds deterministic checkpoint evaluation, a default fixed seed set (1000–1009), terminal-category and touchdown metrics, training-return trend summaries, and headless episode traces. The M3 smoke checkpoint had zero successes and ten unstable touchdowns over the default seeds; mean vertical touchdown speed was 12.505 m/s. The 10,112-step run also had zero successes on ten seeds, all out-of-bounds. These short runs show that the current policy has not landed successfully.

M5 provides optional Matplotlib playback and static image export from M4 traces. Rendering stays outside training and physics. Install with `pip install -e ".[visualization]"`; run `python -m vertex.visualization.viewer <episode_traces.jsonl> --seed <seed>` for animated playback. See EDL-014.

The initial PPO failure diagnosis is recorded in `Docs/BASELINE_FAILURE_REPORT.md`. Deterministic replay shows the saved policy applies sustained positive body-z force and negative moment before all ten episodes exit at the upper boundary. This is the immediate trajectory cause, not yet an explanation for why PPO learned those actions. No environment change or retraining has been performed; preserve the baseline while adding action/reward telemetry and interface invariants before selecting a correction.

The baseline instrumentation and validation work is recorded in `Docs/BASELINE_INSTRUMENTATION_REPORT.md`. Evaluation can optionally emit per-step JSONL diagnostics; the default evaluation and training paths remain unchanged. Interface, reward, boundary, numerical integration, deterministic replay, and scripted physics scenario tests now cover the identified gaps. The original checkpoint and M4 baseline artifacts were preserved.

The saved PPO policy's behavior is analyzed in `Docs/POLICY_BEHAVIOR_ANALYSIS.md`. All ten evaluation seeds start with strong upward body-z force and negative pitch moment; seed 1000's first state proven unable to avoid the upper boundary on the next step is immediately before step 99. A candidate explanation is discounted failure delay (gamma=0.99) combined with weak per-step shaping; this remains a hypothesis. No training or task change has been made.

EXP-001 tested that discount hypothesis with one training run changing only gamma from 0.99 to 1.0 (seed 7, 10,112 collected steps). It still had 0/10 successes and 10/10 out-of-bounds failures (7 upper, 3 horizontal); mean episode length fell from 95.8 to 60.7 steps, while first-30 normalized body-Z action rose from 0.915 to 1.0 and all applied action channels were positive (horizontal saturated at +1 throughout; body-Z and moment were frequently at their positive limits). This weakens the proposed gamma explanation but is only one run per setting. Full configuration and evidence: `Docs/experiments/EXP-001_GAMMA_INVESTIGATION.md`. No physical performance or safety inference follows from this simulation result.

The pretraining engineering audit is recorded in `Docs/PRETRAINING_ENGINEERING_AUDIT.md`. A full-state simulation-only reference controller landed on the ten fixed seeds, confirming feasibility for those sampled states under the toy model. Reward tests ranked success above the tested undesirable behaviors, but gamma=0.99 made delayed upward-force failure slightly better than free fall and left only a small discounted margin over hovering to the time limit. The untrained PPO mean was near zero while its unit Gaussian standard deviation implies frequent stochastic action-bound clipping; existing training logs do not measure those clipped samples. No training or simulator changes were made in this audit.

EXP-002 added opt-in first-window PPO exploration diagnostics and reproduced the original seed-7 gamma=0.99 baseline exactly: trained policy parameters and PPO metrics match, with only elapsed time differing. Of 2,048 sampled actions, 31.54% / 31.10% / 31.01% clipped by channel and 68.60% had at least one clipped channel. Across 16 rollouts, standard deviations stayed near 1 and clipping did not decrease steadily. Details and limits are in `Docs/experiments/EXP-002_EXPLORATION_ANALYSIS.md`; no further experiment was started.

## 5. Engineering principles

- Start with the smallest model that teaches the intended system concepts.
- Preserve clear interfaces between physics, environment, policy, evaluation, and visualization.
- Prefer measurable test results over impressive animation.
- Make training deterministic when seeded and reproducible where practical.
- Keep rendering completely outside the training hot path.
- Record assumptions and decisions instead of silently hard-coding them.
- Introduce realism only when there is a clear reason and a test to justify it.
- Do not equate a successful 2D simulation with a flight-ready controller.
- Preserve a future sim-to-real path, but do not prematurely select real propulsion or flight hardware.

## 6. Definition of success for v0.1

A first version is successful when a developer can:

1. Install the documented environment in a clean virtual environment.
2. Run physics and environment tests without launching any graphical window.
3. Run a short headless PPO training experiment.
4. Save and reload a policy checkpoint.
5. Evaluate that checkpoint over a fixed set of seeded episodes.
6. Obtain machine-readable metrics including landing success rate and failure categories.
7. Launch an optional viewer in a separate command to inspect selected evaluation episodes.
8. Understand the system at a high level from the documentation and concise Codex explanations.

A policy that does not yet learn a successful landing is not automatically a failed software milestone. The milestone succeeds if the environment is correct, tests pass, training is reproducible, and evaluation makes learning progress or failure modes measurable. Do not falsify or weaken success criteria merely to make the agent appear successful.
