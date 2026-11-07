# Development Plan and Milestones

**Project:** PROJECT VERTEX  
**Document ID:** VTX-PLAN-001  
**Version:** 0.1  
**Date:** 2026-11-07

## 1. Development approach

Develop incrementally. Every milestone must produce a working artifact and pass explicit acceptance checks. Do not build the entire future architecture in one pass.

The owner wants to learn the system at an overview level. At each milestone, Codex should briefly explain:

- What was built.
- Why it is needed.
- How data moves through it.
- What tests were run and their results.
- What limitation remains.
- What the next milestone adds.

Avoid long equations and ML derivations unless requested.

## 2. Milestone roadmap

### M0 — Requirements and workspace review

**Goal:** Verify the baseline and prepare a minimal, maintainable Python project.

Tasks:
- Read all project documents.
- Inspect the existing repository before changing files.
- Check OS, Python versions, GPU/driver state, and current toolchain.
- Select a currently supported Python version and dependency set.
- Create a virtual environment and dependency metadata.
- Create package skeleton, test setup, `.gitignore`, and basic README.
- Record setup decisions.

Exit criteria:
- Reproducible environment setup is documented.
- A simple import and test command works.
- No physics, RL, renderer, or hardware complexity has been introduced prematurely.

### M1 — 2D physics model

**Goal:** Implement and verify the smallest useful rigid-body simulation model.

Tasks:
- Agree and document coordinate axes, sign conventions, units, and state structure.
- Implement the state data structure.
- Implement the simplified 2D rigid-body dynamics with constant mass and gravity.
- Choose and document a basic numerical integration method.
- Make the model step deterministically without graphics.
- Add basic scripted tests and a tiny command-line smoke test.

Exit criteria:
- Unit tests pass.
- Known limiting cases behave as expected.
- Repeated runs from the same state and input are reproducible.
- Physics module imports without visualization or RL dependencies.

### M2 — Landing environment

**Goal:** Wrap the physics model in a Gymnasium-compatible task.

Tasks:
- Define a bounded initial-condition distribution.
- Define the landing target and episode boundaries.
- Define compact observations and documented normalization.
- Select abstract bounded simulation action channels and document their semantics.
- Define a readable reward with separate terminal success criteria.
- Implement `reset()` and `step()` with correct Gymnasium semantics.
- Add deterministic seeding and environment API tests.

Exit criteria:
- Gymnasium checker passes.
- Reset and step shapes/dtypes/bounds are correct.
- Terminal reasons are exposed in `info`.
- No viewer is needed to run the environment.

### M3 — PPO integration and smoke training

**Goal:** Connect a single PPO agent to the environment.

Tasks:
- Use Stable-Baselines3 PPO only.
- Start with a small policy network and conservative runtime configuration.
- Train headlessly in a short smoke run.
- Log reward and basic throughput.
- Save model and run metadata.
- Add a model reload test.

Exit criteria:
- Short training run completes.
- Checkpoint and logs are written.
- Model loads in a fresh process.
- Training does not import or initialize visualization.

Important: M3 verifies integration, not successful learning.

### M4 — Evaluation and learning assessment

**Goal:** Establish whether PPO is learning the task.

Tasks:
- Implement evaluation over fixed seeds.
- Define/report terminal success criteria and failure categories.
- Report success rate and touchdown-state metrics.
- Compare training reward trends with evaluation outcomes.
- Review reward design only when evaluation shows a justified problem.
- Save per-episode traces for selected runs.

Exit criteria:
- A reproducible evaluation report is generated.
- Success is measured by task criteria, not reward alone.
- Failure modes can be identified from metrics.
- The result is honestly reported, even if the initial policy fails.

### M5 — Optional visualization

**Goal:** Visualize selected evaluation episodes without affecting training speed.

Tasks:
- Implement a simple 2D viewer or plotter that reads episode traces.
- Provide a separate command.
- Keep graphics dependencies optional if practical.
- Verify headless tests and training still work when graphics packages are absent.

Exit criteria:
- A selected evaluation episode can be inspected.
- Viewer is not imported by training or ordinary evaluation.
- No real-time rendering is required during training.

### M6 — Simulation robustness study

**Goal:** Learn what the policy can and cannot handle within the simplified model.

Possible tasks, each introduced one at a time:
- Vary initial conditions.
- Add bounded disturbances.
- Add observation noise or bias.
- Add abstract actuator lag/limits.
- Randomize selected model parameters.

Exit criteria:
- Every variation has a documented purpose and controlled test.
- Baseline results remain available for comparison.
- The report distinguishes robustness within the toy model from real-world robustness.

### M7 — Simulator study: Newton / Isaac Lab

**Goal:** Learn the simulator ecosystem and decide whether it offers value for the project.

Tasks:
- Read current official Newton and Isaac Lab documentation.
- Make a small unrelated or generic rigid-body/contact example first.
- Check platform and hardware support before installation.
- Compare model fidelity, setup complexity, and simulation throughput.
- Write a decision record before integrating it.

Exit criteria:
- A short technical note explains what Newton does, what Isaac Lab does, and whether a port is justified.
- No dependency is added to the core project without a clear benefit.

### M8 — Physical-system research and non-propulsive validation

**Goal:** Begin a safe physical validation workstream after simulation milestones are complete.

Possible areas:
- Bench-level sensor and computing validation.
- Policy inference interface testing with dummy inputs.
- Independent safety monitor at software-interface level.
- Non-propulsive captive/guided test fixture.
- Passive lander/drop-test structure.

This milestone does not authorize or define construction, propulsion, ignition, launch, or autonomous powered flight. Any future powered-vehicle programme needs specialist engineering, legal/regulatory review, facilities, and independent safety approval.

## 3. Immediate Codex task list

Codex should start with these tasks only:

1. Inspect the repository and confirm whether it is empty or already contains work.
2. Read `docs/README.md`, `PROJECT_CONTEXT.md`, `SRD.md`, `PRD.md`, and `SYSTEM_ARCHITECTURE.md`.
3. Report any conflicts between the existing repo and the project baseline.
4. Check local Python and package compatibility. Do not assume the system default Python is appropriate.
5. Propose the smallest M0/M1 implementation sequence.
6. Implement M0 only after reporting the plan; then implement M1 in small increments.
7. Run tests and report exact commands and results.

Do not start PPO until M1 and M2 are tested.

## 4. Experiment discipline

For every training experiment, record:

- Experiment ID.
- Git commit / source revision when available.
- Config snapshot.
- Dependency versions.
- Random seed(s).
- Total training steps.
- Elapsed time and environment throughput.
- Checkpoint path.
- Evaluation seed set and metrics.
- Known failures or anomalies.

Do not compare two runs unless the task configuration and evaluation procedure are clear.

## 5. Review gates

Before advancing milestones, the owner should be able to explain at a high level:

- What the current module does.
- What has been verified.
- What has not been verified.
- Why the next complexity increase is needed.

No milestone advances based solely on visual appearance or training reward.
