# Product Requirements Document (PRD)

**Project:** PROJECT VERTEX  
**Document ID:** VTX-PRD-001  
**Version:** 0.1  
**Date:** 2026-11-07

## 1. Product overview

The first product is a local developer tool that lets the project owner create a simple vertical-lander simulation, train a PPO policy without rendering, evaluate it reproducibly, and optionally inspect evaluation episodes.

The product should feel like a small engineering workbench, not a complex research platform.

## 2. Primary user

A beginner in reinforcement learning who thinks from a systems-engineering perspective. The user wants to understand:

- What each subsystem is responsible for.
- Why a particular library or model was selected.
- What data passes between modules.
- What was tested and what the result means.
- What limitations remain before moving to the next fidelity level.

The user does not want long mathematical derivations or an ML theory course unless explicitly requested.

## 3. User stories

- As the project owner, I can run a physics smoke test without opening a window so I can verify the model quickly.
- As the project owner, I can train PPO in headless mode so compute is spent on simulation and learning rather than graphics.
- As the project owner, I can change documented simulation settings without editing physics source code.
- As the project owner, I can stop and resume or reload a saved training checkpoint where supported by the library.
- As the project owner, I can evaluate a saved policy on fixed seeds and receive objective metrics.
- As the project owner, I can optionally visualize an evaluation episode without affecting the training pipeline.
- As the project owner, I can understand the purpose and limitation of every major subsystem from concise documentation.

## 4. Product scope for v0.1

### In scope

- CLI commands for environment tests, training, evaluation, and optional viewing.
- Simple YAML or TOML configuration.
- 2D rigid-body physics with six core state quantities.
- One Gymnasium environment.
- One PPO training pipeline.
- Logs, checkpoints, and machine-readable evaluation results.
- Automated tests and a concise getting-started guide.

### Out of scope

- GUI dashboard.
- Web application.
- Photorealistic 3D rendering.
- Multiple RL algorithms.
- Computer vision.
- Hardware deployment.
- Real propulsion or powered flight.
- Automatic purchase or selection of hardware components.

## 5. Experience and interface requirements

Keep the command-line workflow straightforward. Suggested command names may be refined by Codex, but the experience should resemble:

```text
python -m vertex.cli check
python -m vertex.cli train --config config/default.yaml
python -m vertex.cli evaluate --model models/latest.zip --episodes 100
python -m vertex.cli view --episode logs/run_x/episode_0001.json
```

These are examples, not frozen module names. Codex should choose a clean package structure and document the final commands.

Every command should:

- Print a short purpose statement.
- Print the active configuration path and random seed where relevant.
- Fail with an understandable message if inputs are missing or incompatible.
- Avoid starting a viewer unless the user explicitly requests it.

## 6. Product behaviour

### 6.1 Physics and environment

The environment represents a planar rigid body and a landing target. The initial task begins from a bounded set of simulated initial states and attempts a controlled upright landing. The action interface is abstract and simulation-only; it must not be described or exported as a real motor/actuator command interface.

### 6.2 Training

- PPO is the only RL algorithm in v0.1.
- Training is headless by default.
- Training configuration is explicit and saved with run artifacts.
- Training reports progress at a sensible interval without excessive console output.
- Training saves a final checkpoint and may save periodic checkpoints.
- Training output includes total timesteps, elapsed time, throughput, and reward summaries.

### 6.3 Evaluation

- Evaluation must use a model checkpoint and a defined seed set.
- Evaluation must be separate from training.
- Deterministic policy inference should be used for the main scorecard unless a different evaluation mode is explicitly documented.
- The result must include per-episode outcomes and aggregate metrics.
- Evaluation must not alter the saved model.

### 6.4 Visualization

- The viewer is optional.
- The viewer should consume an episode trace or run only as an explicitly requested evaluation mode.
- The viewer must not be imported by normal training or headless evaluation.
- A simple 2D plot or lightweight animation is sufficient. Visual quality is not a product goal in v0.1.

## 7. Acceptance criteria

### AC-01: Environment setup

A clean virtual environment can install documented dependencies and run the test suite.

### AC-02: Physics model

Unit tests demonstrate that the documented coordinate conventions and state-transition behaviour are consistent. Tests cover basic limiting cases and deterministic outputs.

### AC-03: Gymnasium compliance

The environment passes the relevant Gymnasium API checker and has correctly shaped, bounded observation/action spaces.

### AC-04: Headless behaviour

The physics, environment, training, and evaluation commands run without opening a window or requiring a display context.

### AC-05: PPO smoke training

A short training run completes, writes logs and a model checkpoint, and records the configuration used. The smoke test verifies integration, not learning success.

### AC-06: Model reload

A saved policy can be loaded in a fresh process and used for evaluation.

### AC-07: Reproducible evaluation

A fixed model and fixed evaluation seed list produce a recorded evaluation report. Any expected nondeterminism is documented.

### AC-08: Outcome metrics

The report includes landing success rate and explicit failure categories. It does not use average reward as the only measure of success.

### AC-09: Optional viewer

A user can inspect a recorded evaluation episode with a separate command. The viewer is not required for any other acceptance criterion.

### AC-10: Documentation

The README explains setup, commands, project modules, basic concepts, known limitations, and the next milestone in plain engineering language.

## 8. Success metrics

Primary product metrics:

- Environment steps per second in headless mode.
- Short-run setup and execution reliability.
- Percentage of tests passing.
- Ability to save and reload a checkpoint.
- Evaluation reproducibility.
- Policy landing-success rate once learning has been demonstrated.

Do not choose an arbitrary minimum success percentage before the baseline task and its terminal thresholds have been defined. Establish a meaningful target in the M3/M4 review and record it in the decision log.

## 9. Product non-goals

The software must not imply that a policy is ready for hardware because it achieved a high training reward or passed a toy-simulation evaluation. Any hardware use requires a separate safety case, model-fidelity analysis, independent safety mechanisms, and authorized test conditions.
