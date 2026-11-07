# System Requirements Document (SRD)

**Project:** PROJECT VERTEX  
**Document ID:** VTX-SRD-001  
**Version:** 0.1  
**Date:** 2026-11-07  
**Status:** Baseline for initial software development

## 1. Purpose

Define the system-level requirements for the first software increment of PROJECT VERTEX and preserve the constraints needed for future physical-system research.

This SRD covers a 2D simulation and reinforcement-learning development system. It does not specify a buildable rocket, propulsion system, motor, propellant, launch sequence, or operational flight-control design.

## 2. Mission statement

Develop and evaluate a simulated autonomous vertical-landing capability using reinforcement learning, beginning with a 2D planar rigid-body model. Keep the software architecture modular so future research can investigate higher-fidelity simulation, state estimation, avionics interfaces, and safe physical validation.

The long-term aspiration is a real powered vertical-landing vehicle, but the first software version is a research prototype only.

## 3. Scope

### 3.1 Included in v0.1

- A 2D planar rigid-body vehicle representation.
- A configurable, simplified physics model.
- A headless simulation loop.
- A Gymnasium-compatible environment.
- A continuous-action interface using abstract simulation controls.
- One PPO training workflow using Stable-Baselines3.
- Seeded evaluation and quantitative metrics.
- Optional, separate visualization of recorded or evaluated episodes.
- Unit and integration tests.
- Documentation of assumptions, limitations, and decisions.

### 3.2 Explicitly excluded from v0.1

- 3D flight dynamics.
- Detailed propulsion, combustion, motor, propellant, ignition, or engine design.
- Selection or procurement of rocket motors, fuel, oxidizers, or launch hardware.
- Instructions or software for autonomous powered flight of a home-built rocket.
- Real-world motor-control mappings or operational flight envelopes.
- Flight certification, launch authorization, or range-safety procedures.
- Computer vision or camera-based navigation.
- A photorealistic or real-time 3D scene.
- Newton, Isaac Sim, or Isaac Lab as required runtime dependencies.
- Hardware purchase or physical testing in the initial software milestone.

## 4. Operational concept

The initial software has four separate operating activities:

1. **Physics verification:** run deterministic tests and small scripted cases against the mathematical model.
2. **Training:** run many simulation steps without graphics; the RL agent receives observations and returns abstract actions.
3. **Evaluation:** load a saved policy and execute a fixed, seeded evaluation suite; collect metrics and episode traces.
4. **Visualization:** optionally load evaluation traces and display them. Rendering must not be required for physics, training, or evaluation.

The physics implementation is the source of truth for simulated state transitions. The Gymnasium environment wraps it. The RL library trains a policy through that interface. The viewer consumes recorded data or a separate evaluation stream; it does not own physics logic.

## 5. System boundary and principal interfaces

### 5.1 External inputs

- Versioned configuration for the simulated vehicle and environment.
- Initial-condition and randomization settings.
- Training configuration.
- Evaluation seed list.
- User-selected optional viewer settings.

### 5.2 External outputs

- Trained policy checkpoint and metadata.
- Training logs.
- Evaluation summary and per-episode metrics.
- Optional episode trace for visualization.
- Test results and software environment report.

### 5.3 Internal interfaces

- `VehicleDynamics`: accepts the current simulated state and abstract control input; returns the next state or state derivative according to the selected integration scheme.
- `LandingEnv`: implements Gymnasium `reset()` and `step()` and exposes observation, action, reward, termination, truncation, and info.
- `Training`: creates the environment and PPO model, trains, logs, and saves artifacts.
- `Evaluation`: loads a checkpoint, executes deterministic evaluation episodes, and calculates metrics.
- `Viewer`: reads evaluation traces and displays them without being imported by the training path.

Exact data classes and action semantics must be documented in the architecture and implemented consistently.

## 6. Initial vehicle model

### 6.1 State representation

The first model is a planar rigid body with at least the following conceptual state:

- Horizontal position.
- Vertical position / height above the simulated landing surface.
- Horizontal velocity.
- Vertical velocity.
- Pitch angle.
- Pitch angular velocity.

Use SI units internally. Angles should use radians internally. Choose and document a single coordinate convention, positive direction convention, and reference frame before implementing equations.

### 6.2 Initial simplifications

- Constant vehicle mass during an episode.
- Constant gravitational acceleration.
- No wind in the baseline environment.
- No sensor noise in the first baseline environment.
- No actuator delay in the first baseline environment.
- Simplified ground contact / terminal landing logic.
- Abstract control inputs, not real motor or actuator commands.

All simplifications must be explicit configuration or documented assumptions, not hidden behaviour.

### 6.3 Future model-fidelity additions (not v0.1)

Potential future additions include disturbance models, sensor imperfections, actuator lag, parameter variation, contact/landing-leg models, additional rotational axes, and a 3D model. Each addition requires a separate decision and validation plan.

## 7. Functional requirements

| ID | Requirement | Priority |
|---|---|---|
| SR-F-001 | The software shall represent a 2D planar rigid-body state including position, velocity, pitch, and pitch angular velocity. | Must |
| SR-F-002 | The physics model shall use SI units internally and document coordinate conventions. | Must |
| SR-F-003 | The physics model shall be callable without importing or initializing a renderer. | Must |
| SR-F-004 | The environment shall implement the current Gymnasium environment API. | Must |
| SR-F-005 | The environment shall expose bounded, continuous, abstract simulation actions. | Must |
| SR-F-006 | The environment shall expose normalized or otherwise well-scaled observations, with scaling documented. | Must |
| SR-F-007 | The environment shall return reward, `terminated`, `truncated`, and diagnostic `info` values correctly. | Must |
| SR-F-008 | Episode reset shall support deterministic seeding. | Must |
| SR-F-009 | PPO training shall run headlessly. | Must |
| SR-F-010 | Training shall save checkpoints and configuration metadata. | Must |
| SR-F-011 | Evaluation shall load a saved checkpoint independently of the training command. | Must |
| SR-F-012 | Evaluation shall calculate success rate and failure categories over a fixed evaluation suite. | Must |
| SR-F-013 | Visualization shall be an optional separate command or module. | Must |
| SR-F-014 | Training and evaluation shall not require Pygame, a display server, or a graphics context. | Must |
| SR-F-015 | The software shall provide automated tests for dynamics, environment API, seeding, and evaluation. | Must |
| SR-F-016 | The software shall record dependency versions and run configuration with each training run. | Should |
| SR-F-017 | The software should support a configurable number of parallel environments only after the single-environment baseline is verified. | Should |

## 8. Non-functional requirements

| ID | Requirement | Priority |
|---|---|---|
| SR-NF-001 | The initial workflow shall be usable on the owner's reported Windows laptop without requiring a dedicated workstation. | Must |
| SR-NF-002 | The baseline model shall be lightweight and CPU-capable. | Must |
| SR-NF-003 | Rendering shall not run in the training hot path. | Must |
| SR-NF-004 | Runs shall be reproducible to the extent supported by the selected stack and documented seeds. | Must |
| SR-NF-005 | Code shall be modular, typed where practical, and documented at public interfaces. | Should |
| SR-NF-006 | Configuration values shall not be scattered as unexplained literals across source files. | Must |
| SR-NF-007 | Metrics shall be exportable in a machine-readable format such as JSON or CSV. | Must |
| SR-NF-008 | Errors shall be actionable and avoid silently falling back to different behaviour. | Must |
| SR-NF-009 | No GPU requirement shall be imposed until a measured benchmark demonstrates a benefit. | Must |

## 9. Landing outcome and evaluation requirements

The task is a controlled upright landing, not simply reaching the ground.

The landing-success definition must be a single configurable set of terminal conditions based on:

- Contact with the designated landing region.
- Horizontal distance from the target.
- Vertical and horizontal touchdown speed.
- Pitch deviation from upright.
- Angular rate at touchdown.
- Any configured episode safety or boundary conditions.

Numerical thresholds are intentionally not frozen in this document. They must be selected as simulation-task parameters, justified and recorded before evaluating a policy. They must not be presented as real-rocket safe limits.

Evaluation must report at least:

- Number of episodes.
- Landing success rate.
- Target position error.
- Touchdown speed metrics.
- Pitch error at touchdown.
- Angular-rate metric at touchdown.
- Crash / out-of-bounds / timeout / unstable-touchdown counts.
- Mean and distribution of episode return.
- Random seed and model checkpoint identifier.

Reward is a learning signal; it is not a substitute for terminal success criteria. A high reward does not by itself prove a successful landing.

## 10. Performance requirements

The first priority is correctness and repeatability, not maximum throughput. Establish a simple baseline benchmark after physics tests pass.

Measure:

- Environment steps per second with rendering disabled.
- Training wall-clock time and total environment steps.
- CPU and memory usage.
- If tested, CPU vs GPU policy-training performance.

Do not assume the RTX 3050 must be used. Small custom environments may be limited by Python/CPU simulation overhead, and GPU transfer overhead can outweigh benefits. Benchmark before selecting a device.

## 11. Verification approach

- Unit tests for coordinate conventions, state updates, gravity direction, and deterministic behaviour.
- Gymnasium environment checker or equivalent API compliance check.
- Tests for reset reproducibility and episode termination behaviour.
- Smoke test for a short headless PPO run.
- Save/load round-trip test for a model checkpoint.
- Evaluation test using a fixed seed list.
- Import test to ensure training does not import the viewer or create graphics resources.

## 12. Assumptions and constraints

- Initial spending is zero.
- The owner is new to RL but has general software/electronics experience.
- The owner prefers short, engineering-focused explanations.
- A 2D model is an educational first step, not a sufficient representation of a real rocket.
- A physical demonstrator starts non-propulsively.
- Real powered-flight work, if ever pursued, requires qualified propulsion expertise, legal/regulatory review, appropriate test facilities, independent safety review, and separate authorization.

## 13. Open system questions

These do not block the first implementation:

- Exact simplified action abstraction for the first simulator.
- Initial-condition ranges and landing pad geometry.
- Numerical terminal thresholds for the toy simulation task.
- Whether a later higher-fidelity environment should use Newton/Isaac Lab, MuJoCo, or another engine.
- Future sensor suite and state-estimation architecture.
- Feasible physical demonstrator form and available supervised facilities.
