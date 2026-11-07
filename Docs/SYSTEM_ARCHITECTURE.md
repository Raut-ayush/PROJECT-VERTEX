# System Architecture — PROJECT VERTEX

**Document ID:** VTX-ARCH-001  
**Version:** 0.1  
**Date:** 2026-11-07

## 1. Architecture goals

1. Keep the first implementation small and understandable.
2. Separate physics from the RL framework.
3. Make training and evaluation headless by default.
4. Keep visualization optional and isolated.
5. Preserve clean interfaces for future model-fidelity improvements.
6. Avoid a premature dependency on heavyweight robotics simulators.

## 2. Logical architecture

```text
                 CONFIGURATION
                       |
                       v
              +------------------+
              | Vehicle / World  |
              | Configuration    |
              +------------------+
                       |
                       v
              +------------------+
              | Physics Model     |
              | 2D Rigid Body     |
              +------------------+
                       |
                       v
              +------------------+
              | Gymnasium Env     |
              | reset / step      |
              | obs / action      |
              | reward / done     |
              +------------------+
                       |
             +---------+---------+
             |                   |
             v                   v
     +---------------+   +----------------+
     | PPO Training  |   | Policy          |
     | (SB3)         |   | Evaluation      |
     +---------------+   +----------------+
             |                   |
             v                   v
     +---------------+   +----------------+
     | Checkpoints   |   | Metrics /       |
     | and Logs      |   | Episode Traces  |
     +---------------+   +----------------+
                                   |
                                   v
                           +----------------+
                           | Optional Viewer|
                           +----------------+
```

The diagram is a logical view, not a required package layout.

## 3. Recommended initial technology choices

| Concern | Initial choice | Reason |
|---|---|---|
| Language | Python | Familiar, broad ML ecosystem |
| Numeric operations | NumPy | Lightweight CPU physics model |
| RL environment API | Gymnasium | Standard reset/step interface |
| RL algorithm | Stable-Baselines3 PPO | Supported continuous Box action spaces and established implementation |
| ML framework | PyTorch (through SB3) | Standard backend for SB3 |
| Tests | pytest | Simple automated testing |
| Metrics/logging | TensorBoard plus JSON/CSV summaries | Training diagnostics and portable reports |
| Viewer | Matplotlib first; Pygame only if useful | Keep visuals simple and separate |
| Configuration | YAML or TOML | Human-readable parameters |

Do not pin versions from this document. Inspect current official installation guides and select a mutually compatible set. Record exact resolved versions in the lock or requirements file and in each experiment's metadata.

## 4. Module responsibilities

### 4.1 Configuration

Owns validated parameters for the vehicle, world, environment, training, evaluation, and logging. It should reject invalid or physically nonsensical simulation settings with clear messages. It should not contain real motor specifications.

### 4.2 Physics model

Owns state representation and state transitions. It must not depend on Gymnasium, Stable-Baselines3, Matplotlib, Pygame, or a rendering system.

Responsibilities:

- Store the planar state.
- Apply the declared simplified physical model.
- Advance the model using a documented integration method and time step.
- Expose state in a predictable data structure.
- Keep coordinate and unit conventions consistent.
- Provide no graphics or RL-specific reward logic.

### 4.3 Sensor abstraction

For the first baseline, the environment may expose the simulated state directly as a perfect observation. A separate sensor abstraction can be introduced later, but do not add fake complexity to M1. Keep the architecture ready for a later distinction between true simulated state and measured observation.

### 4.4 Actuator abstraction

The initial RL action is an abstract, bounded, dimensionless simulation input. It is not a motor command, thrust command, gimbal command, valve command, or hardware interface. The precise number and semantic meaning of action channels must be selected and documented during M1/M2 design review. Do not map these values to physical propulsion hardware.

### 4.5 Gymnasium environment

Wraps the physics model and defines the task:

- Reset initial conditions using a seeded RNG.
- Convert state to observation.
- Validate and apply abstract actions through the simulation model.
- Calculate reward.
- Detect success and failure terminal states.
- Return diagnostic information.

The environment should implement the current Gymnasium API and be tested with the Gymnasium environment checker.

### 4.6 PPO training

Builds the environment, configures PPO, runs headless learning, writes logs, and saves checkpoints. It must not own or import the viewer.

Use the smallest reasonable policy network and initial training configuration. Avoid premature hyperparameter sweeps. A short smoke run first verifies the pipeline; learning quality is assessed separately.

### 4.7 Evaluation

Loads a saved checkpoint, executes a fixed set of seeded episodes, and writes aggregate and per-episode results. It should support deterministic inference for a stable scorecard. Evaluation must not change the checkpoint.

### 4.8 Visualization

Reads episode traces or uses a clearly separate evaluation path. It displays the vehicle state, target, and relevant measurements. It is not needed for training or tests. Prefer simple 2D visualization over a 3D rendering engine.

## 5. Initial state and unit conventions

The six primary state quantities are:

- `x`: horizontal position, metres.
- `z`: vertical position relative to the landing-surface reference, metres.
- `vx`: horizontal velocity, metres per second.
- `vz`: vertical velocity, metres per second.
- `theta`: pitch angle, radians.
- `omega`: pitch angular velocity, radians per second.

The exact coordinate direction (for example, whether positive vertical is up) must be stated in one location and used everywhere. The landing-surface reference and upright-angle convention must also be explicit.

Internally use SI units. Avoid mixing degrees and radians or metres and centimetres in internal calculations. Convert only at presentation boundaries.

## 6. Observation design

The baseline observation should include the information needed to characterize the state and relative landing error. Keep the observation compact. If absolute and target-relative coordinates are both included, document why and avoid accidental duplicate features.

Normalize or scale observations so typical values have comparable magnitudes. Store normalization parameters with the model. Do not normalize the state differently during evaluation or inference.

The first baseline may use perfect simulated state observations. Later, add sensor noise or estimated state as an explicit model-fidelity increment.

## 7. Action design

Use a bounded continuous Gymnasium `Box` space. Action channels are normalized abstract simulation controls only. They must be documented with their bounds, rate assumptions, and effect on the simulated model.

Do not claim that the action channels correspond to a physically realizable rocket actuation configuration. Mapping a policy to a real flight vehicle is a separate systems-engineering problem and is out of scope for v0.1.

## 8. Reward and termination design

Reward should encourage the task outcome without making reward the only performance metric. The reward design should be readable and documented by term and purpose, without requiring the user to study equations.

Terminal states should distinguish at least:

- Successful controlled touchdown.
- Unsuccessful / unstable touchdown.
- Out-of-bounds or simulated crash condition.
- Episode time limit.

Use separate `terminated` and `truncated` semantics in the Gymnasium API. Expose a human-readable terminal reason in `info`.

## 9. Training and compute strategy

- Headless training is the default and required mode.
- Start with one environment and establish correctness.
- Add vectorized environments only after the single-environment results are correct and measured.
- Benchmark CPU and GPU rather than assuming GPU is faster.
- Keep the physics model NumPy-based initially unless profiling identifies a bottleneck.
- Do not add Newton, Isaac Lab, or GPU physics to solve a performance problem that has not been measured.
- Keep periodic logging light; avoid storing every frame of every training episode.

## 10. Artifact and run management

Each run should have a unique directory with:

- Run identifier and timestamp.
- Source revision / git commit when available.
- Resolved dependency versions.
- Full configuration snapshot.
- Random seed(s).
- Training logs.
- Final and optional periodic policy checkpoints.
- Evaluation summary and per-episode records.
- Optional selected episode trace for viewer use.

Do not commit large model checkpoints or generated logs to Git by default. Add suitable `.gitignore` rules and explain artifact locations.

## 11. Suggested repository layout

```text
PROJECT_VERTEX/
├── AGENTS.md
├── README.md
├── pyproject.toml                 # preferred source of package/dependency metadata
├── requirements.lock.txt          # or chosen lock mechanism
├── config/
│   ├── vehicle.yaml
│   ├── environment.yaml
│   └── training.yaml
├── src/
│   └── vertex/
│       ├── config/
│       ├── simulation/
│       │   ├── state.py
│       │   └── dynamics.py
│       ├── environment/
│       │   └── landing_env.py
│       ├── learning/
│       │   ├── train.py
│       │   └── evaluate.py
│       ├── visualization/
│       │   └── viewer.py
│       └── cli.py
├── tests/
├── models/                        # ignored by Git
├── logs/                          # ignored by Git
└── docs/
    └── (this handoff pack)
```

Codex may adjust the layout if there is a clear benefit, but it must preserve the module boundaries.

## 12. Future simulator evaluation: Newton

Newton is an open-source physics engine built on NVIDIA Warp and OpenUSD, designed for robotics and physical simulation. It is compatible with robotics learning workflows and is integrated with Isaac Lab. Isaac Lab's documentation describes Newton as a backend and documents several solver options.

Newton is a **future investigation**, not the first implementation choice. Before adopting it, evaluate:

- Whether the actual task requires contact-rich or higher-fidelity physics beyond the custom model.
- Windows/Linux/WSL compatibility for the exact versions being considered.
- NVIDIA driver, GPU, CUDA, Python, and Isaac Lab compatibility.
- Installation and maintenance overhead.
- Whether the chosen Newton/Isaac Lab setup genuinely improves fidelity or throughput for this 2D task.

Do not add Newton merely because it is technically impressive. The current small model should remain the fast regression baseline.

Official references:

- Newton overview: https://developer.nvidia.com/newton-physics
- Isaac Lab Newton backend: https://docs.nvidia.com/isaaclab/concepts/newton
- Isaac Lab overview: https://developer.nvidia.com/isaac/lab

## 13. Sim-to-real architecture principle

The v0.1 policy interface should be modular, but the software must not suggest direct deployment to a real rocket. A future transfer programme would need additional evidence and components, such as a higher-fidelity 3D model, measured parameter ranges, sensor/estimator design, actuator characterization, latency analysis, independent safety logic, and authorized validation. These are future requirements, not implementation tasks for the current milestone.
