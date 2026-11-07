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

M3 adds Stable-Baselines3 PPO 2.9.2 with PyTorch 2.15.0+cpu. Training is explicitly CPU-only pending any future benchmark. A 256-step integration smoke run saved and reloaded a checkpoint; its reward is not evidence of landing capability. See EDL-012.

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
