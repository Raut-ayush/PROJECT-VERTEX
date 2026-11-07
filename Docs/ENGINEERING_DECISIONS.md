# Engineering Decision Log (EDL)

**Project:** PROJECT VERTEX  
**Document ID:** VTX-EDL-001  
**Version:** 0.1  
**Date:** 2026-11-07

Use this file to record important engineering choices. New decisions should include date, status, rationale, alternatives considered, consequences, and evidence. Do not silently change a frozen choice.

## 1. Accepted decisions

### EDL-001 — Initial project objective

- **Status:** Accepted
- **Decision:** Build a simulation-first autonomous vertical-landing research project, with an eventual safe non-propulsive physical demonstrator. Preserve the long-term aspiration of a real powered vertical-landing rocket.
- **Rationale:** The project owner wants a meaningful aerospace systems project rather than a generic RL exercise, but is new to RL and wants complexity introduced gradually.
- **Consequence:** Initial work must be useful for learning and future architecture without claiming it is flight-ready.

### EDL-002 — Initial vehicle representation

- **Status:** Accepted
- **Decision:** Use a 2D planar rigid-body model with horizontal/vertical position, horizontal/vertical velocity, pitch, and pitch angular velocity.
- **Rationale:** More informative than a 1D point mass while significantly simpler than a full 3D model.
- **Consequence:** The first model is educational and cannot validate a real vehicle's full dynamics or flight behaviour.

### EDL-003 — First RL algorithm

- **Status:** Accepted
- **Decision:** Use PPO only in v0.1.
- **Rationale:** Keep the first RL project understandable and reduce algorithm-selection overhead.
- **Consequence:** Do not add SAC, TD3, DDPG, imitation learning, or algorithm comparisons unless a later decision approves them.

### EDL-004 — Headless training

- **Status:** Accepted
- **Decision:** Rendering is optional and must be separated from training.
- **Rationale:** Preserve compute and improve throughput; visualization is useful for inspection but is not required for learning.
- **Consequence:** Training and evaluation must work without a graphics window or graphics context.

### EDL-005 — Initial compute budget

- **Status:** Accepted
- **Decision:** Begin software-only with an initial spend of ₹0.
- **Rationale:** Prove the software concept before buying hardware.
- **Consequence:** No component procurement or hardware integration in M0–M5.

### EDL-006 — Initial simulation approach

- **Status:** Accepted
- **Decision:** Build a lightweight custom Python/NumPy physics model first. Keep Newton as a later investigation.
- **Rationale:** The first task is small and the owner wants to understand the system. A custom model provides transparent assumptions and fast headless iteration.
- **Consequence:** Do not install or integrate Newton as a v0.1 dependency. Revisit only when a fidelity or scaling need is demonstrated.

### EDL-007 — Learning and communication style

- **Status:** Accepted
- **Decision:** Explain concepts from a systems-engineering perspective, with concise overviews and practical importance. Avoid deep mathematics unless requested.
- **Rationale:** The owner is new to RL but wants to understand subsystem purpose, interfaces, trade-offs, and validation.
- **Consequence:** Documentation should explain why a step exists and what it verifies, not only show code.

### EDL-008 — Physical validation sequence

- **Status:** Accepted
- **Decision:** Start physical validation with a safe non-propulsive approach. The owner is open to a guided/captive fixture and a passive drop-test lander; begin with whichever is simpler and eventually consider both.
- **Rationale:** Validate selected electronics, sensing, inference, and landing mechanics before considering higher-risk activities.
- **Consequence:** No powered rocket hardware or propulsion design in the initial project scope.

## 2. Technical choices and milestone decisions

The table lists proposed choices that are not yet accepted. The EDL records below it are accepted milestone decisions.

| Topic | Recommendation | Status |
|---|---|---|
| Language | Python | Proposed |
| Numeric library | NumPy | Proposed |
| Environment interface | Gymnasium | Proposed |
| RL implementation | Stable-Baselines3 PPO | Proposed |
| ML backend | PyTorch via SB3 | Proposed |
| Tests | pytest | Proposed |
| Metrics | TensorBoard + JSON/CSV | Proposed |
| Visualization | Matplotlib first; simple 2D only | Proposed |
| Configuration | YAML or TOML | Open |
| Python version | Choose a currently supported version after compatibility check; avoid blindly using default interpreter | Open |

### EDL-009 — M1 coordinate convention and numerical baseline

- **Date:** 2026-11-07
- **Status:** Accepted for M1
- **Decision:** Use an inertial x-z plane with x positive to the right and z positive upward. `theta = 0` is upright, with the body +x axis pointing right; positive theta rotates +x toward +z. Angular rate and applied moment are positive in the same rotational direction. State angles are radians and all quantities use SI units. Resolve generic applied force components from body axes into the inertial frame.
- **Integration:** Use semi-implicit (symplectic) Euler with a caller-supplied positive fixed time step. Update velocities/rates first, then positions/angles using the updated rates. This corrects the previous method name; the numerical update is unchanged.
- **Rationale:** This is a compact, explicit convention that supports sign tests without introducing an RL or graphics dependency.
- **Alternatives considered:** Positive z downward; higher-order integration; forces directly in world axes. These are deferred to avoid ambiguity and scope growth.
- **Consequences / limitations:** Euler integration has timestep-dependent error and this model has no ground contact, aerodynamic effects, changing mass, wind, or actuator behavior. These are educational simulation assumptions, not real vehicle parameters.
- **Verification:** Focused tests cover free fall, rotated force direction, positive moment/pitch, invalid timestep, and invalid mass.

### EDL-010 — M0 Python and dependency baseline

- **Date:** 2026-11-07
- **Status:** Accepted for M0/M1
- **Decision:** Target CPython >=3.12,<3.14; use CPython 3.12 in `.venv`. Keep M0/M1 runtime dependencies empty and pytest as a development extra. Do not add a graphics dependency.
- **Rationale:** CPython 3.12 is installed and broadly compatible with planned ecosystem choices; this phase does not need numerical/RL packages.
- **Consequences:** Recheck current Gymnasium/Stable-Baselines3/PyTorch support before M2/M3. GPU presence does not establish that GPU training will be faster.
- **Environment evidence (2026-11-07):** Windows 10 Home Single Language, CPython 3.12 and 3.14; NVIDIA GeForce RTX 3050 Laptop GPU reported by `nvidia-smi`, driver 616.56, CUDA UMD 13.4. WMI inventory was access denied. No current virtual environment existed before this work; `.venv` was created. PyTorch/CUDA package availability was not tested.

### EDL-011 — M2 landing task interface and educational thresholds

- **Date:** 2026-11-07
- **Status:** Accepted for the baseline M2 task
- **Decision:** Keep Gymnasium in the environment layer; physics remains independently callable. Use three normalized abstract actions in [-1, 1]: body-x force, body-z force, and pitch moment, scaled by configurable educational constants. They are generic simulation inputs and do not represent a real actuator or propulsion layout.
- **Observation:** Return eight float32 values in this order: target-relative x / 10 m; z / 20 m; vx / 5 m/s; vz / 10 m/s; sin(theta); cos(theta); omega / 2 rad/s; fraction of the 500-step episode budget remaining. Clip each to [-1, 1].
- **Initial state:** Sample x in [-1, 1] m, z in [8, 10] m, vx in [-0.5, 0.5] m/s, vz in [-0.5, 0] m/s, theta in [-0.1, 0.1] rad, and omega in [-0.05, 0.05] rad/s using Gymnasium's seeded RNG. Tests may provide a `PlanarState` via reset options.
- **Terminal conditions:** Contact is represented by z <= 0. Success requires target error <= 0.5 m, |vx| and |vz| <= 0.5 m/s, |theta| <= 0.15 rad, and |omega| <= 0.25 rad/s. Other contacts are `unstable_touchdown`. Target-relative |x| > 10 m or z > 20 m is `out_of_bounds`. A 500-step limit (10 s at dt=0.02 s) is truncation with reason `time_limit`.
- **Reward:** Per-step readable penalty of 0.01 times the sum of the normalized absolute state terms (using the observation scales except sin/cos, where |theta| and |omega|/2 are used); terminal success adds 10 and other terminal failures subtract 10. Reward is a training signal, not the definition of success.
- **Rationale:** Small and inspectable task boundary supports later PPO integration while preserving a clean physics/RL separation.
- **Alternatives considered:** More action channels, realistic actuator mappings, complex contact, noisy sensors, and detailed reward shaping. Deferred to avoid implying hardware correspondence or adding unvalidated complexity.
- **Consequences / limitations:** Thresholds and action scales are arbitrary toy-task choices. They are not safe limits or vehicle specifications. Environment correctness does not establish that these values produce a useful learning task.
- **Dependency evidence:** `gymnasium>=1.1,<2` was installed in the CPython 3.12 environment; resolved Gymnasium 1.4.0 and NumPy 2.5.4. Current API behavior was checked with Gymnasium's `check_env`.

### EDL-012 — M3 PPO integration baseline

- **Date:** 2026-11-07
- **Status:** Accepted for the M3 integration baseline
- **Decision:** Use Stable-Baselines3 PPO only, with a small MLP (64, 64), one environment, CPU device, seed 7 by default, rollout length 128, batch size 64, five epochs, and 10,000 default timesteps. Keep training output in a new run directory with a checkpoint, CSV progress log, and JSON metadata. Install PyTorch's CPU build for this baseline.
- **Rationale:** Confirm the complete environment-to-policy-to-checkpoint path with low setup and compute overhead. CPU is the predictable baseline for a small non-CNN policy, and SB3's PPO guidance describes CPU as the primary choice for this kind of policy.
- **Alternatives considered:** GPU execution, parallel environments, and hyperparameter searches. Deferred until a measured throughput comparison or a validated need exists.
- **Compatibility evidence:** Official SB3 2.9.2 installation docs require Python 3.10+ and PyTorch >=2.8. This workspace is CPython 3.12.3; resolved versions are SB3 2.9.2 and PyTorch 2.15.0+cpu. PyTorch official Windows instructions support Python 3.10–3.14 and provide a CPU-only install selection.
- **Verification:** 128-timestep test run saved and reloaded a checkpoint with a bounded action prediction. A 256-step CLI smoke run completed on CPU at approximately 1,711.8 steps/s. After the CLI default fix, the default training command completed its requested 10,000 steps as 10,112 actual rollout steps (SB3 collects whole 128-step rollouts), at 1,337.6 steps/s. Its checkpoint evaluated on the ten default seeds with zero successes; all ten ended out of bounds. This verifies training and evaluation integration, while showing that this run did not learn successful landings.
- **Consequences / limitations:** The short smoke run is not a learning assessment. Reward and terminal thresholds remain educational, and the task/model are not flight-ready. Do not claim successful landing from training reward.

### EDL-013 — M4 deterministic evaluation baseline

- **Date:** 2026-11-07
- **Status:** Accepted for the baseline evaluation workflow
- **Decision:** Evaluate saved PPO checkpoints using deterministic actions on the fixed seed list 1000–1009 by default. Save aggregate JSON, per-episode CSV, and state traces in JSONL. Report success rate and terminal failure categories separately from reward; calculate touchdown-state summaries only for episodes that touch down. Summarize the SB3 progress CSV trend when available.
- **Rationale:** Separate policy evaluation from training and provide repeatable task outcome evidence for later learning assessment.
- **Alternatives considered:** Visual evaluation during training and reward-only scoring. Rejected because rendering is reserved for separate M5 tooling and reward does not define landing success.
- **Verification:** The 256-step smoke checkpoint on default seeds 1000–1009 had 0/10 successes, all unstable touchdowns; mean return -10.908 and mean episode length 70.4 steps. The later 10,112-step training run on the same default seeds had 0/10 successes, all out-of-bounds failures. Evaluation artifacts were produced for both runs; the end-to-end test covers checkpoint load, two episodes, categories, reward trend extraction, CSV, and traces.
- **Consequences / limitations:** Ten episodes are an initial assessment only and cannot establish robust performance. Results show no successful landing in this sample. The high vertical touchdown speed is a task failure under the current criteria. M5 can consume saved traces without changing training.

### EDL-014 — M5 optional trace viewer

- **Date:** 2026-11-07
- **Status:** Accepted for M5
- **Decision:** Use Matplotlib as an optional visualization extra. The viewer reads M4 JSONL traces and supports interactive animated playback or static PNG export. Import Matplotlib only when a render function is called; keep the viewer outside training and physics imports.
- **Rationale:** Provide visual inspection without putting graphics in the training hot path.
- **Alternatives considered:** Live rendering during PPO training and a 3D renderer. Deferred to preserve headless throughput and match the 2D model.
- **Display convention:** Draw the body +z axis as a short centerline glyph, using the M1 pitch convention. Its displayed length is illustrative and is not modeled vehicle geometry. Plot the simulated target band and z=0 surface.
- **Verification:** Matplotlib 3.11.2 was installed as an optional extra. Static image export was rendered and visually inspected; tests cover trace selection, older report metadata, and PNG creation. Interactive GUI playback is available via the documented command.
- **Consequences / limitations:** The viewer only depicts recorded toy-model state and terminal outcomes. It does not add physical geometry, sensor cues, or safety evidence.

## 3. Remaining decisions for later milestones

- Whether M2 task scales and terminal thresholds need revision after environment validation and later evaluation. Any revision should remain explicitly educational and must be recorded.
- PPO configuration and training/evaluation protocol (M3/M4).
- Optional viewer approach (M5).
- Robustness variations (M6).
- Whether a later simulator study is justified (M7).

Do not invent realistic rocket specifications to fill gaps. PPO remains deferred until M1 and M2 validation is complete.

## 4. Future decisions (not needed now)

- Whether to port a model to Newton / Isaac Lab or another simulator.
- Whether to build a 3D model and what fidelity is necessary.
- Sensor suite and state-estimation architecture.
- Future onboard compute architecture.
- Future non-propulsive test fixture details.
- Feasibility, specialist support, compliance and approvals for any future powered-vehicle programme.

## 5. Decision change process

When changing an accepted decision:

1. Explain what new requirement or evidence motivates the change.
2. Identify impacts on code, tests, performance, dependencies, and future interfaces.
3. Record alternatives considered.
4. Ask the project owner to approve any material scope or architecture change.
5. Update this log and the affected requirements document.
