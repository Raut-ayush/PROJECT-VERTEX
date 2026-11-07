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

## 2. Proposed but not yet frozen technical choices

The following are recommendations that Codex may implement after checking current compatibility and documenting the result:

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

## 3. Open decisions for M1/M2

- Coordinate convention and sign of positive vertical axis.
- Exact initial state and coordinate data structures.
- Time step and integration scheme.
- Landing target geometry.
- Initial-condition distribution.
- Action-space channel count and abstract semantics.
- Observation vector contents and normalization.
- Reward terms and relative weights.
- Terminal success/failure criteria and thresholds.

These should be settled in a short design review before PPO training. Do not invent realistic rocket specifications to fill gaps.

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
