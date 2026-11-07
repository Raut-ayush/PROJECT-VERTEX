# PROJECT VERTEX — Codex Handoff Pack

**Document pack:** VERTEX v0.1  
**Prepared:** 2026-11-07  
**Status:** Initial project baseline; suitable for starting software-only development  
**Primary audience:** Codex in VS Code and the project owner

## Purpose

This folder transfers the current project intent, decisions, requirements, architecture, development sequence, and collaboration rules to Codex.

PROJECT VERTEX is a long-term systems-engineering project focused on learning and validating autonomous vertical landing. The long-term aspiration is a real reusable powered vertical-landing rocket demonstrator. The immediate work is deliberately limited to a fast, headless 2D rigid-body simulation and a beginner-friendly reinforcement-learning workflow, followed by carefully staged validation.

The project must remain **hardware-conscious without pretending that a 2D simulation is flight-ready**. A trained policy from the first simulator is a research result, not an operational flight controller.

## Read these files in order

1. `PROJECT_CONTEXT.md` — concise source of truth for intent and user preferences.
2. `SRD.md` — system-level requirements and constraints.
3. `PRD.md` — first software product definition and acceptance criteria.
4. `SYSTEM_ARCHITECTURE.md` — proposed modules, interfaces, and data flow.
5. `DEVELOPMENT_PLAN.md` — milestones and the first implementation tasks.
6. `ENGINEERING_DECISIONS.md` — decisions already made and open decisions.
7. `LEARNING_GUIDE.md` — how to explain concepts to the project owner.
8. `SAFETY_AND_SCOPE.md` — boundaries and staged physical-validation approach.
9. `CODEX_RULES.md` — working rules for Codex.

`ROOT_AGENTS_TEMPLATE.md` is intended to be copied to `PROJECT_VERTEX/AGENTS.md` (the repository root). `AGENTS.md` in the root is important because it can guide Codex across `simulation/`, `reinforcement_learning/`, `tests/`, and other project directories. This handoff pack itself is intended to be copied into `PROJECT_VERTEX/docs/`.

## Current frozen decisions

- Goal: simulation plus an eventual safe, non-propulsive physical demonstrator; long-term aspiration remains a real powered vertical-landing vehicle.
- First vehicle model: 2D planar rigid body with horizontal/vertical position, horizontal/vertical velocity, pitch, and pitch angular velocity.
- Initial training: headless; visualization optional and never part of the training hot path.
- Initial RL algorithm: PPO only. No algorithm bake-off in v0.1.
- Initial learning approach: beginner-friendly, engineering-oriented explanations rather than mathematical derivations.
- Initial budget: ₹0; software only.
- Initial simulator direction: small custom Python model. Newton is a later investigation, not a dependency for the first milestone.
- No physical components, materials, motors, propellants, or launch hardware are to be selected in the first software milestone.

## Important first instruction to Codex

Do not immediately generate a large codebase. Start by reviewing this pack, checking the workspace, and producing a short implementation plan for Milestone M1. Then implement in small, tested increments. Keep the first environment intentionally simple and deterministic. Do not add rendering, Newton, 3D, hardware interfaces, or propulsion-specific modeling to the first milestone.

## Source documentation

Use current official documentation when installing or using dependencies; do not assume package versions from this handoff are still current.

- Gymnasium custom environments: https://gymnasium.farama.org/main/tutorials/environment_creation/
- Stable-Baselines3 PPO: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
- NVIDIA Newton: https://developer.nvidia.com/newton-physics
- Isaac Lab Newton backend documentation: https://docs.nvidia.com/isaaclab/concepts/newton
- NVIDIA Isaac Lab: https://developer.nvidia.com/isaac/lab
- PyTorch installation selector: https://pytorch.org/get-started/locally/

These links were checked while preparing this pack on 2026-11-07. Always re-check platform compatibility and current installation instructions at implementation time.
