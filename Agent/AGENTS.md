# PROJECT VERTEX — Root AGENTS.md Template

Copy this file to `PROJECT_VERTEX/AGENTS.md` (the repository root). Keep the full project-specific instructions here so Codex applies them to source code, tests, configuration, and documentation.

## First read

Before material work, read the project baseline in `docs/`:

- `docs/README.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/SRD.md`
- `docs/PRD.md`
- `docs/SYSTEM_ARCHITECTURE.md`
- `docs/DEVELOPMENT_PLAN.md`
- `docs/ENGINEERING_DECISIONS.md`
- `docs/LEARNING_GUIDE.md`
- `docs/SAFETY_AND_SCOPE.md`
- `docs/CODEX_RULES.md`

## Mission

Build PROJECT VERTEX incrementally as a simulation-first, systems-engineering project for autonomous vertical-landing research. The long-term aspiration is a real powered vertical-landing rocket, but the current implementation scope is a headless 2D rigid-body simulation, Gymnasium environment, PPO training/evaluation, and later safe non-propulsive validation.

## Required working behaviour

- Start with M0/M1 only; inspect the repo before editing.
- Do not build an oversized framework or add features outside the current milestone.
- Keep physics independent of Gymnasium, RL, and visualization.
- Keep training and evaluation headless by default. Visualization is optional and separate.
- Use SI units and explicitly document coordinate conventions.
- Begin with a 2D planar rigid body: x/z position, x/z velocity, pitch, and pitch angular velocity.
- Use PPO only in v0.1. Do not add other algorithms without an approved decision.
- Do not begin PPO training before the physics and Gymnasium environment pass tests.
- Use current official package documentation and verify Python/dependency compatibility.
- Benchmark before choosing GPU or adding parallel simulation.
- Record configuration, seeds, versions, checkpoints, and evaluation metrics.
- Explain major work in concise, beginner-friendly systems-engineering language.
- Do not equate training reward with landing success or simulation success with real-world readiness.
- Do not overwrite existing work without inspection and a clear reason.

## Safety boundary

Current scope is abstract simulation and safe non-propulsive physical validation. Do not create DIY propellant, motor-construction, ignition, propulsion, or autonomous powered-flight instructions, code, or hardware mappings. Do not map simulation action values to real propulsion commands. Mark simulation values as educational parameters, not real vehicle design limits. Future powered-vehicle activity requires qualified engineering, facilities, approvals, and independent safety review.

## Change control

Frozen project decisions are listed in `docs/ENGINEERING_DECISIONS.md`. Explain material deviations and obtain owner approval before changing mission, scope, algorithm, simulator dependency, or hardware direction. Record approved changes in the decision log.
