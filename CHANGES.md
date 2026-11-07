# Change log

## 2026-11-07 — M3 PPO integration

- Added an optional Stable-Baselines3 PPO dependency extra and a headless training module using a small MLP policy and explicit CPU execution.
- Training creates a new run folder, CSV progress log, PPO checkpoint, and JSON metadata with source revision/dirty state, seed, configuration, versions, elapsed time, and throughput.
- Added a short training-and-reload integration test and documented the official CPU-only PyTorch setup.
- Environment: CPython 3.12.3, Stable-Baselines3 2.9.2, PyTorch 2.15.0+cpu, Gymnasium 1.4.0, NumPy 2.5.4.
- Smoke run: 256 timesteps, seed 123, CPU, approximately 1,711.8 steps/s. This is integration evidence only; no landing-learning claim.

## 2026-11-07 — M2 landing environment

- Added a headless Gymnasium environment around the independent physics step function.
- Defined a three-channel normalized simulation action, eight-value normalized observation, seeded initial-condition distribution, toy-task touchdown/out-of-bounds/time-limit criteria, and readable reward.
- Added optional initial-state reset injection for deterministic scenario verification.
- Added environment API, seeding, bounds, terminal-condition, and episode-truncation tests; included Gymnasium's environment checker.
- Added Gymnasium as an environment-layer dependency and recorded resolved versions and M2 task choices in `Docs/ENGINEERING_DECISIONS.md` EDL-011.

## 2026-11-07 — Initial M0/M1 implementation

- Added setuptools project metadata, Python 3.12 virtual-environment setup guidance, pytest configuration, and `.gitignore`.
- Added the `vertex` source package and headless `vertex.simulation` module.
- Implemented fixed-mass planar state advancement with constant gravity, generic simulation-only applied force/moment, and explicit forward Euler integration.
- Added focused tests for package imports, deterministic free fall, body/world force direction, pitch/moment signs, and invalid inputs.
- Added a root README describing setup, subsystem interface, and simulation/safety limitations.
- Recorded coordinate/integration conventions and inspected host environment in `Docs/ENGINEERING_DECISIONS.md`.

### Environment and evidence

- OS: Windows 10 Home Single Language.
- Python: CPython 3.12 and 3.14 are installed; `.venv` uses CPython 3.12.
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU, driver 616.56, CUDA UMD 13.4 as shown by `nvidia-smi`. No GPU use or speed claim was made.
- Package versions installed in `.venv`: pytest 9.2.0 and the package. Runtime dependencies: none.
- WMI access was denied; GPU inventory came from `nvidia-smi`. No prior project virtual environment was present. PyTorch/CUDA compatibility remains unchecked because M0/M1 do not require it.
