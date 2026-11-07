# PROJECT VERTEX

PROJECT VERTEX is a small, headless simulation and learning project. Its long-term ambition is a real vertical-landing rocket demonstrator, but this code is an educational toy model and is not flight-ready. The first physical validation, when reached, will be non-propulsive.

## Development setup

Use CPython 3.12 for the current baseline. From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

If PowerShell blocks activation, use `\.venv\Scripts\python.exe` in place of `python`; no activation policy change is required. Gymnasium is used by the environment layer; the M0/M1 physics module itself still imports without Gymnasium or NumPy. PPO is an optional extra. Graphics and hardware interfaces are not implemented.

## Current subsystem

`vertex.simulation` owns the planar state and deterministic time-step model. Its inputs are a state, fixed configuration, and an abstract simulation-only force/moment; its output is the next state. It imports no RL or visualization packages. Tests check units, signs, and known motion cases.

`vertex.environment.LandingEnv` wraps that model in a headless Gymnasium reset/step interface. It defines bounded abstract actions, normalized observations, seeded initial conditions, toy-task terminal checks, and reward. Its threshold and scale choices are documented in `Docs/ENGINEERING_DECISIONS.md` EDL-011.

```powershell
python -m pytest
```

## PPO integration smoke run

For the CPU-only M3 baseline on Windows, install PyTorch's CPU wheel, then the optional PPO dependency:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,rl]"
python -m vertex.learning.train --timesteps 256 --seed 123 --output-dir runs/m3-smoke
```

The run writes `ppo_vertex.zip`, `run_metadata.json`, and `logs/progress.csv`. The output folder must be new. A short run verifies integration and checkpoint loading; it does not assess whether the policy can land successfully. PPO is explicitly configured for CPU. Do not switch to GPU without benchmarking.

Simulation values and inputs are generic educational model parameters. They do not specify a vehicle or map to physical propulsion commands. Simulation success does not establish physical safety.

