# Learning Guide — Systems Engineer's View

**Project:** PROJECT VERTEX  
**Version:** 0.1  
**Audience:** Project owner; beginner in reinforcement learning

## 1. How to learn during the project

Do not study every subject in advance. Learn each idea when it becomes necessary to make or review an engineering decision.

For each new concept, explain:

1. **What it is:** one or two plain-language sentences.
2. **Why we need it:** the project requirement it supports.
3. **What goes in and what comes out:** its system interface.
4. **How we verify it:** a test, metric, or observable result.
5. **What it does not prove:** its limitations.

Avoid lengthy mathematical derivations by default. Use simple examples and diagrams when useful.

## 2. The minimum RL overview

Reinforcement learning is a way to train a decision-making system by letting it interact with an environment and learn from feedback.

In VERTEX:

- **Agent:** the learned policy being trained.
- **Environment:** the simulated vehicle and landing task.
- **Observation:** the information provided to the agent at each step.
- **Action:** the bounded abstract control request returned by the agent.
- **Reward:** feedback that encourages behaviour useful for the task.
- **Episode:** one simulated attempt, from reset to landing, failure, or time limit.
- **Policy:** the trained mapping from observations to actions.
- **Training:** repeatedly collecting experience and updating the policy.
- **Evaluation:** testing a saved policy on episodes not used for the training update.

The agent does not understand rockets. It learns patterns that improve the reward in the environment we created. If the environment is wrong or incomplete, the learned behaviour may also be wrong or incomplete.

## 3. Why PPO first

PPO (Proximal Policy Optimization) is an established policy-gradient reinforcement-learning algorithm. Stable-Baselines3 provides a maintained implementation and supports continuous `Box` action spaces.

The owner only needs the initial practical understanding:

- PPO tries actions in the environment.
- It gathers batches of experience.
- It updates a neural-network policy while limiting excessively large policy changes.
- Repeating this process can improve task performance.

We are using PPO to reduce implementation work, not because it is guaranteed to be the best algorithm for a rocket-landing problem.

## 4. Why Gymnasium

Gymnasium standardizes how an agent interacts with an environment. The important operations are:

- `reset()`: start a new episode and return the first observation.
- `step(action)`: apply one action and return the next observation, reward, termination flags, and diagnostic information.

This boundary allows us to replace or improve the physics model without rewriting the PPO implementation, provided the environment interface remains consistent.

## 5. Simulator overview

A simulator is not just an animation. It is a model that predicts how a system changes when time advances and inputs or disturbances are applied.

For VERTEX, we first build a transparent mathematical model in Python. Later, we can evaluate a dedicated physics engine if we need more realistic contacts, constraints, or complex multi-body behaviour.

### Newton and Isaac Lab

- **Newton** is a physics engine. It supplies physics simulation capabilities.
- **Isaac Lab** is a robot-learning framework that organizes tasks, simulation, sensors, environments, and policy-training workflows; it can work with several physics backends, including Newton.
- Neither one automatically provides a rocket-landing task. We would still need to define and validate the vehicle, environment, control interface, and objective.

The owner should learn Newton through a small standalone example before deciding whether to use it for VERTEX.

## 6. System engineering concepts to learn at the right time

| Concept | Why it matters here | When to learn |
|---|---|---|
| Coordinate frames and sign conventions | Prevents physics and sensor interpretation mistakes | M1 |
| Rigid-body state | Describes vehicle motion and attitude | M1 |
| Numerical integration | Advances the simulated state over time | M1 |
| Observation and action interfaces | Defines what the agent knows and can influence | M2 |
| Reward vs success criteria | Prevents confusing a high score with a good landing | M2–M4 |
| Training vs evaluation | Measures generalization rather than memorization | M3–M4 |
| Sensor noise and bias | Makes observations less idealized | M6 |
| State estimation | Converts imperfect sensor measurements into useful state estimates | Later |
| Sim-to-real gap | Explains why a simulation policy may fail on hardware | Later |
| Hardware-in-the-loop | Tests software against real electronics without a live powered flight | Later |

## 7. How to interpret results

- A training reward curve is a learning diagnostic, not proof of landing success.
- A successful episode is one sample, not proof of reliability.
- A success rate must include the number of evaluation episodes and their initial-condition distribution.
- A model can perform well in one simulator and fail in another because the physics and interfaces differ.
- A 2D model cannot establish 3D flight stability.
- A passive or captive physical demonstration does not validate powered-flight performance.

## 8. Preferred explanation format from Codex

For a major implementation step, use this compact format:

**What we built** — one short paragraph.  
**Why it matters** — the requirement or engineering problem it addresses.  
**How it works** — a high-level data-flow explanation.  
**Verification** — exact test command and result.  
**Limitations / next step** — what remains unproven.

Do not turn every code change into a lecture. Explain only enough for an informed engineering review.
