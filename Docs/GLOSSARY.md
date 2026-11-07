# PROJECT VERTEX — Glossary

| Term | Meaning in this project |
|---|---|
| Agent | The learning system that chooses actions. |
| Action | A bounded abstract input applied to the simulation. In v0.1 it is not a real actuator or motor command. |
| Actuator model | A simulation component that represents how an abstract requested input affects the simulated vehicle. |
| Episode | One complete simulated attempt, from reset to terminal condition. |
| Evaluation | Testing a saved policy on a defined set of episodes to measure task performance. |
| Gymnasium | A Python interface standard for environments that agents interact with. |
| Headless | Running without opening a graphical window or rendering frames. |
| Isaac Lab | NVIDIA's open-source framework for robot simulation and learning workflows. |
| Newton | An open-source physics engine built on NVIDIA Warp and OpenUSD. |
| Observation | The data presented to the agent at each decision step. |
| Policy | The learned function that maps observations to actions. |
| PPO | Proximal Policy Optimization, the selected initial RL algorithm. |
| Reward | Feedback used during training to encourage desired simulated behaviour. |
| Rigid body | A simplified body whose shape is treated as fixed while it translates and rotates. |
| Sim-to-real | The challenge of transferring behaviour learned in simulation to a physical system. |
| State | The simulated vehicle's internal condition, such as position, velocity, pitch, and angular velocity. |
| Termination | An episode-ending event defined by the task, such as successful or unsuccessful touchdown. |
| Truncation | An episode ending due to an external limit, such as a time limit. |
