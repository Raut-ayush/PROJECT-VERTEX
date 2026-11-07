# Safety, Scope, and Physical-Validation Boundaries

**Project:** PROJECT VERTEX  
**Version:** 0.1  
**Date:** 2026-11-07

## 1. Purpose

Preserve the long-term project ambition while keeping the initial software work appropriately scoped and safe.

The project owner has stated that the ultimate ambition is to make and fly a real vertical rocket. This document does not dismiss that ambition. It establishes that the current codebase is a simulation and learning project, not a design package or authorization for building or flying a powered rocket.

## 2. Current allowed project scope

The current development work can cover:

- Abstract 2D and later 3D simulation.
- General rigid-body modelling concepts.
- Reinforcement-learning concepts and training workflows.
- Software architecture and reproducibility.
- Non-operational simulated sensor and actuator abstractions.
- High-level avionics architecture.
- Bench-level software and sensor-interface testing.
- Non-propulsive captive/guided or passive drop-test demonstrators.
- Risk assessment, test planning, documentation, and compliance research.

## 3. Excluded from this initial project work

Do not create or request:

- DIY propellant or fuel/oxidizer recipes, preparation, processing, or mixing.
- Rocket motor construction, modification, or performance optimization instructions.
- Ignition, firing, or motor-control implementation instructions.
- Detailed instructions for an autonomous powered home-built rocket flight.
- Operational flight envelopes, launch procedures, or field test instructions.
- A direct mapping from the v0.1 abstract RL actions to real propulsion hardware.

If future work approaches propulsion or powered flight, stop and identify the need for qualified propulsion engineers, appropriate test facilities, independent safety review, and applicable legal/regulatory authorization. Do not treat a software milestone as permission to proceed with physical flight work.

## 4. Simulation safety assumptions

- All actions in v0.1 are abstract simulation actions.
- Simulation parameters are educational modelling parameters, not real vehicle design values.
- Simulated success thresholds are not real-world safe limits.
- A policy checkpoint is not flight-ready software.
- The 2D rigid-body model is not adequate for proving real vehicle stability or controllability.

## 5. Physical validation boundary

The first physical demonstrator should be non-propulsive. Candidate directions include:

1. A passive lander or test article to examine landing structure, touchdown sensing, and impact behaviour.
2. A captive/guided fixture to test selected sensors, computation, policy inference, software interfaces, and safe low-energy actuation where appropriate.

Choose the simpler, lower-risk experiment first. These tests should be designed so an unexpected command or software fault cannot create a free-flying powered vehicle.

A non-propulsive demonstrator can validate only the subsystems actually exercised. It cannot validate powered-rocket propulsion, aerodynamic flight, or autonomous powered landing.

## 6. Future gate before any powered-vehicle programme

A separate, independently reviewed programme would need to address at least:

- Regulatory and site authorization.
- Qualified propulsion and aerospace engineering supervision.
- Ground-test facilities and exclusion zones.
- Structural, thermal, vibration, and environmental qualification.
- Independent flight-safety mechanisms.
- Fault handling and safe abort behaviour.
- Verified sensor, estimator, actuator, and timing behaviour.
- 3D model validity and uncertainty analysis.
- Hardware-in-the-loop and staged non-flight verification.
- Formal go/no-go criteria and independent review.

This list is a high-level governance reminder, not a technical procedure or flight checklist.

## 7. Codex behaviour

Codex must keep the current implementation within the software-only simulation scope. It should flag scope creep if a task introduces real propulsion, ignition, operational flight, or hardware deployment. It should suggest a safe simulation or non-propulsive alternative where appropriate.
