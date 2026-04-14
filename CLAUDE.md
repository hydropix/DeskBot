# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

DeskBot is a MuJoCo-based simulator for a self-balancing two-wheeled desktop robot (~25 cm tall). The goal is to develop control algorithms that are portable to real hardware.

## Commands

```bash
# Setup (or just run run.bat which auto-creates venv)
python -m venv .venv && .venv/Scripts/activate && pip install -r requirements.txt

# Run the simulator
python -m deskbot
```

There are no tests or linters configured yet.

## Architecture

The codebase enforces a strict **sensor barrier**: the controller never reads simulator internals (`data.qpos`, `data.qvel`, etc.). All state flows through a realistic sensor pipeline:

```
MuJoCo physics → SensorModel (noise/bias) → SensorReadings → StateEstimator → BalanceController
```

### Key modules

- **`robot.py`** — Physical constants (`WHEEL_RADIUS`, `WHEEL_SEPARATION`, `MAX_TORQUE`) and paths to MJCF model files. Constants must stay in sync with `models/deskbot.xml`.
- **`sensors.py`** — `SensorModel` reads MuJoCo sensor data and adds MPU6050-like IMU noise (bias drift, white noise) and encoder tick quantization. `SensorReadings` is the only data the controller may consume.
- **`control.py`** — `StateEstimator` fuses accelerometer + gyroscope via complementary filter and derives velocity/yaw from encoder odometry. `BalanceController` runs a cascaded loop: velocity PI (outer) → pitch PID (inner) + yaw P+D (differential steering).
- **`sim.py`** — Real-time viewer loop at 60 fps / 500 Hz physics. Handles keyboard input (arrow keys for motion, Space for random push, R for reset).

### MJCF models (`deskbot/models/`)

- **`deskbot.xml`** — Robot-only definition (chassis, wheels, actuators, sensors). Uses MuJoCo defaults classes (`wheel`, `hub`, `visual`).
- **`scene.xml`** — Includes `deskbot.xml` and adds ground plane, lighting, and visual markers. This is the file loaded by the simulator.

Coordinate system: **X=forward, Y=left, Z=up**. Angles in radians.

## Critical Constraints

- **Never bypass the sensor barrier.** Controller and estimator code must only use `SensorReadings` fields (accel, gyro, encoder_left, encoder_right). Reading `data.qpos`/`data.qvel` in control code breaks the sim-to-real transfer goal.
- **Sign conventions matter.** Pitch: positive = leaning forward. Torque: positive = drives wheel forward. Yaw: positive = turning left. Getting these wrong causes the balance loop to diverge instantly.
- **Accelerometer includes gravity.** The `SensorModel` converts MuJoCo's coordinate acceleration to proper acceleration (what a real IMU measures) by adding gravity rotated into the body frame. The estimator uses `atan2(-ax, az)` for pitch — don't change this without understanding why the negation is there.

## Development Journal (MANDATORY)

A project journal is maintained in `JOURNAL.md` at the repository root. It documents the complete development journey: what was tried, what worked, what failed, and the lessons learned. **Claude Code is an active participant in this project** -- not just a tool, but a design partner. The journal should reflect this: decisions are made collaboratively, and Claude's reasoning, suggestions, and mistakes are part of the story.

### Update rules

1. **When to update**: After every session that produces meaningful changes (new features, bug fixes, architectural decisions, failed experiments). Update the journal **at the same time** as saving to auto-memory -- these are complementary:
   - **Memory** = internal recall for future Claude conversations (structured, machine-oriented).
   - **Journal** = human-readable project history (narrative, meant to be read by Bruno or anyone reviewing the project).

2. **What to record for each session**:
   - **Session header**: `## Session N -- YYYY-MM-DD : Short Title`
   - **What was done**: Bullet list of concrete changes (files created/modified, features added).
   - **What worked**: Things that succeeded, especially if the reason is non-obvious.
   - **What failed / lessons learned**: Honest account of failures, dead ends, and why. This is the most valuable part -- don't skip it or sugarcoat it.
   - Update the **Roadmap** section at the bottom (check off completed items, add new ones).

3. **Style**:
   - Write in past tense, factual, concise.
   - Include specific numbers (success rates, parameter values, measurements) when available.
   - Reference file names but don't paste code -- the journal is a narrative, not a code dump.
   - Failed experiments are as valuable as successes. Always explain *why* something failed, not just *that* it failed.

4. **Never delete history.** Failed approaches stay in the journal. They document the reasoning process and prevent revisiting dead ends.

## Pedagogical Mode (MANDATORY)


Every response MUST end with a **"What to learn from this"** section.


### Purpose


The user is not just a client — he is also a learner who wants to deeply understand what was done and why. Every interaction is a training opportunity. The goal is to build lasting software engineering knowledge over time.