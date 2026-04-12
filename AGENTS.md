# DeskBot — Agent Reference Guide

> This document provides essential context for AI coding agents working on the DeskBot project. It complements the human-oriented README.md and the detailed CLAUDE.md (which contains Claude-specific guidance and pedagogical requirements).

## Project Overview

DeskBot is a **MuJoCo-based physics simulator** for a self-balancing two-wheeled desktop robot (~25 cm tall). The project follows a **simulation-first** approach: develop and validate control algorithms in simulation before building physical hardware. The code is designed to be portable to a real microcontroller (ESP32-S3 with MPU6500 IMU, VL53L0X rangefinders, and Hall encoders).

### Core Philosophy: The Sensor Barrier

The codebase enforces a strict **sensor barrier** — the controller never reads simulator internals (`data.qpos`, `data.qvel`, etc.). All state flows through a realistic sensor pipeline:

```
MuJoCo physics → SensorModel (adds noise/bias) → SensorReadings → StateEstimator → Controller
```

This is the project's most important architectural constraint. Violating it breaks the sim-to-real transfer goal.

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Physics Engine | MuJoCo 3.0+ |
| Language | Python 3.10+ |
| Dependencies | mujoco, numpy, scipy |
| GUI | tkinter (separate thread) |
| Robot Model | MJCF (MuJoCo XML format) |

---

## Project Structure

```
deskbot/                    # Main package
  __init__.py               # (empty marker)
  __main__.py               # Entry point: python -m deskbot
  robot.py                  # Physical constants, scene paths
  sensors.py                # IMU + encoder + rangefinder + collision models
  control.py                # StateEstimator + BalanceController (PID) + LQRController
  sim.py                    # Main simulation loop + MuJoCo viewer integration
  navigation.py             # Bug2-inspired navigator with occupancy grid
  perception.py             # GroundGeometry: pitch-compensated rangefinder processing
  localization.py           # WiFi RSSI + wall-geometry heading correction
  gui.py                    # tkinter control panel (joystick, nav input)
  mapviz.py                 # Occupancy grid visualization
  models/
    deskbot.xml             # Robot-only MJCF (chassis, wheels, sensors, actuators)
    scene.xml               # Flat ground scene
    scene_skatepark.xml     # Ramps, bumps, rails, obstacles
    scene_apartment.xml     # T2 apartment layout (~50 m², 5 rooms)
    scene_nav_test.xml      # 3m corridor with 3 obstacles for nav testing

scripts/                    # Benchmarking and diagnostic scripts
  benchmark_random.py       # Randomized obstacle corridor testing
  benchmark_heading.py      # Navigation heading test
  benchmark_nav.py          # General navigation benchmark
  sweep_angle.py            # Sensor placement angle optimization
  optimize_sensors.py       # Sensor noise parameter optimization
  eval_mapping.py           # Mapping evaluation
  diag_*.py                 # Diagnostic scripts (yaw sources, drift, etc.)
  test_*.py                 # Component tests (perception, mapviz)

docs/                       # Design documents and research plans
  plan_astar_local.md       # A* local planner research plan
  teb_evaluation.md         # TEB planner evaluation document
```

---

## Build and Run Commands

### Quick Start (Windows)
```bash
run.bat                     # Auto-creates venv, installs deps, runs simulator
```

### Manual Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Simulator
```bash
python -m deskbot                    # Default flat scene
python -m deskbot --scene skatepark  # Skatepark with ramps/obstacles
python -m deskbot --scene apartment  # Apartment layout with localization
python -m deskbot --random           # Random obstacle corridor (press R for new terrain)
python -m deskbot --random --pid     # Use legacy PID instead of LQR
python -m deskbot --corridor 1.0     # Narrower corridor (default 1.5m half-width)
```

### Batch Shortcuts
```bash
apartment.bat               # Runs --scene apartment
skatepark.bat               # Runs --scene skatepark
nav_test.bat                # Runs --scene nav_test
random_test.bat             # Runs --random with default settings
```

---

## Code Architecture

### Key Modules

#### `robot.py`
- **Purpose**: Physical constants and model paths
- **Key exports**: `WHEEL_RADIUS`, `WHEEL_SEPARATION`, `MAX_TORQUE`, `SCENES`
- **Critical constraint**: Constants must stay in sync with `models/deskbot.xml`

#### `sensors.py`
- **Purpose**: Realistic sensor models (the sensor barrier implementation)
- **Key classes**:
  - `SensorReadings`: Data container for all sensor data the controller may use
  - `SensorModel`: Adds MPU6050-like IMU noise and encoder quantization
  - `CollisionDetector`: High-pass filter on accelerometer for impact detection
- **Critical constraint**: Controller must ONLY use `SensorReadings` fields

#### `control.py`
- **Purpose**: State estimation and balance control
- **Key classes**:
  - `StateEstimator`: Complementary filter (accel + gyro fusion), encoder odometry with encoder-pitch compensation
  - `BalanceController`: Cascaded PID (velocity outer → pitch inner + yaw)
  - `LQRController`: Optimal LQR with 6-state model `[pitch, pitch_rate, velocity, position, yaw_error, yaw_rate]`, CARE solution via scipy
- **Default controller**: LQR (use `--pid` flag for legacy PID)

#### `navigation.py`
- **Purpose**: Autonomous navigation using Bug2 + occupancy grid
- **Key classes**:
  - `Navigator`: Main navigation FSM (GO_HEADING, CONTOUR, REVERSE, IDLE)
  - `OccupancyGrid`: 60×60 log-odds grid (8cm resolution, egocentric, Bresenham raycasting)
  - `NavState`: Observable state for GUI display
- **Key insight**: Heading-based API (`set_heading(degrees)`) avoids dead reckoning position drift

#### `perception.py`
- **Purpose**: Ground geometry model for rangefinder interpretation
- **Key class**: `GroundGeometry` — pitch-compensated, vector-based ground projection
- **Output**: Classifies readings as `flat`, `obstacle`, `hole`, `no_reading`

#### `localization.py`
- **Purpose**: Indoor localization systems
- **Key classes**:
  - `WiFiLocalizer`: RSSI fingerprinting with log-distance path loss model
  - `HeadingCorrector`: Wall-geometry heading correction using side rangefinders

#### `sim.py`
- **Purpose**: Main simulation loop (60 fps display / 500 Hz physics)
- **Key function**: `run()` — the main entry point for interactive simulation
- **Threading**: Runs MuJoCo viewer + physics in main thread, tkinter GUI in daemon thread

---

## Development Conventions

### Code Style

1. **Docstrings**: Google-style docstrings for all modules and public functions
2. **Type hints**: Use type hints for function signatures
3. **Constants**: UPPER_CASE for module-level constants
4. **Private members**: Leading underscore for internal methods/attributes
5. **Comments**: Use `# ── Section headers ──` for visual grouping

### Coordinate System

- **X**: Forward
- **Y**: Left  
- **Z**: Up
- **Angles**: Radians (positive pitch = leaning forward)

### Sign Conventions (CRITICAL)

These are non-negotiable and must be respected:

| Variable | Positive Direction | Note |
|----------|-------------------|------|
| `pitch` | Leaning forward | `atan2(-ax, az)` for accel pitch |
| `torque` | Drives wheel forward | Left torque subtracts for right turn |
| `yaw_rate` | Turning left | Encoder differential: `(vr - vl) / WHEEL_SEPARATION` |
| `heading` | Counter-clockwise from +X | 0° = +X, 90° = +Y |

**Common bug**: Sign errors in LQR cause immediate divergence (unlike PID which tolerates them).

---

## Testing and Validation

### No Formal Test Suite

There are no unit tests or linters configured. Validation is done through:

1. **Interactive simulation**: Visual assessment in MuJoCo viewer
2. **Benchmark scripts**: Headless Monte Carlo testing

### Running Benchmarks

```bash
# Randomized obstacle benchmark (100 episodes)
python scripts/benchmark_random.py --episodes 100

# Specific navigation test
python scripts/benchmark_heading.py --heading 45

# Sensor angle sweep
python scripts/sweep_angle.py

# Mapping evaluation
python scripts/eval_mapping.py --episodes 15
```

### Success Metrics

The navigation system targets:
- **82% success rate** on randomized obstacle corridors
- **0 falls** in normal operation
- **<10° heading drift** over 40 seconds

---

## Hardware Mapping

The simulator models this physical hardware (BOM ordered):

| Simulated Component | Hardware Target |
|--------------------|-----------------|
| IMU (MPU6050-like noise) | MPU6500 (6-axis, better noise) |
| 7× rangefinders | 7× VL53L0X via TCA9548A I2C mux |
| Wheel encoders | JGA25-371 Hall encoders (1440 ticks/rev) |
| Motors | JGA25-371 12V 280RPM |
| Motor driver | TB6612FNG (MOSFET, 90%+ efficiency) |
| Microcontroller | ESP32-S3 N16R8 (dual-core 240MHz, WiFi) |
| Battery | NiMH 9.6V (8× AA) |

---

## Documentation and Development Journal

### JOURNAL.md (Mandatory)

All significant changes must be documented in `JOURNAL.md` at the repository root. This is a **human-AI collaboration journal** that tracks:
- What was done in each session
- What worked and why
- What failed and lessons learned
- Roadmap updates

**Format**: Session-based entries with the header `## Session N -- YYYY-MM-DD : Title`

### CLAUDE.md

Contains Claude-specific guidance including:
- Pedagogical mode requirements (every response ends with "What to learn from this")
- Memory system conventions
- Detailed architectural notes

### docs/ Directory

Research plans and evaluation documents:
- `plan_astar_local.md`: A* local planner research plan
- `teb_evaluation.md`: TEB planner evaluation and lessons from failed attempts

---

## Security Considerations

- **No network services**: The simulator has no network attack surface
- **Local execution only**: All code runs locally with standard user permissions
- **Input validation**: Command-line arguments are validated via `argparse`

---

## Critical Constraints for Agents

1. **Never bypass the sensor barrier**: Controller code must only use `SensorReadings` fields (accel, gyro, encoders, rangefinders, collision flags)

2. **Keep robot.py constants in sync**: Any MJCF model change requires updating `robot.py` constants (wheel radius, separation, max torque)

3. **Respect sign conventions**: MuJoCo conventions differ from textbook robotics. When in doubt, verify empirically with a short test script.

4. **Test navigation changes with benchmarks**: Single-scene testing is insufficient. Always use `scripts/benchmark_random.py` with 100+ episodes.

5. **Update JOURNAL.md**: Document all significant changes, failures, and lessons learned after every meaningful session.

6. **No new dependencies without justification**: The project uses only mujoco, numpy, scipy. Any additional dependency must be discussed.

7. **Validate against hardware constraints**: Code should be portable to ESP32-S3 (240 MHz, limited RAM). Avoid heavy allocations in the control loop.

---

## References

- **MuJoCo docs**: https://mujoco.readthedocs.io/
- **MJCF format**: https://mujoco.readthedocs.io/en/stable/XMLreference.html
- **Bug2 algorithm**: Lumelsky & Stepanov, 1987
- **LQR/Optimal Control**: `scipy.linalg.solve_continuous_are` for solving the Continuous Algebraic Riccati Equation
