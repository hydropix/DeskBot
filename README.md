# DeskBot

A small self-balancing two-wheeled desktop robot (~25 cm tall), built as an exploratory robotics project.

The approach is **simulation-first**: develop and validate control algorithms in [MuJoCo](https://mujoco.org/) before building the physical robot.

![Status](https://img.shields.io/badge/status-simulation%20phase-blue)

## Why

Most hobby robot projects start with hardware and then struggle with control. DeskBot flips this: the physics simulator enforces realistic sensor constraints (noisy IMU, quantized encoders) so that code developed here transfers directly to real hardware — an Arduino or ESP32 reading an MPU6050 and magnetic encoders.

## Quick start

```bash
# Clone and run (Windows)
git clone https://github.com/hydropix/DeskBot.git
cd DeskBot
run.bat

# Or manually
python -m venv .venv && .venv/Scripts/activate && pip install -r requirements.txt
python -m deskbot
```

### Controls

| Key | Action |
|-----|--------|
| Arrow Up/Down | Move forward/backward |
| Arrow Left/Right | Turn |
| S | Immediate stop |
| Space | Random push |
| R | Reset position |
| ESC | Quit |
| Mouse | Rotate/zoom camera |

### Scenes

```bash
python -m deskbot --scene flat        # Default flat ground
python -m deskbot --scene skatepark   # Ramps, bumps, rails, obstacles
```

## Architecture

The codebase enforces a strict **sensor barrier** — the controller never reads simulator internals. All state flows through a realistic sensor pipeline:

```
MuJoCo physics
     |
SensorModel          IMU noise/drift + encoder quantization
     |
SensorReadings       What a real microcontroller would see
     |
StateEstimator       Complementary filter + odometry
     |
BalanceController    Cascaded PID (velocity → pitch → torque)
     |
Motor commands
```

## Project structure

```
deskbot/
  models/
    deskbot.xml            Robot-only MJCF model
    scene.xml              Flat ground scene
    scene_skatepark.xml    Skatepark with ramps and obstacles
  robot.py                 Physical constants and model paths
  sensors.py               IMU + encoder noise models (MPU6050-like)
  control.py               State estimator + balance controller
  sim.py                   Real-time viewer loop (60 fps / 500 Hz physics)
```

## Roadmap

- [x] MuJoCo simulation with realistic sensor pipeline
- [x] Cascaded PID balance controller
- [x] Separated MJCF model/scene (Menagerie convention)
- [ ] LQR controller
- [ ] Domain randomization (mass, friction, sensor noise)
- [ ] Gymnasium integration for RL
- [ ] MJPC predictive control
- [ ] Manipulator arm
- [ ] RL training with MuJoCo Playground
- [ ] Physical build and sim-to-real transfer

## Requirements

- Python 3.10+
- MuJoCo 3.0+
- NumPy

## License

This project is for personal exploration and learning.