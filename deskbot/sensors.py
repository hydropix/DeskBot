"""
Realistic sensor models for the DeskBot.

Simulates the output of real hardware:
  - IMU 6-axis (MPU6050-like): accelerometer + gyroscope with noise & bias drift
  - Wheel encoders: discrete ticks with quantization noise

The controller must ONLY use SensorReadings — never simulator internals.
"""
import math
import numpy as np
import mujoco

# ── IMU noise parameters (MPU6050 datasheet-inspired) ──────────────
# Accelerometer
ACCEL_NOISE_DENSITY = 0.004       # m/s² / sqrt(Hz)  (~400 µg/√Hz)
ACCEL_BIAS_STABILITY = 0.02       # m/s²  (slow random walk)

# Gyroscope
GYRO_NOISE_DENSITY = 0.005        # rad/s / sqrt(Hz)  (~0.3 °/s/√Hz)
GYRO_BIAS_STABILITY = 0.0002      # rad/s  (slow drift ~0.01 °/s)

# ── Encoder parameters ─────────────────────────────────────────────
ENCODER_TICKS_PER_REV = 360       # typical magnetic encoder resolution
TICK_SIZE = 2 * math.pi / ENCODER_TICKS_PER_REV  # rad per tick


class SensorReadings:
    """What the robot's microcontroller actually receives each cycle."""
    __slots__ = (
        "accel",         # [ax, ay, az] in body frame (m/s²)
        "gyro",          # [wx, wy, wz] in body frame (rad/s)
        "encoder_left",  # wheel angular velocity (rad/s), quantized
        "encoder_right",
    )

    def __init__(self):
        self.accel = np.zeros(3)
        self.gyro = np.zeros(3)
        self.encoder_left = 0.0
        self.encoder_right = 0.0


class SensorModel:
    """
    Reads MuJoCo sensor data and adds realistic noise/bias.
    This is the ONLY bridge between the simulator and the controller.
    """
    def __init__(self, model: mujoco.MjModel, dt: float):
        self.dt = dt
        self._sqrt_dt = math.sqrt(dt)
        self._sample_rate = 1.0 / dt
        self._sqrt_rate = math.sqrt(self._sample_rate)

        # Sensor indices in data.sensordata
        self._accel_idx = model.sensor_adr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "accel")
        ]
        self._gyro_idx = model.sensor_adr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")
        ]
        self._enc_l_idx = model.sensor_adr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "enc_L")
        ]
        self._enc_r_idx = model.sensor_adr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "enc_R")
        ]
        # IMU site id (to get rotation matrix for gravity compensation)
        self._imu_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "imu"
        )
        self._gravity = np.array([0.0, 0.0, -model.opt.gravity[2]])  # [0, 0, 9.81]

        # Persistent bias state (random walk)
        self._accel_bias = np.zeros(3)
        self._gyro_bias = np.zeros(3)

        # Encoder accumulator for tick quantization
        self._enc_l_accum = 0.0
        self._enc_r_accum = 0.0

    def reset(self):
        self._accel_bias = np.zeros(3)
        self._gyro_bias = np.zeros(3)
        self._enc_l_accum = 0.0
        self._enc_r_accum = 0.0

    def read(self, data: mujoco.MjData) -> SensorReadings:
        """Sample all sensors with realistic noise."""
        r = SensorReadings()

        # ── IMU accelerometer ──
        # MuJoCo gives coordinate acceleration (0 at rest).
        # A real IMU measures proper acceleration = coord_accel + g_in_body_frame.
        coord_accel = data.sensordata[self._accel_idx: self._accel_idx + 3].copy()
        site_rot = data.site_xmat[self._imu_site_id].reshape(3, 3)
        gravity_body = site_rot.T @ self._gravity  # g rotated into sensor frame
        true_accel = coord_accel + gravity_body
        # White noise + bias drift
        accel_noise = np.random.randn(3) * ACCEL_NOISE_DENSITY * self._sqrt_rate
        self._accel_bias += np.random.randn(3) * ACCEL_BIAS_STABILITY * self._sqrt_dt
        r.accel = true_accel + accel_noise + self._accel_bias

        # ── IMU gyroscope ──
        true_gyro = data.sensordata[self._gyro_idx: self._gyro_idx + 3].copy()
        gyro_noise = np.random.randn(3) * GYRO_NOISE_DENSITY * self._sqrt_rate
        self._gyro_bias += np.random.randn(3) * GYRO_BIAS_STABILITY * self._sqrt_dt
        r.gyro = true_gyro + gyro_noise + self._gyro_bias

        # ── Wheel encoders (quantized ticks) ──
        true_vel_l = float(data.sensordata[self._enc_l_idx])
        true_vel_r = float(data.sensordata[self._enc_r_idx])

        # Accumulate angle, quantize to ticks
        self._enc_l_accum += true_vel_l * self.dt
        self._enc_r_accum += true_vel_r * self.dt

        ticks_l = math.floor(self._enc_l_accum / TICK_SIZE)
        ticks_r = math.floor(self._enc_r_accum / TICK_SIZE)

        self._enc_l_accum -= ticks_l * TICK_SIZE
        self._enc_r_accum -= ticks_r * TICK_SIZE

        # Convert ticks back to angular velocity
        r.encoder_left = (ticks_l * TICK_SIZE) / self.dt
        r.encoder_right = (ticks_r * TICK_SIZE) / self.dt

        return r
