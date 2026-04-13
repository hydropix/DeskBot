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
ENCODER_TICKS_PER_REV = 1440      # 360-line encoder with 4x quadrature decoding
TICK_SIZE = 2 * math.pi / ENCODER_TICKS_PER_REV  # rad per tick

# ── Rangefinder parameters (VL53L0X) ──────────────────────────────
# 10-sensor foveal array, wired through 2× TCA9548A I²C multiplexers.
# Order below follows the azimuth sweep (−180° → +180°) for readability
# only — the code never assumes any ordering.
#
#   fovea (3 parallel beams covering the robot's straight-ahead column):
#       rf_C, rf_FL, rf_FR
#   mid-forward ±25°:
#       rf_L, rf_R
#   wide-forward ±55°:
#       rf_WL, rf_WR
#   pure side ±90° (corridor tracking):
#       rf_SL, rf_SR
#   rear 180°:
#       rf_B
RF_NAMES = [
    "rf_C", "rf_FL", "rf_FR",
    "rf_L", "rf_R",
    "rf_WL", "rf_WR",
    "rf_SL", "rf_SR",
    "rf_B",
]
RF_MAX_RANGE = 2.0        # meters (cutoff in MJCF)
RF_NOISE_SIGMA = 0.005    # 5mm gaussian noise
RF_NOISE_PERCENT = 0.03   # 3% of reading


class SensorReadings:
    """What the robot's microcontroller actually receives each cycle."""
    __slots__ = (
        "accel",              # [ax, ay, az] in body frame (m/s²)
        "gyro",               # [wx, wy, wz] in body frame (rad/s)
        "encoder_left",       # wheel angular velocity (rad/s), quantized
        "encoder_right",
        "rangefinders",       # dict: name -> distance in meters (-1.0 = no hit)
        "collision_detected", # bool: impact detected this cycle
        "collision_magnitude",# float: impact strength (m/s², 0 if no collision)
        "collision_direction",# [dx, dy, dz] unit vector of impact in body frame
    )

    def __init__(self):
        self.accel = np.zeros(3)
        self.gyro = np.zeros(3)
        self.encoder_left = 0.0
        self.encoder_right = 0.0
        self.rangefinders = {name: -1.0 for name in RF_NAMES}
        self.collision_detected = False
        self.collision_magnitude = 0.0
        self.collision_direction = np.zeros(3)


# ── Collision detection parameters ────────────────────────────────
#
# COLLISION_THRESHOLD was originally 8 m/s² but empirical profiling
# (see JOURNAL session on false-positive phantom stamps) showed that
# normal balance dynamics produce regular HP-filter spikes of 10-20 m/s²
# during turn-in-place and aggressive forward-turn regimes — 80+ false
# triggers per second on the old 8 m/s² threshold. MuJoCo stiff contacts
# produce peaks well above 100 m/s² on any real impact, so 25 m/s² sits
# comfortably above balance noise and far below real hits.
#
# `COLLISION_REFRACTORY_S` is a one-shot lockout after a detection: the
# HP filter rings for several ms after a spike, so without a cooldown a
# single impact stamps the grid multiple times as the filter decays
# through the threshold repeatedly.
COLLISION_THRESHOLD = 25.0    # m/s² — impact magnitude to trigger detection
COLLISION_HP_ALPHA = 0.95     # high-pass filter coefficient (higher = more filtering of DC)
COLLISION_REFRACTORY_S = 0.25 # seconds — suppress re-triggers after an impact
GRAVITY_NOMINAL = 9.81        # m/s² — expected gravity magnitude at rest


class CollisionDetector:
    """
    Detects collisions using a high-pass filter on the accelerometer signal.

    The accelerometer constantly reads ~9.81 m/s² (gravity) plus balance
    corrections (~2-5 m/s² DC, up to ~20 m/s² transients at 10-30 Hz).
    Real impacts produce much sharper broadband spikes (>>100 m/s² on
    stiff contacts). A first-order high-pass filter separates gravity
    and DC, a generous amplitude threshold separates impacts from balance
    dynamics, and a refractory period prevents multi-stamping from a
    single event as the filter rings back down through the threshold.

    High-pass filter: y[n] = alpha * (y[n-1] + x[n] - x[n-1])
    """

    def __init__(self, dt: float):
        self.dt = dt
        self._prev_accel = np.zeros(3)
        self._hp_accel = np.zeros(3)  # high-pass filtered output
        self._refractory = 0.0

    def reset(self):
        self._prev_accel = np.zeros(3)
        self._hp_accel = np.zeros(3)
        self._refractory = 0.0

    def process(self, readings: 'SensorReadings'):
        """Update collision fields in a SensorReadings object."""
        accel = readings.accel

        # High-pass filter: removes gravity + slow balance oscillations
        self._hp_accel = COLLISION_HP_ALPHA * (
            self._hp_accel + accel - self._prev_accel
        )
        self._prev_accel = accel.copy()

        magnitude = float(np.linalg.norm(self._hp_accel))

        if self._refractory > 0.0:
            self._refractory = max(0.0, self._refractory - self.dt)
            readings.collision_detected = False
            readings.collision_magnitude = 0.0
            readings.collision_direction = np.zeros(3)
            return

        if magnitude > COLLISION_THRESHOLD:
            readings.collision_detected = True
            readings.collision_magnitude = magnitude
            readings.collision_direction = self._hp_accel / magnitude
            self._refractory = COLLISION_REFRACTORY_S
        else:
            readings.collision_detected = False
            readings.collision_magnitude = 0.0
            readings.collision_direction = np.zeros(3)


class SensorModel:
    """
    Reads MuJoCo sensor data and adds realistic noise/bias.
    This is the ONLY bridge between the simulator and the controller.
    """
    def __init__(self, model: mujoco.MjModel, dt: float):
        self._model = model
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

        # Rangefinder site ids (raycasting done via mj_ray, not built-in sensor)
        self._rf_site_ids = {}
        for name in RF_NAMES:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site_id >= 0:
                self._rf_site_ids[name] = site_id

        # Chassis body id — excluded from raycasting via bodyexclude
        self._chassis_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "chassis"
        )
        self._rf_geomid_buf = np.zeros(1, dtype=np.int32)
        self._gravity = np.array([0.0, 0.0, -model.opt.gravity[2]])  # [0, 0, 9.81]

        # Persistent bias state (random walk)
        self._accel_bias = np.zeros(3)
        self._gyro_bias = np.zeros(3)

        # Encoder accumulator for tick quantization
        self._enc_l_accum = 0.0
        self._enc_r_accum = 0.0

        # Collision detector
        self._collision_detector = CollisionDetector(dt)

    def reset(self):
        self._accel_bias = np.zeros(3)
        self._gyro_bias = np.zeros(3)
        self._enc_l_accum = 0.0
        self._enc_r_accum = 0.0
        self._collision_detector.reset()

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

        # ── Rangefinders (VL53L0X via mj_ray) ──
        # Sensors are physically mounted on the sensor pod, outside all robot
        # geoms. bodyexclude=chassis_id skips chassis-body geoms; the sensor
        # pod placement guarantees rays clear the wheels without hacks.
        for name, site_id in self._rf_site_ids.items():
            origin = data.site_xpos[site_id].copy()
            direction = data.site_xmat[site_id].reshape(3, 3)[:, 2]

            dist = mujoco.mj_ray(
                self._model, data, origin, direction,
                None, 1, self._chassis_id, self._rf_geomid_buf,
            )
            if dist < 0 or dist > RF_MAX_RANGE:
                r.rangefinders[name] = -1.0
            else:
                noise = (dist * RF_NOISE_PERCENT + RF_NOISE_SIGMA) * np.random.randn()
                r.rangefinders[name] = max(0.0, dist + noise)

        # ── Collision detection (high-pass filter on accel) ──
        self._collision_detector.process(r)

        return r
