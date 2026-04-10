"""
Balance controller for the two-wheeled DeskBot.

Uses ONLY realistic sensor data (SensorReadings):
  - IMU accelerometer + gyroscope → pitch estimation via complementary filter
  - Wheel encoders → forward velocity & yaw rate

This code is designed to be portable to a real microcontroller.
"""
import math
import numpy as np

from deskbot.sensors import SensorReadings
from deskbot.robot import WHEEL_RADIUS, WHEEL_SEPARATION


class StateEstimator:
    """
    Estimates robot state from raw sensor data.
    Complementary filter fuses accelerometer (long-term) and gyroscope (short-term).
    Odometry from wheel encoders gives forward velocity and yaw rate.
    """
    def __init__(self, dt: float, alpha: float = 0.96):
        self.dt = dt
        self.alpha = alpha
        self.pitch = 0.0
        self.pitch_rate = 0.0
        self.yaw_rate = 0.0
        self.forward_vel = 0.0
        self.fallen = False
        self._initialized = False

    def reset(self):
        self.pitch = 0.0
        self.pitch_rate = 0.0
        self.yaw_rate = 0.0
        self.forward_vel = 0.0
        self.fallen = False
        self._initialized = False

    def update(self, s: SensorReadings):
        """Process one sensor cycle."""
        self.pitch_rate = s.gyro[1]

        accel_pitch = math.atan2(-s.accel[0], s.accel[2])

        if not self._initialized:
            self.pitch = accel_pitch
            self._initialized = True
        else:
            gyro_pitch = self.pitch + self.pitch_rate * self.dt
            self.pitch = self.alpha * gyro_pitch + (1.0 - self.alpha) * accel_pitch

        vel_left = s.encoder_left * WHEEL_RADIUS
        vel_right = s.encoder_right * WHEEL_RADIUS
        self.forward_vel = (vel_left + vel_right) / 2.0
        self.yaw_rate = (vel_right - vel_left) / WHEEL_SEPARATION

        self.fallen = abs(self.pitch) > math.radians(40)


class BalanceController:
    """
    Cascaded controller:
      - Inner loop (balance): Kp*pitch + Ki*integral + Kd*gyro_rate
      - Outer loop (velocity): PI with rate limiter
      - Yaw: proportional + damping for differential torque
    """
    def __init__(self, max_torque: float = 3.0):
        self.max_torque = max_torque

        # ── Balance loop (inner, fast) ──
        self.kp = 6.5
        self.ki = 0.3
        self.kd = 0.9
        self._pitch_integral = 0.0
        self._integral_decay = 0.998       # per-step decay (~1s time constant at 500Hz)
        self._integral_deadzone = 0.003    # rad (~0.17°) — don't integrate tiny errors

        # ── Velocity loop (outer, responsive) ──
        self.vel_kp = 0.20
        self.vel_ki = 0.03
        self._vel_integral = 0.0
        self._target_pitch = 0.0
        self._pitch_rate_limit = math.radians(12)  # 12 deg/s — snappy transitions

        # ── Position hold (encoder odometry, active only when stopped) ──
        self.pos_kp = 0.15
        self._position = 0.0       # integrated displacement from encoders (m)
        self._holding = False

        # ── Yaw ──
        self.yaw_kp = 0.06
        self.yaw_kd = 0.04

    def reset(self):
        self._pitch_integral = 0.0
        self._vel_integral = 0.0
        self._target_pitch = 0.0
        self._position = 0.0
        self._holding = False

    def compute(
        self,
        est: StateEstimator,
        target_velocity: float,
        target_yaw_rate: float,
        dt: float,
    ) -> tuple[float, float]:
        """Returns (left_torque, right_torque). Uses only estimated state."""

        # 1. Velocity loop → target pitch
        stopped = abs(target_velocity) < 0.01

        # Position hold: integrate displacement when stopped, reset when moving
        if stopped:
            if not self._holding:
                self._position = 0.0  # start tracking from here
                self._holding = True
            self._position += est.forward_vel * dt
            # Position correction feeds into velocity target
            effective_target = -self.pos_kp * self._position
        else:
            self._holding = False
            effective_target = target_velocity

        vel_error = effective_target - est.forward_vel
        self._vel_integral += vel_error * dt
        self._vel_integral = np.clip(self._vel_integral, -2.0, 2.0)

        # Decay velocity integral when stopped (prevents creep)
        if stopped and abs(est.forward_vel) < 0.03:
            self._vel_integral *= 0.99

        raw_target = self.vel_kp * vel_error + self.vel_ki * self._vel_integral
        max_pitch = math.radians(10)
        raw_target = float(np.clip(raw_target, -max_pitch, max_pitch))

        # Rate limiter
        delta = raw_target - self._target_pitch
        max_delta = self._pitch_rate_limit * dt
        self._target_pitch += float(np.clip(delta, -max_delta, max_delta))

        # 2. Balance loop: PI on pitch error + D from gyro
        pitch_error = est.pitch - self._target_pitch

        # Integral with dead zone and decay
        if abs(pitch_error) > self._integral_deadzone:
            self._pitch_integral += pitch_error * dt
        self._pitch_integral *= self._integral_decay
        max_i = self.max_torque / max(self.ki, 1e-9)
        self._pitch_integral = np.clip(self._pitch_integral, -max_i, max_i)

        base_torque = (
            self.kp * pitch_error
            + self.ki * self._pitch_integral
            + self.kd * est.pitch_rate
        )
        base_torque = float(np.clip(base_torque, -self.max_torque, self.max_torque))

        # 3. Yaw → differential torque (only when commanded)
        yaw_torque = 0.0
        if abs(target_yaw_rate) > 0.01:
            yaw_error = target_yaw_rate - est.yaw_rate
            yaw_torque = self.yaw_kp * yaw_error - self.yaw_kd * est.yaw_rate
            yaw_torque = float(np.clip(
                yaw_torque, -self.max_torque * 0.15, self.max_torque * 0.15
            ))

        left = float(np.clip(base_torque - yaw_torque, -self.max_torque, self.max_torque))
        right = float(np.clip(base_torque + yaw_torque, -self.max_torque, self.max_torque))
        return left, right
