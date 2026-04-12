"""
Balance controller for the two-wheeled DeskBot.

Uses ONLY realistic sensor data (SensorReadings):
  - IMU accelerometer + gyroscope → pitch estimation via complementary filter
  - Wheel encoders → forward velocity & yaw rate

This code is designed to be portable to a real microcontroller.
The LQR gains are computed once at startup from the physical model,
then the control loop uses only estimated state (sensor barrier preserved).
"""
import math
import numpy as np
from scipy.linalg import solve_continuous_are

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
        self.yaw_rate = 0.0        # from encoder differential — used by LQR/PID
        self.yaw_rate_gyro = 0.0   # from gyro_z — used by navigator DR
        self.forward_vel = 0.0
        self.fallen = False
        self._initialized = False

    def reset(self):
        self.pitch = 0.0
        self.pitch_rate = 0.0
        self.yaw_rate = 0.0
        self.yaw_rate_gyro = 0.0
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
        self.forward_vel = (vel_left + vel_right) / 2.0 + self.pitch_rate * WHEEL_RADIUS

        # Two yaw-rate estimates, each used in a different place:
        #
        #  - self.yaw_rate (encoder differential): used as feedback by
        #    the balance controller (LQR Q/R was tuned against this). It
        #    represents the *commanded* wheel rotation and is what the
        #    controller's linearized model assumes. Do NOT change the
        #    source here without re-tuning the LQR gains — a mismatch
        #    between the feedback noise character and Q/R makes the
        #    robot fall within a few seconds (verified empirically).
        #
        #  - self.yaw_rate_gyro (raw gyro_z): used by Navigator for
        #    dead-reckoning heading integration. Gyro measures chassis
        #    rotation directly and is immune to the wheel slip that
        #    corrupts encoder-based yaw during contour maneuvers.
        #    Measured on 7 random seeds (diag_yaw_sources.py):
        #        encoder yaw integration error = 13.5° mean
        #        gyro    yaw integration error =  8.1° mean
        #    Gyro has residual bias drift (~0.5°/s) but no slip error.
        self.yaw_rate = (vel_right - vel_left) / WHEEL_SEPARATION
        self.yaw_rate_gyro = s.gyro[2]

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
        self._pitch_rate_limit = math.radians(40)  # 40 deg/s — fast response

        # ── Position hold (encoder odometry, active only when stopped) ──
        self.pos_kp = 0.15
        self._position = 0.0       # integrated displacement from encoders (m)
        self._holding = False

        # ── Yaw (increased for navigation responsiveness) ──
        self.yaw_kp = 0.18
        self.yaw_kd = 0.08

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
        max_pitch = math.radians(25)
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
                yaw_torque, -self.max_torque * 0.25, self.max_torque * 0.25
            ))

        left = float(np.clip(base_torque - yaw_torque, -self.max_torque, self.max_torque))
        right = float(np.clip(base_torque + yaw_torque, -self.max_torque, self.max_torque))
        return left, right


class LQRController:
    """
    Optimal LQR controller for the self-balancing robot.

    Replaces the cascaded PID with a single matrix multiply per step.
    State vector: x = [pitch, pitch_rate, velocity, position, yaw_error, yaw_rate]
    Control:      u = [torque_avg, torque_diff]

    The linearized dynamics are derived from the Euler-Lagrange equations
    of a 2D inverted pendulum on wheels, with a separate yaw subsystem.

    Physical parameters are extracted from the MuJoCo compiled model once
    at construction time (offline design). The control loop uses only
    estimated state from sensors (sensor barrier preserved).
    """

    def __init__(self, mj_model=None, max_torque: float = 3.0):
        self.max_torque = max_torque
        self._position = 0.0
        self._holding = False
        # Low-pass filter on pitch_rate to attenuate gyro noise
        # (~0.11 rad/s per sample at 500 Hz)
        self._filtered_pitch_rate = 0.0
        self._pitch_rate_alpha = 0.8  # smoothing factor (higher = more filtering)

        if mj_model is not None:
            self._compute_gains_from_model(mj_model)
        else:
            self._use_default_gains()

    def _compute_gains_from_model(self, model):
        """Extract physical parameters from compiled MuJoCo model and solve CARE."""
        import mujoco

        g = 9.81
        R = WHEEL_RADIUS
        L = WHEEL_SEPARATION

        # ── Extract body parameters from compiled model ──
        chassis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        wheel_l_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wheel_L")

        # Total body mass (chassis = everything except wheels)
        mb = float(model.body_mass[chassis_id])
        # Wheel mass (per wheel)
        mw = float(model.body_mass[wheel_l_id])

        # CoM height above axle (chassis body frame origin = axle)
        # model.body_ipos gives CoM relative to body frame
        l = float(model.body_ipos[chassis_id][2])  # Z component
        if l < 0.01:
            l = 0.038  # fallback: computed from MJCF geom positions

        # Body moment of inertia about pitch axis (Y-axis in body frame)
        # model.body_inertia gives diagonal [Ixx, Iyy, Izz] at CoM
        Ib_com = float(model.body_inertia[chassis_id][1])  # Iyy
        # Parallel axis theorem: Ib about axle = Ib_com + mb * l²
        Ib = Ib_com + mb * l ** 2

        # Wheel MOI about rotation axis (Y-axis, hinge axis)
        # Cylinder: I = 0.5 * m * r²
        Iw = 0.5 * mw * R ** 2

        # Yaw MOI: Izz of chassis + wheel contributions
        Iz_com = float(model.body_inertia[chassis_id][2])  # Izz
        Iz = Iz_com + 2 * mw * (L / 2) ** 2  # + wheels at distance L/2

        self._build_lqr(mb, mw, l, Ib, Iw, Iz, R, L, g)

    def _use_default_gains(self):
        """Fallback gains computed from nominal DeskBot parameters."""
        g = 9.81
        R = WHEEL_RADIUS
        L = WHEEL_SEPARATION

        mb = 0.602   # kg (sum of all chassis geom masses)
        mw = 0.05    # kg per wheel
        l = 0.038    # m (CoM above axle)
        Ib_com = 0.0015  # approximate Iyy at CoM
        Ib = Ib_com + mb * l ** 2
        Iw = 0.5 * mw * R ** 2
        Iz = 0.002 + 2 * mw * (L / 2) ** 2

        self._build_lqr(mb, mw, l, Ib, Iw, Iz, R, L, g)

    def _build_lqr(self, mb, mw, l, Ib, Iw, Iz, R, L, g):
        """Build linearized state-space model and solve the Riccati equation."""

        # ── Balance subsystem (pitch, pitch_rate, velocity, position) ──
        # Euler-Lagrange for inverted pendulum on wheels:
        #   a*θ̈ + b*φ̈ = d*θ - τ     (body)
        #   b*θ̈ + c*φ̈ = τ            (wheels)
        # where φ = average wheel angle, τ = average torque
        a = Ib                                      # body inertia about axle
        b = mb * l * R                              # coupling
        c = (mb + 2 * mw) * R ** 2 + 2 * Iw        # wheel effective inertia
        d = mb * g * l                              # gravity torque

        det = a * c - b ** 2  # determinant of mass matrix

        # Solve for θ̈ and v̇ (v = R*φ̇):
        # θ̈ = (c*d/det)*θ - ((c+b)/det)*τ
        # v̇ = (-R*b*d/det)*θ + (R*(a+b)/det)*τ

        # State: x_bal = [θ, θ̇, v, p]
        A_bal = np.zeros((4, 4))
        A_bal[0, 1] = 1.0                    # θ̇
        A_bal[1, 0] = c * d / det            # θ̈ from θ
        A_bal[2, 0] = -R * b * d / det       # v̇ from θ
        A_bal[3, 2] = 1.0                    # ṗ = v

        B_bal = np.zeros((4, 1))
        B_bal[1, 0] = -(c + b) / det         # θ̈ from τ
        B_bal[2, 0] = R * (a + b) / det      # v̇ from τ

        # ── Yaw subsystem (yaw_error, yaw_rate) ──
        # τ_diff = (right - left) / 2
        # ψ̈ = τ_diff * L / (R * Iz_eff)
        Iz_eff = Iz + 2 * Iw * (L / (2 * R)) ** 2
        yaw_gain = L / (R * Iz_eff)

        A_yaw = np.array([[0.0, 1.0],
                          [0.0, 0.0]])
        B_yaw = np.array([[0.0],
                          [yaw_gain]])

        # ── Combined 6-state system ──
        # x = [pitch, pitch_rate, velocity, position, yaw_error, yaw_rate]
        # u = [τ_avg, τ_diff]
        A = np.zeros((6, 6))
        A[:4, :4] = A_bal
        A[4:, 4:] = A_yaw

        B = np.zeros((6, 2))
        B[:4, 0:1] = B_bal
        B[4:, 1:2] = B_yaw

        # ── Cost matrices (Q penalizes state, R penalizes effort) ──
        # Tuned for robustness to sensor noise:
        #   - Gyro noise ~0.11 rad/s per sample → keep pitch_rate Q low
        #   - R_avg high enough to prevent noise amplification
        #   - Small robot inertia (det ~1e-5) means B elements are huge (~780)
        #     so gains must be conservative to avoid noise-driven oscillation
        Q = np.diag([
            80.0,    # pitch — keep upright
            0.5,     # pitch_rate — LOW: gyro noise ~0.11 rad/s per sample
            4.0,     # velocity — track speed commands
            2.0,     # position — hold position when stopped
            5.0,     # yaw_error — track heading
            0.3,     # yaw_rate — smooth turns
        ])
        R_cost = np.diag([
            8.0,     # τ_avg — HIGH: penalize effort to reduce noise sensitivity
            10.0,    # τ_diff — yaw torque (even more conservative)
        ])

        # ── Solve continuous algebraic Riccati equation ──
        P = solve_continuous_are(A, B, Q, R_cost)
        self.K = np.linalg.inv(R_cost) @ B.T @ P

        # Store for diagnostics
        self._A = A
        self._B = B
        self._Q = Q
        self._R_cost = R_cost
        self._params = {
            "mb": mb, "mw": mw, "l": l, "Ib": Ib, "Iw": Iw, "Iz": Iz,
            "det": det, "yaw_gain": yaw_gain,
        }

    def reset(self):
        self._position = 0.0
        self._holding = False
        self._filtered_pitch_rate = 0.0

    def compute(
        self,
        est: 'StateEstimator',
        target_velocity: float,
        target_yaw_rate: float,
        dt: float,
    ) -> tuple[float, float]:
        """
        Returns (left_torque, right_torque). Uses only estimated state.

        u = -K @ x  where x is the error state relative to setpoint.
        Then: left = u_avg - u_diff, right = u_avg + u_diff.
        """
        # Position hold: integrate displacement when stopped
        stopped = abs(target_velocity) < 0.01
        if stopped:
            if not self._holding:
                self._position = 0.0
                self._holding = True
            self._position += est.forward_vel * dt
        else:
            self._holding = False
            self._position = 0.0

        # Low-pass filter on pitch_rate to reduce gyro noise amplification
        a = self._pitch_rate_alpha
        self._filtered_pitch_rate = a * self._filtered_pitch_rate + (1 - a) * est.pitch_rate

        # State error vector (deviation from desired equilibrium)
        x = np.array([
            est.pitch,                             # pitch error (want 0)
            self._filtered_pitch_rate,             # filtered pitch rate (want 0)
            est.forward_vel - target_velocity,     # velocity error
            self._position,                        # position error (when holding)
            0.0,                                   # yaw error (handled below)
            est.yaw_rate - target_yaw_rate,        # yaw rate error
        ])

        # Compute optimal control: u = -K @ x
        u = -self.K @ x

        u_avg = float(np.clip(u[0], -self.max_torque, self.max_torque))
        u_diff = float(np.clip(u[1], -self.max_torque * 0.4, self.max_torque * 0.4))

        left = float(np.clip(u_avg - u_diff, -self.max_torque, self.max_torque))
        right = float(np.clip(u_avg + u_diff, -self.max_torque, self.max_torque))
        return left, right
