"""
Ground geometry model for rangefinder readings.

Given each laser's mounting (position + direction in the chassis body frame)
and the current body pitch from the IMU, computes the distance the ray
would travel on a perfectly flat floor. Any deviation between this
expectation and the measurement flags a hole (measured > expected) or an
obstacle (measured < expected).

Mounting geometry is extracted once at init from the compiled MuJoCo model
using a proper world → body frame change of basis — no assumption on the
initial pose. The pivot point (wheel axle) is read from the wheel joint,
not assumed to coincide with the chassis body origin. The axle height
above the ground is read from the tire geom radius.

The runtime API takes only pitch + measured distance; the sensor barrier
between the controller and simulator internals is preserved.

Known limitation: roll is ignored (the estimator does not expose it). This
is acceptable for a two-wheeled balancer where roll stays below ~2°.
"""
import math
from dataclasses import dataclass

import numpy as np
import mujoco

from deskbot.sensors import (
    RF_NAMES,
    RF_MAX_RANGE,
    RF_NOISE_SIGMA,
    RF_NOISE_PERCENT,
)


# ── Pitch uncertainty (1-sigma) used by classify() ──────────────────
# Derived from the complementary filter performance on a dynamic balancer:
#   - Accelerometer bias ~0.02 m/s² → ~0.002 rad steady-state angle error
#   - Gyro integration lag during active balancing adds ~0.005-0.01 rad
#   - Filter transient on impulses ~0.01 rad
# A 1-sigma of 0.02 rad (~1.1°) envelopes these sources for a well-tuned
# complementary filter in active balancing. Values above ~0.03 rad cause
# the propagated tolerance near the horizon to swamp real obstacles.
DEFAULT_SIGMA_PITCH = 0.02

# Relative tolerance cap — physical precision ceiling.
#
# Near the horizon (ray almost horizontal), ∂(expected)/∂pitch diverges:
# a 1° pitch change can move the floor-hit distance by a meter. In that
# regime the propagated tolerance becomes larger than the signal itself,
# and every obstacle inside the "uncertain band" would be suppressed as
# flat ground. The cap enforces that relative uncertainty can never
# exceed this fraction of the expected distance — beyond it, the sensor
# is considered ill-conditioned and classification falls back to the
# raw sensor noise only (more conservative toward flagging obstacles).
#
# 0.15 means we trust the geometry to ~15% relative accuracy.
MAX_RELATIVE_PITCH_TOL = 0.15

# Half-step used for the symmetric finite difference ∂(expected)/∂pitch.
# Must be small enough to be locally linear, large enough to dominate
# floating-point noise. 1 milliradian (~0.057°) is a good compromise.
_PITCH_DERIV_STEP = 1e-3


@dataclass(frozen=True)
class GroundReading:
    name: str
    measured: float       # raw sensor reading (m); -1.0 means no hit within range
    expected: float       # flat-ground expectation (m); +inf if ray never hits ground
    deviation: float      # measured - expected (signed); NaN if either side unknown
    tolerance: float      # total 1-sigma tolerance used for classification (m)
    kind: str             # "flat" | "obstacle" | "hole" | "no_reading" | "no_ground_expected"


def _rotation_y(theta: float) -> np.ndarray:
    """Right-handed rotation matrix around the body-frame Y axis (pitch)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [c,  0.0, s ],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c ],
    ])


class GroundGeometry:
    """
    Precomputes each laser's mounting in the chassis body frame (relative
    to the wheel axle, the pivot point of the pitch rotation) and, on
    demand, computes the distance each ray would measure on a perfectly
    flat floor at z=0.

    Model assumptions:
      - Body pitches around the wheel axle.
      - Wheels remain in contact with the ground; axle height above the
        floor is constant and equal to the tire radius.
      - Roll is negligible; ground is the horizontal plane z=0.

    The pivot point (`axle_in_body`) and axle height (`axle_height`) are
    read from the compiled model — no hardcoded values, no assumption on
    the initial qpos or chassis orientation.
    """

    def __init__(self, model: mujoco.MjModel):
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)

        chassis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        chassis_pos_world = data.xpos[chassis_id].copy()
        R_chassis_world = data.xmat[chassis_id].reshape(3, 3).copy()

        # ── Axle in chassis body frame ──
        # Read from the wheel joint anchor. We take the average of both
        # wheel joints in case the MJCF is asymmetric. Each wheel body's
        # position is stored in chassis frame (body_pos), and the joint
        # anchor (jnt_pos) is in the child body's local frame.
        axle_local_points = []
        for wheel_name in ("wheel_L", "wheel_R"):
            wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, wheel_name)
            if wheel_id < 0:
                continue
            wheel_pos_in_chassis = model.body_pos[wheel_id].copy()
            jnt_adr = model.body_jntadr[wheel_id]
            if jnt_adr >= 0:
                jnt_pos_in_child = model.jnt_pos[jnt_adr].copy()
            else:
                jnt_pos_in_child = np.zeros(3)
            axle_local_points.append(wheel_pos_in_chassis + jnt_pos_in_child)

        if not axle_local_points:
            raise RuntimeError("GroundGeometry: no wheel_L/wheel_R bodies found")
        self._axle_in_body = np.mean(axle_local_points, axis=0)

        # ── Axle height above ground ──
        # Read from the tire geom radius. Cylinder with euler="1.5708 0 0"
        # has its symmetry axis along Y, so geom_size[0] is the radius.
        tire_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tire_L")
        if tire_id < 0:
            raise RuntimeError("GroundGeometry: geom 'tire_L' not found")
        self._axle_height = float(model.geom_size[tire_id][0])

        # ── Per-sensor mounting in body frame ──
        # Proper change of basis: for any world-frame vector v,
        #   v_body = R_chassis^T · (v_world - chassis_pos_world)
        # This works regardless of the chassis orientation at init time.
        self._mounts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in RF_NAMES:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site_id < 0:
                continue
            site_pos_world = data.site_xpos[site_id]
            site_rot_world = data.site_xmat[site_id].reshape(3, 3)

            # Sensor position relative to the axle, in chassis body frame.
            p_body = R_chassis_world.T @ (site_pos_world - chassis_pos_world) \
                     - self._axle_in_body
            # Ray direction (site local +Z) in chassis body frame.
            R_site_in_body = R_chassis_world.T @ site_rot_world
            d_body = R_site_in_body[:, 2].copy()
            # Normalize to guard against numerical drift in the MJCF.
            d_body /= np.linalg.norm(d_body)

            self._mounts[name] = (p_body.copy(), d_body)

    @property
    def sensors(self) -> list[str]:
        return list(self._mounts.keys())

    @property
    def axle_height(self) -> float:
        return self._axle_height

    def expected_distance(self, name: str, pitch: float) -> float:
        """
        Distance the sensor should read on perfectly flat ground at z=0.

        Returns math.inf when the rotated ray does not intersect the
        ground within RF_MAX_RANGE (points horizontally, upward, or to a
        point beyond the sensor's reach).
        """
        mount = self._mounts.get(name)
        if mount is None:
            return math.inf
        p_body, d_body = mount

        # Rotate the body-frame mounting around Y by pitch theta.
        Ry = _rotation_y(pitch)
        p_rot = Ry @ p_body   # sensor position relative to axle, in world frame
        d_rot = Ry @ d_body   # unit ray direction in world frame

        sensor_z = self._axle_height + p_rot[2]
        ray_dz = d_rot[2]

        # Ray must point strictly downward and start above the floor.
        if ray_dz >= -1e-9 or sensor_z <= 0.0:
            return math.inf

        distance = -sensor_z / ray_dz
        if distance > RF_MAX_RANGE:
            return math.inf
        return distance

    def classify(
        self,
        name: str,
        measured: float,
        pitch: float,
        sigma_pitch: float = DEFAULT_SIGMA_PITCH,
        extra_tol: float = 0.0,
    ) -> GroundReading:
        """
        Compare a measured distance with the flat-ground expectation.

        The tolerance combines three independent sources (added as 1-sigma
        Gaussian, assuming independence):
          1. Rangefinder noise: RF_NOISE_SIGMA + RF_NOISE_PERCENT * expected
          2. Pitch estimation uncertainty, propagated through the expected
             distance via a symmetric finite difference:
                 sigma_d_from_pitch = |d(expected)/d(pitch)| * sigma_pitch
          3. Optional user-supplied margin (safety factor).

        At the boundary where the ray is nearly horizontal, a small pitch
        perturbation can flip the expected distance between finite and
        infinite. In that regime we cannot meaningfully classify the
        reading, so the tolerance is widened to `expected` itself — which
        effectively disables the flat/obstacle/hole distinction.

        Kinds:
          flat               — within tolerance of expectation
          obstacle           — closer than expected by more than tolerance
          hole               — farther than expected (or no hit when one was expected)
          no_reading         — sensor returned -1 and no ground hit expected
          no_ground_expected — ray never reaches the floor; any hit is an obstacle
        """
        expected = self.expected_distance(name, pitch)
        ground_expected = not math.isinf(expected)

        if measured < 0.0:
            if ground_expected:
                # Ray should have hit the ground but didn't → hole (or missing return).
                return GroundReading(
                    name, measured, expected, math.inf, float("nan"), "hole"
                )
            return GroundReading(
                name, measured, expected, float("nan"), float("nan"), "no_reading"
            )

        if not ground_expected:
            # Ray was never going to see the floor; any hit is an obstacle.
            return GroundReading(
                name, measured, expected, float("nan"), float("nan"), "obstacle"
            )

        # ── Propagate pitch uncertainty into the tolerance ──
        # Symmetric finite difference for ∂(expected)/∂pitch.
        d_plus = self.expected_distance(name, pitch + _PITCH_DERIV_STEP)
        d_minus = self.expected_distance(name, pitch - _PITCH_DERIV_STEP)

        if math.isinf(d_plus) or math.isinf(d_minus):
            # Near the horizon: derivative undefined, geometry ill-
            # conditioned. Cap pitch_tol at its physical ceiling.
            pitch_tol_raw = math.inf
        else:
            d_expected_d_pitch = (d_plus - d_minus) / (2.0 * _PITCH_DERIV_STEP)
            pitch_tol_raw = abs(d_expected_d_pitch) * sigma_pitch

        # Cap the pitch contribution at a fixed fraction of `expected`.
        # This enforces a minimum relative accuracy for classification —
        # if the linearized propagation would exceed it, we trust the
        # geometry only to that level and rely on sensor noise for the
        # rest. Prevents the uncertainty band from swallowing real
        # obstacles in the ill-conditioned near-horizon regime.
        pitch_tol = min(pitch_tol_raw, MAX_RELATIVE_PITCH_TOL * expected)

        sensor_tol = RF_NOISE_SIGMA + RF_NOISE_PERCENT * expected
        # Quadrature sum of independent Gaussian sources + user margin.
        tol = math.sqrt(sensor_tol * sensor_tol + pitch_tol * pitch_tol) + extra_tol

        deviation = measured - expected
        if deviation > tol:
            kind = "hole"
        elif deviation < -tol:
            kind = "obstacle"
        else:
            kind = "flat"
        return GroundReading(name, measured, expected, deviation, tol, kind)

    def horizontal_distance(self, name: str, pitch: float, measured: float) -> float:
        """
        Project a measurement onto the horizontal plane, measured from the
        robot pivot axis (wheel axle projected onto Z=0), not from the
        sensor itself.

        Navigation code reasons in 2D around the robot pivot: the same hit
        seen by two different sensors mounted at different offsets must
        yield the same planar distance. Returning a sensor-local value
        creates a systematic offset of several centimeters (the sensor
        pod extends 2-5 cm ahead and above the axle).

        Returns -1.0 for invalid/missing measurements.
        """
        if measured < 0.0:
            return -1.0
        mount = self._mounts.get(name)
        if mount is None:
            return measured
        p_body, d_body = mount

        Ry = _rotation_y(pitch)
        p_rot = Ry @ p_body
        d_rot = Ry @ d_body
        # Hit point expressed in world frame, relative to the axle (which
        # sits at XY=(0,0) in the robot's local navigation frame).
        hit_rel = p_rot + measured * d_rot
        return float(math.hypot(hit_rel[0], hit_rel[1]))

    def classify_all(
        self,
        rangefinders: dict[str, float],
        pitch: float,
        sigma_pitch: float = DEFAULT_SIGMA_PITCH,
        extra_tol: float = 0.0,
    ) -> dict[str, GroundReading]:
        """Classify every known rangefinder reading in one call."""
        return {
            name: self.classify(
                name, rangefinders.get(name, -1.0), pitch, sigma_pitch, extra_tol
            )
            for name in self._mounts
        }
