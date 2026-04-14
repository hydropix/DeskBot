"""
DeskBot robot constants and model paths.

Physical dimensions target a small desk robot (~25cm tall).
Coordinate system: X=forward, Y=left, Z=up.

The MJCF model is defined in models/deskbot.xml (robot only)
and models/scene.xml (robot + environment).
"""
import math
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
MODEL_XML = MODELS_DIR / "deskbot.xml"

SCENES = {
    "flat":      MODELS_DIR / "scene.xml",
    "skatepark": MODELS_DIR / "scene_skatepark.xml",
    "apartment": MODELS_DIR / "scene_apartment.xml",
    "nav_test":  MODELS_DIR / "scene_nav_test.xml",
}
DEFAULT_SCENE = "flat"

# Physical constants (must match deskbot.xml)
WHEEL_RADIUS = 0.10          # 10 cm (reduced from 12cm for sensor clearance)
WHEEL_WIDTH = 0.018          # 1.8 cm
WHEEL_SEPARATION = 0.16     # 16 cm axle-to-axle
WHEEL_MASS = 0.05            # 50 g each

MAX_TORQUE = 3.0             # Nm per motor

# Sensor pod geometry — kept in sync with deskbot.xml.
# Chassis free-body origin is at world Z=0.10; each rangefinder site is at
# local Z=0.112 → world Z=0.212 m above ground when the robot is upright.
SENSOR_POD_HEIGHT = 0.212            # metres above ground at rest
LASER_GROUND_TARGET = 2.0            # metres — horizontal distance where
                                     # every beam should intersect the floor


def ground_impact_pitch(sensor_height: float,
                        ground_distance: float = LASER_GROUND_TARGET) -> float:
    """
    Downward pitch angle (radians) so a laser mounted at ``sensor_height``
    above the ground hits the floor at ``ground_distance`` horizontal metres
    from its origin — independent of the beam's yaw direction.

    The geometry is a right triangle:
        vertical leg   = sensor_height      (above the floor)
        horizontal leg = ground_distance    (along the ray's XY projection)
        pitch          = atan2(sensor_height, ground_distance)

    A beam rotated by this pitch has a unit direction::

        zaxis = (cos(yaw) * cos(pitch),
                 sin(yaw) * cos(pitch),
                -sin(pitch))

    regardless of how it is aimed horizontally. Applied uniformly to every
    VL53L0X beam on the sensor pod: all beams reach the ground at the same
    range, giving the state estimator a consistent flat-ground expectation.
    """
    return math.atan2(sensor_height, ground_distance)
