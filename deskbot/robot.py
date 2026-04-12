"""
DeskBot robot constants and model paths.

Physical dimensions target a small desk robot (~25cm tall).
Coordinate system: X=forward, Y=left, Z=up.

The MJCF model is defined in models/deskbot.xml (robot only)
and models/scene.xml (robot + environment).
"""
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
