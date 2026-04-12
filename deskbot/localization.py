"""
Indoor localization for DeskBot.

Two complementary systems:
  1. WiFi RSSI fingerprinting — room-level position (~1-3m accuracy)
     Simulates ESP32-S3 WiFi scanning against virtual access points.
  2. Wall-geometry heading correction — recalibrate heading from
     detected wall alignment vs known map angles.

Both use ONLY sensor data + pre-configured AP/map info. No simulator internals.
"""
import math
import numpy as np
from dataclasses import dataclass, field


# -- WiFi RSSI simulation parameters ------------------------------
# Log-distance path loss model: RSSI = TX_POWER - 10*n*log10(d/d0) + noise
TX_POWER_DBM = -30.0       # RSSI at 1m reference distance
PATH_LOSS_EXP = 3.0        # path loss exponent (2=free space, 3=indoor typical)
RSSI_NOISE_STD = 4.0       # dBm gaussian noise (realistic indoor WiFi)
RSSI_FLOOR = -90.0         # minimum detectable signal
SCAN_INTERVAL = 2.0        # seconds between WiFi scans (ESP32 limitation)

# Fingerprint grid resolution for matching
FP_GRID_RES = 0.5          # meters between fingerprint sample points


@dataclass
class AccessPoint:
    """A simulated WiFi access point."""
    ssid: str
    x: float           # world position
    y: float
    z: float = 2.0     # typical ceiling mount height
    tx_power: float = TX_POWER_DBM


@dataclass
class WiFiScan:
    """Result of a WiFi scan: dict of SSID -> RSSI (dBm)."""
    readings: dict = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class LocalizationEstimate:
    """Position estimate from WiFi + heading correction."""
    wifi_x: float = 0.0       # WiFi-estimated position (noisy, ~1-3m)
    wifi_y: float = 0.0
    wifi_confidence: float = 0.0  # 0..1 (higher = more APs visible)
    heading_correction: float = 0.0  # radians to add to DR heading
    heading_valid: bool = False


class WiFiLocalizer:
    """
    Simulates WiFi RSSI-based indoor localization.

    In simulation: computes RSSI from true distance to virtual APs.
    On real hardware: would read actual WiFi scan results from ESP32.

    Localization uses k-nearest-neighbor fingerprint matching:
    1. Calibration phase: build RSSI fingerprint map (grid of expected RSSI)
    2. Online: compare current scan to fingerprint map, find closest match.
    """

    def __init__(self, access_points: list[AccessPoint]):
        self.aps = access_points
        self._scan_timer = 0.0
        self._last_scan = WiFiScan()
        self._fingerprint_db = {}  # (grid_x, grid_y) -> {ssid: expected_rssi}

    def reset(self):
        self._scan_timer = 0.0
        self._last_scan = WiFiScan()

    def build_fingerprint_map(self, x_range: tuple[float, float],
                              y_range: tuple[float, float]):
        """
        Pre-compute expected RSSI at grid points (calibration phase).
        On a real robot, this would be built by driving around and recording.
        """
        self._fingerprint_db = {}
        xs = np.arange(x_range[0], x_range[1] + FP_GRID_RES, FP_GRID_RES)
        ys = np.arange(y_range[0], y_range[1] + FP_GRID_RES, FP_GRID_RES)

        for gx in xs:
            for gy in ys:
                fp = {}
                for ap in self.aps:
                    rssi = self._compute_rssi(gx, gy, ap, add_noise=False)
                    if rssi > RSSI_FLOOR:
                        fp[ap.ssid] = rssi
                if fp:
                    key = (round(gx / FP_GRID_RES), round(gy / FP_GRID_RES))
                    self._fingerprint_db[key] = fp

    def _compute_rssi(self, x: float, y: float, ap: AccessPoint,
                      add_noise: bool = True) -> float:
        """Log-distance path loss model."""
        dx = x - ap.x
        dy = y - ap.y
        dz = -ap.z  # robot is at ground level
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        dist = max(dist, 0.1)  # avoid log(0)

        rssi = ap.tx_power - 10.0 * PATH_LOSS_EXP * math.log10(dist)
        if add_noise:
            rssi += np.random.randn() * RSSI_NOISE_STD

        return max(rssi, RSSI_FLOOR)

    def scan(self, true_x: float, true_y: float, dt: float) -> WiFiScan | None:
        """
        Simulate a WiFi scan. Returns scan result every SCAN_INTERVAL seconds.

        In simulation, true_x/y come from dead reckoning (not ground truth).
        The RSSI is computed from true distance but with realistic noise,
        so the position estimate will be approximate.
        """
        self._scan_timer += dt
        if self._scan_timer < SCAN_INTERVAL:
            return None

        self._scan_timer = 0.0
        scan = WiFiScan()
        for ap in self.aps:
            rssi = self._compute_rssi(true_x, true_y, ap, add_noise=True)
            if rssi > RSSI_FLOOR:
                scan.readings[ap.ssid] = rssi

        self._last_scan = scan
        return scan

    def estimate_position(self, scan: WiFiScan) -> tuple[float, float, float]:
        """
        k-NN fingerprint matching: find grid position whose fingerprint
        is closest to the current scan (Euclidean distance in RSSI space).

        Returns (x, y, confidence). Confidence = 0..1 based on number of
        matching APs and quality of match.
        """
        if not scan.readings or not self._fingerprint_db:
            return 0.0, 0.0, 0.0

        best_dist = float('inf')
        best_pos = (0.0, 0.0)
        k_best = []  # top-k matches for weighted average

        for (gi, gj), fp in self._fingerprint_db.items():
            # Compute RSSI distance between scan and fingerprint
            common_ssids = set(scan.readings.keys()) & set(fp.keys())
            if len(common_ssids) < 2:
                continue

            sq_sum = 0.0
            for ssid in common_ssids:
                diff = scan.readings[ssid] - fp[ssid]
                sq_sum += diff * diff
            dist = math.sqrt(sq_sum / len(common_ssids))

            wx = gi * FP_GRID_RES
            wy = gj * FP_GRID_RES
            k_best.append((dist, wx, wy, len(common_ssids)))

        if not k_best:
            return 0.0, 0.0, 0.0

        # Weighted average of top-3 matches
        k_best.sort(key=lambda t: t[0])
        k = min(3, len(k_best))
        top_k = k_best[:k]

        # Inverse-distance weighting
        weights = [1.0 / max(d, 0.1) for d, _, _, _ in top_k]
        w_sum = sum(weights)

        est_x = sum(w * x for w, (_, x, _, _) in zip(weights, top_k)) / w_sum
        est_y = sum(w * y for w, (_, _, y, _) in zip(weights, top_k)) / w_sum

        # Confidence: based on match quality and number of APs
        best_match_dist = top_k[0][0]
        n_aps = top_k[0][3]
        confidence = min(1.0, n_aps / len(self.aps)) * max(0.0, 1.0 - best_match_dist / 20.0)

        return est_x, est_y, confidence


class HeadingCorrector:
    """
    Corrects heading drift using wall geometry from rangefinders.

    When the robot detects it's parallel to a wall (side sensors reading
    similar distances), it compares the detected wall angle to known
    wall orientations in the map. This gives a heading correction
    without any additional hardware.

    Known wall angles are configured for each scene (e.g., apartment
    walls are at 0, 90, 180, 270 degrees).
    """

    def __init__(self, known_wall_angles: list[float] = None):
        # Default: axis-aligned walls (typical indoor environment)
        if known_wall_angles is None:
            known_wall_angles = [0.0, math.pi / 2, math.pi, -math.pi / 2]
        self.wall_angles = known_wall_angles

        # Smoothed correction to avoid jumps
        self._correction = 0.0
        self._correction_alpha = 0.05  # slow update rate

    def reset(self):
        self._correction = 0.0

    def update(self, heading: float, rf_compensated: dict) -> tuple[float, bool]:
        """
        Attempt heading correction from wall geometry.

        Detects parallel walls from side sensor readings:
        - If SL reads a consistent distance, robot is parallel to a wall on the left.
        - The wall angle = heading +/- 90 degrees.
        - Compare to nearest known wall angle to get heading error.

        Returns (correction_radians, is_valid).
        """
        # Check side sensors for wall parallelism
        sl = rf_compensated.get("rf_SL", -1.0)
        sr = rf_compensated.get("rf_SR", -1.0)
        fl2 = rf_compensated.get("rf_FL2", -1.0)
        fr2 = rf_compensated.get("rf_FR2", -1.0)

        wall_detected = False
        detected_wall_angle = 0.0

        # Left wall: SL + FL2 readings indicate a wall on the left
        if sl > 0 and fl2 > 0:
            # If both readings are similar (within 30%), we're roughly parallel
            ratio = sl / fl2 if fl2 > 0.01 else 999.0
            if 0.6 < ratio < 1.6:
                # Wall is at heading + 90 degrees
                detected_wall_angle = heading + math.pi / 2
                wall_detected = True

        # Right wall (same logic)
        if not wall_detected and sr > 0 and fr2 > 0:
            ratio = sr / fr2 if fr2 > 0.01 else 999.0
            if 0.6 < ratio < 1.6:
                detected_wall_angle = heading - math.pi / 2
                wall_detected = True

        if not wall_detected:
            return self._correction, False

        # Normalize detected angle
        detected_wall_angle = math.atan2(math.sin(detected_wall_angle),
                                         math.cos(detected_wall_angle))

        # Find nearest known wall angle
        best_err = float('inf')
        best_correction = 0.0
        for known_angle in self.wall_angles:
            err = detected_wall_angle - known_angle
            err = math.atan2(math.sin(err), math.cos(err))
            if abs(err) < abs(best_err):
                best_err = err
                best_correction = -err  # correction = negative of error

        # Only apply if error is small enough to be plausible drift (not a corner)
        if abs(best_err) > math.radians(20):
            return self._correction, False

        # Smooth update
        self._correction = (self._correction_alpha * best_correction
                            + (1 - self._correction_alpha) * self._correction)

        return self._correction, True


# -- Apartment scene AP layout (pre-configured) --------------------
APARTMENT_APS = [
    AccessPoint("Livebox-5G", x=1.0, y=2.0, z=2.5),     # living room
    AccessPoint("Livebox-2G", x=1.0, y=2.0, z=2.5),      # same router, different band
    AccessPoint("Voisin-WiFi", x=-3.0, y=5.0, z=2.5),    # neighbor's AP (through wall)
    AccessPoint("Freebox-4K", x=4.0, y=-1.0, z=2.5),     # bedroom
]


def create_apartment_localizer() -> tuple[WiFiLocalizer, HeadingCorrector]:
    """Factory for apartment scene localization."""
    localizer = WiFiLocalizer(APARTMENT_APS)
    localizer.build_fingerprint_map(
        x_range=(-2.0, 6.0),
        y_range=(-2.0, 6.0),
    )
    heading_corrector = HeadingCorrector()  # axis-aligned walls
    return localizer, heading_corrector
