"""
Occupancy-grid visualization for DeskBot.

Renders a top-down egocentric view of:
  - the log-odds occupancy grid (Navigator.grid)
  - the robot pose (triangle + heading arrow)
  - the rangefinder rays, color-coded by compensated distance
  - the target heading (green dashed line)
  - a debug overlay of ground-truth obstacles (clearly marked cyan)

Two outputs:
  - `render_rgb(frame)` → numpy HxWx3 uint8 array, ready for display or PNG
  - `save_png(path, rgb)` → pure-stdlib PNG writer (zlib only, no Pillow)
  - `MapWindow` → tkinter Toplevel displaying the frame live via PPM/PhotoImage

Display orientation: world X → image right, world Y → image up
(standard map convention, matches the 3D viewer seen from above).
Image is centered on the robot.

This module is ONLY for visualization/debugging. It may read the MuJoCo
model to extract ground-truth obstacle positions for overlay — that is a
deliberate, clearly-flagged break of the sensor barrier for debug purposes
and must never flow back into control code.
"""
import math
import os
import struct
import time
import zlib
from dataclasses import dataclass, field

import numpy as np

from deskbot.navigation import (
    GRID_SIZE, GRID_RES,
    LOG_ODD_OCCUPIED_THRESHOLD, LOG_ODD_MAX, LOG_ODD_MIN,
    RF_BODY_ANGLES,
    SAFE_DIST, CAUTION_DIST,
)


# ─────────────────────────────────────────────────────────────────────
# Rendering constants
# ─────────────────────────────────────────────────────────────────────

PX_PER_CELL = 10
MAP_PX = GRID_SIZE * PX_PER_CELL        # 600
PX_PER_M = PX_PER_CELL / GRID_RES       # 125 px/m (4.8 m visible)

# Color palette (RGB uint8)
COL_BG          = (20,  22,  28)
COL_UNKNOWN     = (70,  72,  80)
COL_FREE        = (160, 180, 195)
COL_OCCUPIED    = (230, 70,  70)
COL_ROBOT_FILL  = (90,  180, 255)
COL_ROBOT_EDGE  = (230, 240, 255)
COL_HEADING     = (120, 255, 140)
COL_TARGET      = (120, 255, 140)
COL_GT_OBST     = (0,   235, 220)     # cyan = ground truth debug
COL_AXIS        = (50,  55,  65)
COL_SAFE_RING   = (250, 120, 60)
COL_CAUTION_RING= (200, 200, 90)


# ─────────────────────────────────────────────────────────────────────
# Frame dataclass
# ─────────────────────────────────────────────────────────────────────

@dataclass
class MapFrame:
    """All data needed to render one map frame. Snapshot of state."""
    grid: np.ndarray            # log-odds, shape (GRID_SIZE, GRID_SIZE)
    grid_cx: float              # world-coord of grid center, x (meters)
    grid_cy: float              # world-coord of grid center, y (meters)
    robot_x: float              # dead-reckoned robot pos x (meters)
    robot_y: float              # dead-reckoned robot pos y (meters)
    heading: float              # dead-reckoned heading (rad)
    target_heading: float       # desired heading (rad)
    nav_active: bool
    fsm_state: str
    rf_compensated: dict        # {name: horizontal distance or -1}
    gt_obstacles: list = field(default_factory=list)
    # gt_obstacles: list of {"type","x","y",...} — see scripts/benchmark_random.py


# ─────────────────────────────────────────────────────────────────────
# Coordinate transform: world → image pixels
# ─────────────────────────────────────────────────────────────────────

def world_to_px(wx: float, wy: float, robot_x: float, robot_y: float) -> tuple[int, int]:
    """
    Map world coordinates (meters) to image pixel (col, row).

    Image is centered on the robot. World +X → right, world +Y → up.
    The y-axis is flipped because image rows increase downward.
    """
    col = int(round(MAP_PX / 2 + (wx - robot_x) * PX_PER_M))
    row = int(round(MAP_PX / 2 - (wy - robot_y) * PX_PER_M))
    return col, row


# ─────────────────────────────────────────────────────────────────────
# Primitive drawing (numpy, no dependencies)
# ─────────────────────────────────────────────────────────────────────

def _in_bounds(col: int, row: int, w: int, h: int) -> bool:
    return 0 <= col < w and 0 <= row < h


def draw_pixel(img: np.ndarray, col: int, row: int, color: tuple):
    h, w = img.shape[:2]
    if _in_bounds(col, row, w, h):
        img[row, col] = color


def draw_line(img: np.ndarray, c0: int, r0: int, c1: int, r1: int,
              color: tuple, width: int = 1):
    """Bresenham line with optional thickness (perpendicular brush)."""
    dc = abs(c1 - c0)
    dr = abs(r1 - r0)
    sc = 1 if c0 < c1 else -1
    sr = 1 if r0 < r1 else -1
    err = dc - dr
    c, r = c0, r0
    half = (width - 1) // 2
    while True:
        for oc in range(-half, half + 1):
            for orr in range(-half, half + 1):
                draw_pixel(img, c + oc, r + orr, color)
        if c == c1 and r == r1:
            return
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr


def draw_circle_filled(img: np.ndarray, cc: int, rc: int, radius: int,
                       color: tuple):
    h, w = img.shape[:2]
    r2 = radius * radius
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dc * dc + dr * dr <= r2:
                draw_pixel(img, cc + dc, rc + dr, color)


def draw_circle_outline(img: np.ndarray, cc: int, rc: int, radius: int,
                        color: tuple):
    """Midpoint circle algorithm."""
    x, y = radius, 0
    err = 0
    while x >= y:
        for sx, sy in ((x, y), (y, x), (-y, x), (-x, y),
                       (-x, -y), (-y, -x), (y, -x), (x, -y)):
            draw_pixel(img, cc + sx, rc + sy, color)
        y += 1
        if err <= 0:
            err += 2 * y + 1
        else:
            x -= 1
            err += 2 * (y - x) + 1


def draw_rect_outline(img: np.ndarray, c0: int, r0: int, c1: int, r1: int,
                      color: tuple):
    draw_line(img, c0, r0, c1, r0, color)
    draw_line(img, c1, r0, c1, r1, color)
    draw_line(img, c1, r1, c0, r1, color)
    draw_line(img, c0, r1, c0, r0, color)


def draw_triangle_filled(img: np.ndarray, points: list, color: tuple):
    """Fill triangle by scanlines. Points are (col, row) tuples."""
    pts = sorted(points, key=lambda p: p[1])
    (x0, y0), (x1, y1), (x2, y2) = pts

    def edge(ya, yb, xa, xb, y):
        if yb == ya:
            return xa
        return xa + (xb - xa) * (y - ya) / (yb - ya)

    h, w = img.shape[:2]
    for y in range(max(0, y0), min(h, y2 + 1)):
        if y < y1:
            xa = edge(y0, y1, x0, x1, y)
            xb = edge(y0, y2, x0, x2, y)
        else:
            xa = edge(y1, y2, x1, x2, y)
            xb = edge(y0, y2, x0, x2, y)
        xs = int(round(min(xa, xb)))
        xe = int(round(max(xa, xb)))
        for x in range(max(0, xs), min(w, xe + 1)):
            img[y, x] = color


# ─────────────────────────────────────────────────────────────────────
# Grid → image
# ─────────────────────────────────────────────────────────────────────

def _lerp_color(c0: tuple, c1: tuple, t: float) -> tuple:
    t = max(0.0, min(1.0, t))
    return (int(c0[0] + (c1[0] - c0[0]) * t),
            int(c0[1] + (c1[1] - c0[1]) * t),
            int(c0[2] + (c1[2] - c0[2]) * t))


def rasterize_grid(img: np.ndarray, frame: MapFrame):
    """
    Paint the log-odds grid onto `img`, centered on the robot.

    Grid cell (ci, cj) has world center:
        wx = grid_cx + (ci - GRID_SIZE/2 + 0.5) * GRID_RES
        wy = grid_cy + (cj - GRID_SIZE/2 + 0.5) * GRID_RES
    which we then convert to pixels via world_to_px().
    """
    g = frame.grid
    for ci in range(GRID_SIZE):
        wx = frame.grid_cx + (ci - GRID_SIZE / 2 + 0.5) * GRID_RES
        for cj in range(GRID_SIZE):
            wy = frame.grid_cy + (cj - GRID_SIZE / 2 + 0.5) * GRID_RES
            col, row = world_to_px(wx, wy, frame.robot_x, frame.robot_y)
            if not (0 <= col < MAP_PX and 0 <= row < MAP_PX):
                continue
            lv = float(g[ci, cj])

            if lv > LOG_ODD_OCCUPIED_THRESHOLD:
                # Occupied: map [0.5 … LOG_ODD_MAX] → [unknown, bright red]
                t = (lv - LOG_ODD_OCCUPIED_THRESHOLD) / (LOG_ODD_MAX - LOG_ODD_OCCUPIED_THRESHOLD + 1e-9)
                color = _lerp_color(COL_UNKNOWN, COL_OCCUPIED, t)
            elif lv < -0.1:
                # Free: [LOG_ODD_MIN, -0.1] → [bright free, unknown]
                t = 1.0 - (lv - LOG_ODD_MIN) / (-0.1 - LOG_ODD_MIN + 1e-9)
                color = _lerp_color(COL_FREE, COL_UNKNOWN, t)
            else:
                color = COL_UNKNOWN

            # Fill the cell's pixel block
            c0 = col - PX_PER_CELL // 2
            r0 = row - PX_PER_CELL // 2
            c1 = c0 + PX_PER_CELL
            r1 = r0 + PX_PER_CELL
            c0 = max(0, c0); r0 = max(0, r0)
            c1 = min(MAP_PX, c1); r1 = min(MAP_PX, r1)
            img[r0:r1, c0:c1] = color


# ─────────────────────────────────────────────────────────────────────
# Overlays
# ─────────────────────────────────────────────────────────────────────

def draw_axes(img: np.ndarray):
    """Thin crosshair at center + 1 m range rings."""
    cc, rc = MAP_PX // 2, MAP_PX // 2
    # Cross
    draw_line(img, 0, rc, MAP_PX - 1, rc, COL_AXIS)
    draw_line(img, cc, 0, cc, MAP_PX - 1, COL_AXIS)
    # Range rings (every 1 m)
    for d_m in (1.0, 2.0):
        r_px = int(round(d_m * PX_PER_M))
        draw_circle_outline(img, cc, rc, r_px, COL_AXIS)


def draw_safety_rings(img: np.ndarray):
    """SAFE_DIST (hard) and CAUTION_DIST (soft) visualization at robot."""
    cc, rc = MAP_PX // 2, MAP_PX // 2
    draw_circle_outline(img, cc, rc, int(round(SAFE_DIST * PX_PER_M)), COL_SAFE_RING)
    draw_circle_outline(img, cc, rc, int(round(CAUTION_DIST * PX_PER_M)), COL_CAUTION_RING)


def draw_robot(img: np.ndarray, frame: MapFrame):
    """Filled triangle pointing along the robot's body X axis."""
    cc, rc = MAP_PX // 2, MAP_PX // 2
    h = frame.heading
    cos_h, sin_h = math.cos(h), math.sin(h)

    # Triangle tip 15px forward, base 10px wide perpendicular
    tip = (cc + int(round(15 * cos_h)), rc - int(round(15 * sin_h)))
    left = (cc + int(round(-6 * cos_h - 8 * sin_h)),
            rc - int(round(-6 * sin_h + 8 * cos_h)))
    right = (cc + int(round(-6 * cos_h + 8 * sin_h)),
             rc - int(round(-6 * sin_h - 8 * cos_h)))
    draw_triangle_filled(img, [tip, left, right], COL_ROBOT_FILL)
    # Outline
    draw_line(img, tip[0], tip[1], left[0], left[1], COL_ROBOT_EDGE)
    draw_line(img, left[0], left[1], right[0], right[1], COL_ROBOT_EDGE)
    draw_line(img, right[0], right[1], tip[0], tip[1], COL_ROBOT_EDGE)


def draw_rangefinder_rays(img: np.ndarray, frame: MapFrame):
    """
    Draw each ray as a line from the robot along its world direction, to
    the hit distance (or to grid edge if no hit). Color-coded by distance.
    """
    robot_c, robot_r = MAP_PX // 2, MAP_PX // 2
    for name, body_angle in RF_BODY_ANGLES.items():
        d = frame.rf_compensated.get(name, -1.0)
        world_angle = frame.heading + body_angle
        if d > 0:
            wx = frame.robot_x + d * math.cos(world_angle)
            wy = frame.robot_y + d * math.sin(world_angle)
            color = _distance_color(d)
        else:
            wx = frame.robot_x + 2.0 * math.cos(world_angle)
            wy = frame.robot_y + 2.0 * math.sin(world_angle)
            color = (60, 60, 70)
        end_c, end_r = world_to_px(wx, wy, frame.robot_x, frame.robot_y)
        draw_line(img, robot_c, robot_r, end_c, end_r, color)
        if d > 0:
            draw_circle_filled(img, end_c, end_r, 3, color)


def _distance_color(d: float) -> tuple:
    if d < 0.20:
        return (255, 40, 40)
    if d < 0.50:
        t = (d - 0.20) / 0.30
        return (255, int(40 + 160 * t), 40)
    if d < 1.0:
        t = (d - 0.50) / 0.50
        return (int(255 - 150 * t), 255, 40)
    return (80, 255, 80)


def draw_target_heading(img: np.ndarray, frame: MapFrame):
    """Dashed line from robot along the target heading, 2 m long."""
    if not frame.nav_active:
        return
    cc, rc = MAP_PX // 2, MAP_PX // 2
    cos_h = math.cos(frame.target_heading)
    sin_h = math.sin(frame.target_heading)

    # Dashed segments every 10 px
    for d_px in range(15, int(2.5 * PX_PER_M), 18):
        c0 = cc + int(round((d_px)     * cos_h))
        r0 = rc - int(round((d_px)     * sin_h))
        c1 = cc + int(round((d_px + 8) * cos_h))
        r1 = rc - int(round((d_px + 8) * sin_h))
        draw_line(img, c0, r0, c1, r1, COL_TARGET, width=2)


def draw_gt_obstacles(img: np.ndarray, frame: MapFrame):
    """
    Overlay ground-truth obstacle outlines (cyan).

    DEBUG ONLY. This is not used by the navigator — it lets the human
    operator see whether the perceived occupancy matches reality.
    """
    for o in frame.gt_obstacles:
        t = o.get("type")
        if t == "box":
            x, y, sx, sy = o["x"], o["y"], o["sx"], o["sy"]
            c0, r0 = world_to_px(x - sx, y - sy, frame.robot_x, frame.robot_y)
            c1, r1 = world_to_px(x + sx, y + sy, frame.robot_x, frame.robot_y)
            draw_rect_outline(img, min(c0, c1), min(r0, r1),
                              max(c0, c1), max(r0, r1), COL_GT_OBST)
        elif t == "cylinder":
            x, y, r = o["x"], o["y"], o["r"]
            cc, rc = world_to_px(x, y, frame.robot_x, frame.robot_y)
            draw_circle_outline(img, cc, rc, int(round(r * PX_PER_M)), COL_GT_OBST)
        elif t == "wall":
            x, y, sy = o["x"], o["y"], o["sy"]
            sx = 0.05
            c0, r0 = world_to_px(x - sx, y - sy, frame.robot_x, frame.robot_y)
            c1, r1 = world_to_px(x + sx, y + sy, frame.robot_x, frame.robot_y)
            draw_rect_outline(img, min(c0, c1), min(r0, r1),
                              max(c0, c1), max(r0, r1), COL_GT_OBST)


# ─────────────────────────────────────────────────────────────────────
# Full render
# ─────────────────────────────────────────────────────────────────────

def render_rgb(frame: MapFrame) -> np.ndarray:
    """Render one full map frame. Returns HxWx3 uint8 RGB."""
    img = np.full((MAP_PX, MAP_PX, 3), COL_BG, dtype=np.uint8)
    rasterize_grid(img, frame)
    draw_axes(img)
    draw_gt_obstacles(img, frame)
    draw_safety_rings(img)
    draw_rangefinder_rays(img, frame)
    draw_target_heading(img, frame)
    draw_robot(img, frame)
    return img


# ─────────────────────────────────────────────────────────────────────
# PNG writer (pure stdlib zlib, no Pillow)
# ─────────────────────────────────────────────────────────────────────

def save_png(path: str, rgb: np.ndarray):
    """
    Minimal PNG writer. Supports RGB (no alpha).

    Format reference: PNG spec, filter type 0 (None) per row.
    Each row is prefixed with a 0 byte (filter type), then raw RGB bytes.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"save_png expects HxWx3 uint8, got {rgb.shape} {rgb.dtype}")
    h, w = rgb.shape[:2]

    # Build raw byte stream: each row is filter(0) + RGB bytes
    raw = bytearray()
    for row in rgb:
        raw.append(0)
        raw.extend(row.tobytes())
    compressed = zlib.compress(bytes(raw), level=6)

    def chunk(tag: bytes, data: bytes) -> bytes:
        length = struct.pack('>I', len(data))
        crc = struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
        return length + tag + data + crc

    signature = b'\x89PNG\r\n\x1a\n'
    ihdr = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)  # 8-bit, truecolor
    png_bytes = (signature
                 + chunk(b'IHDR', ihdr)
                 + chunk(b'IDAT', compressed)
                 + chunk(b'IEND', b''))
    with open(path, 'wb') as f:
        f.write(png_bytes)


def rgb_to_ppm(rgb: np.ndarray) -> bytes:
    """Binary PPM (P6) bytes, directly loadable by tkinter.PhotoImage."""
    h, w = rgb.shape[:2]
    header = f"P6\n{w} {h}\n255\n".encode("ascii")
    return header + rgb.tobytes()


# ─────────────────────────────────────────────────────────────────────
# Ground-truth obstacle extraction from MuJoCo model (debug only)
# ─────────────────────────────────────────────────────────────────────

def extract_gt_obstacles(mj_model) -> list:
    """
    Return a list of obstacle dicts compatible with MapFrame.gt_obstacles,
    by iterating every world-body geom that is collidable and not the floor.

    Selection rule:
      - geom_bodyid == 0            → attached to the world body
      - geom_type in {box,cylinder} → something the robot can hit in 2D
      - contype != 0                → actually collidable (skip decorative)
      - name != "floor"             → not the ground plane
      - not starting with "wall_"   → skip corridor side walls for the
                                       debug overlay (they clutter the view)

    This is for debug overlay ONLY. The navigator never calls this.
    """
    import mujoco
    obstacles = []
    for gid in range(mj_model.ngeom):
        if int(mj_model.geom_bodyid[gid]) != 0:
            continue
        gtype = int(mj_model.geom_type[gid])
        if gtype not in (int(mujoco.mjtGeom.mjGEOM_BOX),
                         int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
            continue
        if int(mj_model.geom_contype[gid]) == 0:
            continue

        name_adr = int(mj_model.name_geomadr[gid])
        name = bytes(mj_model.names[name_adr:]).split(b'\x00', 1)[0].decode('ascii', 'replace')
        if name == "floor":
            continue

        pos = mj_model.geom_pos[gid]
        size = mj_model.geom_size[gid]

        if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
            sx, sy = float(size[0]), float(size[1])
            # Narrow box (either very thin in X or in Y) is visualized as a wall
            if sx < 0.08 or sy < 0.08:
                obstacles.append({
                    "type": "box",
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "sx": sx,
                    "sy": sy,
                })
            else:
                obstacles.append({
                    "type": "box",
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "sx": sx,
                    "sy": sy,
                })
        elif gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            obstacles.append({
                "type": "cylinder",
                "x": float(pos[0]),
                "y": float(pos[1]),
                "r": float(size[0]),
            })
    return obstacles


# ─────────────────────────────────────────────────────────────────────
# Snapshot helper
# ─────────────────────────────────────────────────────────────────────

def default_snapshot_dir() -> str:
    """Return project-root/snapshots, creating it if needed."""
    here = os.path.dirname(os.path.abspath(__file__))
    proj = os.path.dirname(here)
    out = os.path.join(proj, "snapshots")
    os.makedirs(out, exist_ok=True)
    return out


def snapshot(frame: MapFrame, out_dir: str | None = None,
             tag: str = "") -> str:
    """Render `frame` and save as PNG with a timestamped filename."""
    if out_dir is None:
        out_dir = default_snapshot_dir()
    img = render_rgb(frame)
    ts = time.strftime("%Y%m%d_%H%M%S")
    tag_part = f"_{tag}" if tag else ""
    path = os.path.join(out_dir, f"map_{ts}{tag_part}.png")
    save_png(path, img)
    return path


# ─────────────────────────────────────────────────────────────────────
# Live tkinter window
# ─────────────────────────────────────────────────────────────────────

class MapWindow:
    """
    Toplevel tkinter window that displays the latest MapFrame.

    Call `update_frame(frame)` from the main loop; the window is refreshed
    via a scheduled tk `after` callback. The window owns its own tk
    PhotoImage and swaps it each refresh.

    Thread safety: tkinter is NOT thread-safe, so the caller must update
    `self._latest_frame` from the same thread that runs `after`. In our
    setup, ControlPanel already runs its own tk mainloop in a background
    thread — the simplest pattern is to let MapWindow live inside that
    thread and feed it via a lock-protected attribute.
    """

    def __init__(self, parent, get_frame_callable):
        """
        parent: a tk.Tk or tk.Toplevel (used to spawn our own Toplevel)
        get_frame_callable: zero-arg callable returning the latest MapFrame
                            or None. Called from the tk thread.
        """
        import tkinter as tk
        self._tk = tk
        self._get_frame = get_frame_callable
        self._top = tk.Toplevel(parent)
        self._top.title("DeskBot Map")
        self._top.resizable(False, False)

        self._canvas = tk.Canvas(
            self._top, width=MAP_PX, height=MAP_PX,
            bg="#141418", highlightthickness=0,
        )
        self._canvas.pack()
        self._photo = tk.PhotoImage(width=MAP_PX, height=MAP_PX)
        self._canvas.create_image(0, 0, image=self._photo, anchor="nw")

        # Status bar
        self._status = tk.Label(
            self._top, text="(no frame)",
            font=("Consolas", 9), anchor="w", fg="#ccc", bg="#202024",
            padx=6, pady=3,
        )
        self._status.pack(fill="x")

        # Snapshot button
        btn_frame = tk.Frame(self._top, bg="#202024")
        btn_frame.pack(fill="x")
        tk.Button(btn_frame, text="Snapshot PNG",
                  command=self._on_snapshot).pack(side="left", padx=4, pady=4)

        self._refresh_ms = 100  # 10 Hz refresh
        self._schedule()

    def _schedule(self):
        self._top.after(self._refresh_ms, self._refresh)

    def _refresh(self):
        frame = self._get_frame()
        if frame is not None:
            rgb = render_rgb(frame)
            ppm = rgb_to_ppm(rgb)
            new_photo = self._tk.PhotoImage(data=ppm, format="PPM")
            self._canvas.itemconfig(
                self._canvas.find_all()[0], image=new_photo
            )
            # Keep ref to prevent GC
            self._photo = new_photo

            self._status.config(
                text=(
                    f"pos=({frame.robot_x:+.2f},{frame.robot_y:+.2f})  "
                    f"h={math.degrees(frame.heading):+.0f}°  "
                    f"tgt={math.degrees(frame.target_heading):+.0f}°  "
                    f"fsm={frame.fsm_state}  "
                    f"nav={'ON' if frame.nav_active else 'OFF'}"
                )
            )
        self._schedule()

    def _on_snapshot(self):
        frame = self._get_frame()
        if frame is None:
            return
        path = snapshot(frame, tag=frame.fsm_state or "idle")
        self._status.config(text=f"Saved: {os.path.basename(path)}")


# ─────────────────────────────────────────────────────────────────────
# Self-test: `python -m deskbot.mapviz`
# ─────────────────────────────────────────────────────────────────────

def _synthetic_frame() -> MapFrame:
    """Build a fake frame with a couple of fake obstacles — for unit test."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    # Put a fake wall at (x=1 m, y=0) in the grid
    for dy_cells in range(-5, 6):
        ci = GRID_SIZE // 2 + int(1.0 / GRID_RES)
        cj = GRID_SIZE // 2 + dy_cells
        grid[ci, cj] = 2.5  # strongly occupied
    # Free cells in front
    for ci in range(GRID_SIZE // 2 + 1, GRID_SIZE // 2 + int(1.0 / GRID_RES)):
        grid[ci, GRID_SIZE // 2] = -1.5

    return MapFrame(
        grid=grid,
        grid_cx=0.0,
        grid_cy=0.0,
        robot_x=0.0,
        robot_y=0.0,
        heading=0.0,
        target_heading=math.radians(5.0),
        nav_active=True,
        fsm_state="go_heading",
        rf_compensated={
            "rf_C":  1.0,
            "rf_FL": 1.2,
            "rf_FR": 1.1,
            "rf_L":  1.0,
            "rf_R":  1.0,
            "rf_WL": 0.8,
            "rf_WR": 0.9,
            "rf_SL": -1.0,
            "rf_SR": -1.0,
            "rf_B":  -1.0,
        },
        gt_obstacles=[
            {"type": "box", "x": 1.0, "y": 0.0, "sx": 0.1, "sy": 0.4},
        ],
    )


if __name__ == "__main__":
    frame = _synthetic_frame()
    img = render_rgb(frame)
    out_dir = default_snapshot_dir()
    path = os.path.join(out_dir, "mapviz_selftest.png")
    save_png(path, img)
    print(f"Self-test: wrote {path}")
    print(f"Image shape: {img.shape}, dtype: {img.dtype}")
