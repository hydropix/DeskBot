"""
Control panel GUI for DeskBot — tkinter-based, runs in a separate thread.

Provides a virtual joystick (click+drag) for velocity/yaw control,
plus buttons for push and reset. Updates a shared Commands object.
Navigation panel allows setting a heading direction for the robot to follow.
"""
import math
import threading
import tkinter as tk
import numpy as np


# Joystick canvas size
JOY_SIZE = 180
JOY_RADIUS = JOY_SIZE // 2 - 10  # usable radius in pixels


class ControlPanel:
    """Tkinter control panel that writes to a shared Commands object."""

    def __init__(self, commands, max_velocity: float, max_yaw_rate: float):
        self.commands = commands
        self.max_velocity = max_velocity
        self.max_yaw_rate = max_yaw_rate
        self._thread = None
        self._root = None
        self._joy_dot = None
        self._dragging = False

    def start(self):
        """Launch the GUI in a background thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        root = tk.Tk()
        self._root = root
        root.title("DeskBot Control")
        root.resizable(False, False)
        root.attributes("-topmost", True)

        # ── Joystick frame ──
        joy_frame = tk.LabelFrame(root, text="Joystick", padx=5, pady=5)
        joy_frame.pack(padx=8, pady=(8, 4))

        canvas = tk.Canvas(
            joy_frame, width=JOY_SIZE, height=JOY_SIZE,
            bg="#2a2a2e", highlightthickness=0,
        )
        canvas.pack()
        self._canvas = canvas

        cx, cy = JOY_SIZE // 2, JOY_SIZE // 2

        # Background circles
        canvas.create_oval(
            cx - JOY_RADIUS, cy - JOY_RADIUS,
            cx + JOY_RADIUS, cy + JOY_RADIUS,
            outline="#555", width=1,
        )
        canvas.create_oval(
            cx - JOY_RADIUS // 2, cy - JOY_RADIUS // 2,
            cx + JOY_RADIUS // 2, cy + JOY_RADIUS // 2,
            outline="#444", width=1, dash=(3, 3),
        )
        # Crosshair
        canvas.create_line(cx - JOY_RADIUS, cy, cx + JOY_RADIUS, cy,
                           fill="#444", width=1)
        canvas.create_line(cx, cy - JOY_RADIUS, cx, cy + JOY_RADIUS,
                           fill="#444", width=1)

        # Axis labels
        canvas.create_text(cx, cy - JOY_RADIUS - 8, text="FWD",
                           fill="#8c8", font=("Consolas", 7))
        canvas.create_text(cx, cy + JOY_RADIUS + 8, text="BWD",
                           fill="#c88", font=("Consolas", 7))
        canvas.create_text(cx - JOY_RADIUS - 12, cy, text="L",
                           fill="#88c", font=("Consolas", 7))
        canvas.create_text(cx + JOY_RADIUS + 12, cy, text="R",
                           fill="#88c", font=("Consolas", 7))

        # Draggable dot
        dot_r = 12
        self._joy_dot = canvas.create_oval(
            cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r,
            fill="#4a9eff", outline="#78b8ff", width=2,
        )

        canvas.bind("<ButtonPress-1>", self._on_joy_press)
        canvas.bind("<B1-Motion>", self._on_joy_drag)
        canvas.bind("<ButtonRelease-1>", self._on_joy_release)

        # ── Status display ──
        status_frame = tk.Frame(root)
        status_frame.pack(padx=8, pady=2, fill="x")

        self._vel_label = tk.Label(
            status_frame, text="Vel: 0.00 m/s", font=("Consolas", 9),
            anchor="w",
        )
        self._vel_label.pack(fill="x")

        self._yaw_label = tk.Label(
            status_frame, text="Yaw: 0.00 rad/s", font=("Consolas", 9),
            anchor="w",
        )
        self._yaw_label.pack(fill="x")

        self._pitch_label = tk.Label(
            status_frame, text="Pitch: 0.00 deg", font=("Consolas", 9),
            anchor="w",
        )
        self._pitch_label.pack(fill="x")

        # ── Buttons ──
        btn_frame = tk.Frame(root)
        btn_frame.pack(padx=8, pady=(4, 8), fill="x")

        tk.Button(
            btn_frame, text="Push", width=8,
            command=self._on_push,
        ).pack(side="left", padx=2)

        tk.Button(
            btn_frame, text="Stop", width=8,
            command=self._on_stop,
        ).pack(side="left", padx=2)

        tk.Button(
            btn_frame, text="Reset", width=8,
            command=self._on_reset,
        ).pack(side="left", padx=2)

        # ── Navigation panel ──
        nav_frame = tk.LabelFrame(root, text="Navigation AI", padx=5, pady=5)
        nav_frame.pack(padx=8, pady=(4, 4), fill="x")

        # Heading input
        heading_row = tk.Frame(nav_frame)
        heading_row.pack(fill="x", pady=2)
        tk.Label(heading_row, text="Cap:", font=("Consolas", 9)).pack(side="left")
        self._heading_var = tk.StringVar(value="0")
        tk.Entry(heading_row, textvariable=self._heading_var, width=5,
                 font=("Consolas", 9)).pack(side="left", padx=4)
        tk.Label(heading_row, text="deg", font=("Consolas", 8),
                 fg="#888").pack(side="left")

        # Quick heading buttons
        quick_row = tk.Frame(nav_frame)
        quick_row.pack(fill="x", pady=2)
        for label, angle in [("+X (0)", 0), ("+Y (90)", 90),
                             ("-X (180)", 180), ("-Y (-90)", -90)]:
            tk.Button(
                quick_row, text=label, width=8, font=("Consolas", 7),
                command=lambda a=angle: self._set_heading_quick(a),
            ).pack(side="left", padx=1)

        # Go / Stop buttons
        nav_btn_row = tk.Frame(nav_frame)
        nav_btn_row.pack(fill="x", pady=2)
        tk.Button(
            nav_btn_row, text="Go", width=10,
            command=self._on_navigate, bg="#2d5a2d", fg="white",
        ).pack(side="left", padx=2)
        tk.Button(
            nav_btn_row, text="Stop Nav", width=10,
            command=self._on_nav_stop, bg="#5a2d2d", fg="white",
        ).pack(side="left", padx=2)

        # Nav status display
        self._nav_status_label = tk.Label(
            nav_frame, text="Nav: idle", font=("Consolas", 8),
            anchor="w", fg="#888",
        )
        self._nav_status_label.pack(fill="x")

        self._nav_pos_label = tk.Label(
            nav_frame, text="Pos: (0.00, 0.00) h=0.0",
            font=("Consolas", 8), anchor="w", fg="#888",
        )
        self._nav_pos_label.pack(fill="x")

        # Start status update loop
        self._update_status()

        root.protocol("WM_DELETE_WINDOW", root.quit)
        root.mainloop()

    # ── Joystick events ──

    def _joy_xy_to_commands(self, event_x, event_y):
        """Convert canvas pixel position to (velocity, yaw_rate)."""
        cx, cy = JOY_SIZE // 2, JOY_SIZE // 2
        dx = (event_x - cx) / JOY_RADIUS   # -1..+1 (left..right)
        dy = (cy - event_y) / JOY_RADIUS    # -1..+1 (back..forward)

        # Clamp to circle
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 1.0:
            dx /= dist
            dy /= dist

        velocity = dy * self.max_velocity
        yaw_rate = -dx * self.max_yaw_rate  # right on screen = negative yaw

        return velocity, yaw_rate, dx, dy

    def _move_dot(self, dx, dy):
        """Move the joystick dot to normalized position (-1..1)."""
        cx, cy = JOY_SIZE // 2, JOY_SIZE // 2
        px = cx + dx * JOY_RADIUS
        py = cy - dy * JOY_RADIUS
        dot_r = 12
        self._canvas.coords(
            self._joy_dot,
            px - dot_r, py - dot_r, px + dot_r, py + dot_r,
        )

    def _on_joy_press(self, event):
        self._dragging = True
        vel, yaw, dx, dy = self._joy_xy_to_commands(event.x, event.y)
        self.commands.target_velocity = vel
        self.commands.target_yaw_rate = yaw
        self._move_dot(dx, dy)

    def _on_joy_drag(self, event):
        if not self._dragging:
            return
        vel, yaw, dx, dy = self._joy_xy_to_commands(event.x, event.y)
        self.commands.target_velocity = vel
        self.commands.target_yaw_rate = yaw
        self._move_dot(dx, dy)

    def _on_joy_release(self, event):
        self._dragging = False
        self.commands.target_velocity = 0.0
        self.commands.target_yaw_rate = 0.0
        self._move_dot(0, 0)

    # ── Buttons ──

    def _on_push(self):
        angle = np.random.uniform(0, 2 * math.pi)
        self.commands.push_dir = np.array(
            [math.cos(angle), math.sin(angle), 0.0]
        )
        self.commands.push_timer = 0.08

    def _on_stop(self):
        self.commands.target_velocity = 0.0
        self.commands.target_yaw_rate = 0.0
        self._move_dot(0, 0)

    def _on_reset(self):
        self.commands.reset_requested = True
        self._move_dot(0, 0)

    # ── Navigation ──

    def _set_heading_quick(self, angle):
        self._heading_var.set(str(angle))
        self._on_navigate()

    def _on_navigate(self):
        """Send heading to navigator."""
        try:
            heading = float(self._heading_var.get())
        except ValueError:
            return
        self.commands.nav_heading_request = heading

    def _on_nav_stop(self):
        """Stop navigation."""
        self.commands.nav_stop_request = True

    # ── Status update ──

    def _update_status(self):
        if self._root is None:
            return
        self._vel_label.config(
            text=f"Vel:   {self.commands.target_velocity:+.2f} m/s"
        )
        self._yaw_label.config(
            text=f"Yaw:   {self.commands.target_yaw_rate:+.2f} rad/s"
        )
        pitch = getattr(self.commands, '_pitch_display', 0.0)
        self._pitch_label.config(
            text=f"Pitch: {math.degrees(pitch):+.1f} deg"
        )

        # Navigation status
        nav = getattr(self.commands, 'navigator', None)
        if nav is not None:
            s = nav.state
            if s.active:
                color = "#4a9eff"
                front_str = f"front={s.min_front_dist:.2f}m"
                if s.behavior == "contour":
                    detail = f"{s.contour_side} dev={math.degrees(s.contour_deviation):+.0f}"
                    status = f"Nav: {s.behavior} | {detail} | {front_str}"
                else:
                    h_err = math.degrees(s.target_heading - s.heading)
                    status = f"Nav: {s.behavior} | err={h_err:+.0f} | {front_str}"
            else:
                color = "#888"
                status = "Nav: idle"
            self._nav_status_label.config(text=status, fg=color)
            self._nav_pos_label.config(
                text=f"Pos: ({s.pos_x:.2f}, {s.pos_y:.2f}) h={math.degrees(s.heading):.0f}"
            )

        self._root.after(50, self._update_status)
