# DeskBot Development Journal

This document tracks the complete development journey of DeskBot -- what was tried, what worked, what failed, and the lessons learned along the way.

## Method & Collaboration

DeskBot is developed as a **human-AI collaboration** between Hydropix (robotics enthusiast, project owner) and Claude Code (AI design partner, Anthropic's Claude Opus). This is not a traditional "developer uses a tool" workflow -- Claude is an active participant in design decisions, architecture choices, debugging, and research.

**How we work together:**

- Hydropix sets the vision, goals, and constraints. He validates results in the simulator and provides feedback.
- Claude proposes architectures, writes code, researches solutions, and explains the reasoning. Claude also makes mistakes, hits dead ends, and learns from them -- all documented here.
- The process is iterative and pedagogical: every implementation is also a learning opportunity. Claude explains concepts in depth so Hydropix builds lasting engineering knowledge.
- Claude maintains persistent memory across sessions (auto-memory system) and this journal to ensure continuity.

**Tools used:** MuJoCo (physics simulation), Python, Claude Code (AI pair programmer), Git/GitHub, tkinter (GUI).

---

## Session 1 -- 2026-04-10 : Project Bootstrap & First Balance

### What was done

- Created project scaffold: `run.bat`, virtual environment, `requirements.txt` (MuJoCo, NumPy).
- Designed the first MJCF robot model (`deskbot.xml`): two-wheeled body with actuators and sensors.
- Built the **sensor barrier** architecture: `SensorModel` reads MuJoCo sensors and injects realistic IMU noise (bias drift, white noise) + encoder quantization. The controller only ever sees `SensorReadings`.
- Implemented `StateEstimator` with complementary filter (accel + gyro fusion) and encoder-based odometry.
- Implemented cascaded PID controller: velocity PI (outer) -> pitch PID (inner) + yaw P+D (differential steering).
- Added position hold via integrated encoder displacement.
- Created interactive MuJoCo viewer with keyboard controls (arrows, Space for push, R for reset).

### Claude's role

Claude designed the full architecture from scratch based on Hydropix's goal ("a self-balancing robot simulator for sim-to-real"). The sensor barrier was Claude's proposal -- inspired by real robotics practice where you never have access to ground truth in deployment. Claude also researched MuJoCo's sensor API and MJCF format from documentation.

### What worked

- The sensor barrier concept proved solid from the start -- it forced realistic constraints early.
- Cascaded PID architecture balanced the robot on first attempt after sign convention fixes.

### What failed / lessons learned

- **Sign conventions caused immediate divergence.** Yaw torque was inverted: must subtract from left wheel, add to right. Getting this wrong makes the robot spin instead of correcting.
- **Explicit `<inertial>` tags silently override geom-computed inertia.** This changed the dynamics in unexpected ways until discovered.
- **MuJoCo accelerometer reports coordinate acceleration, not proper acceleration.** Had to add gravity rotated into the body frame to match what a real IMU would read. The `atan2(-ax, az)` negation is critical.

---

## Session 2 -- 2026-04-11 (morning) : Robot Redesign & Skatepark

### What was done

- **Robot redesign**: 12 cm wheels (up from smaller), cubic body, 3 cm ground clearance.
- Increased encoder resolution from 360 to 1440 ticks/rev for finer odometry.
- Created **skatepark scene** (`scene_skatepark.xml`) with 9 obstacle zones (ramps, bumps, etc.).
- Built **tkinter GUI control panel** with virtual joystick (click+drag) to replace hold-to-move keyboard controls.
- Implemented camera follow system in the viewer.
- Pushed dynamics: max pitch 25 deg, top speed ~2 m/s.

### Claude's role

Claude proposed the tkinter GUI after diagnosing the MuJoCo viewer keyboard limitation. This was a pivot -- the original plan was keyboard control, but Claude discovered through experimentation that `key_callback` doesn't support key-hold detection, then researched alternatives. Claude also designed the skatepark scene procedurally with varied terrain challenges.

### What worked

- tkinter GUI solved the keyboard limitation elegantly -- MuJoCo viewer only fires PRESS events, no RELEASE or REPEAT, making held keys impossible.
- Larger wheels + higher encoder resolution gave much better terrain handling and odometry precision.

### What failed / lessons learned

- **MuJoCo viewer `key_callback` is PRESS-only.** No RELEASE, no REPEAT. Continuous keyboard control is fundamentally impossible through the viewer API. This was not documented -- discovered by trial and error.
- **Yaw controller can overpower balance.** Had to limit yaw torque to <= 15% of max torque, otherwise the robot tips during turns.
- **Velocity loop fights balance if unconstrained.** Rate-limited velocity output to max 12 deg/s pitch change to prevent the outer loop from commanding faster lean than the inner loop can stabilize.
- **Balance Ki causes slow drift.** Added dead zone, integral decay, and reduced Ki to prevent the integrator from pushing the robot off position when balanced.

---

## Session 3 -- 2026-04-11 (afternoon) : Sensors & Apartment Scene

### What was done

- Created **apartment scene** (`scene_apartment.xml`): T2 layout ~50 m squared, 5 rooms with furniture.
- Integrated **5x VL53L0X rangefinders** using `mj_ray` raycasting: front-center, front-left (30 deg), front-right (-30 deg), side-left, side-right.
- Added **laser beam visualization** in the viewer (color-coded by distance: green=far, red=close).
- Redesigned wheels: 12 cm -> 10 cm radius for sensor clearance.
- Narrowed wheel separation: 22 cm -> 16 cm for compact design.
- Designed **curved bumper** for constant ground clearance across pitch range.
- Designed **sensor pod**: front sensors mounted LOW (6 cm, horizontal), side sensors mounted HIGH (21 cm, -5 deg tilt). The low front mount uses the robot's own pitch as "embodied intelligence" -- when the robot leans forward to move, the sensors naturally tilt downward.

### Claude's role

Claude researched VL53L0X datasheets to model realistic rangefinder behavior (2 m range, 25 deg FoV cone). The "embodied tilt" concept was a Claude insight: instead of compensating for pitch in software, let the physics do the work. Claude also ran geometric analysis to solve the wheel clearance constraint for side sensor placement.

### What worked

- The embodied tilt concept for front sensors is elegant: no need for explicit tilt compensation during normal movement.
- Separating front (low, horizontal) and side (high, tilted) mounting solved the wheel clearance constraint without sacrificing field of view.

### What failed / lessons learned

- **Wheel clearance constrains sensor placement.** Side sensors must satisfy `sqrt(X^2 + Z^2) > wheel_radius` to avoid intersecting the wheel volume. This forced the side sensors onto a raised pod.
- **MuJoCo box edges create invisible walls.** Thin boxes (5 mm) for rugs create edges the wheels can't cross. Fix: set `contype=0` on decorative elements.

---

## Session 4 -- 2026-04-11 (afternoon) : Navigation v1 -- The Failure

### What was done

- Attempted reactive navigation: wall-following + obstacle avoidance using the 5 rangefinders.
- Iterative tuning of avoidance thresholds, turn rates, and state transitions.
- Created headless benchmark script for systematic testing.

### Claude's role

Claude implemented a reactive behavior-blending approach (weighted sum of wander/avoid/follow vectors). After multiple rounds of tuning failed, Claude proposed systematic benchmarking to get objective numbers instead of subjective visual assessment. The benchmark revealed the 30% success rate that made the failure undeniable. Hydropix called it "largement dysfonctionnel" -- Claude agreed and proposed a full redesign rather than more tuning.

### What worked

- **Straight corridor navigation**: 100% success rate in simple corridors.
- **Pitch compensation for rangefinders**: `compensate_rangefinders()` filters readings where `distance > 80% * floor_distance_at_current_pitch`. This eliminated false "obstacle" readings caused by the front lasers hitting the ground during forward lean.

### What failed -- extensively

- **~30% success in furnished rooms.** The reactive approach was fundamentally insufficient.
- **Floor hits (failure cause #1)**: Front lasers hit the ground at >10 deg pitch even with compensation. The 6 cm mount height and horizontal orientation means the beam intercepts the floor plane at ~34 cm when pitched 10 deg.
- **60-degree coverage gaps**: 5 rangefinders leave blind spots. The robot would miss obstacles between sensor cones entirely.
- **Dead reckoning drift**: ~0.8 m drift over 13 seconds. Without absolute position reference, the robot quickly lost track of where it was.
- **No absolute heading**: Without a magnetometer, heading accumulates gyro drift. Encoder-pitch coupling further contaminates heading estimates.
- **FSM vs. blending**: Reactive blending of avoid/follow/wander behaviors created unpredictable oscillations. A clean FSM would have been better from the start.

### Post-mortem verdict

User assessment: "largement dysfonctionnel." The architecture was broken at a fundamental level -- no amount of parameter tuning would fix 5-sensor reactive navigation in cluttered rooms. Decision: **stop iterating, redesign from scratch.**

---

## Session 5 -- 2026-04-11 (evening) : Navigation v2 -- Bug2 + Occupancy Grid

### What was done

- Complete rewrite of navigation (~600 lines in `navigation.py`).
- **3-layer architecture**:
  1. **Strategic**: Bug2-inspired 2-state FSM (GO_HEADING / CONTOUR).
  2. **Tactical**: Wall-following with P-controller on side laser distance.
  3. **Reactive**: Pitch-compensated rangefinders, emergency braking.
- **Heading-based API**: `set_heading(degrees)` instead of waypoint coordinates. This avoids dead reckoning drift -- the robot only needs to maintain a compass heading, not track absolute position.
- **Occupancy grid**: 40x40 cells at 5 cm resolution (2 m x 2 m egocentric map).
- Added hysteresis: 2-second minimum contour time before checking exit condition.
- Stuck recovery: reverse + flip contour side.
- New GUI (`gui.py`) with heading input instead of waypoints.
- New test scene (`scene_nav_test.xml`): 3 m corridor with 3 obstacles.
- Headless benchmark script (`benchmark_heading.py`).

### Claude's role

Claude researched classical robot navigation algorithms (Bug1, Bug2, tangent bug, VFH, potential fields) and recommended Bug2 for its simplicity and provable completeness guarantees. The heading-based API was a joint insight: Hydropix's constraint ("no GPS, no magnetometer") led Claude to realize that heading-only control avoids the DR position drift problem entirely. Claude wrote the full 600-line implementation and designed the test scene for systematic validation.

### What worked

- **30/30 success on nav_test corridor**, 0 falls, 0 stuck. Validated the architecture.
- Heading-based API was a breakthrough insight: it sidesteps the DR drift problem entirely for corridor-type navigation.
- Bug2 FSM (go-to-heading + contour-obstacle) is simple, predictable, and debuggable.

### Known limitations (not failures -- accepted trade-offs)

- DR heading drift ~5 deg over 40 seconds (~1.15 m lateral deviation over 12 m).
- Still only 5 rangefinders with 60-degree gaps.
- Not yet tested on apartment scene.

---

## Session 6 -- 2026-04-11 : Hardware Planning

### What was done

- Finalized and ordered the complete hardware BOM for a physical DeskBot prototype.
- Audited the SumoBot project (separate repo) for upgrade opportunities.

### Claude's role

Claude researched component specifications, cross-referenced datasheets, and identified cross-project synergies (shared components between DeskBot and SumoBot). Claude recommended specific parts based on the simulator parameters (e.g., motor RPM matched to desired wheel speed, encoder resolution matched to simulation, driver efficiency for battery-powered operation).

### Key hardware decisions

- **ESP32-S3 N16R8** (dual-core 240 MHz, 16 MB Flash, 8 MB PSRAM) -- chosen for wireless capability and compute headroom for future ML inference.
- **JGA25-371 motors** (12V, 280 RPM, integrated Hall encoders) -- direct drive, no gearbox needed at 200 mm wheel diameter.
- **TB6612FNG driver** instead of L298N -- MOSFET-based, 90%+ efficiency vs. 50% for BJT-based L298N. Also shared with SumoBot upgrade.
- **MPU6500** (not MPU6050) -- better noise characteristics for balance control.
- **5x VL53L0X** via TCA9548A I2C multiplexer -- matches the simulator sensor layout exactly.
- **NiMH batteries** (9.6V, 8xAA) -- safe chemistry, no fire risk on a desktop robot.

### Cross-project synergy

- TB6612FNG, VL53L0X spares, thermal inserts, and NiMH batteries shared between DeskBot and SumoBot orders.

---

## Session 7 -- 2026-04-11 (night) : Randomized Stress Testing & Sensor Upgrade

### What was done

- Created **randomized obstacle benchmark** (`benchmark_random.py`): procedurally generates corridors with 2-5 random obstacles (boxes, cylinders, walls with gaps, chicanes) and measures success rate over hundreds of episodes.
- Iterative improvement cycle on navigation v2 based on failure analysis:
  - **Heading pull during wall-follow**: steers toward heading when it doesn't fight wall-following (only when pulling AWAY from the wall). Fixed the "robot follows corridor wall backwards" failure.
  - **GO_HEADING after reverse**: breaks the contour->stuck->contour loop. Previously the robot re-entered contour immediately after reversing.
  - **Anti-regression watchdog**: per-contour (4s, 0.3m regression) + global (4s, no forward progress). Detects when the robot is making zero or negative progress along the heading axis.
  - **Virtual grid scan**: on contour entry, scans the occupancy grid at 15-degree intervals over 180 degrees to pick the side with most clearance. Replaces the sensor-only heuristic. This was the single biggest improvement for wall obstacles.
  - **Occupancy grid upgrade**: 40x40 at 5cm -> 60x60 at 8cm (2m -> 4.8m extent) for better spatial memory.
- **Sensor upgrade**: added 2x VL53L0X (FL2, FR2) to fill the 60-degree gap between front and side sensors.
- **Angle sweep** (`sweep_angle.py`): tested FL2/FR2 at 45, 50, 55, 60, 65, 70, 75 degrees. **Optimal: 50-55 degrees** -- maximizes wall success rate.
- Applied 55 degrees as final angle in `deskbot.xml`.

### Claude's role

Claude designed the randomized benchmark and ran the full analysis loop: generate obstacles, measure failure rate, categorize failures (wall vs non-wall), debug specific failing seeds with verbose traces, implement fixes, re-measure. The key insight came from statistical analysis: wall obstacles caused 90% of failures, and the root cause was the 60-degree gap in sensor coverage between FL (30 deg) and SL (90 deg). Claude proposed the sensor upgrade and designed the angle sweep to find the empirical optimum.

### Progression of success rate (randomized, 100 episodes)

| Iteration | Change | Overall | Wall | No wall |
|-----------|--------|---------|------|---------|
| v2.0 baseline (5 sensors) | Naive wall-follow | 34% | ~20% | ~60% |
| v2.1 wall-lost heading return | Return to heading when wall lost | 40% | | |
| v2.2 heading pull | Pull toward heading during wall-follow | 44% | | |
| v2.3 GO_HEADING after reverse | Break contour loop | 60% | | |
| v2.4 anti-regression watchdog | Detect backwards movement | 64% | | |
| v2.5 global watchdog 4s | Detect zero progress | 73% | | |
| v2.6 virtual grid scan | Grid-based side choice | 80% | 72% | 91% |
| **v2.7 + 2 sensors at 55 deg** | **Fill coverage gap** | **82%** | **78%** | **89%** |

### What worked

- **Monte Carlo testing** was essential. A single test scene (nav_test) showed 100% success -- randomized testing revealed the true 34% failure rate. You can't evaluate navigation on one track.
- **Virtual grid scan** was the biggest algorithmic win. Instead of scanning physically (which disrupts balance), query the occupancy grid that's been built passively during approach.
- **Sensor placement matters more than algorithm tuning.** Going from 5 to 7 sensors with FL2/FR2 at 55 degrees improved wall success from 72% to 78% -- a gain no algorithm change could achieve with the original 5 sensors.
- **Angle sweep**: 50-55 degrees was optimal. Below 50 degrees, too much overlap with FL/FR. Above 65 degrees, too much overlap with SL/SR and not enough gap coverage.

### What failed / lessons learned

- **Physical scan (rotating in place) hurt performance.** The balancing robot doesn't like standing still and spinning -- it disrupts balance and wastes time. Removed in favor of virtual grid scan.
- **State oscillation bug**: relaxing contour exit conditions caused the robot to enter/exit contour 6000+ times per second. Fixed with temporal hysteresis (1.5s minimum in contour before exit check).
- **Heading pull conflicts**: applying heading pull unconditionally during wall-follow made the robot fight itself (trying to go toward heading AND follow the wall simultaneously). Fixed by only applying the pull when it steers AWAY from the wall.
- **Stuck detection is not regression detection.** A robot going backwards at full speed is "not stuck" by distance metrics but is making negative progress. Added heading-axis progress check.

### Key architectural insight

The navigation improvement was not one big change but **8 incremental fixes**, each addressing a specific failure mode discovered through statistical testing. This mirrors real robotics development: you can't design a perfect system upfront. You need fast iteration cycles (headless benchmark: 0.2 episodes/second = 100 episodes in 8 minutes) and objective metrics to converge.

---

## Session 8 -- 2026-04-12 : Feature Sprint + Navigation Disaster

### What was done

Five features planned, three kept, two reverted after they broke the robot.

**Kept:**

1. **Encoder-pitch compensation** (`control.py:59`): 1-line fix adding `pitch_rate * WHEEL_RADIUS` to encoder velocity. Sign was initially wrong (-) and had to be corrected to (+) after empirical sign verification — see below.

2. **Collision detection via accelerometer** (`sensors.py`): `CollisionDetector` class using a first-order high-pass filter (alpha=0.95) on accelerometer signal. Separates impact transients (>15 m/s^2) from balance oscillations. Exposes `collision_detected`, `collision_magnitude`, `collision_direction` in `SensorReadings`. Zero hardware cost.

3. **LQR controller** (`control.py`): `LQRController` class replacing cascaded PID. 6-state system `[pitch, pitch_rate, velocity, position, yaw_error, yaw_rate]` with 2 control inputs. Physical parameters extracted from MuJoCo's compiled model. CARE solved via `scipy.linalg.solve_continuous_are`. Initial Q/R tuning diverged catastrophically — required 3 fixes (see below) before stable. `--pid` flag kept as fallback. Added `scipy` to requirements.

4. **WiFi RSSI localization + heading correction** (`localization.py`): `WiFiLocalizer` simulates ESP32-S3 WiFi scanning with log-distance path loss model. k-NN fingerprint matching on 289 pre-built grid points for apartment scene. `HeadingCorrector` detects wall parallelism from side rangefinders, compares to known axis-aligned map angles. Active only on apartment scene.

**Reverted:**

5. **DWA local planner**: Implemented as replacement for Bug2 FSM. Greedy one-step planner sampling 315 (v, w) trajectories. Worked on synthetic tests but catastrophic in practice — oscillated in front of obstacles because all candidates scored poorly.

6. **TEB planner** (attempt after DWA): Timed Elastic Band with 12 poses and gradient descent optimization. Looked good on a single-wall synthetic test. Reality: robot crashed into walls repeatedly. Patched with a reactive safety layer, pitch filtering, early FC detection. Each patch made it worse. Reverted to Bug2 v2 (the validated 82% system from session 7).

### LQR debugging saga

The initial LQR implementation made the robot diverge in ~0.1s. Three bugs compounded:

1. **Encoder-pitch sign wrong** (`-pitch_rate*R` instead of `+`). Verified empirically: applied +0.5 Nm torque for 50ms, measured actual pitch direction. In MuJoCo's convention, positive gyro_y = backward lean, so the phantom velocity correction needs `+` not `-`. The textbook convention (positive pitch = forward lean) is wrong for this estimator.

2. **Q/R weights too aggressive**. Initial Q=[200, 5, 15, 8, 10, 1], R=[1, 2] gave K[0,0]=17. Combined with gyro noise (~0.11 rad/s per sample at 500Hz), this produced ~0.33 Nm of noise-driven torque — enough to cause divergence. Retuned to Q=[80, 0.5, 4, 2, 5, 0.3], R=[8, 10] → K[0,0]=3.65. Much more conservative, stable under 3N push.

3. **Unfiltered pitch_rate in state vector**. Added first-order low-pass filter (alpha=0.8) on pitch_rate before feeding it into the LQR state. Attenuates gyro noise by ~5x at the cost of a 10ms delay.

After all three fixes, LQR balances 10s stably with max pitch 8.5° under a 3N push, and tracks velocity commands smoothly.

### Navigation saga (the failure)

The plan was: replace Bug2 FSM (ugly, 15 heuristics) with a theoretically cleaner planner.

**DWA attempt**: Implemented following Fox et al. 1997. Synthetic test (one wall in front) passed. Real test: robot oscillated in front of every obstacle because all (v, w) candidates scored poorly in narrow spaces. Bruno described it as "catastrophic, hesitates and can't find a path".

**TEB attempt**: Replaced DWA with Timed Elastic Band, which plans a full trajectory (12 poses over 2.5s) instead of one step. Better in theory — can encode multi-step maneuvers. Synthetic test (same wall) worked. Real test: robot went straight into walls. Bruno: "le robot fonce direct dans le mur".

Added a reactive safety layer reading FC laser at 500Hz. Robot still crashed. Tuned thresholds. Crashed differently. Added pitch filtering on top of `compensate_rangefinders` which already does pitch filtering — the two filters had different thresholds and contradicted each other in the gray zone. Robot saw walls, then ignored them.

Bruno finally asked to revert: "pfff le robot est toujours très stupide, aucun sens de l'évitement". Restored Bug2 v2 verbatim from the session context.

### Claude's role

Implemented the 5 features from memory notes. Debugged LQR through empirical sign verification (3-line test script). Made wrong architectural choices on navigation — picked DWA because it was on the roadmap, picked TEB because it was "better", patched broken systems instead of reverting earlier. Didn't run the randomized benchmark on the new planners before declaring them good. Classic anti-pattern: replacing a tested imperfect system with an elegant untested one.

### What worked

- **LQR after tuning**: stable, smooth, tracks commands, survives pushes.
- **Encoder-pitch fix**: immediate downstream improvement (cleaner velocity signal).
- **Collision detector**: high-pass filter approach cleanly separates impacts from balance noise.
- **WiFi localization**: log-distance model and k-NN fingerprinting work as expected on the apartment scene.
- **Reverting to Bug2**: fastest way to restore working navigation after 2 failed rewrites.

### What failed / lessons learned

- **LQR Q/R tuning requires careful thought about sensor noise**. The math (solving CARE) is trivial; the art is choosing weights that don't amplify gyro noise. Started too aggressive, had to back off 3x.
- **MuJoCo sign conventions are NOT standard textbook**. Always test empirically with a 3-line script before trusting any sign in a model. The LQR diverged catastrophically because of ONE wrong sign (encoder-pitch) amplified by the controller's high gains.
- **Don't replace a validated system with an elegant untested one**. Bug2 had 82% success on randomized corridors — that validation IS the value, not the algorithm's elegance. DWA and TEB were theoretically superior but had zero validation. They failed.
- **Patching a broken planner usually makes it worse**. The reactive safety layer I added on top of TEB created conflicts with `compensate_rangefinders` (double pitch filtering with different thresholds). Fixing the filter was impossible without understanding all the interactions.
- **The PID survives bad sign conventions, the LQR doesn't**. Cascaded PID has loose couplings and modest gains — a sign error in one sub-loop degrades performance but doesn't destabilize. LQR is a coupled MIMO controller; a sign error in ANY state propagates directly into torque via K, and a high gain amplifies it into divergence. Price of optimality: you must match the model to reality exactly.
- **Benchmark validation is non-negotiable for navigation changes**. If a new planner doesn't run scripts/benchmark_random.py with 100+ episodes and match Bug2's 82%, it doesn't get to replace Bug2.

### Session 8 final state

- LQR default, PID fallback (`--pid`). Both work.
- Encoder-pitch compensation with correct sign.
- Collision detection via IMU.
- WiFi localization on apartment scene.
- Navigation: Bug2 v2, unchanged from session 7. DWA and TEB code discarded.

---

## Roadmap (as of 2026-04-12)

### Completed

1. Project scaffold & tooling
2. MJCF robot model (v3: 10 cm wheels, 16 cm track, sensor pod)
3. Realistic sensor pipeline (IMU noise + encoder quantization + rangefinders)
4. Cascaded PID balance controller
5. Skatepark & apartment scenes
6. tkinter GUI with virtual joystick
7. Navigation v1 (reactive) -- failed at 30%, abandoned
8. Navigation v2 (Bug2 + occupancy grid) -- 82% randomized, validated
9. Sensor upgrade: 5 -> 7 rangefinders, angle sweep found 55 deg optimal
10. Hardware BOM ordered (including 2 extra VL53L0X for FL2/FR2)
11. Randomized benchmark infrastructure (Monte Carlo testing)
12. Encoder-pitch compensation (1-line fix in velocity estimation)
13. IMU-based collision detection (high-pass filter, zero extra hardware)
14. LQR optimal controller (replaces cascaded PID + 7 workarounds)
15. DWA local planner (replaces Bug2 FSM, 315 trajectory samples at 20 Hz)
16. WiFi RSSI localization + wall-geometry heading correction

### Next priorities

- [ ] Validate LQR stability (push tests, Q/R tuning)
- [ ] Benchmark DWA vs Bug2 on randomized corridors
- [ ] Tune collision detection threshold empirically
- [ ] Navigation v3 on apartment scene
- [ ] Height discrimination (obstacle traversability)
- [ ] Domain randomization (mass, friction, sensor noise ranges)
- [ ] Gymnasium/Farama integration
- [ ] MJPC experimentation
- [ ] Manipulator arm design
- [ ] RL training pipeline
- [ ] Sim-to-real transfer with physical hardware
