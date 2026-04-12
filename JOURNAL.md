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

## Session 9 -- 2026-04-12 : A* Local Planner as Bug2 Contour Helper

### Context

Session 8 ended with Bug2 v2 restored at 82 % after the DWA/TEB double failure. The post-mortem (`docs/teb_evaluation.md`) identified that Bug2's ~22 % of wall-gap failures come from its virtual scan being too short-sighted: 13 discrete rays cannot reason about a two-step detour. Bruno and Claude wrote `docs/plan_astar_local.md` proposing a classical A\* on the existing occupancy grid as a *helper* for the CONTOUR entry -- not a replacement for Bug2. The plan explicitly forbade touching the FSM, the pitch filter, or adding any reactive layer. Every step had a gate and a fallback.

### What was done

1. **`deskbot/astar_local.py` (new)** -- standalone 8-connected A\* with:
   - Octile heuristic (admissible for uniform 8-conn grids).
   - Chebyshev dilation of the log-odds mask by `INFLATE_CELLS = 2` (~16 cm, robot half-width 13 cm + 5 cm margin at 8 cm/cell).
   - No-corner-cutting rule: diagonal step requires both axial shoulder cells to be free.
   - Hard budget `max_iterations = 2000`, tie-breaker counter so the heap never compares tuples.
   - Helpers `nearest_free_cell` (Chebyshev-ring BFS to nudge a blocked start/goal) and `path_initial_tangent` (average direction over the first N cells).
   - Full mathematical specification in the module docstring (frames, cost, heuristic, complexity).
2. **`scripts/test_astar_local.py` (new)** -- 7 synthetic tests: empty grid straight line, wall with gap, gap closed by inflation, goal nudging, tangent sign around an asymmetric wall, budget timeout, full seal (no path). All pass.
3. **`deskbot/navigation.py`** -- minimal surgical changes:
   - New `use_astar` constructor flag (default False, preserves Bug2).
   - `_enter_contour` extracted into two helpers: `_contour_side_from_virtual_scan` (the legacy code, untouched) and `_contour_side_from_astar` (new). A\* builds its own `AStarPlanner` over `self.grid.grid`, plans from the robot's cell to a cell `LOCAL_HORIZON = 1.5 m` ahead along the target heading, nudges both endpoints out of inflated obstacles, then derives the contour side from the path tangent. On failure the function returns None and the caller silently falls back to the virtual scan.
   - Nothing else moved: FSM states, transitions, watchdogs, pitch filter, wall-follow controller are all byte-identical to session 8.
4. **`scripts/benchmark_random.py`** -- added `--planner {bug2,astar}` flag (default `bug2`), threaded into `run_episode` and both worker paths.
5. **`scripts/eval_mapping.py`** -- same `--planner` flag for the sanity check in step 7.
6. **`docs/benchmark_astar_v1.txt` (new)** -- full raw benchmark numbers, gate checklist, parameters.

### Step-by-step results

Each step had its own gate; the plan said to stop if A\* regressed by more than 5 points or if I was tempted to add a rustine.

- **Step 1 (A\* + unit tests)**. First draft had three bugs surfaced by the tests: an overly ambitious "narrow passage" test that actually expected no-path, a start cell inside its own inflation in the tangent test, and an ill-sealed no-path test. All three were fixture bugs, not planner bugs, and all tests pass after fixing them.
- **Step 2 (start/goal selection)**. Implemented as `nearest_free_cell` helper. Covered by the "goal nudged" test.
- **Step 3 (integration)**. Added `_contour_side_from_astar`. The threshold for side selection is ±10° off the heading; ambiguous tangents fall through to the sensor-biased virtual scan, preserving the existing tie-break logic.
- **Step 4 (A/B at 50 episodes, seed 42)**. First run with `path_initial_tangent(look_ahead=4)` regressed by 4 pts (82 % vs 86 %). Diagnosis: 4 cells = 32 cm is too short on an 8-connected staircase; a single diagonal near the start can flip the measured tangent and commit the wall-follower to the wrong side. Doubled the look-ahead to 10 cells (~80 cm). New result: **A\* 90 % vs Bug2 86 %** over 3 × 50 eps; **gate passed (+4 pts)**.
- **Step 5 (periodic replanning)**. **SKIPPED by design**. The plan allowed step 5 only as an enhancement once step 4 gate was met. A\* was already net positive without it; the plan's own red-line rule is "never rewrite validated nav"; adding replanning would have re-entered the territory of biasing the wall-follow P-controller -- exactly what broke TEB in session 8. Stayed with single-shot A\* at contour entry.
- **Step 6 (final benchmark, 100 eps × 3 runs seed 42)**. Bug2 mean 84.0 % (80/87/85), A\* mean **89.0 % (86/91/90)**, delta **+5 pts**, zero falls in every run, A\* variance (σ 2.6) tighter than Bug2 (σ 3.6). Run-to-run jitter comes from `SensorModel` not reseeding per episode -- noted as follow-up but out of scope.
- **Step 7 (mapping sanity check, `eval_mapping.py --episodes 15`)**. Aggregate IoU 0.380 (A\*) vs 0.384 (Bug2), -0.4 pt, inside the ±1 pt gate. A\* actually improves precision (0.399 vs 0.395) and lowers FP density (0.0796 vs 0.0841); recall drops 4 pts because A\*'s route hits a slightly smaller fraction of the GT surfaces in 15 s.

### What worked

- **Treating A\* as a helper, not a replacement.** The whole integration is one new method plus a flag. Bug2's FSM, watchdogs, and virtual scan are byte-identical. If A\* ever returns None (budget exceeded, start or goal blocked, tangent too ambiguous), the robot keeps its session-8 behavior. Zero regression surface.
- **Octile heuristic and binary inflation.** Admissible heuristic → optimal paths; binary inflation → debuggable. No ad-hoc continuous cost on obstacle proximity (the plan explicitly forbade it, and in hindsight it would have muddied the tangent direction signal).
- **Writing unit tests before benchmark.** The 4 → 10 cell look-ahead bug would have been painful to chase from benchmark numbers alone. The `test_tangent_choice_around_left_wall` test would have caught the wrong answer directly if I had wired it to compare magnitudes -- adding that to the test suite is low-hanging fruit.
- **Multiple runs per planner to characterize variance.** Running 3 × 50 episodes revealed that bug2 varies across 82-90 % on the same seed (because sensor noise is unseeded). A single-run gate would have been misleading.

### What failed / lessons learned

- **First look-ahead of 4 cells was wrong, and I would have missed it without the benchmark.** The planner was mathematically correct; the side-decision logic was not robust to the 8-conn staircase artefact. On a diagonal staircase the first segment can point 45° off from the bulk direction. 10 cells averages that out. **Transferable rule**: any "initial direction" extraction from a discrete grid path should integrate over a length that is large compared to the step size but small compared to the full plan.
- **Run-to-run variance is ±6 pts on 50 eps and ±3 pts on 100 eps** because `SensorModel` uses the process-global numpy RNG. Both planners see the same noise so A/B comparisons are valid, but absolute rates jitter. Follow-up: thread an `ep_seed`-derived RNG into `SensorModel` so benchmark results are reproducible.
- **The "murs vs hors-murs" split from the plan was not implemented.** The benchmark does not tag episodes by dominant obstacle type; adding it would have required a benchmark refactor outside the planner change. The global rate is comfortably above the gate and falls are zero, so I accepted that gap and documented it in `docs/benchmark_astar_v1.txt`. If a future session needs the split, a single field on `EpisodeResult` would suffice.
- **SKIP decision for step 5 is worth capturing**: periodic replanning was an *optional* plan step, conditional on the rest working. The rest *did* work at single-shot. Taking the step anyway would have been *gratuitous complexity*, exactly the anti-pattern session 8 warned about. This is the first time in this project that I stopped at "good enough" instead of chasing the next optimization; it felt uncomfortable, and the plan's explicit "stop if tempted to touch the FSM" rule is what anchored the decision.

### Claude's role

Read `docs/plan_astar_local.md`, `docs/teb_evaluation.md`, and the relevant parts of `deskbot/navigation.py`. Implemented the module, the tests, the integration, and the benchmark wiring. Hit and diagnosed the 4-cell look-ahead regression through the A/B gate rather than by guessing. Chose to skip step 5 rather than add optional complexity. The whole session was a straightforward plan execution with one surprise (look-ahead) that was caught by the gate as designed.

### Session 9 final state

- `deskbot/astar_local.py` (new, ~230 LOC) with 7 passing unit tests.
- `deskbot/navigation.py` gains `use_astar` flag and two small helper methods. Default remains `bug2`.
- `scripts/benchmark_random.py` and `scripts/eval_mapping.py` gain `--planner`.
- A\* score: **89 % global success @ 100 eps × 3 runs, 0 falls**, vs Bug2 **84 %** on the same conditions. Mapping IoU essentially unchanged (−0.4 pt, inside gate).
- Bug2 stays the *default planner* until a user opts in with `--planner astar`, pending a broader validation run on other seeds.

---

## Session 10 -- 2026-04-12 : Early Avoidance -- Nav Wins, Mapping Loses

### Context

Session 9 closed with Bug2 + A* at 89 % on 100 random episodes, 0 falls, and the `_contour_side_from_astar` helper only consulted once the robot crossed SAFE_DIST = 0.36 m. The remaining ~11 % of failures came from two patterns: tight chicanes where CONTOUR pivoted late, and narrow-gap walls where a two-step detour was needed. Bruno and Claude wrote `docs/plan_early_avoidance.md` proposing a *continuous* yaw bias in GO_HEADING, proportional to the inverse of the frontal distance, applied *before* the SAFE_DIST trigger. The intuition was that a tiny bias (a few deg/s at 2 m) integrated over ~2 m of approach would bend the trajectory by ~15-25 degrees and steer past the obstacle without ever engaging CONTOUR. The plan also expected the continuous rotation to sweep the FC ray laterally and densify the grid -- a free mapping boost on the side.

### What was done

1. **`deskbot/early_avoidance.py` (new)** -- pure, stateless guidance law plus a `EarlySideCache` dataclass and a tick-level `update_side_cache` helper. All five constants (`k`, `d_sat`, `d_trig`, `d_cut`, `t_hold`, `t_clear`) documented with physical justification in the module docstring. No new dependency.
2. **`scripts/test_early_avoidance.py` (7 unit tests)** on the pure law: cutoff, saturation, trigger handoff, sign, monotonicity. All pass.
3. **`scripts/test_early_side_state.py` (8 unit tests)** on the side cache state machine: FSM gating, hold hysteresis, clear dwell, intermittent noise, handoff below `d_trig`. All pass.
4. **`deskbot/navigation.py`** -- surgical additions:
   - Constructor flag `use_early_avoid` + optional `early_k` override for the step 6 sweep.
   - `_update_early_side(rf, dt)` called once per tick after the occupancy-grid update.
   - `_early_side_picker(rf)` (new direct-rangefinder selector -- see "What failed").
   - One additive term in `_state_go_heading`, gated by `self._use_early_avoid and self._early_cache.side is not None`. All existing Bug2 + A* code paths are byte-identical when the flag is off.
5. **`scripts/benchmark_random.py`, `scripts/eval_mapping.py`** -- both gain `--early-avoid {off,on}`; the random benchmark also gains `--early-k FLOAT` for the sweep. Defaults `off`.
6. **`scripts/benchmark_chicanes.py` (new)** -- chicane-only variant of the random benchmark for step 5.
7. **`scripts/diag_early_avoid.py` (new)** -- per-tick debug tracer used to diagnose the side-selector bug described below.
8. **`docs/benchmark_early_avoid_v1.txt` (new)** -- full raw numbers for every gate.

### Step-by-step results

- **Step 0 (flag-neutral baseline)**. Three runs of `--planner astar` without the flag (87/87/88 %, mean 87.3) and three with `--early-avoid off` (88/88/90, mean 88.7). Delta +1.4 pts within the documented ±3-5 pt SensorModel-noise variance. Gate passed.
- **Step 1 (pure law)**. 7/7 unit tests pass at first try.
- **Step 2 (cache state machine)**. 8/8 tests pass. Hysteresis + clear dwell behave as specified.
- **Step 3 (smoke run)**. First 3-episode smoke run was terrible: 1/3 success, 14+ contours on the successful episode (3× baseline), 2 failures. The plan's fallback clause said "stop and diagnose". I did.
- **Diagnosis (session 10's surprise).** `scripts/diag_early_avoid.py` traced seed 42 tick by tick and revealed that `_contour_side_from_virtual_scan` returns *systematically* the same side (-1) at long range for two compounding reasons:
  1. `OccupancyGrid.clearance_in_direction` walks at most `GRID_RAY_MAX = 1.5 m`, and discretized ray cells miss any obstacle beyond ~1.44 m. So at 1.5-2.0 m *all 13 scan directions* tie at max clearance, collapsing to the straight-ahead tie-breaker.
  2. The tie-breaker uses `fl - fr` with undetected rays (`-1`) coerced to `0 m`. A grazing detection on FL plus no detection on FR then reads as "left has obstacle at 1.7 m, right has obstacle at 0 m -- right is CLOSER", so the code returns "go left" -- into the obstacle.

   On seed 42 the obstacle box was at y=+0.52, exactly to the left of the robot's path. The biased trajectory pushed the robot straight into it. Classic sign/convention inversion hidden by a subtle boundary condition in a validated helper.
- **Step 3 fix (formulation change, not a rustine)**. Added `_early_side_picker(rf)` -- a direct-rangefinder selector that weighs `rf_FL/FL2` vs `rf_FR/FR2` by inverse distance, treats undetected rays as *no information* (weight 0), and returns **None** when there is no asymmetry. The guarantee: early avoidance only picks a side when it has actual evidence. If the only thing visible is `rf_FC` (narrow obstacle straight ahead), the cache stays unlocked, no bias is applied, and the robot continues on its straight path until CONTOUR takes over. This respects the plan's "correct the formulation, don't add a rustine" rule: `_contour_side_from_virtual_scan` is *not touched*, it's simply bypassed by early avoidance in favour of a more honest-at-long-range selector. Re-running the seed 42 diag confirmed: the robot curves smoothly right around obstacle 1, then again around obstacle 2, with zero contour entries in 12 sim-seconds.
- **Step 4 (real A/B benchmark, 3 × 100 eps per configuration)**. Seed 42: off 88/88/90 → on 90/91/91 (+2.0 pts mean). Seed 43: off 85/85/86 → on 92/93/91 (+6.7 pts mean). Zero falls in all 12 runs (1200 episodes). Stuck recoveries down 20-30 %. Gate passed cleanly on both seeds.
- **Step 5 (chicane-only benchmark)**. 30 forced chicane scenes at seed 1000. Off: 29/30 with avg 2.2 contours. On: 30/30 with avg 1.1 contours. The single `off` failure was absorbed by early avoidance steering the robot past the first box with a better alignment for the second. Gate passed.
- **Step 6 (k sweep, 3 × 100 eps at seed 42)**. k=0.05 mean 91.7, k=0.08 mean 90.7, k=0.12 mean 92.3. All three pass ≥ 89 %, zero falls everywhere. The gaps between means (max 1.6 pts) are inside the inter-run variance (σ 2.4-2.5 pts for k=0.05/0.12, σ 0.5 for k=0.08). Retained **k = 0.08** because it is the only value also validated at seed 43 (from step 4) and because it has the lowest run-to-run sigma. A wider validation at seed 43 for the other k values would have needed another 24 min of compute to beat the noise floor.
- **Step 7 (mapping eval, 15 eps at seed 42)**. This is where the story turns. IoU 0.372 → 0.327 (**-0.045**, gate was -0.010). Recall 0.923 → 0.856 (**-0.067**, gate wanted +0.020). Precision 0.384 → 0.346. FP density 0.0821 → 0.0882. Early avoidance *regresses* the grid quality by more than 2 IoU points -- the plan's explicit stopping criterion. Per-episode deltas: 6 improved, 9 worsened; one severe outlier (seed 47, IoU 0.103 vs 0.381 off), but the mean regression holds even without it.

### What worked

- **Diagnosing before patching.** 3-episode smoke result (14 contours vs baseline 5) was a clear red flag. Instead of tuning `k` down or inventing a safety layer, I built `diag_early_avoid.py` with a tick-level trace and *watched* a single seed. Found the exact double-bug in `_contour_side_from_virtual_scan` in ten minutes. The plan's explicit "5 interdictions" kept me honest: no reactive filter, no safety layer, no FSM touch.
- **The honest-information selector.** `_early_side_picker` returns `None` when the picker has nothing to go on. That `None` result propagates naturally through `update_side_cache` into "no bias", which means the system degrades gracefully to plain Bug2 whenever the long-range info is ambiguous. A wrong decision at 1.8 m compounds over a whole approach; no decision is strictly better.
- **Cache hysteresis with an escape.** `T_hold` keeps the bias stable on near-symmetric obstacles (session 8 taught that oscillating side choices cause massive damage). `T_clear` lets the cache forget a past obstacle once the robot has cleared it, so the next chicane gets a fresh decision. Chicane benchmark (+1 seed, avg contours halved) is the visible payoff of that balance.
- **Parameterising `k` through `dataclasses.replace`.** Adding `--early-k FLOAT` to the benchmark threaded through `Navigator.early_k` into a one-off `EarlyAvoidanceParams` instance via `dataclasses.replace` -- 6 lines total, no module-level mutation, no monkey-patch, fully compatible with the `ProcessPoolExecutor` workers.

### What failed / lessons learned

- **The plan's mapping theory did not survive contact with reality.** The expected "continuous rotation sweeps the FC ray sideways, densifies the grid" was wrong on two counts. (i) The retained bias k=0.08 peaks at 0.16 rad/s -- roughly 2-3° of body-yaw deviation over a typical approach, not enough to meaningfully re-aim FC. (ii) Eliminating CONTOUR events is exactly what removes the main source of *wide-angle* grid updates: CONTOUR wall-follows sweep the side lasers across an obstacle's entire extent. Fewer contours = fewer viewpoints = lower recall. Navigation and mapping are in tension, not synergy, and early avoidance buys nav success at the cost of mapping fidelity. The theory ignored that the grid's recall budget was being funded by contour events, not by straight-line cruising.
- **Signature bug in a session-9 helper, surfaced in a regime where session 9 never operated.** `_contour_side_from_virtual_scan` treats undetected `-1` rays as `fr_val = 0 m` in its tie-break. At close range (SAFE_DIST) both FL and FR usually see the obstacle, so the bug is invisible. At 1.5-2.0 m, only one side sees the obstacle and the bug inverts the answer. Session 9's A* version did not exercise the long-range regime either. The lesson: a helper is only validated *within the regime it was tested in*. Re-using it in a new regime (early avoidance's 1-2 m range) re-exposes hidden assumptions.
- **Choice between k=0.08 and k=0.12 is noise.** On seed 42 alone, k=0.12 scores higher mean (+1.6 pts) but with 4× the inter-run sigma. Picking k=0.12 to "squeeze out the last point" would have been sweep-hacking: a 1.6-pt difference on three runs is indistinguishable from the 3-5 pt SensorModel variance. The rational call was k=0.08 because it is also validated on seed 43 and has the tightest distribution. This was the harder call of the session because the mean comparison was psychologically tempting.
- **Pedagogical moment on sign conventions.** The plan's literal formula `ω_bias = k · side · 1/D` expects `side = +1` → positive yaw → turn left. But `_contour_side_from_virtual_scan` uses the opposite convention (`+1` = "obstacle on left, go right" = negative yaw). Resolved by documenting the convention explicitly in `compute_bias_yaw` and negating once in `_update_early_side` via `-contour_side`. Then the bug above forced me to replace the selector entirely, but the resolution pattern -- "keep the pure law in one convention, negate at the integration boundary once with a comment" -- is the cleanest way to handle sign mismatches without polluting either side.

### Claude's role

Read `docs/plan_early_avoidance.md` end to end before touching code. Built the pure law + its tests before integration. Hit the 14-contour smoke regression, resisted the temptation to tune `k` down, built a tick-level diagnostic, found the dual root cause in 10 minutes, and replaced *only* the side selector -- not any watchdog, not `_state_contour`, not the pitch filter, not the LQR. All four of the plan's absolute "do not touch" lines were respected. When the mapping gate failed at step 7, did not retry with a bigger k or a wider margin; documented the regression honestly and kept the flag `off` by default per the plan's fallback clause.

### Session 10 final state

- `deskbot/early_avoidance.py` (new, ~160 LOC) with 7 + 8 = 15 passing unit tests.
- `deskbot/navigation.py` gains a side picker, a cache updater, a one-line additive bias in `_state_go_heading`, and a `use_early_avoid` constructor flag (~80 LOC added, zero existing lines modified in the already-validated paths).
- `scripts/benchmark_random.py`, `scripts/benchmark_chicanes.py` (new), `scripts/eval_mapping.py`, `scripts/diag_early_avoid.py` (new), `scripts/test_early_avoidance.py` (new), `scripts/test_early_side_state.py` (new).
- Navigation success: **seed 42 88.7 → 90.7 (+2.0)**, **seed 43 85.3 → 92.0 (+6.7)**, **chicanes 97 → 100**, **0 falls across 1200 + 60 + 600 + ~450 episodes**, contours -12 %, stuck recoveries -28 %.
- Mapping: **IoU 0.372 → 0.327 (-0.045)**, **recall 0.923 → 0.856 (-0.067)**. Mapping gate fails.
- Default remains `--early-avoid off` (Bug2 + A* from session 9). Feature ships as opt-in. A future session could explore either raising `k` enough to produce real angular sweep (stability TBD) or re-introducing a targeted CONTOUR-alike sweep whenever the robot enters the 0.6-1.2 m range, but neither is in scope for this session.

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
17. A\* local planner as Bug2 CONTOUR helper (opt-in, +5 pts over Bug2 on 100 eps)
18. Early avoidance guidance law (opt-in, +2/+7 pts over A\* on seed 42/43, 100 % on chicanes, but mapping IoU regresses -4.5 pts so it stays opt-in)

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
