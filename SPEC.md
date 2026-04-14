Below is a concrete implementation brief you can hand to Codex.

On the X-Plane side, build the deterministic controller around the SDK’s dataref and flight-loop model: plugins can read and write datarefs, can publish their own custom datarefs, and X-Plane includes sample code showing use of override datarefs to disable internal control logic. Flight-loop callbacks can run before or after the flight model; the SDK docs note that the callback timing parameters are not especially useful, so you should track your own timing. Also, X-Plane exposes a way to check whether a dataref is writable before relying on it. ([X-Plane Developer][1])

For longitudinal control, use a TECS-like split from the start. ArduPilot’s Plane docs describe the core idea cleanly: total aircraft energy is potential plus kinetic energy, throttle is used to manage total energy, and pitch is used to manage the balance between speed and height. That is the right abstraction for climb, descent, approach, and go-around. ([ArduPilot.org][2])

## The architecture

Use four layers.

1. **LLM pilot**
   Reads ATC or mission goals and emits structured commands only.

2. **Procedure executive**
   Deterministic state machine. Owns flight phases, legal transitions, safety rules, and interpretation of things like “extend downwind” and “turn base now.”

3. **Guidance / flight director**
   Converts active procedure state into continuous targets:

   * target ground track or path
   * target altitude / vertical path
   * target speed
   * target configuration
   * target flare profile

4. **Low-level control**
   Deterministic controllers that turn those targets into actuator commands:

   * aileron / rudder
   * elevator / trim
   * throttle
   * brakes / flaps / gear

The LLM must never command raw control surface deflections.

---

## What Codex should build

### Repo layout

```text
sim_pilot/
  README.md
  pyproject.toml

  config/
    aircraft_c172.yaml
    airport_defaults.yaml
    controller_gains.yaml
    safety_limits.yaml

  sim/
    xplane_bridge.py
    datarefs.py
    commands.py
    state_reader.py
    actuator_writer.py
    timing.py

  core/
    types.py
    blackboard.py
    state_estimator.py
    mode_manager.py
    safety_monitor.py
    event_bus.py
    mission_manager.py

  guidance/
    lateral.py
    vertical.py
    route_manager.py
    pattern_manager.py
    runway_geometry.py
    approach_manager.py
    flare_profile.py

  control/
    pid.py
    bank_hold.py
    heading_hold.py
    track_hold.py
    l1_path_follow.py
    pitch_hold.py
    altitude_hold.py
    vs_hold.py
    speed_hold.py
    tecs_lite.py
    centerline_rollout.py

  procedures/
    takeoff.py
    climb.py
    cruise.py
    descent.py
    pattern.py
    approach.py
    landing.py
    go_around.py

  llm/
    command_schema.py
    parser.py
    pilot_agent.py
    atc_adapter.py
    prompt_templates/

  tests/
    test_pattern_geometry.py
    test_mode_transitions.py
    test_guidance_lateral.py
    test_tecs_lite.py
    test_flare_logic.py
    test_safety_monitor.py
```

---

## Core rule

Every layer talks to the next layer through typed objects.

### LLM output schema

```python
class PilotIntent(BaseModel):
    action: Literal[
        "follow_route",
        "direct_to",
        "fly_heading",
        "maintain_altitude",
        "maintain_speed",
        "join_pattern",
        "extend_downwind",
        "turn_base_now",
        "cleared_to_land",
        "go_around",
        "enter_hold",
        "change_runway",
    ]
    params: dict
    confidence: float
    source: Literal["mission", "atc", "operator"]
```

Examples:

```json
{"action":"direct_to","params":{"fix":"SMO"}}
{"action":"join_pattern","params":{"runway":"25L","side":"left","entry":"45_downwind","pattern_alt_ft":1800}}
{"action":"extend_downwind","params":{}}
{"action":"turn_base_now","params":{}}
{"action":"go_around","params":{"runway":"25L","heading_mode":"runway_track","climb_to_ft":3000}}
```

### Guidance contract

```python
class GuidanceTargets(BaseModel):
    lateral_mode: Literal[
        "bank_hold",
        "heading_hold",
        "track_hold",
        "path_follow",
        "centerline_intercept",
        "rollout_centerline"
    ]
    vertical_mode: Literal[
        "pitch_hold",
        "altitude_hold",
        "vs_hold",
        "fpa_hold",
        "tecs",
        "glidepath_track",
        "flare_track"
    ]
    target_bank_deg: float | None = None
    target_heading_deg: float | None = None
    target_track_deg: float | None = None
    target_path: dict | None = None
    target_altitude_ft: float | None = None
    target_vs_fpm: float | None = None
    target_fpa_deg: float | None = None
    target_speed_kt: float | None = None
    target_glideslope_deg: float | None = None
    target_centerline: dict | None = None
    flare_profile: dict | None = None
    flaps_cmd: int | None = None
    gear_down: bool | None = None
    throttle_limit: tuple[float, float] | None = None
```

### Control output contract

```python
class ActuatorCommands(BaseModel):
    aileron: float      # -1..1
    elevator: float     # -1..1
    rudder: float       # -1..1
    throttle: float     # 0..1
    flaps: int | None
    gear_down: bool | None
    brakes: float       # 0..1
```

---

## State estimator

Codex should create a single normalized aircraft state object.

```python
class AircraftState(BaseModel):
    t_sim: float
    dt: float

    lat_deg: float
    lon_deg: float
    alt_msl_ft: float
    alt_agl_ft: float | None

    pitch_deg: float
    roll_deg: float
    yaw_deg: float

    p_rad_s: float
    q_rad_s: float
    r_rad_s: float

    ias_kt: float
    tas_kt: float
    gs_kt: float
    vs_fpm: float

    track_deg: float
    heading_deg: float
    alpha_deg: float | None
    beta_deg: float | None

    wind_dir_deg: float | None
    wind_speed_kt: float | None

    flap_index: int
    gear_down: bool
    on_ground: bool
    wow_main: bool | None

    throttle_pos: float
    runway_id: str | None
    runway_dist_remaining_ft: float | None
```

The state estimator should also compute:

* cross-track error to active path
* along-track distance
* runway-relative coordinates
* local frame `(x_forward, y_right, z_up)` in runway axes
* whether threshold is abeam
* distance to touchdown
* estimated sink energy at touchdown
* stall margin = `ias / vso_current_config`

---

## Mode manager

Implement the deterministic flight state machine first.

```python
enum FlightPhase:
    PREFLIGHT
    TAKEOFF_ROLL
    ROTATE
    INITIAL_CLIMB
    ENROUTE_CLIMB
    CRUISE
    DESCENT
    PATTERN_ENTRY
    DOWNWIND
    BASE
    FINAL
    ROUNDOUT
    FLARE
    ROLLOUT
    TAXI_CLEAR
    GO_AROUND
```

### Transition rules

Codex should encode hard rules, not “AI judgment.”

Examples:

* `PREFLIGHT -> TAKEOFF_ROLL`

  * runway aligned
  * brakes released
  * throttle advanced
  * clearance state true

* `TAKEOFF_ROLL -> ROTATE`

  * IAS > `Vr`
  * centerline deviation < limit

* `ROTATE -> INITIAL_CLIMB`

  * positive climb
  * radio alt or AGL > 20 ft

* `INITIAL_CLIMB -> ENROUTE_CLIMB`

  * AGL > 400 ft
  * climb stabilized

* `PATTERN_ENTRY -> DOWNWIND`

  * intercept downwind leg
  * track error < threshold
  * altitude within band

* `DOWNWIND -> BASE`

  * either automatic trigger:

    * threshold abeam and extension satisfied
  * or explicit ATC trigger:

    * `turn_base_now`

* `BASE -> FINAL`

  * final centerline intercept feasible
  * localizer/centerline error decreasing
  * no excessive overshoot predicted

* `FINAL -> ROUNDOUT`

  * height AGL < `roundout_height_ft`

* `ROUNDOUT -> FLARE`

  * sink rate and height inside flare gate

* `FLARE -> ROLLOUT`

  * main wheels on ground

* `ANY -> GO_AROUND`

  * commanded by LLM / operator
  * unstable approach
  * sink rate too high below gate
  * excessive lateral error below gate
  * too fast / too slow below gate

---

## Guidance design

This is the most important piece.

### Lateral guidance primitives

Codex should implement these:

```python
class StraightLeg:
    start_xy: Vec2
    end_xy: Vec2

class ArcLeg:
    center_xy: Vec2
    radius_ft: float
    start_angle_deg: float
    end_angle_deg: float
    turn: Literal["left","right"]

class RunwayCenterline:
    threshold_xy: Vec2
    course_deg: float

class DownwindOffsetLeg:
    runway_course_deg: float
    threshold_xy: Vec2
    offset_ft: float
    side: Literal["left","right"]
    length_ft: float
```

Then a lateral path follower that accepts one active primitive and outputs:

* desired track
* desired bank, or
* target intercept point

Use an L1-style or lookahead path follower:

* compute cross-track error
* compute along-track velocity
* choose lookahead distance proportional to groundspeed
* command bank from path curvature plus cross-track correction

Do not start with heading-hold-only navigation. It falls apart in wind.

### Vertical guidance primitives

Implement:

```python
class AltitudeConstraint:
    target_alt_ft: float
    tolerance_ft: float

class Glidepath:
    threshold_crossing_height_ft: float
    slope_deg: float
    aimpoint_ft_from_threshold: float

class SpeedSchedule:
    segments: list[tuple[FlightPhase, float]]  # target IAS by phase
```

For the MVP:

* takeoff / climb: TECS with target speed + target altitude
* cruise: TECS with hold altitude + target speed
* descent: TECS with altitude target or FPA target + speed
* final: glidepath tracking + approach speed
* flare: dedicated flare profile

---

## Low-level loops

### Lateral control stack

Use this chain:

```text
path follower -> desired bank angle -> bank controller -> aileron
                                       + coordination helper -> rudder
```

Implement:

```python
class BankController:
    def update(target_bank_deg, roll_deg, p_rad_s, dt) -> float: ...

class CoordinationController:
    def update(target_bank_deg, roll_deg, yaw_rate, beta_deg, dt) -> float: ...
```

MVP version:

* PID on bank error with roll-rate damping
* rudder from slip / yaw damping, or simple coordinated-turn feedforward

### Vertical control stack

Use TECS-lite:

```text
height/speed targets -> TECS-lite -> desired pitch + desired throttle
desired pitch -> pitch controller -> elevator
desired throttle -> throttle controller
```

Implement:

```python
class TECSLite:
    def update(
        target_alt_ft,
        target_speed_kt,
        alt_ft,
        vs_fpm,
        ias_kt,
        pitch_deg,
        throttle_pos,
        dt
    ) -> tuple[float, float]:
        ...
```

### TECS-lite behavior

Use these ideas:

* total energy error ≈ altitude error + speed error term
* throttle responds mostly to total energy error
* pitch responds mostly to energy balance error
* clamp pitch and throttle by flight phase
* below approach gate, bias pitch toward glidepath tracking and throttle toward speed control

A practical simple formulation:

```python
e_h = target_alt_ft - alt_ft
e_v = target_speed_kt - ias_kt

# throttle controls total energy
e_total = k_h_total * e_h + k_v_total * e_v
throttle_cmd = throttle_trim + kp_t * e_total + ki_t * int_e_total

# pitch controls balance: trade height and speed
e_balance = k_h_bal * e_h - k_v_bal * e_v
pitch_cmd = pitch_trim + kp_p * e_balance + kd_p * (-vs_fpm)
```

That is not full TECS, but it is enough to get a stable first system.

### Pitch / bank loops

Implement classical inner loops:

```python
aileron = Kp_bank * bank_err + Kd_bank * roll_rate_err + Ki_bank * bank_int
elevator = Kp_pitch * pitch_err + Kd_pitch * q_rate_err + Ki_pitch * pitch_int
```

Use anti-windup and rate limits.

---

## Pattern manager

Treat the pattern as explicit geometry in runway coordinates.

### Runway frame

Given:

* runway threshold lat/lon
* runway heading
* pattern altitude
* traffic side
* downwind offset

Build a local 2D frame:

* `x`: runway heading direction
* `y`: right of runway
* threshold at `(0,0)`

Then define:

* upwind line along `x`
* crosswind transition
* downwind line at `y = +/- offset`
* base leg from downwind toward centerline
* final centerline aligned with runway

### Pattern entry types

Implement:

* straight-in
* crosswind join
* midfield crosswind join
* 45 to downwind

For the MVP, fully support only:

* 45 to downwind
* straight-in

### Downwind logic

Downwind is not “hold heading runway+180.”
Downwind is:

* follow a line offset from runway centerline
* maintain pattern altitude
* maintain pattern speed
* monitor abeam threshold condition

Define:

```python
abeam = abs(x_threshold_relative) < abeam_window_ft
```

Base turn may occur when:

* `abeam == True`
* extension distance satisfied
* stable speed and altitude
* explicit base delay not active

### Extend downwind

This should only modify one state variable:

```python
pattern_state.base_clearance = False
```

The path follower continues on the downwind leg. No other logic changes.

### Turn base now

This sets:

```python
pattern_state.base_clearance = True
pattern_state.force_base_now = True
```

Then the guidance layer generates a feasible turning intercept from current state to either:

* a nominal base leg, or
* directly to final if geometry says base is too short

---

## Approach and landing

Landing should be a separate procedure module.

### Final approach

Targets:

* centerline
* glidepath
* target IAS = `Vref + additive`

Use:

* lateral: centerline intercept / track
* vertical: glidepath tracking
* throttle: speed hold or TECS approach mode
* flaps / gear: schedule by distance or phase

### Roundout

At `h_agl <= roundout_height_ft`:

* stop chasing the geometric glidepath
* transition to a commanded descent-rate reduction
* gradually reduce sink rate
* smoothly raise pitch toward flare reference

### Flare

Use a dedicated flare profile:

Inputs:

* radio altitude or AGL estimate
* sink rate
* IAS error vs touchdown target
* pitch
* runway slope if available

Outputs:

* target pitch
* throttle to idle
* small centerline corrections

A usable first law:

```python
if h_agl < flare_start_ft:
    throttle_cmd = 0
    target_pitch = flare_pitch_schedule(h_agl, sink_rate, ias_err)
```

Where `flare_pitch_schedule` increases pitch as height decreases, but is limited by:

* max flare pitch
* minimum stall margin
* maximum pitch rate

### Rollout

After touchdown:

* maintain runway centerline with rudder / nosewheel steering
* aileron into crosswind if modeled later
* brakes based on rollout speed
* exit centerline control once groundspeed below threshold

---

## Safety monitor

Codex should implement a hard safety layer that can override the procedure logic.

Triggers:

* bank too high for phase
* stall margin too low
* sink rate too high below 200 ft AGL
* runway alignment too poor below 300 ft AGL
* excessive float beyond touchdown zone
* no landing clearance and runway incursion logic if you add traffic later

Actions:

* clamp bank
* clamp pitch
* inhibit flap extension
* command go-around
* reject landing and re-enter climb mode

Example rules:

```python
if phase in {BASE, FINAL} and abs(roll_deg) > 30:
    bank_limit_active = True

if phase == FINAL and alt_agl_ft < 200 and abs(centerline_error_ft) > 100:
    request_go_around("unstable_lateral")

if phase == FINAL and alt_agl_ft < 100 and stall_margin < 1.15:
    request_go_around("low_energy")

if phase == FINAL and abs(glidepath_error_ft) > max_glidepath_error_ft:
    request_go_around("unstable_vertical")
```

---

## ATC design

The LLM should be an interpreter, not the autopilot.

### Input to LLM

Give it:

* current phase
* airport / runway / route context
* active constraints
* last ATC message
* a short list of legal actions

### Output from LLM

One structured action only, chosen from the schema.

Examples:

* “Extend your downwind, I’ll call your base.”

  * `{"action":"extend_downwind","params":{}}`

* “Turn base now, cleared to land runway 27.”

  * `{"action":"turn_base_now","params":{}}`
  * `{"action":"cleared_to_land","params":{"runway":"27"}}`

* “Fly heading 310, maintain 2500.”

  * `{"action":"fly_heading","params":{"heading_deg":310}}`
  * `{"action":"maintain_altitude","params":{"altitude_ft":2500}}`

### Legal-action filtering

Do not let the LLM emit every command in every phase.

Examples:

* in `FINAL`, legal actions might be:

  * `go_around`
  * `maintain_speed`
  * `change_runway` only if missed approach
* in `DOWNWIND`, legal actions might be:

  * `extend_downwind`
  * `turn_base_now`
  * `maintain_altitude`
  * `maintain_speed`

This matters a lot.

---

## Recommended control update rates

Use separate loops.

* State read / estimation: 20–50 Hz
* Inner control loops: 20–50 Hz
* Guidance recompute: 5–10 Hz
* Procedure state machine: 5–10 Hz
* LLM / ATC interpretation: event-driven, not periodic
* Logging: 5–20 Hz

Do not call the LLM every frame.

---

## X-Plane integration strategy

Use a thin sim bridge and keep the flight logic outside it.

### Preferred layout

* **Plugin bridge**

  * reads/writes datarefs
  * owns flight-loop callback
  * exposes normalized state
  * accepts actuator commands
  * can expose custom datarefs for debugging

* **Pilot core process**

  * guidance, control, procedure logic
  * can run in Python for faster iteration

If you want the simplest first pass, you can keep everything in one process. But architecturally, separating the X-Plane bridge from the pilot core will save you later.

### Important implementation choices

* compute your own `dt` from sim time
* use “before flight model” writes for actuator commands
* reserve “after flight model” reads for logging / derived analysis if needed
* do not rely on X-Plane autopilot modes except maybe as a reference during debugging

The reason is simple: you want one source of truth for control logic.

---

## Configuration

Create an aircraft config file with:

```yaml
aircraft: c172
vr_kt: 55
vx_kt: 62
vy_kt: 74
vapp_kt: 65
vref_kt: 61

pattern:
  altitude_agl_ft: 1000
  downwind_offset_ft: 3500
  abeam_window_ft: 750
  default_extension_ft: 0

limits:
  max_bank_enroute_deg: 25
  max_bank_pattern_deg: 20
  max_bank_final_deg: 15
  max_pitch_up_deg: 12
  max_pitch_down_deg: 10
  min_stall_margin: 1.2

flare:
  roundout_height_ft: 20
  flare_start_ft: 10
  max_flare_pitch_deg: 9

controllers:
  bank:
    kp: 0.08
    kd: 0.03
    ki: 0.01
  pitch:
    kp: 0.10
    kd: 0.04
    ki: 0.01
  tecs:
    kp_total: 0.003
    ki_total: 0.0008
    kp_balance: 0.02
    kd_balance: 0.002
```

---

## Minimum viable milestones

### Milestone 1

No LLM yet.

Implement:

* takeoff roll
* rotate
* climb to target altitude
* direct-to waypoint navigation
* descent to pattern altitude
* 45-to-downwind pattern entry
* downwind / base / final
* flare and rollout
* one airport, one aircraft

### Milestone 2

Add:

* structured command interface
* ATC instruction parser
* `extend_downwind`
* `turn_base_now`
* `go_around`

### Milestone 3

Add:

* cross-country route manager
* direct-to fix
* heading vectors
* altitude changes
* runway changes
* missed approach / re-sequencing

---

## Acceptance tests

Codex should build test harnesses for these.

### Geometry tests

* downwind line generated on correct side
* base turn point moves correctly after extension
* final centerline intercept remains feasible

### Mode transition tests

* no skipping from downwind directly to flare
* go-around can be triggered from base/final/flare
* turn-base-now works even after long extension

### Control tests

* direct-to waypoint converges in crosswind
* altitude hold after climb settles without large oscillation
* final approach speed remains inside band
* flare touchdown sink rate below threshold

### Scenario tests

* take off, fly 15 nm direct-to, join pattern, land
* same mission with 10 kt crosswind
* extend downwind by 20 seconds, then base now
* unstable final triggers go-around
* cross-country with one mid-flight heading vector

### Success metrics

For the first version, something like:

* touchdown within first third of runway
* centerline error at touchdown < 20 ft
* approach speed within ±5 kt on short final
* no bank exceedance in final
* no stall warning in normal landing
* no oscillatory pilot-induced “porpoising”

---

## The concrete behavior by phase

This is the part I would explicitly hand to Codex.

### TAKEOFF_ROLL

Guidance:

* runway centerline hold
* throttle max
* pitch neutral until Vr

Controllers:

* lateral: rollout centerline controller
* vertical: none
* throttle: open-loop max

Exit:

* IAS >= Vr

### ROTATE

Guidance:

* target pitch to initial climb attitude
* maintain centerline track

Controllers:

* pitch hold
* lateral rollout / then track hold

Exit:

* positive climb and AGL > 20 ft

### INITIAL_CLIMB

Guidance:

* runway track
* Vy speed
* climb to safe altitude

Controllers:

* track hold or centerline extension
* TECS-lite

Exit:

* AGL > 400 ft or mission leg active

### ENROUTE / CRUISE

Guidance:

* path follow or direct-to
* target altitude
* target cruise IAS

Controllers:

* L1 path follower
* TECS-lite

Exit:

* arrival setup begins

### PATTERN_ENTRY

Guidance:

* intercept chosen pattern entry geometry
* descend or level at pattern altitude
* pattern speed

Controllers:

* path follow
* TECS-lite

Exit:

* on downwind geometry

### DOWNWIND

Guidance:

* downwind offset line
* pattern altitude
* pattern speed
* flap schedule arm

Controllers:

* path follow
* TECS-lite

Exit:

* base clearance true and base geometry feasible

### BASE

Guidance:

* base intercept leg
* descending toward final intercept altitude
* slower approach speed

Controllers:

* path follow
* TECS-lite with descent bias

Exit:

* final intercept captured

### FINAL

Guidance:

* runway centerline
* glidepath
* approach speed

Controllers:

* centerline intercept / track
* glidepath track
* throttle speed control or TECS approach mode

Exit:

* roundout gate reached

### ROUNDOUT

Guidance:

* reduce sink
* transition from glidepath to runway-referenced height profile

Controllers:

* roundout controller
* throttle idle transition

Exit:

* flare gate reached

### FLARE

Guidance:

* touchdown attitude / sink-rate reduction
* maintain centerline

Controllers:

* flare controller
* centerline rudder control

Exit:

* weight on wheels / touchdown

### ROLLOUT

Guidance:

* runway centerline
* decelerate

Controllers:

* rudder / steering
* brakes

Exit:

* safe taxi speed

### GO_AROUND

Guidance:

* full power
* climb attitude
* runway track or assigned heading
* flaps to intermediate
* positive climb cleanup

Controllers:

* TECS climb mode
* track hold

Exit:

* stable climb and new instruction available

---

## What not to do

Do not have Codex implement:

* LLM deciding control surface deflection
* heading-hold-only navigation in wind
* generic altitude hold reused as flare logic
* phase transitions inferred by the LLM
* a single giant controller that does all phases
* direct dependence on X-Plane’s built-in autopilot modes

---

## Prompt you can paste to Codex

```text
Build a deterministic fixed-wing sim pilot for X-Plane with these layers:

1. A typed command interface for a future LLM pilot. The LLM is not allowed to command raw actuators.
2. A deterministic mode manager / procedure executive with phases:
   PREFLIGHT, TAKEOFF_ROLL, ROTATE, INITIAL_CLIMB, ENROUTE_CLIMB, CRUISE, DESCENT, PATTERN_ENTRY, DOWNWIND, BASE, FINAL, ROUNDOUT, FLARE, ROLLOUT, GO_AROUND.
3. A guidance layer that outputs GuidanceTargets for lateral/vertical navigation.
4. A low-level control layer that outputs ActuatorCommands.

Implement the following modules:
- state_estimator
- mode_manager
- route_manager
- pattern_manager
- runway_geometry
- lateral guidance with an L1-style path follower
- TECS-lite longitudinal control
- bank and pitch PID loops
- glidepath/final approach controller
- flare controller
- safety monitor

Implement one complete scenario:
take off, climb, fly direct for ~15 nm, join 45-to-downwind, fly the pattern, land.

Implement support for these future structured pilot intents:
direct_to, fly_heading, maintain_altitude, maintain_speed, join_pattern, extend_downwind, turn_base_now, go_around.

Use clean Python modules and type annotations. Put all aircraft-specific values in YAML config. Add unit tests for geometry, mode transitions, and control stability. Do not let any LLM code command raw aileron/elevator/rudder/throttle.

Assume a C172-like aircraft for the first pass.
```

If you want, I can turn this into a second version that is even more Codex-oriented: a literal implementation spec with class definitions, function signatures, and pseudocode for each module.

[1]: https://developer.x-plane.com/sdk/XPLMDataAccess/ "XPLMDataAccess | X-Plane Developer"
[2]: https://ardupilot.org/plane/docs/tecs-total-energy-control-system-for-speed-height-tuning-guide.html "TECS (Total Energy Control System) for Speed and Height Tuning Guide — Plane  documentation"
