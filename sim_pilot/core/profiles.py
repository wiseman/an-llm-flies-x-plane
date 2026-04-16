from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, Protocol

from sim_pilot.core.config import ConfigBundle
from sim_pilot.core.mode_manager import ModeManager
from sim_pilot.core.safety_monitor import SafetyMonitor
from sim_pilot.core.types import (
    AircraftState,
    FlightPhase,
    Glidepath,
    GuidanceTargets,
    LateralMode,
    StraightLeg,
    TrafficSide,
    VerticalMode,
    Waypoint,
    clamp,
    wrap_degrees_180,
    wrap_degrees_360,
)
from sim_pilot.guidance.lateral import L1PathFollower
from sim_pilot.guidance.pattern_manager import (
    PatternGeometry,
    build_pattern_geometry,
    glidepath_target_altitude_ft,
)
from sim_pilot.guidance.route_manager import RouteManager
from sim_pilot.guidance.runway_geometry import RunwayFrame

if TYPE_CHECKING:
    from sim_pilot.core.mission_manager import PilotCore


class Axis(StrEnum):
    LATERAL = "lateral"
    VERTICAL = "vertical"
    SPEED = "speed"


@dataclass(slots=True)
class ProfileContribution:
    lateral_mode: LateralMode | None = None
    vertical_mode: VerticalMode | None = None
    target_bank_deg: float | None = None
    target_heading_deg: float | None = None
    target_track_deg: float | None = None
    target_path: StraightLeg | None = None
    target_waypoint: Waypoint | None = None
    target_altitude_ft: float | None = None
    target_speed_kt: float | None = None
    target_pitch_deg: float | None = None
    glidepath: Glidepath | None = None
    throttle_limit: tuple[float, float] | None = None
    flaps_cmd: int | None = None
    gear_down: bool | None = None
    brakes: float | None = None
    tecs_phase_override: FlightPhase | None = None


class GuidanceProfile(Protocol):
    name: str
    owns: frozenset[Axis]

    def contribute(
        self,
        state: AircraftState,
        dt: float,
        pilot: "PilotCore",
    ) -> ProfileContribution: ...


class IdleLateralProfile:
    name: ClassVar[str] = "idle_lateral"
    owns: ClassVar[frozenset[Axis]] = frozenset({Axis.LATERAL})

    def contribute(self, state: AircraftState, dt: float, pilot: "PilotCore") -> ProfileContribution:
        return ProfileContribution(
            lateral_mode=LateralMode.BANK_HOLD,
            target_bank_deg=0.0,
        )


class IdleVerticalProfile:
    name: ClassVar[str] = "idle_vertical"
    owns: ClassVar[frozenset[Axis]] = frozenset({Axis.VERTICAL})

    def contribute(self, state: AircraftState, dt: float, pilot: "PilotCore") -> ProfileContribution:
        return ProfileContribution(
            vertical_mode=VerticalMode.PITCH_HOLD,
            target_pitch_deg=0.0,
            throttle_limit=(0.0, 0.0),
        )


class IdleSpeedProfile:
    name: ClassVar[str] = "idle_speed"
    owns: ClassVar[frozenset[Axis]] = frozenset({Axis.SPEED})

    def __init__(self, default_speed_kt: float) -> None:
        self.default_speed_kt = float(default_speed_kt)

    def contribute(self, state: AircraftState, dt: float, pilot: "PilotCore") -> ProfileContribution:
        return ProfileContribution(target_speed_kt=self.default_speed_kt)


class HeadingHoldProfile:
    """Hold a target ground track.

    ``turn_direction`` can be ``"left"``, ``"right"``, or ``None`` (default).
    When None, the profile takes the shortest-path turn to the target. When
    set to ``"left"`` or ``"right"``, the initial turn is forced in that
    direction even if the other way is shorter; once the aircraft is within
    a few degrees of the target the lock clears so normal shortest-path
    tracking resumes (no endless spinning past the target).
    """

    name: ClassVar[str] = "heading_hold"
    owns: ClassVar[frozenset[Axis]] = frozenset({Axis.LATERAL})

    def __init__(
        self,
        heading_deg: float,
        max_bank_deg: float = 25.0,
        turn_direction: str | None = None,
    ) -> None:
        self.heading_deg = float(heading_deg) % 360.0
        self.max_bank_deg = float(max_bank_deg)
        normalized = (turn_direction or "").lower() or None
        if normalized not in (None, "left", "right"):
            raise ValueError(f"turn_direction must be None, 'left', or 'right'; got {turn_direction!r}")
        self._direction_lock: str | None = normalized

    def contribute(self, state: AircraftState, dt: float, pilot: "PilotCore") -> ProfileContribution:
        raw_error = (self.heading_deg - state.track_deg) % 360.0
        short_error = raw_error - 360.0 if raw_error > 180.0 else raw_error
        if abs(short_error) < 5.0:
            self._direction_lock = None
        if self._direction_lock == "right":
            effective_error = raw_error if raw_error > 0.0 else 0.0
        elif self._direction_lock == "left":
            effective_error = raw_error - 360.0 if raw_error > 0.0 else 0.0
        else:
            effective_error = short_error
        target_bank_deg = clamp(effective_error * 0.35, -self.max_bank_deg, self.max_bank_deg)
        return ProfileContribution(
            lateral_mode=LateralMode.TRACK_HOLD,
            target_track_deg=self.heading_deg,
            target_heading_deg=self.heading_deg,
            target_bank_deg=target_bank_deg,
        )


_ALT_HOLD_CAPTURE_BAND_FT = 150.0

# Used to be 400 ft "to avoid chattering". But in practice stay-in-pattern
# missions routinely enter DOWNWIND ~200-300 ft below pattern altitude
# (the crosswind-to-downwind turn sags altitude a bit while banked), and
# at the default 0.55 throttle ceiling TECS can't climb back up before
# the base turn. Observed in sim_pilot-20260415-130505.log: plane rolled
# out on downwind at 700 AGL and never recovered to the 1000 AGL target.
# Lower the threshold to 150 ft so climb-capture kicks in on any real
# altitude deficit and TECS gets the full-climb throttle band.
_PATTERN_CLIMB_CAPTURE_BAND_FT = 150.0


class AltitudeHoldProfile:
    """Hold a target altitude (MSL) via TECS.

    When the aircraft is within ~150 ft of the target this profile asks TECS
    for the default cruise tuning (trim throttle ~0.58, trim pitch ~2 deg),
    which gives smooth small-error corrections. When the error is larger —
    which happens whenever the LLM commands a new altitude hundreds or
    thousands of feet away — it hints a ``tecs_phase_override`` and a
    narrower ``throttle_limit`` so TECS uses climb-capture or
    descent-capture gains. Without this regime switch the CRUISE gains are
    far too weak to catch a 500+ ft setpoint in reasonable time (the
    aircraft just sinks toward the target at cruise trim throttle).
    """

    name: ClassVar[str] = "altitude_hold"
    owns: ClassVar[frozenset[Axis]] = frozenset({Axis.VERTICAL})

    def __init__(self, altitude_ft: float) -> None:
        self.altitude_ft = float(altitude_ft)

    def contribute(self, state: AircraftState, dt: float, pilot: "PilotCore") -> ProfileContribution:
        error_ft = self.altitude_ft - state.alt_msl_ft
        if error_ft > _ALT_HOLD_CAPTURE_BAND_FT:
            # Climb capture: clamp throttle high so TECS has to use near-full
            # power, and hand TECS the ENROUTE_CLIMB trim (0.85 throttle /
            # 6 deg pitch). At max pitch 12 deg and 70-90% throttle the C172
            # actually climbs toward the target instead of sagging.
            return ProfileContribution(
                vertical_mode=VerticalMode.TECS,
                target_altitude_ft=self.altitude_ft,
                throttle_limit=(0.7, 1.0),
                tecs_phase_override=FlightPhase.ENROUTE_CLIMB,
            )
        if error_ft < -_ALT_HOLD_CAPTURE_BAND_FT:
            # Descent capture: force low throttle so the aircraft actually
            # descends, and use DESCENT trim (0.35 throttle / -1.5 deg pitch).
            return ProfileContribution(
                vertical_mode=VerticalMode.TECS,
                target_altitude_ft=self.altitude_ft,
                throttle_limit=(0.1, 0.5),
                tecs_phase_override=FlightPhase.DESCENT,
            )
        return ProfileContribution(
            vertical_mode=VerticalMode.TECS,
            target_altitude_ft=self.altitude_ft,
            throttle_limit=(0.1, 0.9),
        )


class SpeedHoldProfile:
    name: ClassVar[str] = "speed_hold"
    owns: ClassVar[frozenset[Axis]] = frozenset({Axis.SPEED})

    def __init__(self, speed_kt: float) -> None:
        self.speed_kt = float(speed_kt)

    def contribute(self, state: AircraftState, dt: float, pilot: "PilotCore") -> ProfileContribution:
        return ProfileContribution(target_speed_kt=self.speed_kt)


def build_takeoff_roll_guidance(config: ConfigBundle, runway_frame: RunwayFrame) -> GuidanceTargets:
    """Full-power takeoff roll: centerline hold, wings level, pitch neutral, target Vr."""
    return GuidanceTargets(
        lateral_mode=LateralMode.ROLLOUT_CENTERLINE,
        vertical_mode=VerticalMode.PITCH_HOLD,
        target_bank_deg=0.0,
        target_heading_deg=runway_frame.runway.course_deg,
        target_pitch_deg=0.0,
        target_speed_kt=config.performance.vr_kt,
        throttle_limit=(1.0, 1.0),
    )


def build_rotate_guidance(
    config: ConfigBundle,
    runway_frame: RunwayFrame,
    state: AircraftState,
    bank_limit_deg: float,
) -> GuidanceTargets:
    """Rotation: full power, pitch up to initial climb attitude.

    Lateral control depends on whether the wheels are on the ground:

    - Still rolling: keep ``ROLLOUT_CENTERLINE`` (rudder + nosewheel)
      authoritative and emit ``target_bank_deg=0``. Ailerons don't help
      a three-wheel aircraft hold the runway centerline and can induce
      ground-handling surprises.
    - Wheels off: switch to ``TRACK_HOLD`` with a proportional bank
      command toward runway course so the aircraft rolls back onto the
      runway extension immediately after liftoff.

    Regression note: the old implementation emitted ``PATH_FOLLOW`` with
    no ``target_track_deg`` and ``target_bank_deg=0.0``, which appeared
    in the status bus as ``tgt_hdg=—`` and left the bank controller
    holding wings level regardless of any drift. During the KWHP log
    (`output/sim_pilot-20260415-094519.log`) the aircraft rotated ~45°
    off runway heading and PatternFlyProfile had no lateral authority
    during the critical liftoff transition.
    """
    target_course_deg = runway_frame.runway.course_deg
    if state.on_ground:
        return GuidanceTargets(
            lateral_mode=LateralMode.ROLLOUT_CENTERLINE,
            vertical_mode=VerticalMode.PITCH_HOLD,
            target_bank_deg=0.0,
            target_heading_deg=target_course_deg,
            target_track_deg=target_course_deg,
            target_pitch_deg=8.0,
            target_speed_kt=config.performance.vy_kt,
            throttle_limit=(1.0, 1.0),
        )
    track_error_deg = wrap_degrees_180(target_course_deg - state.track_deg)
    target_bank_deg = clamp(track_error_deg * 0.35, -bank_limit_deg, bank_limit_deg)
    return GuidanceTargets(
        lateral_mode=LateralMode.TRACK_HOLD,
        vertical_mode=VerticalMode.PITCH_HOLD,
        target_bank_deg=target_bank_deg,
        target_heading_deg=target_course_deg,
        target_track_deg=target_course_deg,
        target_pitch_deg=8.0,
        target_speed_kt=config.performance.vy_kt,
        throttle_limit=(1.0, 1.0),
    )


class TakeoffProfile:
    """Runs the takeoff sequence: roll, rotate, initial climb straight ahead at Vy.

    Owns all three axes. Uses an internal ``ModeManager`` to advance through
    PREFLIGHT → TAKEOFF_ROLL → ROTATE → INITIAL_CLIMB. Once in INITIAL_CLIMB it
    holds a simple "climb-straight-on-runway-track at Vy" profile indefinitely
    — it does NOT auto-disengage. Transition out by engaging a different
    profile (``heading_hold``, ``altitude_hold``, ``pattern_fly``, …); those
    engagements displace this profile via axis-ownership conflict.
    """

    name: ClassVar[str] = "takeoff"
    owns: ClassVar[frozenset[Axis]] = frozenset({Axis.LATERAL, Axis.VERTICAL, Axis.SPEED})

    def __init__(self, config: ConfigBundle, runway_frame: RunwayFrame) -> None:
        self.config = config
        self.runway_frame = runway_frame
        self.phase: FlightPhase = FlightPhase.PREFLIGHT
        self.mode_manager = ModeManager(config)
        self.safety_monitor = SafetyMonitor(config)
        self._pattern_geometry_stub = build_pattern_geometry(
            runway_frame,
            downwind_offset_ft=config.pattern.downwind_offset_ft,
            extension_ft=config.pattern.default_extension_ft,
        )
        self._route_stub = RouteManager([])

    def contribute(self, state: AircraftState, dt: float, pilot: "PilotCore") -> ProfileContribution:
        safety_status = self.safety_monitor.evaluate(state, self.phase)
        self.phase = self.mode_manager.update(
            self.phase,
            state,
            self._route_stub,
            self._pattern_geometry_stub,
            safety_status,
        )
        if self.phase is FlightPhase.TAKEOFF_ROLL:
            guidance = build_takeoff_roll_guidance(self.config, self.runway_frame)
        elif self.phase is FlightPhase.ROTATE:
            guidance = build_rotate_guidance(
                self.config,
                self.runway_frame,
                state,
                bank_limit_deg=self.safety_monitor.bank_limit_deg(self.phase),
            )
        elif self.phase in {FlightPhase.INITIAL_CLIMB, FlightPhase.ENROUTE_CLIMB, FlightPhase.CRUISE}:
            # Airborne on runway course: compute a bank command from track
            # error so the aircraft actively rolls back onto the runway
            # course when it drifts. Previously this was hard-coded to 0.0
            # which meant the plane just flew whatever heading it left the
            # ground on.
            target_track_deg = self.runway_frame.runway.course_deg
            track_error_deg = wrap_degrees_180(target_track_deg - state.track_deg)
            bank_limit_deg = self.safety_monitor.bank_limit_deg(self.phase)
            target_bank_deg = clamp(track_error_deg * 0.35, -bank_limit_deg, bank_limit_deg)
            guidance = GuidanceTargets(
                lateral_mode=LateralMode.TRACK_HOLD,
                vertical_mode=VerticalMode.TECS,
                target_bank_deg=target_bank_deg,
                target_track_deg=target_track_deg,
                target_heading_deg=target_track_deg,
                target_altitude_ft=self.config.cruise_altitude_ft,
                target_speed_kt=self.config.performance.vy_kt,
                throttle_limit=(0.9, 1.0),
            )
        else:
            guidance = GuidanceTargets(
                lateral_mode=LateralMode.BANK_HOLD,
                vertical_mode=VerticalMode.PITCH_HOLD,
                target_bank_deg=0.0,
                target_pitch_deg=0.0,
                throttle_limit=(0.0, 0.0),
            )
        guidance = self.safety_monitor.apply_limits(guidance, self.phase)
        return _guidance_to_contribution(guidance)


class PatternFlyProfile:
    """Full deterministic mission executor wrapping the phase-machine pilot.

    Covers PREFLIGHT → TAKEOFF_ROLL → CLIMB → CRUISE → DESCENT → PATTERN_ENTRY →
    DOWNWIND → BASE → FINAL → ROUNDOUT → FLARE → ROLLOUT → TAXI_CLEAR, plus
    GO_AROUND. Owns all three axes (lateral, vertical, speed). The LLM engages
    this when it wants the deterministic pilot to fly the whole mission; narrower
    profiles (HeadingHold, AltitudeHold, etc.) can be used in place of it for
    parts of the flight the LLM wants to micromanage.
    """

    name: ClassVar[str] = "pattern_fly"
    owns: ClassVar[frozenset[Axis]] = frozenset({Axis.LATERAL, Axis.VERTICAL, Axis.SPEED})

    def __init__(self, config: ConfigBundle, runway_frame: RunwayFrame) -> None:
        self.config = config
        self.runway_frame = runway_frame
        self.pattern_extension_ft = 0.0
        self._turn_base_trigger = False
        self._force_go_around_trigger = False
        self.cleared_to_land_runway: str | None = None
        self.phase: FlightPhase = FlightPhase.PREFLIGHT
        # Set on the tick a go-around is triggered, so observers (heartbeat
        # pump, status snapshot) can surface why the state machine flipped
        # into GO_AROUND instead of forcing them to reconstruct it from
        # position / altitude history.
        self.last_go_around_reason: str | None = None
        self.mode_manager = ModeManager(config)
        self.safety_monitor = SafetyMonitor(config)
        self.lateral_guidance = L1PathFollower()
        self.pattern = self._build_pattern_geometry()
        # Only one waypoint: the pattern entry point. We used to also have
        # an "outbound" waypoint (default (0, 91142) in world frame = 91 kft
        # due north of the georef) which was meaningless for live runs and
        # was the waypoint GO_AROUND's direct_to() locked on, producing the
        # "go-around turns to 2°" bug. GO_AROUND now has its own runway-
        # heading branch in _guidance_for_phase.
        self.route_manager = RouteManager(
            [
                Waypoint(
                    name="pattern_entry_start",
                    position_ft=runway_frame.to_world_frame(config.airport.mission.entry_start_runway_ft),
                    altitude_ft=config.pattern_altitude_msl_ft,
                ),
            ]
        )

    def turn_base_now(self) -> None:
        self._turn_base_trigger = True

    def go_around(self) -> None:
        self._force_go_around_trigger = True

    def extend_downwind(self, extension_ft: float) -> None:
        self.pattern_extension_ft += max(0.0, float(extension_ft))
        self.pattern = self._build_pattern_geometry()

    def cleared_to_land(self, runway_id: str | None) -> None:
        self.cleared_to_land_runway = runway_id or self.runway_frame.runway.id

    def contribute(self, state: AircraftState, dt: float, pilot: "PilotCore") -> ProfileContribution:
        # When turn_base_now fires while on DOWNWIND, rebuild the base
        # leg so it starts at the aircraft's *current* runway_x instead
        # of whatever base_turn_x the pattern geometry was last built
        # with. Without this, an extend_downwind(big_number) followed
        # by a later turn_base_now would leave the pre-computed base
        # leg way behind the aircraft, and L1's fallback "direct to
        # leg start" would turn the plane *further upwind* toward the
        # stale leg start instead of toward the runway.
        if (
            self._turn_base_trigger
            and self.phase is FlightPhase.DOWNWIND
            and state.runway_x_ft is not None
        ):
            # Solve base_turn_x_ft = -(downwind_offset + extension) for
            # extension such that base_turn_x_ft == state.runway_x_ft.
            downwind_offset_ft = self.config.pattern.downwind_offset_ft
            self.pattern_extension_ft = max(
                0.0,
                -float(state.runway_x_ft) - downwind_offset_ft,
            )
            self.pattern = self._build_pattern_geometry()

        self.route_manager.advance_if_needed(state.position_ft)
        safety_status = self.safety_monitor.evaluate(state, self.phase)
        previous_phase = self.phase
        manual_go_around_triggered = self._force_go_around_trigger
        self.phase = self.mode_manager.update(
            self.phase,
            state,
            self.route_manager,
            self.pattern,
            safety_status,
            turn_base_now=self._turn_base_trigger,
            force_go_around=self._force_go_around_trigger,
            stay_in_pattern=True,
        )
        if previous_phase is FlightPhase.DOWNWIND and self.phase is not FlightPhase.DOWNWIND:
            self._turn_base_trigger = False
        if self.phase is FlightPhase.GO_AROUND:
            self._force_go_around_trigger = False
            # Record why this tick flipped to GO_AROUND — but only on the
            # transition itself, so that subsequent ticks in GO_AROUND
            # don't overwrite with stale reasons.
            if previous_phase is not FlightPhase.GO_AROUND:
                if manual_go_around_triggered:
                    self.last_go_around_reason = "manual_trigger"
                elif safety_status.reason is not None:
                    self.last_go_around_reason = safety_status.reason
                else:
                    self.last_go_around_reason = "unknown"
        guidance = self._guidance_for_phase(state, self.phase)
        guidance = self.safety_monitor.apply_limits(guidance, self.phase)
        # Once the go-around climb has settled near pattern altitude,
        # hand off to three single-axis holds (heading / altitude /
        # speed). The LLM then sees a normal "autopilot hold" state
        # and can decide what to do next (fly another pattern, divert,
        # continue climb). Without this, pattern_fly would stay
        # engaged in GO_AROUND indefinitely, which isn't a state the
        # LLM's prompt vocabulary is set up to handle naturally.
        if self.phase is FlightPhase.GO_AROUND and self._go_around_climb_settled(state):
            self._hand_off_to_holds(pilot)
        return _guidance_to_contribution(guidance)

    def _go_around_climb_settled(self, state: AircraftState) -> bool:
        alt_error_ft = abs(self.config.pattern_altitude_msl_ft - state.alt_msl_ft)
        return alt_error_ft < 100.0 and abs(state.vs_fpm) < 200.0

    def _hand_off_to_holds(self, pilot: "PilotCore") -> None:
        """Replace this profile with HeadingHold + AltitudeHold + SpeedHold.

        Engages all three holds via ``pilot.engage_profile``. The first
        engagement displaces ``pattern_fly`` (because HeadingHold owns
        the LATERAL axis and pattern_fly owns all three); the
        subsequent two fill the orphaned VERTICAL and SPEED axes. After
        the three calls, ``pilot.active_profiles`` contains exactly the
        three holds.
        """
        pilot.engage_profile(
            HeadingHoldProfile(heading_deg=self.runway_frame.runway.course_deg)
        )
        pilot.engage_profile(
            AltitudeHoldProfile(altitude_ft=self.config.pattern_altitude_msl_ft)
        )
        pilot.engage_profile(
            SpeedHoldProfile(speed_kt=self.config.performance.vy_kt)
        )

    def _build_pattern_geometry(self) -> PatternGeometry:
        return build_pattern_geometry(
            self.runway_frame,
            downwind_offset_ft=self.config.pattern.downwind_offset_ft,
            extension_ft=self.config.pattern.default_extension_ft + self.pattern_extension_ft,
        )

    def _guidance_for_phase(self, state: AircraftState, phase: FlightPhase) -> GuidanceTargets:
        bank_limit_deg = self.safety_monitor.bank_limit_deg(phase)
        if phase is FlightPhase.TAKEOFF_ROLL:
            return build_takeoff_roll_guidance(self.config, self.runway_frame)
        if phase is FlightPhase.ROTATE:
            return build_rotate_guidance(
                self.config,
                self.runway_frame,
                state,
                bank_limit_deg=bank_limit_deg,
            )
        if phase is FlightPhase.GO_AROUND:
            # Climb runway heading at Vy and level off at pattern
            # altitude. Stays in GO_AROUND indefinitely — the LLM is
            # responsible for deciding what to do next (fly another
            # pattern, divert, etc.). Explicitly does NOT consult
            # route_manager.active_waypoint(), because direct_to() from
            # a position abeam the runway to a distant waypoint
            # produced nonsense go-around headings in the field.
            runway_course_deg = self.runway_frame.runway.course_deg
            track_error_deg = wrap_degrees_180(runway_course_deg - state.track_deg)
            bank_cmd_deg = clamp(track_error_deg * 0.35, -bank_limit_deg, bank_limit_deg)
            # Climb toward pattern altitude, then level off. The
            # threshold for switching from climb (0.9-1.0 throttle,
            # GO_AROUND trim) to level (0.2-0.55 throttle, PATTERN_ENTRY
            # trim) is 100 ft below target — anticipating the overshoot
            # that comes from TECS integrator wind-up accumulated during
            # the climb. The 0.55 throttle ceiling is just above
            # PATTERN_ENTRY trim (0.45), so any integrator-driven
            # throttle push is capped and the aircraft bleeds speed
            # into altitude rather than continuing to climb.
            pattern_alt_msl_ft = self.config.pattern_altitude_msl_ft
            alt_error_ft = pattern_alt_msl_ft - state.alt_msl_ft
            if alt_error_ft > 100.0:
                throttle_limit = (0.9, 1.0)
                tecs_phase_override = None  # uses GO_AROUND climb trim
            else:
                throttle_limit = (0.2, 0.55)
                tecs_phase_override = FlightPhase.PATTERN_ENTRY
            return GuidanceTargets(
                lateral_mode=LateralMode.TRACK_HOLD,
                vertical_mode=VerticalMode.TECS,
                target_bank_deg=bank_cmd_deg,
                target_track_deg=runway_course_deg,
                target_heading_deg=runway_course_deg,
                target_altitude_ft=pattern_alt_msl_ft,
                target_speed_kt=self.config.performance.vy_kt,
                throttle_limit=throttle_limit,
                tecs_phase_override=tecs_phase_override,
                flaps_cmd=10,
                gear_down=True,
            )
        if phase is FlightPhase.INITIAL_CLIMB:
            # Upwind: hold runway course straight out, climb toward pattern
            # altitude at Vy. Replaces the old ``direct_to(pattern_entry_start)``
            # behavior, which banked the aircraft off runway heading within
            # seconds of wheels-up (seen in the KWHP log at alt_agl=43 ft).
            target_course_deg = self.runway_frame.runway.course_deg
            track_error_deg = wrap_degrees_180(target_course_deg - state.track_deg)
            bank_cmd_deg = clamp(track_error_deg * 0.35, -bank_limit_deg, bank_limit_deg)
            return GuidanceTargets(
                lateral_mode=LateralMode.TRACK_HOLD,
                vertical_mode=VerticalMode.TECS,
                target_bank_deg=bank_cmd_deg,
                target_track_deg=target_course_deg,
                target_heading_deg=target_course_deg,
                target_altitude_ft=self.config.pattern_altitude_msl_ft,
                target_speed_kt=self.config.performance.vy_kt,
                throttle_limit=(0.75, 1.0),
                tecs_phase_override=FlightPhase.ENROUTE_CLIMB,
            )
        if phase is FlightPhase.CROSSWIND:
            # 90° turn away from runway heading (left for left traffic,
            # right for right traffic). Continue climbing toward pattern
            # altitude at Vy — the turn typically completes at or just
            # below pattern altitude.
            side_sign = -1.0 if self.runway_frame.runway.traffic_side is TrafficSide.LEFT else 1.0
            crosswind_course_deg = wrap_degrees_360(
                self.runway_frame.runway.course_deg + (side_sign * 90.0)
            )
            track_error_deg = wrap_degrees_180(crosswind_course_deg - state.track_deg)
            bank_cmd_deg = clamp(track_error_deg * 0.35, -bank_limit_deg, bank_limit_deg)
            return GuidanceTargets(
                lateral_mode=LateralMode.TRACK_HOLD,
                vertical_mode=VerticalMode.TECS,
                target_bank_deg=bank_cmd_deg,
                target_track_deg=crosswind_course_deg,
                target_heading_deg=crosswind_course_deg,
                target_altitude_ft=self.config.pattern_altitude_msl_ft,
                target_speed_kt=self.config.performance.vy_kt,
                throttle_limit=(0.7, 1.0),
                tecs_phase_override=FlightPhase.ENROUTE_CLIMB,
            )
        if phase in {FlightPhase.ENROUTE_CLIMB, FlightPhase.CRUISE, FlightPhase.DESCENT}:
            # Airborne rejoin branch: the LLM engaged PatternFlyProfile
            # mid-flight and needs to navigate back to the pattern entry
            # point before beginning PATTERN_ENTRY. Stay-in-pattern flows
            # that start from TAKEOFF_ROLL go INITIAL_CLIMB → CROSSWIND →
            # DOWNWIND and never enter this branch.
            waypoint = self.route_manager.active_waypoint()
            desired_track_deg = state.track_deg
            bank_cmd_deg = 0.0
            if waypoint is not None:
                desired_track_deg, bank_cmd_deg = self.lateral_guidance.direct_to(state, waypoint, max_bank_deg=bank_limit_deg)
            if phase is FlightPhase.ENROUTE_CLIMB:
                target_altitude_ft = self.config.cruise_altitude_ft
                target_speed_kt = self.config.performance.vy_kt
                throttle_limit = (0.75, 1.0)
            elif phase is FlightPhase.CRUISE:
                target_altitude_ft = self.config.cruise_altitude_ft
                target_speed_kt = self.config.performance.cruise_speed_kt
                throttle_limit = (0.4, 0.8)
            else:
                target_altitude_ft = self.config.pattern_altitude_msl_ft
                target_speed_kt = self.config.performance.descent_speed_kt
                throttle_limit = (0.15, 0.6)
            return GuidanceTargets(
                lateral_mode=LateralMode.TRACK_HOLD,
                vertical_mode=VerticalMode.TECS,
                target_bank_deg=bank_cmd_deg,
                target_track_deg=desired_track_deg,
                target_waypoint=waypoint,
                target_altitude_ft=target_altitude_ft,
                target_speed_kt=target_speed_kt,
                throttle_limit=throttle_limit,
            )
        if phase in {FlightPhase.PATTERN_ENTRY, FlightPhase.DOWNWIND, FlightPhase.BASE, FlightPhase.FINAL}:
            leg = self.pattern.leg_for_phase(phase)
            assert leg is not None
            if phase is FlightPhase.PATTERN_ENTRY:
                # Direct-to the join point rather than following entry_leg
                # as a rigid leg: an aircraft engaging pattern_fly from an
                # arbitrary position (mid-flight rejoin) otherwise has to
                # fly away from the runway to reach the leg start before
                # turning inbound.
                join_waypoint = Waypoint(
                    name="pattern_join",
                    position_ft=self.runway_frame.to_world_frame(self.pattern.join_point_runway_ft),
                    altitude_ft=self.config.pattern_altitude_msl_ft,
                )
                desired_track_deg, bank_cmd_deg = self.lateral_guidance.direct_to(
                    state, join_waypoint, max_bank_deg=bank_limit_deg
                )
            else:
                desired_track_deg, bank_cmd_deg = self.lateral_guidance.follow_leg(state, leg, max_bank_deg=bank_limit_deg)
            # C172 landing flap schedule: clean on pattern entry so we have
            # climb/speed margin for rejoins, first notch abeam the numbers,
            # second on base, full on final. The old schedule had entry/final
            # at 20 and downwind/base at 10, which capped IAS during the
            # climb-to-pattern-altitude rejoin seen in the KWHP log.
            tecs_phase_override: FlightPhase | None = None
            if phase is FlightPhase.PATTERN_ENTRY:
                target_altitude_ft = self.config.pattern_altitude_msl_ft
                target_speed_kt = self.config.performance.downwind_speed_kt
                vertical_mode = VerticalMode.TECS
                glidepath = None
                throttle_limit = (0.2, 0.6)
                flaps_cmd = 0
            elif phase is FlightPhase.DOWNWIND:
                target_altitude_ft = self.config.pattern_altitude_msl_ft
                target_speed_kt = self.config.performance.downwind_speed_kt
                vertical_mode = VerticalMode.TECS
                glidepath = None
                throttle_limit = (0.2, 0.55)
                flaps_cmd = 10
            elif phase is FlightPhase.BASE:
                # Achievable BASE target: 400 ft AGL. This is a realistic
                # altitude for a C172 to reach during a single base leg
                # from pattern altitude, and leaves the final intercept
                # only ~100 ft above a 4° glidepath so TECS can capture
                # with the small remaining error. Using the glidepath
                # altitude at base_end as the target was too aggressive
                # (required ~1200 fpm descent, which a C172 with gear +
                # flaps 20 cannot sustain).
                target_altitude_ft = self.config.airport.field_elevation_ft + 400.0
                target_speed_kt = self.config.performance.base_speed_kt
                vertical_mode = VerticalMode.TECS
                glidepath = None
                throttle_limit = (0.05, 0.5)
                flaps_cmd = 20
            else:
                # 4° final approach slope — steeper than the 3° ILS
                # standard because short-field visual approaches fly
                # steeper, and because at 3° a C172 can't catch the
                # glidepath from above within the available final leg
                # length (~4500 ft for a standard pattern). With the
                # aim point at ground level the threshold crossing
                # height falls out as aim_x * tan(4°) ≈ 70 AGL for a
                # 1000 ft aim point — which matches the "over the
                # numbers at ~70 ft" visual technique.
                final_slope_deg = 4.0
                target_altitude_ft = glidepath_target_altitude_ft(
                    self.runway_frame,
                    runway_x_ft=state.runway_x_ft or -3000.0,
                    field_elevation_ft=self.config.airport.field_elevation_ft,
                    slope_deg=final_slope_deg,
                )
                target_speed_kt = self.config.performance.final_speed_kt
                vertical_mode = VerticalMode.GLIDEPATH_TRACK
                glidepath = Glidepath(
                    slope_deg=final_slope_deg,
                    threshold_crossing_height_ft=0.0,
                    aimpoint_ft_from_threshold=self.runway_frame.touchdown_runway_x_ft,
                )
                throttle_limit = (0.05, 0.65)
                flaps_cmd = 30
            # Climb-capture for PATTERN_ENTRY/DOWNWIND: if the LLM
            # re-engaged pattern_fly from well below pattern altitude
            # (e.g. during the KWHP recovery, where the aircraft was at
            # 400 ft AGL with target 1000 ft AGL), the normal 0.55–0.60
            # throttle ceiling can't simultaneously climb and accelerate.
            # Hint TECS into ENROUTE_CLIMB trim with a higher ceiling
            # until within _PATTERN_CLIMB_CAPTURE_BAND_FT of target.
            # Explicitly skipped on BASE/FINAL (descent phases) because a
            # high throttle ceiling there would mask over-speed problems.
            if phase in {FlightPhase.PATTERN_ENTRY, FlightPhase.DOWNWIND}:
                alt_error_ft = target_altitude_ft - state.alt_msl_ft
                if alt_error_ft > _PATTERN_CLIMB_CAPTURE_BAND_FT:
                    throttle_limit = (0.7, 1.0)
                    tecs_phase_override = FlightPhase.ENROUTE_CLIMB
            # Stable target_heading_deg for the status display. Using the
            # L1-computed desired_track_deg here would cause the display to
            # flicker as the cross-track error closes during path
            # following, which makes it look like "the autopilot keeps
            # changing its mind". The actual L1 track command is still
            # passed as target_track_deg, which is what drives lateral
            # control; target_heading_deg is display-only. See tui.py
            # format_snapshot_display.
            runway_course_deg = self.runway_frame.runway.course_deg
            side_sign = -1.0 if self.runway_frame.runway.traffic_side is TrafficSide.LEFT else 1.0
            if phase is FlightPhase.PATTERN_ENTRY:
                display_heading_deg = wrap_degrees_360(runway_course_deg + 180.0)
            elif phase is FlightPhase.DOWNWIND:
                display_heading_deg = wrap_degrees_360(runway_course_deg + 180.0)
            elif phase is FlightPhase.BASE:
                # Perpendicular base: heading = runway_course + 90° toward
                # the runway from the downwind offset. For left traffic
                # (downwind on the left of the runway course), base turns
                # right relative to downwind heading — that's the runway
                # course minus 90° in world. The formula simplifies to
                # runway_course - side_sign * 90° for "left-traffic left
                # turns".
                display_heading_deg = wrap_degrees_360(runway_course_deg - side_sign * 90.0)
            else:
                display_heading_deg = runway_course_deg
            return GuidanceTargets(
                lateral_mode=LateralMode.PATH_FOLLOW,
                vertical_mode=vertical_mode,
                target_bank_deg=bank_cmd_deg,
                target_track_deg=desired_track_deg,
                target_heading_deg=display_heading_deg,
                target_path=leg,
                target_altitude_ft=target_altitude_ft,
                target_speed_kt=target_speed_kt,
                glidepath=glidepath,
                throttle_limit=throttle_limit,
                flaps_cmd=flaps_cmd,
                tecs_phase_override=tecs_phase_override,
            )
        if phase is FlightPhase.ROUNDOUT:
            desired_track_deg, bank_cmd_deg = self.lateral_guidance.follow_leg(state, self.pattern.final_leg, max_bank_deg=min(bank_limit_deg, 8.0))
            pitch_cmd_deg = 2.5 + max(0.0, (self.config.flare.roundout_height_ft - state.alt_agl_ft) * 0.12)
            return GuidanceTargets(
                lateral_mode=LateralMode.PATH_FOLLOW,
                vertical_mode=VerticalMode.PITCH_HOLD,
                target_path=self.pattern.final_leg,
                target_bank_deg=bank_cmd_deg,
                target_track_deg=desired_track_deg,
                target_pitch_deg=pitch_cmd_deg,
                target_speed_kt=self.config.performance.vref_kt,
                throttle_limit=(0.0, 0.2),
            )
        if phase is FlightPhase.FLARE:
            desired_track_deg, bank_cmd_deg = self.lateral_guidance.follow_leg(state, self.pattern.final_leg, max_bank_deg=min(bank_limit_deg, 6.0))
            return GuidanceTargets(
                lateral_mode=LateralMode.PATH_FOLLOW,
                vertical_mode=VerticalMode.FLARE_TRACK,
                target_path=self.pattern.final_leg,
                target_bank_deg=bank_cmd_deg,
                target_track_deg=desired_track_deg,
                target_speed_kt=self.config.performance.vref_kt - 2.0,
                throttle_limit=(0.0, 0.0),
                brakes=0.0,
            )
        if phase in {FlightPhase.ROLLOUT, FlightPhase.TAXI_CLEAR}:
            # Short-field landing technique: firm braking as soon as
            # the wheels are down, progressively harder as speed
            # bleeds off. The old 0.2/0.45 split was too gentle and
            # left long rollouts on short runways.
            if state.gs_kt > 40.0:
                brakes = 0.5
            elif state.gs_kt > 20.0:
                brakes = 0.8
            else:
                brakes = 1.0
            return GuidanceTargets(
                lateral_mode=LateralMode.ROLLOUT_CENTERLINE,
                vertical_mode=VerticalMode.PITCH_HOLD,
                target_bank_deg=0.0,
                target_heading_deg=self.runway_frame.runway.course_deg,
                target_pitch_deg=0.0,
                throttle_limit=(0.0, 0.0),
                brakes=brakes,
            )
        return GuidanceTargets(
            lateral_mode=LateralMode.BANK_HOLD,
            vertical_mode=VerticalMode.PITCH_HOLD,
            target_bank_deg=0.0,
            target_pitch_deg=0.0,
            throttle_limit=(0.0, 0.0),
        )


class ApproachRunwayProfile:
    name: ClassVar[str] = "approach_runway"
    owns: ClassVar[frozenset[Axis]] = frozenset({Axis.LATERAL, Axis.VERTICAL, Axis.SPEED})

    def __init__(self, runway_id: str) -> None:
        self.runway_id = runway_id

    def contribute(self, state: AircraftState, dt: float, pilot: "PilotCore") -> ProfileContribution:
        raise NotImplementedError("ApproachRunwayProfile is not yet implemented.")


class RouteFollowProfile:
    name: ClassVar[str] = "route_follow"
    owns: ClassVar[frozenset[Axis]] = frozenset({Axis.LATERAL, Axis.VERTICAL, Axis.SPEED})

    def __init__(self, waypoints: list[Waypoint]) -> None:
        self.waypoints = waypoints

    def contribute(self, state: AircraftState, dt: float, pilot: "PilotCore") -> ProfileContribution:
        raise NotImplementedError("RouteFollowProfile is not yet implemented.")


def _guidance_to_contribution(guidance: GuidanceTargets) -> ProfileContribution:
    return ProfileContribution(
        lateral_mode=guidance.lateral_mode,
        vertical_mode=guidance.vertical_mode,
        target_bank_deg=guidance.target_bank_deg,
        target_heading_deg=guidance.target_heading_deg,
        target_track_deg=guidance.target_track_deg,
        target_path=guidance.target_path,
        target_waypoint=guidance.target_waypoint,
        target_altitude_ft=guidance.target_altitude_ft,
        target_speed_kt=guidance.target_speed_kt,
        target_pitch_deg=guidance.target_pitch_deg,
        glidepath=guidance.glidepath,
        throttle_limit=guidance.throttle_limit,
        flaps_cmd=guidance.flaps_cmd,
        gear_down=guidance.gear_down,
        brakes=guidance.brakes,
    )
