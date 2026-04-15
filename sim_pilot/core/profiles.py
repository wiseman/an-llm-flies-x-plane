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

# Wider band than AltitudeHoldProfile: in normal pattern flying a C172 on
# downwind will routinely sit 100-200 ft below target while the TECS
# steady-state settles, so the 150 ft AltitudeHoldProfile threshold would
# chatter in and out of climb-capture on every downwind. Only fire when
# the aircraft is dramatically below target — e.g. LLM re-engaged
# pattern_fly from 400 ft AGL with target 1000 ft AGL (600 ft deficit).
_PATTERN_CLIMB_CAPTURE_BAND_FT = 400.0


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
        return _guidance_to_contribution(guidance)

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
            # Climb runway heading at Vy, retract flaps to takeoff setting,
            # and level off at pattern altitude so the next pattern entry
            # doesn't have to re-descend. Explicitly does NOT consult
            # route_manager.active_waypoint(), because direct_to() from a
            # position abeam the runway to a distant waypoint produced
            # nonsense go-around headings in the field.
            runway_course_deg = self.runway_frame.runway.course_deg
            track_error_deg = wrap_degrees_180(runway_course_deg - state.track_deg)
            bank_cmd_deg = clamp(track_error_deg * 0.35, -bank_limit_deg, bank_limit_deg)
            return GuidanceTargets(
                lateral_mode=LateralMode.TRACK_HOLD,
                vertical_mode=VerticalMode.TECS,
                target_bank_deg=bank_cmd_deg,
                target_track_deg=runway_course_deg,
                target_heading_deg=runway_course_deg,
                target_altitude_ft=self.config.pattern_altitude_msl_ft,
                target_speed_kt=self.config.performance.vy_kt,
                throttle_limit=(0.9, 1.0),
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
                # Direct-to the downwind join point instead of following
                # the rigid entry_leg. The old entry_leg started at a
                # fixed upwind-and-offset point and ran diagonally to the
                # join point, so an aircraft engaging pattern_fly from
                # any other position (the KWHP log scenario: LLM engaged
                # mid-flight from SE of the airport) had to fly AWAY
                # from the runway to reach the entry_leg start before
                # it could turn inbound. Direct-to the join point always
                # produces a sensible inbound bearing regardless of
                # starting position. Once close enough the mode_manager
                # transitions to DOWNWIND on its own.
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
                # Target the glidepath altitude at the END of the base
                # leg. This commands TECS to descend from pattern
                # altitude to the 3° slope intercept so that BASE →
                # FINAL hands off near the glidepath. The old behavior
                # held a flat 600 AGL, which left the aircraft far
                # above the glidepath at final intercept and caused
                # long floats before touchdown (observed on KWHP 4120
                # ft runway: touchdown ~3000 ft past the threshold).
                base_end_runway = self.runway_frame.to_runway_frame(
                    self.pattern.base_leg.end_ft
                )
                target_altitude_ft = glidepath_target_altitude_ft(
                    self.runway_frame,
                    runway_x_ft=base_end_runway.x,
                    field_elevation_ft=self.config.airport.field_elevation_ft,
                )
                target_speed_kt = self.config.performance.base_speed_kt
                vertical_mode = VerticalMode.TECS
                glidepath = None
                throttle_limit = (0.1, 0.5)
                flaps_cmd = 20
            else:
                target_altitude_ft = glidepath_target_altitude_ft(
                    self.runway_frame,
                    runway_x_ft=state.runway_x_ft or -3000.0,
                    field_elevation_ft=self.config.airport.field_elevation_ft,
                )
                target_speed_kt = self.config.performance.final_speed_kt
                vertical_mode = VerticalMode.GLIDEPATH_TRACK
                glidepath = Glidepath(
                    slope_deg=3.0,
                    threshold_crossing_height_ft=50.0,
                    aimpoint_ft_from_threshold=self.runway_frame.touchdown_runway_x_ft,
                )
                throttle_limit = (0.1, 0.65)
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
            return GuidanceTargets(
                lateral_mode=LateralMode.PATH_FOLLOW,
                vertical_mode=vertical_mode,
                target_bank_deg=bank_cmd_deg,
                target_track_deg=desired_track_deg,
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
            brakes = 0.2 if state.gs_kt > 30.0 else 0.45
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
