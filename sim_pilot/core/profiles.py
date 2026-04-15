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
    VerticalMode,
    Waypoint,
    clamp,
    wrap_degrees_180,
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
    name: ClassVar[str] = "heading_hold"
    owns: ClassVar[frozenset[Axis]] = frozenset({Axis.LATERAL})

    def __init__(self, heading_deg: float, max_bank_deg: float = 25.0) -> None:
        self.heading_deg = float(heading_deg) % 360.0
        self.max_bank_deg = float(max_bank_deg)

    def contribute(self, state: AircraftState, dt: float, pilot: "PilotCore") -> ProfileContribution:
        heading_error_deg = wrap_degrees_180(self.heading_deg - state.track_deg)
        target_bank_deg = clamp(heading_error_deg * 0.35, -self.max_bank_deg, self.max_bank_deg)
        return ProfileContribution(
            lateral_mode=LateralMode.TRACK_HOLD,
            target_track_deg=self.heading_deg,
            target_heading_deg=self.heading_deg,
            target_bank_deg=target_bank_deg,
        )


class AltitudeHoldProfile:
    name: ClassVar[str] = "altitude_hold"
    owns: ClassVar[frozenset[Axis]] = frozenset({Axis.VERTICAL})

    def __init__(self, altitude_ft: float) -> None:
        self.altitude_ft = float(altitude_ft)

    def contribute(self, state: AircraftState, dt: float, pilot: "PilotCore") -> ProfileContribution:
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
        self.mode_manager = ModeManager(config)
        self.safety_monitor = SafetyMonitor(config)
        self.lateral_guidance = L1PathFollower()
        self.pattern = self._build_pattern_geometry()
        self.route_manager = RouteManager(
            [
                Waypoint(
                    name="outbound",
                    position_ft=config.airport.mission.outbound_waypoint_ft,
                    altitude_ft=config.cruise_altitude_ft,
                ),
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
        self.phase = self.mode_manager.update(
            self.phase,
            state,
            self.route_manager,
            self.pattern,
            safety_status,
            turn_base_now=self._turn_base_trigger,
            force_go_around=self._force_go_around_trigger,
        )
        if previous_phase is FlightPhase.DOWNWIND and self.phase is not FlightPhase.DOWNWIND:
            self._turn_base_trigger = False
        if self.phase is FlightPhase.GO_AROUND:
            self._force_go_around_trigger = False
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
            return GuidanceTargets(
                lateral_mode=LateralMode.ROLLOUT_CENTERLINE,
                vertical_mode=VerticalMode.PITCH_HOLD,
                target_bank_deg=0.0,
                target_heading_deg=self.runway_frame.runway.course_deg,
                target_pitch_deg=0.0,
                target_speed_kt=self.config.performance.vr_kt,
                throttle_limit=(1.0, 1.0),
            )
        if phase is FlightPhase.ROTATE:
            return GuidanceTargets(
                lateral_mode=LateralMode.PATH_FOLLOW,
                vertical_mode=VerticalMode.PITCH_HOLD,
                target_path=self.runway_frame.departure_leg(),
                target_bank_deg=0.0,
                target_pitch_deg=8.0,
                target_speed_kt=self.config.performance.vy_kt,
                throttle_limit=(1.0, 1.0),
            )
        if phase in {FlightPhase.INITIAL_CLIMB, FlightPhase.ENROUTE_CLIMB, FlightPhase.CRUISE, FlightPhase.DESCENT, FlightPhase.GO_AROUND}:
            waypoint = self.route_manager.active_waypoint()
            desired_track_deg = state.track_deg
            bank_cmd_deg = 0.0
            if waypoint is not None:
                desired_track_deg, bank_cmd_deg = self.lateral_guidance.direct_to(state, waypoint, max_bank_deg=bank_limit_deg)
            if phase in {FlightPhase.INITIAL_CLIMB, FlightPhase.ENROUTE_CLIMB, FlightPhase.GO_AROUND}:
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
            desired_track_deg, bank_cmd_deg = self.lateral_guidance.follow_leg(state, leg, max_bank_deg=bank_limit_deg)
            if phase is FlightPhase.PATTERN_ENTRY:
                target_altitude_ft = self.config.pattern_altitude_msl_ft
                target_speed_kt = self.config.performance.downwind_speed_kt
                vertical_mode = VerticalMode.TECS
                glidepath = None
                throttle_limit = (0.2, 0.6)
            elif phase is FlightPhase.DOWNWIND:
                target_altitude_ft = self.config.pattern_altitude_msl_ft
                target_speed_kt = self.config.performance.downwind_speed_kt
                vertical_mode = VerticalMode.TECS
                glidepath = None
                throttle_limit = (0.2, 0.55)
            elif phase is FlightPhase.BASE:
                target_altitude_ft = self.config.airport.field_elevation_ft + 600.0
                target_speed_kt = self.config.performance.base_speed_kt
                vertical_mode = VerticalMode.TECS
                glidepath = None
                throttle_limit = (0.1, 0.5)
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
                flaps_cmd=10 if phase in {FlightPhase.DOWNWIND, FlightPhase.BASE} else 20,
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
