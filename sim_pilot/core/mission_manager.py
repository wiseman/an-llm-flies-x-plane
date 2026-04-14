from __future__ import annotations

from dataclasses import dataclass

from sim_pilot.control.bank_hold import BankController, CoordinationController
from sim_pilot.control.centerline_rollout import CenterlineRolloutController
from sim_pilot.control.pitch_hold import PitchController
from sim_pilot.control.tecs_lite import TECSLite
from sim_pilot.core.config import ConfigBundle
from sim_pilot.core.mode_manager import ModeManager
from sim_pilot.core.safety_monitor import SafetyMonitor
from sim_pilot.core.state_estimator import estimate_aircraft_state
from sim_pilot.core.types import ActuatorCommands, AircraftState, FlightPhase, Glidepath, GuidanceTargets, LateralMode, VerticalMode, Waypoint, clamp, wrap_degrees_180
from sim_pilot.guidance.flare_profile import FlareController
from sim_pilot.guidance.lateral import L1PathFollower
from sim_pilot.guidance.pattern_manager import PatternGeometry, build_pattern_geometry, glidepath_target_altitude_ft
from sim_pilot.guidance.route_manager import RouteManager
from sim_pilot.guidance.runway_geometry import RunwayFrame
from sim_pilot.sim.simple_dynamics import DynamicsState


@dataclass
class PilotCore:
    config: ConfigBundle

    def __post_init__(self) -> None:
        self.runway_frame = RunwayFrame(self.config.airport.runway)
        self.pattern = build_pattern_geometry(
            self.runway_frame,
            downwind_offset_ft=self.config.pattern.downwind_offset_ft,
            extension_ft=self.config.pattern.default_extension_ft,
        )
        self.route_manager = RouteManager(
            [
                Waypoint(
                    name="outbound",
                    position_ft=self.config.airport.mission.outbound_waypoint_ft,
                    altitude_ft=self.config.cruise_altitude_ft,
                ),
                Waypoint(
                    name="pattern_entry_start",
                    position_ft=self.runway_frame.to_world_frame(self.config.airport.mission.entry_start_runway_ft),
                    altitude_ft=self.config.pattern_altitude_msl_ft,
                ),
            ]
        )
        self.phase = FlightPhase.PREFLIGHT
        self.mode_manager = ModeManager(self.config)
        self.safety_monitor = SafetyMonitor(self.config)
        self.lateral_guidance = L1PathFollower()
        self.bank_controller = BankController(self.config.controllers.bank)
        self.coordination = CoordinationController()
        self.pitch_controller = PitchController(self.config.controllers.pitch)
        self.rollout_controller = CenterlineRolloutController()
        self.tecs = TECSLite(self.config.controllers.tecs)
        self.flare_controller = FlareController(self.config.flare)

    def update(self, raw_state: DynamicsState, dt: float) -> tuple[AircraftState, ActuatorCommands]:
        state = estimate_aircraft_state(raw_state, self.config, self.runway_frame, dt)
        self.route_manager.advance_if_needed(state.position_ft)
        safety_status = self.safety_monitor.evaluate(state, self.phase)
        self.phase = self.mode_manager.update(self.phase, state, self.route_manager, self.pattern, safety_status)
        guidance = self._guidance_for_phase(state, self.phase)
        guidance = self.safety_monitor.apply_limits(guidance, self.phase)
        return state, self._commands_from_guidance(state, guidance)

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
                target_speed_kt = self.config.performance.pattern_speed_kt
                vertical_mode = VerticalMode.TECS
                glidepath = None
                throttle_limit = (0.2, 0.6)
            elif phase is FlightPhase.DOWNWIND:
                target_altitude_ft = self.config.pattern_altitude_msl_ft
                target_speed_kt = self.config.performance.pattern_speed_kt
                vertical_mode = VerticalMode.TECS
                glidepath = None
                throttle_limit = (0.2, 0.55)
            elif phase is FlightPhase.BASE:
                target_altitude_ft = self.config.airport.field_elevation_ft + 600.0
                target_speed_kt = self.config.performance.vapp_kt + 6.0
                vertical_mode = VerticalMode.TECS
                glidepath = None
                throttle_limit = (0.1, 0.5)
            else:
                target_altitude_ft = glidepath_target_altitude_ft(
                    self.runway_frame,
                    runway_x_ft=state.runway_x_ft or -3000.0,
                    field_elevation_ft=self.config.airport.field_elevation_ft,
                )
                target_speed_kt = self.config.performance.vapp_kt + 2.0
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

    def _commands_from_guidance(self, state: AircraftState, guidance: GuidanceTargets) -> ActuatorCommands:
        if guidance.lateral_mode is LateralMode.ROLLOUT_CENTERLINE:
            track_reference_deg = state.heading_deg if state.on_ground else state.track_deg
            track_error_deg = wrap_degrees_180(self.runway_frame.runway.course_deg - track_reference_deg)
            rudder = self.rollout_controller.update(
                centerline_error_ft=state.centerline_error_ft or 0.0,
                track_error_deg=track_error_deg,
                gs_kt=state.gs_kt,
            )
            aileron = self.bank_controller.update(0.0, state.roll_deg, state.p_rad_s, state.dt)
        else:
            target_bank_deg = guidance.target_bank_deg or 0.0
            aileron = self.bank_controller.update(target_bank_deg, state.roll_deg, state.p_rad_s, state.dt)
            rudder = self.coordination.update(target_bank_deg, state.roll_deg, state.r_rad_s, None, state.dt)

        throttle_limit = guidance.throttle_limit or (0.0, 1.0)
        if guidance.vertical_mode in {VerticalMode.TECS, VerticalMode.GLIDEPATH_TRACK}:
            pitch_cmd_deg, throttle_cmd = self.tecs.update(
                phase=self.phase,
                target_alt_ft=guidance.target_altitude_ft or state.alt_msl_ft,
                target_speed_kt=guidance.target_speed_kt or state.ias_kt,
                alt_ft=state.alt_msl_ft,
                vs_fpm=state.vs_fpm,
                ias_kt=state.ias_kt,
                dt=state.dt,
                throttle_limit=throttle_limit,
            )
        elif guidance.vertical_mode is VerticalMode.FLARE_TRACK:
            pitch_cmd_deg = self.flare_controller.target_pitch_deg(
                alt_agl_ft=state.alt_agl_ft,
                sink_rate_fpm=state.vs_fpm,
                ias_error_kt=(guidance.target_speed_kt or state.ias_kt) - state.ias_kt,
            )
            throttle_cmd = 0.0
        else:
            pitch_cmd_deg = guidance.target_pitch_deg or 0.0
            throttle_cmd = throttle_limit[1]

        elevator = self.pitch_controller.update(pitch_cmd_deg, state.pitch_deg, state.q_rad_s, state.dt)
        return ActuatorCommands(
            aileron=clamp(aileron, -1.0, 1.0),
            elevator=clamp(elevator, -1.0, 1.0),
            rudder=clamp(rudder, -1.0, 1.0),
            throttle=clamp(throttle_cmd, 0.0, 1.0),
            flaps=guidance.flaps_cmd,
            gear_down=True,
            brakes=guidance.brakes,
        )
