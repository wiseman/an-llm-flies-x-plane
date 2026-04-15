from __future__ import annotations

from dataclasses import dataclass

from sim_pilot.core.config import ConfigBundle
from sim_pilot.core.safety_monitor import SafetyStatus
from sim_pilot.core.types import AircraftState, FlightPhase, clamp, wrap_degrees_180
from sim_pilot.guidance.pattern_manager import PatternGeometry
from sim_pilot.guidance.route_manager import RouteManager


@dataclass(slots=True)
class ModeManager:
    config: ConfigBundle

    def update(
        self,
        phase: FlightPhase,
        state: AircraftState,
        route_manager: RouteManager,
        pattern: PatternGeometry,
        safety_status: SafetyStatus,
        *,
        turn_base_now: bool = False,
        force_go_around: bool = False,
        stay_in_pattern: bool = False,
    ) -> FlightPhase:
        if force_go_around or safety_status.request_go_around:
            return FlightPhase.GO_AROUND

        if phase is FlightPhase.PREFLIGHT:
            # A profile engaged while already airborne should skip the
            # ground phases entirely. Without this, a mid-flight
            # engagement spends one tick in TAKEOFF_ROLL emitting
            # rollout/Vr guidance (wrong for 90 kt at 300 ft AGL) before
            # the airborne bailout below fires on the next tick. Seen in
            # output/sim_pilot-20260415-094519.log where the LLM engaged
            # TakeoffProfile in mid-air as a recovery and got `tgt_spd=55`
            # for one tick.
            if not state.on_ground and state.ias_kt >= self.config.performance.vr_kt:
                return FlightPhase.INITIAL_CLIMB
            return FlightPhase.TAKEOFF_ROLL
        if phase is FlightPhase.TAKEOFF_ROLL:
            # Airborne bailout: if the wheels have somehow left the ground
            # at or above Vr, the state machine MUST advance — rollout can
            # no longer act on the aircraft and rotate/climb guidance is
            # strictly safer than a stuck TAKEOFF_ROLL. Seen in the KWHP
            # live log where the aircraft ended up airborne at 100 kt and
            # 2800 ft off centerline because yoke_heading_ratio writes
            # weren't applied (the rollout rudder was a no-op).
            if not state.on_ground and state.ias_kt >= self.config.performance.vr_kt:
                return FlightPhase.ROTATE
            # Normal rotate trigger. Centerline tolerance relaxed from 25 ft
            # (tight enough to trap a takeoff in any real crosswind) to
            # unstable_centerline_error_ft / 2 — 50 ft by default. Below
            # this we're tracking the runway; above it we're not yet ready
            # to rotate and want more time for the rollout controller (or
            # a Safety-monitor abort) to react.
            centerline_limit_ft = self.config.limits.unstable_centerline_error_ft * 0.5
            if (
                state.ias_kt >= self.config.performance.vr_kt
                and abs(state.centerline_error_ft or 0.0) <= centerline_limit_ft
            ):
                return FlightPhase.ROTATE
            return phase
        if phase is FlightPhase.ROTATE:
            if (not state.on_ground) and state.alt_agl_ft >= 20.0 and state.vs_fpm > 100.0:
                return FlightPhase.INITIAL_CLIMB
            return phase
        if phase is FlightPhase.INITIAL_CLIMB:
            if stay_in_pattern:
                # Stay-in-pattern missions turn crosswind "within 300 ft
                # below pattern altitude" (AC 90-66C). For a 1000 AGL TPA
                # that's 700 AGL minimum.
                crosswind_turn_agl_ft = max(
                    400.0, self.config.pattern.altitude_agl_ft - 300.0
                )
                if state.alt_agl_ft >= crosswind_turn_agl_ft and state.vs_fpm > 250.0:
                    return FlightPhase.CROSSWIND
                return phase
            if state.alt_agl_ft >= 400.0 and state.vs_fpm > 250.0:
                return FlightPhase.ENROUTE_CLIMB
            return phase
        if phase is FlightPhase.CROSSWIND:
            # Turn downwind when we've both captured the crosswind heading
            # AND reached the full downwind offset. Requiring heading-
            # captured prevents the old bug where the 80%-offset check
            # fired while the plane was still in the middle of its turn
            # from upwind to crosswind, so the downwind leg started with
            # 100° of remaining turn and L1 had to drag the plane around
            # at a shallow bank while altitude sagged below pattern
            # altitude (observed in sim_pilot-20260415-130505.log: the
            # plane rolled out on downwind at 700 AGL instead of 1000).
            if state.runway_y_ft is None:
                return phase
            downwind_offset_ft = abs(pattern.downwind_y_ft)
            runway_course_deg = pattern.runway_frame.runway.course_deg
            side_sign = -1.0 if pattern.downwind_y_ft < 0.0 else 1.0
            crosswind_course_deg = (runway_course_deg + side_sign * 90.0) % 360.0
            heading_error_deg = abs(wrap_degrees_180(state.track_deg - crosswind_course_deg))
            heading_captured = heading_error_deg <= 15.0
            at_offset = abs(state.runway_y_ft) >= downwind_offset_ft
            if heading_captured and at_offset:
                return FlightPhase.DOWNWIND
            return phase
        if phase is FlightPhase.ENROUTE_CLIMB:
            if state.alt_msl_ft >= self.config.cruise_altitude_ft - 150.0:
                return FlightPhase.CRUISE
            return phase
        if phase is FlightPhase.CRUISE:
            waypoint = route_manager.active_waypoint()
            if waypoint is not None and waypoint.name == "pattern_entry_start":
                return FlightPhase.DESCENT
            return phase
        if phase is FlightPhase.DESCENT:
            waypoint = route_manager.active_waypoint()
            if waypoint is not None and waypoint.name == "pattern_entry_start":
                if state.position_ft.distance_to(waypoint.position_ft) <= 2500.0 and abs(state.alt_msl_ft - self.config.pattern_altitude_msl_ft) <= 450.0:
                    return FlightPhase.PATTERN_ENTRY
            return phase
        if phase is FlightPhase.PATTERN_ENTRY:
            if state.runway_x_ft is not None and state.runway_y_ft is not None:
                join_reached = (
                    state.runway_x_ft <= (pattern.join_point_runway_ft.x + 400.0)
                    and abs(state.runway_y_ft - pattern.downwind_y_ft) <= 900.0
                )
                if join_reached or pattern.is_established_on_downwind(state.runway_x_ft, state.runway_y_ft, state.track_deg):
                    return FlightPhase.DOWNWIND
            return phase
        if phase is FlightPhase.DOWNWIND:
            if turn_base_now or self._downwind_base_turn_ready(state, pattern):
                return FlightPhase.BASE
            return phase
        if phase is FlightPhase.BASE:
            # Fire at 65% along-track so L1 has ~1225 ft of anticipation
            # room to start the 90° turn onto final before reaching the
            # extended centerline (~1 turn radius at 25° bank / 65 kt).
            if pattern.is_established_on_final(state.runway_x_ft, state.runway_y_ft, state.track_deg):
                return FlightPhase.FINAL
            leg = pattern.base_leg
            path = leg.end_ft - leg.start_ft
            rel = state.position_ft - leg.start_ft
            path_length = max(path.length(), 1.0)
            along_track_ft = rel.dot(path.normalized())
            if along_track_ft >= path_length * 0.65:
                return FlightPhase.FINAL
            return phase
        if phase is FlightPhase.FINAL:
            if state.alt_agl_ft <= self.config.flare.roundout_height_ft:
                return FlightPhase.ROUNDOUT
            return phase
        if phase is FlightPhase.ROUNDOUT:
            if state.alt_agl_ft <= self.config.flare.flare_start_ft:
                return FlightPhase.FLARE
            return phase
        if phase is FlightPhase.FLARE:
            if state.on_ground:
                return FlightPhase.ROLLOUT
            return phase
        if phase is FlightPhase.ROLLOUT:
            if state.gs_kt <= 5.0:
                return FlightPhase.TAXI_CLEAR
            return phase
        if phase is FlightPhase.GO_AROUND:
            # Stay in GO_AROUND indefinitely: climb to pattern altitude
            # on runway course and then hold there until the LLM decides
            # what to do next (typically "fly another pattern" via
            # engage_pattern_fly, or divert). The old transition was
            # GO_AROUND → ENROUTE_CLIMB at 400 AGL, which then targeted
            # cruise altitude (3000 MSL) — so a go-around from a bad
            # final approach would send the aircraft climbing toward
            # cruise altitude instead of holding pattern altitude.
            return phase
        return phase

    def _downwind_base_turn_ready(self, state: AircraftState, pattern: PatternGeometry) -> bool:
        if state.runway_x_ft is None:
            return False

        # Grow the turn window toward the abeam-numbers point for slow
        # aircraft, so they start the turn earlier and don't overshoot
        # during the base descent.
        nominal_turn_x_ft = pattern.base_turn_x_ft
        earliest_turn_x_ft = self.config.pattern.abeam_window_ft
        speed_shortfall_kt = max(
            0.0, self.config.performance.downwind_speed_kt - state.gs_kt
        )
        max_relief_ft = max(0.0, earliest_turn_x_ft - nominal_turn_x_ft)
        relief_ft = clamp(speed_shortfall_kt * 140.0, 0.0, max_relief_ft)
        adaptive_turn_x_ft = nominal_turn_x_ft + relief_ft
        return state.runway_x_ft <= adaptive_turn_x_ft
