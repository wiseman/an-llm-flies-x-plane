from __future__ import annotations

from dataclasses import dataclass

from sim_pilot.core.config import ConfigBundle
from sim_pilot.core.safety_monitor import SafetyStatus
from sim_pilot.core.types import AircraftState, FlightPhase
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
    ) -> FlightPhase:
        if safety_status.request_go_around:
            return FlightPhase.GO_AROUND

        if phase is FlightPhase.PREFLIGHT:
            return FlightPhase.TAKEOFF_ROLL
        if phase is FlightPhase.TAKEOFF_ROLL:
            if state.ias_kt >= self.config.performance.vr_kt and abs(state.centerline_error_ft or 0.0) <= 25.0:
                return FlightPhase.ROTATE
            return phase
        if phase is FlightPhase.ROTATE:
            if (not state.on_ground) and state.alt_agl_ft >= 20.0 and state.vs_fpm > 100.0:
                return FlightPhase.INITIAL_CLIMB
            return phase
        if phase is FlightPhase.INITIAL_CLIMB:
            if state.alt_agl_ft >= 400.0 and state.vs_fpm > 250.0:
                return FlightPhase.ENROUTE_CLIMB
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
            if pattern.base_turn_ready(state.runway_x_ft):
                return FlightPhase.BASE
            return phase
        if phase is FlightPhase.BASE:
            if state.position_ft.distance_to(pattern.base_leg.end_ft) <= 1400.0 or pattern.is_established_on_final(state.runway_x_ft, state.runway_y_ft, state.track_deg):
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
            if state.alt_agl_ft >= 400.0:
                return FlightPhase.ENROUTE_CLIMB
        return phase
