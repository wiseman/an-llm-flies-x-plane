from __future__ import annotations

from dataclasses import dataclass
import math

from sim_pilot.core.types import AircraftState, StraightLeg, Waypoint, clamp, course_between, wrap_degrees_180, wrap_degrees_360


def _leg_projection(state: AircraftState, leg: StraightLeg) -> tuple[float, float, float]:
    path = leg.end_ft - leg.start_ft
    rel = state.position_ft - leg.start_ft
    path_length = max(path.length(), 1.0)
    path_unit = path.normalized()
    along_track_ft = rel.dot(path_unit)
    cross_track_ft = -path_unit.cross(rel)
    return along_track_ft, cross_track_ft, path_length


@dataclass(slots=True)
class L1PathFollower:
    lookahead_time_s: float = 10.0
    bank_gain: float = 1.6

    def follow_leg(self, state: AircraftState, leg: StraightLeg, max_bank_deg: float) -> tuple[float, float]:
        along_track_ft, cross_track_ft, _ = _leg_projection(state, leg)
        ground_speed_ft_s = max(state.gs_kt * 1.6878098571011957, 55.0)
        lookahead_ft = max(ground_speed_ft_s * self.lookahead_time_s, 800.0)
        course_deg = course_between(leg.start_ft, leg.end_ft)
        intercept_deg = math.degrees(math.atan2(cross_track_ft, lookahead_ft))
        desired_track_deg = wrap_degrees_360(course_deg - intercept_deg)
        track_error_deg = wrap_degrees_180(desired_track_deg - state.track_deg)
        bank_cmd_deg = clamp(
            (track_error_deg * self.bank_gain) - clamp(cross_track_ft / 180.0, -8.0, 8.0),
            -max_bank_deg,
            max_bank_deg,
        )
        if along_track_ft < -200.0:
            desired_track_deg = course_between(state.position_ft, leg.start_ft)
            track_error_deg = wrap_degrees_180(desired_track_deg - state.track_deg)
            bank_cmd_deg = clamp(track_error_deg * self.bank_gain, -max_bank_deg, max_bank_deg)
        return desired_track_deg, bank_cmd_deg

    def direct_to(self, state: AircraftState, waypoint: Waypoint, max_bank_deg: float) -> tuple[float, float]:
        desired_track_deg = course_between(state.position_ft, waypoint.position_ft)
        track_error_deg = wrap_degrees_180(desired_track_deg - state.track_deg)
        bank_cmd_deg = clamp(track_error_deg * self.bank_gain, -max_bank_deg, max_bank_deg)
        return desired_track_deg, bank_cmd_deg
