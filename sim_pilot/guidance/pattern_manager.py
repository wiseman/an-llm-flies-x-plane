from __future__ import annotations

from dataclasses import dataclass

from sim_pilot.core.types import FlightPhase, StraightLeg, TrafficSide, Vec2, clamp, wrap_degrees_180, wrap_degrees_360
from sim_pilot.guidance.runway_geometry import RunwayFrame


@dataclass(slots=True, frozen=True)
class PatternGeometry:
    runway_frame: RunwayFrame
    entry_leg: StraightLeg
    downwind_leg: StraightLeg
    base_leg: StraightLeg
    final_leg: StraightLeg
    join_point_runway_ft: Vec2
    base_turn_x_ft: float
    downwind_y_ft: float
    extension_ft: float

    @property
    def reciprocal_course_deg(self) -> float:
        return wrap_degrees_360(self.runway_frame.runway.course_deg + 180.0)

    def leg_for_phase(self, phase: FlightPhase) -> StraightLeg | None:
        if phase is FlightPhase.PATTERN_ENTRY:
            return self.entry_leg
        if phase is FlightPhase.DOWNWIND:
            return self.downwind_leg
        if phase is FlightPhase.BASE:
            return self.base_leg
        if phase is FlightPhase.FINAL:
            return self.final_leg
        return None

    def is_established_on_downwind(self, runway_x_ft: float | None, runway_y_ft: float | None, track_deg: float) -> bool:
        if runway_x_ft is None or runway_y_ft is None:
            return False
        altitude_band_ok = abs(runway_y_ft - self.downwind_y_ft) <= 350.0
        track_ok = abs(wrap_degrees_180(track_deg - self.reciprocal_course_deg)) <= 20.0
        return altitude_band_ok and track_ok and runway_x_ft <= self.join_point_runway_ft.x

    def base_turn_ready(self, runway_x_ft: float | None) -> bool:
        if runway_x_ft is None:
            return False
        return runway_x_ft <= self.base_turn_x_ft

    def is_established_on_final(self, runway_x_ft: float | None, runway_y_ft: float | None, track_deg: float) -> bool:
        if runway_x_ft is None or runway_y_ft is None:
            return False
        return (
            runway_x_ft <= -500.0
            and abs(runway_y_ft) <= 120.0
            and abs(wrap_degrees_180(track_deg - self.runway_frame.runway.course_deg)) <= 15.0
        )


def build_pattern_geometry(
    runway_frame: RunwayFrame,
    downwind_offset_ft: float,
    extension_ft: float,
) -> PatternGeometry:
    side_sign = -1.0 if runway_frame.runway.traffic_side is TrafficSide.LEFT else 1.0
    downwind_y_ft = side_sign * downwind_offset_ft
    join_x_ft = downwind_offset_ft * 1.1
    base_turn_x_ft = -(downwind_offset_ft + 1500.0 + extension_ft)
    final_intercept_x_ft = -max(4500.0, downwind_offset_ft)
    final_start_x_ft = -max(10000.0, downwind_offset_ft * 2.5)

    entry_start_runway_ft = Vec2(join_x_ft + downwind_offset_ft, downwind_y_ft + (side_sign * downwind_offset_ft))
    join_point_runway_ft = Vec2(join_x_ft, downwind_y_ft)
    downwind_end_runway_ft = Vec2(base_turn_x_ft, downwind_y_ft)
    base_end_runway_ft = Vec2(final_intercept_x_ft, 0.0)

    entry_leg = StraightLeg(
        start_ft=runway_frame.to_world_frame(entry_start_runway_ft),
        end_ft=runway_frame.to_world_frame(join_point_runway_ft),
    )
    downwind_leg = StraightLeg(
        start_ft=runway_frame.to_world_frame(join_point_runway_ft),
        end_ft=runway_frame.to_world_frame(downwind_end_runway_ft),
    )
    base_leg = StraightLeg(
        start_ft=runway_frame.to_world_frame(downwind_end_runway_ft),
        end_ft=runway_frame.to_world_frame(base_end_runway_ft),
    )
    final_leg = runway_frame.final_leg(start_x_ft=final_start_x_ft)
    return PatternGeometry(
        runway_frame=runway_frame,
        entry_leg=entry_leg,
        downwind_leg=downwind_leg,
        base_leg=base_leg,
        final_leg=final_leg,
        join_point_runway_ft=join_point_runway_ft,
        base_turn_x_ft=base_turn_x_ft,
        downwind_y_ft=downwind_y_ft,
        extension_ft=extension_ft,
    )


def glidepath_target_altitude_ft(
    runway_frame: RunwayFrame,
    runway_x_ft: float,
    field_elevation_ft: float,
    slope_deg: float = 3.0,
    threshold_crossing_height_ft: float = 50.0,
) -> float:
    distance_from_threshold_ft = max(0.0, runway_frame.touchdown_runway_x_ft - runway_x_ft)
    path_height_ft = threshold_crossing_height_ft + (distance_from_threshold_ft * clamp(slope_deg / 57.2958, 0.0, 0.2))
    return field_elevation_ft + path_height_ft
