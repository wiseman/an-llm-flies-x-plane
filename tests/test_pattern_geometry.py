from __future__ import annotations

import unittest

from sim_pilot.core.config import load_default_config_bundle
from sim_pilot.guidance.pattern_manager import (
    build_pattern_geometry,
    glidepath_target_altitude_ft,
)
from sim_pilot.guidance.runway_geometry import RunwayFrame


class PatternGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_default_config_bundle()
        self.runway_frame = RunwayFrame(self.config.airport.runway)

    def test_downwind_line_generated_on_correct_side(self) -> None:
        pattern = build_pattern_geometry(
            self.runway_frame,
            downwind_offset_ft=self.config.pattern.downwind_offset_ft,
            extension_ft=0.0,
        )
        start = self.runway_frame.to_runway_frame(pattern.downwind_leg.start_ft)
        end = self.runway_frame.to_runway_frame(pattern.downwind_leg.end_ft)
        self.assertAlmostEqual(start.y, -self.config.pattern.downwind_offset_ft)
        self.assertAlmostEqual(end.y, -self.config.pattern.downwind_offset_ft)

    def test_base_turn_point_moves_after_extension(self) -> None:
        nominal = build_pattern_geometry(
            self.runway_frame,
            downwind_offset_ft=self.config.pattern.downwind_offset_ft,
            extension_ft=0.0,
        )
        extended = build_pattern_geometry(
            self.runway_frame,
            downwind_offset_ft=self.config.pattern.downwind_offset_ft,
            extension_ft=2000.0,
        )
        self.assertLess(extended.base_turn_x_ft, nominal.base_turn_x_ft)
        self.assertAlmostEqual(extended.base_turn_x_ft - nominal.base_turn_x_ft, -2000.0)

    def test_final_intercept_stays_on_centerline(self) -> None:
        pattern = build_pattern_geometry(
            self.runway_frame,
            downwind_offset_ft=self.config.pattern.downwind_offset_ft,
            extension_ft=0.0,
        )
        base_end = self.runway_frame.to_runway_frame(pattern.base_leg.end_ft)
        final_start = self.runway_frame.to_runway_frame(pattern.final_leg.start_ft)
        final_end = self.runway_frame.to_runway_frame(pattern.final_leg.end_ft)
        self.assertAlmostEqual(base_end.y, 0.0)
        self.assertAlmostEqual(final_start.y, 0.0)
        self.assertAlmostEqual(final_end.y, 0.0)
        self.assertLess(final_start.x, base_end.x)
        self.assertLess(base_end.x, final_end.x)


class GlidepathTargetAltitudeTests(unittest.TestCase):
    """Regression tests for the glidepath altitude math. The aim point
    is at ground level by default (aim_point_height_agl_ft=0), and the
    glidepath rises along the slope going backward from the aim point.
    Past the aim point the target continues down toward ground, clamped
    at field elevation."""

    def setUp(self) -> None:
        self.config = load_default_config_bundle()
        self.runway_frame = RunwayFrame(self.config.airport.runway)
        self.field_elev_ft = self.config.airport.field_elevation_ft

    def test_altitude_at_aim_point_is_ground_level_by_default(self) -> None:
        aim_x = self.runway_frame.touchdown_runway_x_ft
        target = glidepath_target_altitude_ft(
            self.runway_frame,
            runway_x_ft=aim_x,
            field_elevation_ft=self.field_elev_ft,
        )
        self.assertAlmostEqual(target, self.field_elev_ft)

    def test_altitude_rises_before_the_aim_point(self) -> None:
        aim_x = self.runway_frame.touchdown_runway_x_ft
        # 2000 ft before the aim point at 3° slope = 2000 * tan(3°)
        # ≈ 104.7 ft above the aim-point altitude (ground level).
        target = glidepath_target_altitude_ft(
            self.runway_frame,
            runway_x_ft=aim_x - 2000.0,
            field_elevation_ft=self.field_elev_ft,
        )
        self.assertAlmostEqual(
            target - self.field_elev_ft, 2000.0 * 3.0 / 57.2958, places=1
        )

    def test_threshold_crossing_height_is_implicit(self) -> None:
        # With the aim 1000 ft past the threshold and a 3° slope, the
        # threshold crossing height falls out as 1000 * tan(3°) ≈ 52 ft
        # AGL — this is what a standard visual approach profile looks
        # like.
        aim_x = self.runway_frame.touchdown_runway_x_ft
        target = glidepath_target_altitude_ft(
            self.runway_frame,
            runway_x_ft=0.0,
            field_elevation_ft=self.field_elev_ft,
        )
        expected_tch_ft = aim_x * 3.0 / 57.2958
        self.assertAlmostEqual(target - self.field_elev_ft, expected_tch_ft, places=1)

    def test_altitude_descends_past_the_aim_point(self) -> None:
        aim_x = self.runway_frame.touchdown_runway_x_ft
        target_400_past = glidepath_target_altitude_ft(
            self.runway_frame,
            runway_x_ft=aim_x + 400.0,
            field_elevation_ft=self.field_elev_ft,
        )
        # 400 ft past the aim point would mathematically be 21 ft
        # below ground; the formula clamps at field elevation.
        self.assertAlmostEqual(target_400_past, self.field_elev_ft)

    def test_altitude_clamps_at_ground_level(self) -> None:
        aim_x = self.runway_frame.touchdown_runway_x_ft
        target = glidepath_target_altitude_ft(
            self.runway_frame,
            runway_x_ft=aim_x + 5000.0,
            field_elevation_ft=self.field_elev_ft,
        )
        self.assertAlmostEqual(target, self.field_elev_ft)

    def test_non_zero_aim_point_height_adds_offset(self) -> None:
        # An instrument-style approach with a 50 ft aim-point height
        # lifts the whole glidepath by 50 ft.
        aim_x = self.runway_frame.touchdown_runway_x_ft
        target = glidepath_target_altitude_ft(
            self.runway_frame,
            runway_x_ft=aim_x,
            field_elevation_ft=self.field_elev_ft,
            aim_point_height_agl_ft=50.0,
        )
        self.assertAlmostEqual(target - self.field_elev_ft, 50.0)


if __name__ == "__main__":
    unittest.main()
