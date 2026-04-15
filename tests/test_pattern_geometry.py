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
    """Regression tests for the glidepath altitude floor.

    The old implementation clamped distance_from_aimpoint at zero once
    the aircraft was past the aim point, so target altitude froze at
    field_elev + threshold_crossing_height (≈50 AGL) forever. On a
    short runway (KWHP, 4120 ft) this trapped the aircraft at 50 AGL
    over a thousand feet of runway before the roundout trigger fired,
    and touchdown happened ~3000 ft past the threshold. The fix: keep
    descending along the 3° slope past the aim point, clamped at
    ground level."""

    def setUp(self) -> None:
        self.config = load_default_config_bundle()
        self.runway_frame = RunwayFrame(self.config.airport.runway)
        self.field_elev_ft = self.config.airport.field_elevation_ft

    def test_altitude_at_aim_point_equals_threshold_crossing_height(self) -> None:
        aim_x = self.runway_frame.touchdown_runway_x_ft
        target = glidepath_target_altitude_ft(
            self.runway_frame,
            runway_x_ft=aim_x,
            field_elevation_ft=self.field_elev_ft,
        )
        self.assertAlmostEqual(target - self.field_elev_ft, 50.0)

    def test_altitude_rises_before_the_aim_point(self) -> None:
        aim_x = self.runway_frame.touchdown_runway_x_ft
        # 2000 ft before the aim point at 3° slope = 104.7 ft above the
        # aim-point altitude = 154.7 ft AGL.
        target = glidepath_target_altitude_ft(
            self.runway_frame,
            runway_x_ft=aim_x - 2000.0,
            field_elevation_ft=self.field_elev_ft,
        )
        self.assertAlmostEqual(target - self.field_elev_ft, 50.0 + 2000.0 * 3.0 / 57.2958, places=1)

    def test_altitude_descends_past_the_aim_point(self) -> None:
        # Past the aim point the target must continue to decrease, not
        # freeze at 50 AGL. Regression for the KWHP "long landing" bug.
        aim_x = self.runway_frame.touchdown_runway_x_ft
        target_400_past = glidepath_target_altitude_ft(
            self.runway_frame,
            runway_x_ft=aim_x + 400.0,
            field_elevation_ft=self.field_elev_ft,
        )
        self.assertLess(target_400_past - self.field_elev_ft, 50.0)
        # At 3° slope the altitude drops by 400 * tan(3°) ≈ 21 ft below
        # the aim-point altitude, so 50 - 21 = 29 AGL.
        self.assertAlmostEqual(target_400_past - self.field_elev_ft, 50.0 - 400.0 * 3.0 / 57.2958, places=1)

    def test_altitude_clamps_at_ground_level(self) -> None:
        # Far past the aim point the formula would go below ground — it
        # must clamp at field elevation, not command a subterranean
        # altitude to TECS.
        aim_x = self.runway_frame.touchdown_runway_x_ft
        target = glidepath_target_altitude_ft(
            self.runway_frame,
            runway_x_ft=aim_x + 5000.0,
            field_elevation_ft=self.field_elev_ft,
        )
        self.assertAlmostEqual(target, self.field_elev_ft)


if __name__ == "__main__":
    unittest.main()
