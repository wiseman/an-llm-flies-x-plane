from __future__ import annotations

import unittest

from sim_pilot.core.config import load_default_config_bundle
from sim_pilot.guidance.pattern_manager import build_pattern_geometry
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


if __name__ == "__main__":
    unittest.main()
