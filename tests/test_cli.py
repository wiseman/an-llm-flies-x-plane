from __future__ import annotations

import unittest

from sim_pilot.__main__ import resolve_scenario_name
from sim_pilot.core.types import Vec2


class CLITests(unittest.TestCase):
    def test_nominal_scenario_name_when_no_wind(self) -> None:
        self.assertEqual(resolve_scenario_name(None, Vec2(0.0, 0.0)), "takeoff_to_pattern_landing")

    def test_crosswind_scenario_name_when_only_x_wind(self) -> None:
        self.assertEqual(
            resolve_scenario_name(None, Vec2(10.0, 0.0)),
            "takeoff_to_pattern_landing_crosswind_10.0kt",
        )

    def test_explicit_scenario_name_takes_precedence(self) -> None:
        self.assertEqual(resolve_scenario_name("pattern_debug", Vec2(10.0, 5.0)), "pattern_debug")


if __name__ == "__main__":
    unittest.main()
