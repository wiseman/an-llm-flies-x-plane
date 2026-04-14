from __future__ import annotations

import unittest

from sim_pilot.core.config import load_default_config_bundle
from sim_pilot.core.types import FlightPhase, Vec2
from sim_pilot.sim.scenario import ScenarioRunner


class ScenarioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_default_config_bundle()

    def test_nominal_mission_completes_takeoff_to_rollout(self) -> None:
        result = ScenarioRunner(self.config).run()
        self.assertTrue(result.success)
        self.assertIn(FlightPhase.TAKEOFF_ROLL, result.phases_seen)
        self.assertIn(FlightPhase.ROTATE, result.phases_seen)
        self.assertIn(FlightPhase.PATTERN_ENTRY, result.phases_seen)
        self.assertIn(FlightPhase.DOWNWIND, result.phases_seen)
        self.assertIn(FlightPhase.BASE, result.phases_seen)
        self.assertIn(FlightPhase.FINAL, result.phases_seen)
        self.assertIn(FlightPhase.FLARE, result.phases_seen)
        self.assertIn(result.final_phase, {FlightPhase.ROLLOUT, FlightPhase.TAXI_CLEAR})

    def test_crosswind_mission_still_tracks_and_lands(self) -> None:
        result = ScenarioRunner(self.config, wind_vector_kt=Vec2(10.0, 0.0)).run()
        self.assertTrue(result.success)
        self.assertLess(abs(result.touchdown_centerline_ft or 999.0), 20.0)
        self.assertLess(abs(result.max_final_bank_deg), self.config.limits.max_bank_final_deg + 0.5)


if __name__ == "__main__":
    unittest.main()
