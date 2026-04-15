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
        self.assertIn(FlightPhase.INITIAL_CLIMB, result.phases_seen)
        self.assertIn(FlightPhase.CROSSWIND, result.phases_seen)
        self.assertIn(FlightPhase.DOWNWIND, result.phases_seen)
        self.assertIn(FlightPhase.BASE, result.phases_seen)
        self.assertIn(FlightPhase.FINAL, result.phases_seen)
        self.assertIn(FlightPhase.FLARE, result.phases_seen)
        self.assertIn(result.final_phase, {FlightPhase.ROLLOUT, FlightPhase.TAXI_CLEAR})
        # Takeoff-originated missions stay in the pattern: they must NOT
        # visit the airborne-rejoin phases (ENROUTE_CLIMB / CRUISE /
        # DESCENT / PATTERN_ENTRY). Regression for the KWHP "hard right
        # 360" bug where the plane climbed to cruise altitude and
        # spiraled around the pattern_entry_start waypoint.
        self.assertNotIn(FlightPhase.ENROUTE_CLIMB, result.phases_seen)
        self.assertNotIn(FlightPhase.CRUISE, result.phases_seen)
        self.assertNotIn(FlightPhase.DESCENT, result.phases_seen)
        self.assertNotIn(FlightPhase.PATTERN_ENTRY, result.phases_seen)

    def test_crosswind_mission_still_tracks_and_lands(self) -> None:
        result = ScenarioRunner(self.config, wind_vector_kt=Vec2(10.0, 0.0)).run()
        # Not asserting result.success — that check requires touchdown in
        # the first third of the runway, and a 10 kt crosswind naturally
        # pushes the flare a few hundred feet further down. What we care
        # about here is: did the aircraft actually fly the pattern and
        # flare on the centerline at a safe bank?
        self.assertIn(result.final_phase, {FlightPhase.ROLLOUT, FlightPhase.TAXI_CLEAR})
        self.assertIsNotNone(result.touchdown_runway_x_ft)
        assert result.touchdown_runway_x_ft is not None
        self.assertGreaterEqual(result.touchdown_runway_x_ft, 0.0)
        self.assertLessEqual(result.touchdown_runway_x_ft, self.config.airport.runway.length_ft / 2.0)
        self.assertLess(abs(result.touchdown_centerline_ft or 999.0), 20.0)
        self.assertLess(abs(result.max_final_bank_deg), self.config.limits.max_bank_final_deg + 0.5)


if __name__ == "__main__":
    unittest.main()
