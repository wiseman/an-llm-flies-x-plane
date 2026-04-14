from __future__ import annotations

import unittest

from sim_pilot.control.tecs_lite import TECSLite
from sim_pilot.core.config import load_default_config_bundle
from sim_pilot.core.types import FlightPhase


class TECSLiteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = TECSLite(load_default_config_bundle().controllers.tecs)

    def test_below_target_altitude_and_slow_commands_more_pitch_and_power(self) -> None:
        pitch_cmd_deg, throttle_cmd = self.controller.update(
            phase=FlightPhase.ENROUTE_CLIMB,
            target_alt_ft=3000.0,
            target_speed_kt=74.0,
            alt_ft=2200.0,
            vs_fpm=400.0,
            ias_kt=66.0,
            dt=0.2,
            throttle_limit=(0.75, 1.0),
        )
        self.assertGreater(pitch_cmd_deg, 6.0)
        self.assertGreater(throttle_cmd, 0.85)

    def test_above_target_and_fast_commands_descent_and_power_reduction(self) -> None:
        pitch_cmd_deg, throttle_cmd = self.controller.update(
            phase=FlightPhase.DESCENT,
            target_alt_ft=1500.0,
            target_speed_kt=85.0,
            alt_ft=2500.0,
            vs_fpm=-300.0,
            ias_kt=100.0,
            dt=0.2,
            throttle_limit=(0.1, 0.6),
        )
        self.assertLess(pitch_cmd_deg, 0.0)
        self.assertLess(throttle_cmd, 0.35)


if __name__ == "__main__":
    unittest.main()
