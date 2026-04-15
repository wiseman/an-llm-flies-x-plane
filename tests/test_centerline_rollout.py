from __future__ import annotations

import unittest

from sim_pilot.control.centerline_rollout import CenterlineRolloutController


class CenterlineRolloutSignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = CenterlineRolloutController()

    def test_right_of_centerline_commands_left_rudder(self) -> None:
        rudder = self.controller.update(
            centerline_error_ft=50.0,
            track_error_deg=0.0,
            yaw_rate_deg_s=0.0,
            gs_kt=80.0,
        )
        self.assertLess(rudder, 0.0)

    def test_left_of_centerline_commands_right_rudder(self) -> None:
        rudder = self.controller.update(
            centerline_error_ft=-50.0,
            track_error_deg=0.0,
            yaw_rate_deg_s=0.0,
            gs_kt=80.0,
        )
        self.assertGreater(rudder, 0.0)

    def test_heading_left_of_runway_commands_right_rudder(self) -> None:
        # runway_course > heading → track_error positive → need right rudder
        rudder = self.controller.update(
            centerline_error_ft=0.0,
            track_error_deg=10.0,
            yaw_rate_deg_s=0.0,
            gs_kt=80.0,
        )
        self.assertGreater(rudder, 0.0)

    def test_heading_right_of_runway_commands_left_rudder(self) -> None:
        rudder = self.controller.update(
            centerline_error_ft=0.0,
            track_error_deg=-10.0,
            yaw_rate_deg_s=0.0,
            gs_kt=80.0,
        )
        self.assertLess(rudder, 0.0)

    def test_yaw_rate_damping_opposes_current_yaw(self) -> None:
        # At zero errors but yawing right: damping should give left rudder
        rudder = self.controller.update(
            centerline_error_ft=0.0,
            track_error_deg=0.0,
            yaw_rate_deg_s=5.0,
            gs_kt=80.0,
        )
        self.assertLess(rudder, 0.0)

    def test_damping_reduces_rudder_when_already_turning_toward_target(self) -> None:
        # Aircraft is right of centerline (want left rudder, negative) and
        # already yawing left at 3 deg/s (damping should oppose → less negative).
        # Use a small enough error to stay off the output clamp.
        undamped = self.controller.update(
            centerline_error_ft=15.0,
            track_error_deg=0.0,
            yaw_rate_deg_s=0.0,
            gs_kt=80.0,
        )
        damped = self.controller.update(
            centerline_error_ft=15.0,
            track_error_deg=0.0,
            yaw_rate_deg_s=-3.0,
            gs_kt=80.0,
        )
        self.assertGreater(damped, undamped)


if __name__ == "__main__":
    unittest.main()
