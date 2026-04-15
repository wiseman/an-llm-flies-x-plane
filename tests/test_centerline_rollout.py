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
            dt=0.1,
        )
        self.assertLess(rudder, 0.0)

    def test_left_of_centerline_commands_right_rudder(self) -> None:
        rudder = self.controller.update(
            centerline_error_ft=-50.0,
            track_error_deg=0.0,
            yaw_rate_deg_s=0.0,
            gs_kt=80.0,
            dt=0.1,
        )
        self.assertGreater(rudder, 0.0)

    def test_heading_left_of_runway_commands_right_rudder(self) -> None:
        # runway_course > heading → track_error positive → need right rudder
        rudder = self.controller.update(
            centerline_error_ft=0.0,
            track_error_deg=10.0,
            yaw_rate_deg_s=0.0,
            gs_kt=80.0,
            dt=0.1,
        )
        self.assertGreater(rudder, 0.0)

    def test_heading_right_of_runway_commands_left_rudder(self) -> None:
        rudder = self.controller.update(
            centerline_error_ft=0.0,
            track_error_deg=-10.0,
            yaw_rate_deg_s=0.0,
            gs_kt=80.0,
            dt=0.1,
        )
        self.assertLess(rudder, 0.0)

    def test_yaw_rate_damping_opposes_current_yaw(self) -> None:
        # At zero errors but yawing right: damping should give left rudder
        rudder = self.controller.update(
            centerline_error_ft=0.0,
            track_error_deg=0.0,
            yaw_rate_deg_s=5.0,
            gs_kt=80.0,
            dt=0.1,
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
            dt=0.1,
        )
        damped = CenterlineRolloutController().update(  # fresh controller to isolate from integrator state
            centerline_error_ft=15.0,
            track_error_deg=0.0,
            yaw_rate_deg_s=-3.0,
            gs_kt=80.0,
            dt=0.1,
        )
        undamped_fresh = CenterlineRolloutController().update(
            centerline_error_ft=15.0,
            track_error_deg=0.0,
            yaw_rate_deg_s=0.0,
            gs_kt=80.0,
            dt=0.1,
        )
        self.assertGreater(damped, undamped_fresh)

    def test_integrator_accumulates_for_steady_state_bias(self) -> None:
        """Regression for the live-X-Plane takeoff-roll heading drift: with a
        constant left yaw bias (e.g. P-factor), the P term alone gives the
        same rudder every tick, but the integrator should build up over
        several ticks and produce progressively more right rudder to fight
        the sustained disturbance."""
        controller = CenterlineRolloutController()
        # Simulate 20 ticks of sustained 10-deg nose-left track error with
        # zero yaw rate (as if the plane is physically pinned left)
        first = controller.update(
            centerline_error_ft=0.0, track_error_deg=10.0,
            yaw_rate_deg_s=0.0, gs_kt=80.0, dt=0.1,
        )
        for _ in range(19):
            last = controller.update(
                centerline_error_ft=0.0, track_error_deg=10.0,
                yaw_rate_deg_s=0.0, gs_kt=80.0, dt=0.1,
            )
        # Rudder should grow over time as the integrator builds up
        self.assertGreater(last, first)
        self.assertGreater(last, 0.0)  # right rudder for left-pointing nose

    def test_reset_clears_integrator(self) -> None:
        controller = CenterlineRolloutController()
        for _ in range(20):
            controller.update(
                centerline_error_ft=0.0, track_error_deg=10.0,
                yaw_rate_deg_s=0.0, gs_kt=80.0, dt=0.1,
            )
        before_reset = controller.update(
            centerline_error_ft=0.0, track_error_deg=0.0,
            yaw_rate_deg_s=0.0, gs_kt=80.0, dt=0.1,
        )
        # The integrator should still be positive from the prior ticks
        self.assertGreater(before_reset, 0.0)
        controller.reset()
        after_reset = controller.update(
            centerline_error_ft=0.0, track_error_deg=0.0,
            yaw_rate_deg_s=0.0, gs_kt=80.0, dt=0.1,
        )
        self.assertAlmostEqual(after_reset, 0.0, places=5)

    def test_moderate_error_does_not_saturate_rudder(self) -> None:
        """Regression for the oscillation: with a 25 deg track error (which
        the old gains saturated at), the new controller should leave
        headroom so damping can do its job."""
        controller = CenterlineRolloutController()
        rudder = controller.update(
            centerline_error_ft=0.0, track_error_deg=25.0,
            yaw_rate_deg_s=0.0, gs_kt=80.0, dt=0.1,
        )
        self.assertLess(abs(rudder), 0.95)


if __name__ == "__main__":
    unittest.main()
