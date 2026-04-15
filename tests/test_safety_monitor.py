from __future__ import annotations

import unittest

from sim_pilot.core.config import load_default_config_bundle
from sim_pilot.core.safety_monitor import SafetyMonitor
from sim_pilot.core.types import AircraftState, FlightPhase, KT_TO_FPS, Vec2, heading_to_vector


def make_state(**overrides: object) -> AircraftState:
    defaults: dict[str, object] = {
        "t_sim": 0.0,
        "dt": 0.2,
        "position_ft": Vec2(0.0, 0.0),
        "alt_msl_ft": 650.0,
        "alt_agl_ft": 150.0,
        "pitch_deg": 0.0,
        "roll_deg": 5.0,
        "heading_deg": 0.0,
        "track_deg": 0.0,
        "p_rad_s": 0.0,
        "q_rad_s": 0.0,
        "r_rad_s": 0.0,
        "ias_kt": 65.0,
        "tas_kt": 65.0,
        "gs_kt": 65.0,
        "vs_fpm": -400.0,
        "ground_velocity_ft_s": heading_to_vector(0.0, 65.0 * KT_TO_FPS),
        "flap_index": 20,
        "gear_down": True,
        "on_ground": False,
        "throttle_pos": 0.3,
        "runway_id": "36",
        "runway_dist_remaining_ft": None,
        "runway_x_ft": -1200.0,
        "runway_y_ft": 0.0,
        "centerline_error_ft": 0.0,
        "threshold_abeam": False,
        "distance_to_touchdown_ft": 2500.0,
        "stall_margin": 1.35,
    }
    defaults.update(overrides)
    return AircraftState(**defaults)


class SafetyMonitorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.monitor = SafetyMonitor(load_default_config_bundle())

    def test_unstable_lateral_alignment_requests_go_around(self) -> None:
        # At 150 ft AGL the scaled limit is 300 ft; 400 ft is above it.
        status = self.monitor.evaluate(
            make_state(alt_agl_ft=150.0, centerline_error_ft=400.0),
            FlightPhase.FINAL,
        )
        self.assertTrue(status.request_go_around)
        assert status.reason is not None
        self.assertIn("unstable_lateral", status.reason)
        self.assertIn("cle=400ft", status.reason)
        self.assertIn("agl=150ft", status.reason)
        self.assertIn("limit=300ft", status.reason)

    def test_low_stall_margin_requests_go_around(self) -> None:
        status = self.monitor.evaluate(
            make_state(alt_agl_ft=40.0, stall_margin=1.05),
            FlightPhase.FINAL,
        )
        self.assertTrue(status.request_go_around)
        assert status.reason is not None
        self.assertIn("low_energy", status.reason)
        self.assertIn("stall_margin=1.05", status.reason)

    def test_centerline_limit_scales_with_altitude(self) -> None:
        # At 200 ft AGL (high end), 300 ft off is within the 400 ft limit.
        high_alt = self.monitor.evaluate(
            make_state(alt_agl_ft=199.0, centerline_error_ft=300.0),
            FlightPhase.FINAL,
        )
        self.assertFalse(high_alt.request_go_around)
        # At 40 ft AGL the scaled limit is 80 ft; 100 ft is over it.
        low_alt = self.monitor.evaluate(
            make_state(alt_agl_ft=40.0, centerline_error_ft=100.0, stall_margin=2.0),
            FlightPhase.FINAL,
        )
        self.assertTrue(low_alt.request_go_around)
        assert low_alt.reason is not None
        self.assertIn("unstable_lateral", low_alt.reason)

    def test_centerline_limit_has_30_foot_floor_near_ground(self) -> None:
        # At 5 ft AGL, the scaled limit (2 * 5 = 10 ft) must be clamped
        # to the 30 ft floor — otherwise the floor doesn't protect us in
        # the flare and every slight deviation trips GA.
        status = self.monitor.evaluate(
            make_state(alt_agl_ft=5.0, centerline_error_ft=25.0, stall_margin=2.0),
            FlightPhase.FINAL,
        )
        self.assertFalse(status.request_go_around)

    def test_moderate_intercept_at_high_final_altitude_is_allowed(self) -> None:
        # Regression: the old flat 100 ft threshold rejected legitimate
        # intercepts. At 180 ft AGL with 150 ft of centerline offset
        # (below the 360 ft scaled limit), safety monitor must not GA.
        status = self.monitor.evaluate(
            make_state(alt_agl_ft=180.0, centerline_error_ft=150.0, stall_margin=2.0),
            FlightPhase.FINAL,
        )
        self.assertFalse(status.request_go_around)

    def test_on_ground_short_circuits_all_checks(self) -> None:
        # Regression: in live X-Plane runs the AGL reading can still
        # say 30-50 ft on touchdown due to field-elevation vs
        # terrain-under-aircraft mismatch. If the low-energy check
        # fires on a touched-down aircraft, it requests a go-around
        # from a plane that's already on the runway decelerating.
        status = self.monitor.evaluate(
            make_state(
                alt_agl_ft=35.0,  # stale altitude reading
                stall_margin=1.05,  # would normally trip low_energy
                centerline_error_ft=500.0,  # would normally trip unstable_lateral
                vs_fpm=-1500.0,  # would normally trip unstable_vertical
                on_ground=True,
            ),
            FlightPhase.FINAL,
        )
        self.assertFalse(status.request_go_around)
        self.assertIsNone(status.reason)


if __name__ == "__main__":
    unittest.main()
