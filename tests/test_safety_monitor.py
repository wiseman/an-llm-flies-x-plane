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
        status = self.monitor.evaluate(
            make_state(centerline_error_ft=160.0),
            FlightPhase.FINAL,
        )
        self.assertTrue(status.request_go_around)
        self.assertEqual(status.reason, "unstable_lateral")

    def test_low_stall_margin_requests_go_around(self) -> None:
        status = self.monitor.evaluate(
            make_state(alt_agl_ft=40.0, stall_margin=1.05),
            FlightPhase.FINAL,
        )
        self.assertTrue(status.request_go_around)
        self.assertEqual(status.reason, "low_energy")


if __name__ == "__main__":
    unittest.main()
