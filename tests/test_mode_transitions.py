from __future__ import annotations

import unittest

from sim_pilot.core.config import load_default_config_bundle
from sim_pilot.core.mode_manager import ModeManager
from sim_pilot.core.safety_monitor import SafetyStatus
from sim_pilot.core.types import AircraftState, FlightPhase, KT_TO_FPS, Vec2, heading_to_vector
from sim_pilot.guidance.pattern_manager import build_pattern_geometry
from sim_pilot.guidance.route_manager import RouteManager
from sim_pilot.guidance.runway_geometry import RunwayFrame


def make_state(**overrides: object) -> AircraftState:
    defaults: dict[str, object] = {
        "t_sim": 0.0,
        "dt": 0.2,
        "position_ft": Vec2(0.0, 0.0),
        "alt_msl_ft": 1500.0,
        "alt_agl_ft": 1000.0,
        "pitch_deg": 0.0,
        "roll_deg": 0.0,
        "heading_deg": 0.0,
        "track_deg": 0.0,
        "p_rad_s": 0.0,
        "q_rad_s": 0.0,
        "r_rad_s": 0.0,
        "ias_kt": 80.0,
        "tas_kt": 80.0,
        "gs_kt": 80.0,
        "vs_fpm": 0.0,
        "ground_velocity_ft_s": heading_to_vector(0.0, 80.0 * KT_TO_FPS),
        "flap_index": 0,
        "gear_down": True,
        "on_ground": False,
        "throttle_pos": 0.5,
        "runway_id": "36",
        "runway_dist_remaining_ft": None,
        "runway_x_ft": 0.0,
        "runway_y_ft": 0.0,
        "centerline_error_ft": 0.0,
        "threshold_abeam": False,
        "distance_to_touchdown_ft": 2000.0,
        "stall_margin": 1.5,
    }
    defaults.update(overrides)
    return AircraftState(**defaults)


class ModeTransitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_default_config_bundle()
        self.mode_manager = ModeManager(self.config)
        self.runway_frame = RunwayFrame(self.config.airport.runway)
        self.pattern = build_pattern_geometry(
            self.runway_frame,
            downwind_offset_ft=self.config.pattern.downwind_offset_ft,
            extension_ft=0.0,
        )
        self.route_manager = RouteManager([])
        self.safe = SafetyStatus(False, None, self.config.limits.max_bank_pattern_deg)

    def test_downwind_does_not_skip_directly_to_flare(self) -> None:
        phase = self.mode_manager.update(
            FlightPhase.DOWNWIND,
            make_state(alt_agl_ft=8.0, runway_x_ft=self.pattern.base_turn_x_ft + 500.0, runway_y_ft=self.pattern.downwind_y_ft),
            self.route_manager,
            self.pattern,
            self.safe,
        )
        self.assertEqual(phase, FlightPhase.DOWNWIND)

    def test_pattern_sequence_advances_one_phase_at_a_time(self) -> None:
        phase = self.mode_manager.update(
            FlightPhase.DOWNWIND,
            make_state(runway_x_ft=self.pattern.base_turn_x_ft - 10.0, runway_y_ft=self.pattern.downwind_y_ft, track_deg=180.0),
            self.route_manager,
            self.pattern,
            self.safe,
        )
        self.assertEqual(phase, FlightPhase.BASE)

        phase = self.mode_manager.update(
            FlightPhase.BASE,
            make_state(runway_x_ft=-1500.0, runway_y_ft=40.0, track_deg=0.0),
            self.route_manager,
            self.pattern,
            self.safe,
        )
        self.assertEqual(phase, FlightPhase.FINAL)

        phase = self.mode_manager.update(
            FlightPhase.FINAL,
            make_state(alt_agl_ft=self.config.flare.roundout_height_ft - 1.0, runway_x_ft=-300.0),
            self.route_manager,
            self.pattern,
            self.safe,
        )
        self.assertEqual(phase, FlightPhase.ROUNDOUT)

        phase = self.mode_manager.update(
            FlightPhase.ROUNDOUT,
            make_state(alt_agl_ft=self.config.flare.flare_start_ft - 1.0, runway_x_ft=50.0),
            self.route_manager,
            self.pattern,
            self.safe,
        )
        self.assertEqual(phase, FlightPhase.FLARE)


if __name__ == "__main__":
    unittest.main()
