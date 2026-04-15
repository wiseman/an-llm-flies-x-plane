from __future__ import annotations

import unittest

from sim_pilot.core.config import load_default_config_bundle
from sim_pilot.core.mission_manager import PilotCore
from sim_pilot.core.mode_manager import ModeManager
from sim_pilot.core.profiles import PatternFlyProfile
from sim_pilot.core.safety_monitor import SafetyStatus
from sim_pilot.core.types import AircraftState, FlightPhase, KT_TO_FPS, Vec2, heading_to_vector
from sim_pilot.guidance.pattern_manager import build_pattern_geometry
from sim_pilot.guidance.route_manager import RouteManager
from sim_pilot.guidance.runway_geometry import RunwayFrame
from sim_pilot.live_runner import apply_bootstrap_sample_to_config
from sim_pilot.sim.xplane_bridge import BootstrapSample, PositionSample


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


class LiveIntentIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_default_config_bundle()

    def test_extend_downwind_rebuilds_pattern_geometry(self) -> None:
        pilot = PilotCore(self.config)
        profile = PatternFlyProfile(self.config, pilot.runway_frame)
        pilot.engage_profile(profile)
        original_extension_ft = profile.pattern.extension_ft
        original_base_turn_x_ft = profile.pattern.base_turn_x_ft
        profile.extend_downwind(2400.0)
        self.assertEqual(profile.pattern.extension_ft, original_extension_ft + 2400.0)
        self.assertLess(profile.pattern.base_turn_x_ft, original_base_turn_x_ft)

    def test_turn_base_now_override_skips_waiting_for_nominal_turn_point(self) -> None:
        mode_manager = ModeManager(self.config)
        runway_frame = RunwayFrame(self.config.airport.runway)
        pattern = build_pattern_geometry(
            runway_frame,
            downwind_offset_ft=self.config.pattern.downwind_offset_ft,
            extension_ft=0.0,
        )
        phase = mode_manager.update(
            FlightPhase.DOWNWIND,
            make_state(
                runway_x_ft=1200.0,
                runway_y_ft=pattern.downwind_y_ft,
                gs_kt=80.0,
                ias_kt=80.0,
            ),
            RouteManager([]),
            pattern,
            SafetyStatus(False, None, self.config.limits.max_bank_pattern_deg),
            turn_base_now=True,
        )
        self.assertEqual(phase, FlightPhase.BASE)

    def test_force_go_around_override_preempts_normal_sequence(self) -> None:
        mode_manager = ModeManager(self.config)
        runway_frame = RunwayFrame(self.config.airport.runway)
        pattern = build_pattern_geometry(
            runway_frame,
            downwind_offset_ft=self.config.pattern.downwind_offset_ft,
            extension_ft=0.0,
        )
        phase = mode_manager.update(
            FlightPhase.FINAL,
            make_state(alt_agl_ft=300.0, runway_x_ft=-2000.0, runway_y_ft=10.0),
            RouteManager([]),
            pattern,
            SafetyStatus(False, None, self.config.limits.max_bank_final_deg),
            force_go_around=True,
        )
        self.assertEqual(phase, FlightPhase.GO_AROUND)

    def test_bootstrap_sample_derives_runway_metadata_from_live_heading(self) -> None:
        bootstrapped = apply_bootstrap_sample_to_config(
            self.config,
            sample=BootstrapSample(
                posi=PositionSample(
                    lat_deg=47.449,
                    lon_deg=-122.309,
                    altitude_msl_m=132.0,
                    roll_deg=0.0,
                    pitch_deg=0.0,
                    heading_deg=163.0,
                ),
                alt_agl_ft=0.5,
                on_ground=True,
            ),
            airport="KSEA",
            runway_id=None,
            runway_course_deg=None,
            field_elevation_ft=None,
        )
        self.assertEqual(bootstrapped.airport.airport, "KSEA")
        self.assertEqual(bootstrapped.airport.runway.id, "16")
        self.assertEqual(bootstrapped.airport.runway.course_deg, 160.0)
        self.assertAlmostEqual(bootstrapped.airport.field_elevation_ft, (132.0 * 3.280839895013123) - 0.5)

    def test_bootstrap_requires_on_runway_or_explicit_field_elevation(self) -> None:
        with self.assertRaises(ValueError):
            apply_bootstrap_sample_to_config(
                self.config,
                sample=BootstrapSample(
                    posi=PositionSample(
                        lat_deg=47.449,
                        lon_deg=-122.309,
                        altitude_msl_m=300.0,
                        roll_deg=0.0,
                        pitch_deg=0.0,
                        heading_deg=163.0,
                    ),
                    alt_agl_ft=150.0,
                    on_ground=False,
                ),
                airport="KSEA",
                runway_id=None,
                runway_course_deg=None,
                field_elevation_ft=None,
            )


if __name__ == "__main__":
    unittest.main()
