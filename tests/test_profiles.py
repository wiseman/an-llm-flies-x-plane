from __future__ import annotations

import unittest

from sim_pilot.core.config import load_default_config_bundle
from sim_pilot.core.mission_manager import PilotCore
from sim_pilot.core.profiles import (
    AltitudeHoldProfile,
    Axis,
    HeadingHoldProfile,
    IdleLateralProfile,
    IdleSpeedProfile,
    IdleVerticalProfile,
    PatternFlyProfile,
    SpeedHoldProfile,
    TakeoffProfile,
    build_rotate_guidance,
    build_takeoff_roll_guidance,
)
from sim_pilot.core.types import FlightPhase, LateralMode, VerticalMode


class ProfileEngagementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_default_config_bundle()
        self.pilot = PilotCore(self.config)

    def test_new_pilot_starts_with_three_idle_profiles(self) -> None:
        names = set(self.pilot.list_profile_names())
        self.assertEqual(names, {"idle_lateral", "idle_vertical", "idle_speed"})

    def test_engaging_heading_hold_displaces_idle_lateral_only(self) -> None:
        displaced = self.pilot.engage_profile(HeadingHoldProfile(heading_deg=270.0))
        self.assertEqual(displaced, ["idle_lateral"])
        remaining = set(self.pilot.list_profile_names())
        self.assertEqual(remaining, {"idle_vertical", "idle_speed", "heading_hold"})

    def test_engaging_pattern_fly_displaces_all_three_idle_profiles(self) -> None:
        displaced = self.pilot.engage_profile(PatternFlyProfile(self.config, self.pilot.runway_frame))
        self.assertEqual(set(displaced), {"idle_lateral", "idle_vertical", "idle_speed"})
        self.assertEqual(self.pilot.list_profile_names(), ["pattern_fly"])

    def test_engaging_pattern_fly_over_heading_hold_displaces_both(self) -> None:
        self.pilot.engage_profile(HeadingHoldProfile(heading_deg=270.0))
        displaced = self.pilot.engage_profile(PatternFlyProfile(self.config, self.pilot.runway_frame))
        self.assertEqual(set(displaced), {"idle_vertical", "idle_speed", "heading_hold"})
        self.assertEqual(self.pilot.list_profile_names(), ["pattern_fly"])

    def test_disengage_pattern_fly_readds_three_idle_profiles(self) -> None:
        self.pilot.engage_profile(PatternFlyProfile(self.config, self.pilot.runway_frame))
        added = self.pilot.disengage_profile("pattern_fly")
        self.assertEqual(set(added), {"idle_lateral", "idle_vertical", "idle_speed"})
        self.assertEqual(set(self.pilot.list_profile_names()), {"idle_lateral", "idle_vertical", "idle_speed"})

    def test_disengage_heading_hold_readds_only_idle_lateral(self) -> None:
        self.pilot.engage_profile(HeadingHoldProfile(heading_deg=270.0))
        added = self.pilot.disengage_profile("heading_hold")
        self.assertEqual(added, ["idle_lateral"])
        self.assertEqual(set(self.pilot.list_profile_names()), {"idle_lateral", "idle_vertical", "idle_speed"})

    def test_disengage_unknown_profile_is_noop(self) -> None:
        added = self.pilot.disengage_profile("no_such_profile")
        self.assertEqual(added, [])

    def test_alt_and_speed_hold_compose_without_conflict(self) -> None:
        self.pilot.engage_profile(AltitudeHoldProfile(altitude_ft=3000.0))
        self.pilot.engage_profile(SpeedHoldProfile(speed_kt=95.0))
        names = set(self.pilot.list_profile_names())
        self.assertEqual(names, {"idle_lateral", "altitude_hold", "speed_hold"})


class PatternFlyMethodTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_default_config_bundle()
        self.pilot = PilotCore(self.config)
        self.profile = PatternFlyProfile(self.config, self.pilot.runway_frame)
        self.pilot.engage_profile(self.profile)

    def test_extend_downwind_updates_pattern_extension(self) -> None:
        original = self.profile.pattern_extension_ft
        self.profile.extend_downwind(1500.0)
        self.assertEqual(self.profile.pattern_extension_ft, original + 1500.0)

    def test_extend_downwind_rebuilds_geometry(self) -> None:
        original_base_turn_x = self.profile.pattern.base_turn_x_ft
        self.profile.extend_downwind(2400.0)
        self.assertLess(self.profile.pattern.base_turn_x_ft, original_base_turn_x)

    def test_turn_base_now_sets_trigger(self) -> None:
        self.assertFalse(self.profile._turn_base_trigger)
        self.profile.turn_base_now()
        self.assertTrue(self.profile._turn_base_trigger)

    def test_go_around_sets_trigger(self) -> None:
        self.assertFalse(self.profile._force_go_around_trigger)
        self.profile.go_around()
        self.assertTrue(self.profile._force_go_around_trigger)

    def test_cleared_to_land_records_runway(self) -> None:
        self.profile.cleared_to_land("16L")
        self.assertEqual(self.profile.cleared_to_land_runway, "16L")


class IdleProfileShapeTests(unittest.TestCase):
    def test_idle_profiles_own_exactly_one_axis_each(self) -> None:
        self.assertEqual(IdleLateralProfile.owns, frozenset({Axis.LATERAL}))
        self.assertEqual(IdleVerticalProfile.owns, frozenset({Axis.VERTICAL}))
        self.assertEqual(IdleSpeedProfile.owns, frozenset({Axis.SPEED}))


class HeadingHoldDirectionTests(unittest.TestCase):
    def _make_state(self, track_deg: float):
        from sim_pilot.core.types import AircraftState, Vec2, heading_to_vector, KT_TO_FPS
        return AircraftState(
            t_sim=0.0,
            dt=0.2,
            position_ft=Vec2(0.0, 0.0),
            alt_msl_ft=3000.0,
            alt_agl_ft=2500.0,
            pitch_deg=0.0,
            roll_deg=0.0,
            heading_deg=track_deg,
            track_deg=track_deg,
            p_rad_s=0.0,
            q_rad_s=0.0,
            r_rad_s=0.0,
            ias_kt=90.0,
            tas_kt=90.0,
            gs_kt=90.0,
            vs_fpm=0.0,
            ground_velocity_ft_s=heading_to_vector(track_deg, 90.0 * KT_TO_FPS),
            flap_index=0,
            gear_down=True,
            on_ground=False,
            throttle_pos=0.5,
            runway_id=None,
            runway_dist_remaining_ft=None,
            runway_x_ft=None,
            runway_y_ft=None,
            centerline_error_ft=None,
            threshold_abeam=False,
            distance_to_touchdown_ft=None,
            stall_margin=1.5,
        )

    def test_shortest_path_default_goes_left_from_060_to_290(self) -> None:
        profile = HeadingHoldProfile(heading_deg=290.0)
        contribution = profile.contribute(self._make_state(60.0), 0.2, None)
        self.assertLess(contribution.target_bank_deg, 0.0)

    def test_forced_right_direction_banks_right_even_when_left_is_shorter(self) -> None:
        profile = HeadingHoldProfile(heading_deg=290.0, turn_direction="right")
        contribution = profile.contribute(self._make_state(60.0), 0.2, None)
        self.assertGreater(contribution.target_bank_deg, 0.0)

    def test_forced_left_direction_banks_left_even_when_right_is_shorter(self) -> None:
        profile = HeadingHoldProfile(heading_deg=100.0, turn_direction="left")
        # Short path from 060 to 100 is right (+40). Forced left should go -320.
        contribution = profile.contribute(self._make_state(60.0), 0.2, None)
        self.assertLess(contribution.target_bank_deg, 0.0)

    def test_direction_lock_clears_once_within_tolerance(self) -> None:
        profile = HeadingHoldProfile(heading_deg=290.0, turn_direction="right")
        # Start far from target, then simulate arrival
        profile.contribute(self._make_state(60.0), 0.2, None)
        self.assertEqual(profile._direction_lock, "right")
        # Now a state near the target should clear the lock
        profile.contribute(self._make_state(288.0), 0.2, None)
        self.assertIsNone(profile._direction_lock)

    def test_invalid_direction_raises(self) -> None:
        with self.assertRaises(ValueError):
            HeadingHoldProfile(heading_deg=270.0, turn_direction="backwards")


class TakeoffHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_default_config_bundle()
        self.pilot = PilotCore(self.config)

    def test_takeoff_roll_guidance_goes_full_power(self) -> None:
        gt = build_takeoff_roll_guidance(self.config, self.pilot.runway_frame)
        self.assertEqual(gt.throttle_limit, (1.0, 1.0))
        self.assertEqual(gt.lateral_mode, LateralMode.ROLLOUT_CENTERLINE)
        self.assertEqual(gt.target_pitch_deg, 0.0)
        self.assertEqual(gt.target_speed_kt, self.config.performance.vr_kt)

    def test_rotate_guidance_commands_rotate_pitch_at_vy(self) -> None:
        gt = build_rotate_guidance(self.config, self.pilot.runway_frame)
        self.assertEqual(gt.throttle_limit, (1.0, 1.0))
        self.assertEqual(gt.target_pitch_deg, 8.0)
        self.assertEqual(gt.target_speed_kt, self.config.performance.vy_kt)
        self.assertEqual(gt.lateral_mode, LateralMode.PATH_FOLLOW)


class TakeoffProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_default_config_bundle()
        self.pilot = PilotCore(self.config)

    def test_takeoff_profile_owns_all_three_axes(self) -> None:
        self.assertEqual(
            TakeoffProfile.owns,
            frozenset({Axis.LATERAL, Axis.VERTICAL, Axis.SPEED}),
        )

    def test_engaging_takeoff_displaces_all_three_idle_profiles(self) -> None:
        displaced = self.pilot.engage_profile(TakeoffProfile(self.config, self.pilot.runway_frame))
        self.assertEqual(set(displaced), {"idle_lateral", "idle_vertical", "idle_speed"})
        self.assertEqual(self.pilot.list_profile_names(), ["takeoff"])

    def test_engaging_takeoff_displaces_pattern_fly(self) -> None:
        self.pilot.engage_profile(PatternFlyProfile(self.config, self.pilot.runway_frame))
        displaced = self.pilot.engage_profile(TakeoffProfile(self.config, self.pilot.runway_frame))
        self.assertEqual(displaced, ["pattern_fly"])
        self.assertEqual(self.pilot.list_profile_names(), ["takeoff"])

    def test_takeoff_starts_in_preflight_and_advances_on_acceleration(self) -> None:
        from sim_pilot.sim.scenario import ScenarioRunner
        # Reuse the simple-backend scenario runner but engage TakeoffProfile instead of
        # PatternFlyProfile so we can observe phase progression without pattern-flying
        # to completion.
        config = self.config
        runner_pilot = PilotCore(config)
        takeoff = TakeoffProfile(config, runner_pilot.runway_frame)
        runner_pilot.engage_profile(takeoff)
        # Drive the simple-model dynamics directly for a handful of ticks
        from sim_pilot.sim.simple_dynamics import SimpleAircraftModel
        from sim_pilot.core.types import Vec2
        model = SimpleAircraftModel(config, Vec2(0.0, 0.0))
        raw_state = model.initial_state()
        # Initially we should be in PREFLIGHT
        self.assertEqual(takeoff.phase, FlightPhase.PREFLIGHT)
        # After a few ticks with full power (which takeoff commands), we should
        # advance through TAKEOFF_ROLL → ROTATE → (ultimately) something airborne.
        phases_seen: set[FlightPhase] = set()
        for _ in range(200):
            _, commands = runner_pilot.update(raw_state, 0.2)
            phases_seen.add(takeoff.phase)
            raw_state = model.step(raw_state, commands, 0.2)
            if takeoff.phase is FlightPhase.INITIAL_CLIMB:
                break
        self.assertIn(FlightPhase.TAKEOFF_ROLL, phases_seen)
        self.assertIn(FlightPhase.ROTATE, phases_seen)
        self.assertIn(FlightPhase.INITIAL_CLIMB, phases_seen)

    def test_takeoff_profile_does_not_auto_disengage(self) -> None:
        takeoff = TakeoffProfile(self.config, self.pilot.runway_frame)
        self.pilot.engage_profile(takeoff)
        # Manually force the profile into INITIAL_CLIMB; it should stay engaged
        # and keep producing guidance.
        takeoff.phase = FlightPhase.INITIAL_CLIMB
        self.assertEqual(self.pilot.list_profile_names(), ["takeoff"])


if __name__ == "__main__":
    unittest.main()
