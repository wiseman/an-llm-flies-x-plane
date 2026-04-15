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
)


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


if __name__ == "__main__":
    unittest.main()
