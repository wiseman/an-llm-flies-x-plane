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


class PatternFlyGuidanceTests(unittest.TestCase):
    """Regression tests for guidance emitted per phase. Exercises the
    per-phase branches of PatternFlyProfile._guidance_for_phase directly
    by forcing self.profile.phase and calling contribute()."""

    def setUp(self) -> None:
        self.config = load_default_config_bundle()
        self.pilot = PilotCore(self.config)
        self.profile = PatternFlyProfile(self.config, self.pilot.runway_frame)
        self.pilot.engage_profile(self.profile)

    def _airborne_state(self, *, phase_hint_alt_ft: float = 1500.0):
        from sim_pilot.core.types import AircraftState, Vec2
        return AircraftState(
            t_sim=0.0,
            dt=0.2,
            position_ft=Vec2(0.0, 0.0),
            alt_msl_ft=phase_hint_alt_ft,
            alt_agl_ft=max(0.0, phase_hint_alt_ft - self.config.airport.field_elevation_ft),
            pitch_deg=2.0,
            roll_deg=0.0,
            heading_deg=self.config.airport.runway.course_deg,
            track_deg=self.config.airport.runway.course_deg,
            p_rad_s=0.0,
            q_rad_s=0.0,
            r_rad_s=0.0,
            ias_kt=80.0,
            tas_kt=80.0,
            gs_kt=80.0,
            vs_fpm=0.0,
            ground_velocity_ft_s=Vec2(0.0, 80.0),
            flap_index=0,
            gear_down=True,
            on_ground=False,
            throttle_pos=0.6,
            runway_id=self.config.airport.runway.id,
            runway_dist_remaining_ft=None,
            runway_x_ft=-3000.0,
            runway_y_ft=0.0,
            centerline_error_ft=0.0,
            threshold_abeam=False,
            distance_to_touchdown_ft=None,
            stall_margin=1.5,
        )

    def test_flap_schedule_per_phase_follows_c172_sop(self) -> None:
        # Regression for the log-observed schedule of entry=20, downwind=10,
        # base=10, final=20 which kept the C172 from climbing on rejoin and
        # gave it too little landing drag. The corrected schedule is:
        # entry clean, downwind first notch, base second, final full.
        cases = {
            FlightPhase.PATTERN_ENTRY: 0,
            FlightPhase.DOWNWIND: 10,
            FlightPhase.BASE: 20,
            FlightPhase.FINAL: 30,
        }
        for phase, expected_flaps in cases.items():
            with self.subTest(phase=phase):
                guidance = self.profile._guidance_for_phase(self._airborne_state(), phase)
                self.assertEqual(guidance.flaps_cmd, expected_flaps)

    def test_go_around_targets_runway_course_not_a_waypoint(self) -> None:
        # Regression for the "go-around turns to 2°" bug: with route_manager
        # starting at "outbound" = (0, 91142) in world frame, the old
        # GO_AROUND branch did direct_to(outbound) and got a bearing that
        # had no relationship to the runway. GO_AROUND must steer runway
        # heading independent of any waypoint.
        from sim_pilot.core.types import AircraftState, Vec2, LateralMode
        state = self._airborne_state()
        # Simulate a sideways aircraft to prove the target is runway course,
        # not current track.
        import dataclasses
        state = dataclasses.replace(state, track_deg=200.0, heading_deg=200.0)
        guidance = self.profile._guidance_for_phase(state, FlightPhase.GO_AROUND)
        self.assertEqual(guidance.lateral_mode, LateralMode.TRACK_HOLD)
        self.assertAlmostEqual(
            guidance.target_track_deg,
            self.config.airport.runway.course_deg,
            places=2,
        )

    def test_go_around_below_pattern_altitude_commands_full_climb(self) -> None:
        # Go-around procedure from well below pattern altitude: climb at
        # Vy with full throttle, flaps retracted to takeoff setting, gear
        # stays down for C172.
        low_state = self._airborne_state(phase_hint_alt_ft=700.0)  # 200 AGL
        guidance = self.profile._guidance_for_phase(low_state, FlightPhase.GO_AROUND)
        self.assertEqual(guidance.flaps_cmd, 10)
        self.assertEqual(guidance.throttle_limit, (0.9, 1.0))
        self.assertEqual(guidance.target_speed_kt, self.config.performance.vy_kt)
        self.assertEqual(guidance.target_altitude_ft, self.config.pattern_altitude_msl_ft)

    def test_go_around_at_pattern_altitude_levels_off(self) -> None:
        # Regression: once the go-around climb reaches pattern altitude
        # the aircraft must level off rather than continue climbing. Old
        # behavior used a fixed (0.9, 1.0) throttle floor plus GO_AROUND
        # climb trim (85% / 6°), which kept the aircraft climbing past
        # the target. The fix lowers the throttle ceiling and hands TECS
        # a PATTERN_ENTRY tecs_phase_override once within 200 ft of
        # pattern altitude.
        at_alt_state = self._airborne_state(
            phase_hint_alt_ft=self.config.pattern_altitude_msl_ft
        )
        guidance = self.profile._guidance_for_phase(at_alt_state, FlightPhase.GO_AROUND)
        self.assertNotEqual(guidance.throttle_limit, (0.9, 1.0))
        self.assertLess(guidance.throttle_limit[1], 0.9)
        self.assertEqual(guidance.target_altitude_ft, self.config.pattern_altitude_msl_ft)
        self.assertEqual(guidance.tecs_phase_override, FlightPhase.PATTERN_ENTRY)

    def test_pattern_fly_route_has_only_pattern_entry_waypoint(self) -> None:
        # Regression for task #10: the obsolete "outbound" waypoint is
        # gone; the only remaining destination is "pattern_entry_start".
        names = [wp.name for wp in self.profile.route_manager.waypoints]
        self.assertEqual(names, ["pattern_entry_start"])

    def test_go_around_hands_off_to_holds_when_settled(self) -> None:
        # After the go-around climb reaches pattern altitude and vs
        # has bled off, pattern_fly should displace itself with three
        # single-axis holds (heading_hold, altitude_hold, speed_hold)
        # so the LLM gets a clean "autopilot hold" handoff point.
        self.profile.phase = FlightPhase.GO_AROUND
        state = self._airborne_state(
            phase_hint_alt_ft=self.config.pattern_altitude_msl_ft
        )
        import dataclasses
        state = dataclasses.replace(state, vs_fpm=50.0)  # near-level
        self.profile.contribute(state, 0.2, self.pilot)
        names = set(self.pilot.list_profile_names())
        self.assertNotIn("pattern_fly", names)
        self.assertEqual(names, {"heading_hold", "altitude_hold", "speed_hold"})

    def test_go_around_holds_inherit_runway_course_and_pattern_altitude(self) -> None:
        self.profile.phase = FlightPhase.GO_AROUND
        state = self._airborne_state(
            phase_hint_alt_ft=self.config.pattern_altitude_msl_ft
        )
        import dataclasses
        state = dataclasses.replace(state, vs_fpm=50.0)
        self.profile.contribute(state, 0.2, self.pilot)
        heading_hold = self.pilot.find_profile("heading_hold")
        altitude_hold = self.pilot.find_profile("altitude_hold")
        speed_hold = self.pilot.find_profile("speed_hold")
        assert isinstance(heading_hold, HeadingHoldProfile)
        assert isinstance(altitude_hold, AltitudeHoldProfile)
        assert isinstance(speed_hold, SpeedHoldProfile)
        self.assertAlmostEqual(
            heading_hold.heading_deg, self.config.airport.runway.course_deg, places=2
        )
        self.assertAlmostEqual(
            altitude_hold.altitude_ft, self.config.pattern_altitude_msl_ft, places=2
        )
        self.assertAlmostEqual(
            speed_hold.speed_kt, self.config.performance.vy_kt, places=2
        )

    def test_go_around_still_climbing_does_not_hand_off(self) -> None:
        # If the aircraft is still well below pattern altitude, the
        # handoff must wait — otherwise we lose the climb trim and
        # throttle floor that pattern_fly's GO_AROUND guidance provides.
        self.profile.phase = FlightPhase.GO_AROUND
        low_state = self._airborne_state(
            phase_hint_alt_ft=self.config.pattern_altitude_msl_ft - 500.0
        )
        self.profile.contribute(low_state, 0.2, self.pilot)
        self.assertIn("pattern_fly", self.pilot.list_profile_names())

    def test_go_around_at_altitude_but_still_climbing_does_not_hand_off(self) -> None:
        # Even at the target altitude, if vs_fpm is large (e.g. the
        # aircraft is about to overshoot), we wait for the climb to
        # settle before handing off.
        self.profile.phase = FlightPhase.GO_AROUND
        state = self._airborne_state(
            phase_hint_alt_ft=self.config.pattern_altitude_msl_ft
        )
        import dataclasses
        state = dataclasses.replace(state, vs_fpm=500.0)  # still climbing hard
        self.profile.contribute(state, 0.2, self.pilot)
        self.assertIn("pattern_fly", self.pilot.list_profile_names())

    def _pattern_state(
        self,
        *,
        alt_msl_ft: float,
        ias_kt: float = 80.0,
    ):
        from sim_pilot.core.types import AircraftState, Vec2
        return AircraftState(
            t_sim=0.0,
            dt=0.2,
            position_ft=Vec2(0.0, 0.0),
            alt_msl_ft=alt_msl_ft,
            alt_agl_ft=max(0.0, alt_msl_ft - self.config.airport.field_elevation_ft),
            pitch_deg=2.0,
            roll_deg=0.0,
            heading_deg=self.config.airport.runway.course_deg,
            track_deg=self.config.airport.runway.course_deg,
            p_rad_s=0.0,
            q_rad_s=0.0,
            r_rad_s=0.0,
            ias_kt=ias_kt,
            tas_kt=ias_kt,
            gs_kt=ias_kt,
            vs_fpm=0.0,
            ground_velocity_ft_s=Vec2(0.0, 80.0),
            flap_index=0,
            gear_down=True,
            on_ground=False,
            throttle_pos=0.5,
            runway_id=self.config.airport.runway.id,
            runway_dist_remaining_ft=None,
            runway_x_ft=-3000.0,
            runway_y_ft=0.0,
            centerline_error_ft=0.0,
            threshold_abeam=False,
            distance_to_touchdown_ft=None,
            stall_margin=ias_kt / self.config.performance.vso_landing_kt,
        )

    def test_pattern_entry_from_well_below_triggers_climb_capture(self) -> None:
        # Regression for the KWHP log: LLM re-engaged pattern_fly at
        # ~400 ft AGL (600 ft below target) and 54 kt. The old throttle
        # ceiling of 0.6 couldn't climb AND accelerate, so the aircraft
        # mushed around at 54 kt/400 ft AGL for the entire pattern.
        # With the fix, far-below-target states hint ENROUTE_CLIMB trim
        # and raise the throttle ceiling to 1.0.
        agl_ft = 400.0
        state = self._pattern_state(
            alt_msl_ft=self.config.airport.field_elevation_ft + agl_ft,
            ias_kt=54.0,
        )
        guidance = self.profile._guidance_for_phase(state, FlightPhase.PATTERN_ENTRY)
        self.assertEqual(guidance.tecs_phase_override, FlightPhase.ENROUTE_CLIMB)
        assert guidance.throttle_limit is not None
        self.assertEqual(guidance.throttle_limit[1], 1.0)
        self.assertGreaterEqual(guidance.throttle_limit[0], 0.7)

    def test_downwind_from_well_below_triggers_climb_capture(self) -> None:
        state = self._pattern_state(
            alt_msl_ft=self.config.pattern_altitude_msl_ft - 500.0,  # 500 ft low
            ias_kt=60.0,
        )
        guidance = self.profile._guidance_for_phase(state, FlightPhase.DOWNWIND)
        self.assertEqual(guidance.tecs_phase_override, FlightPhase.ENROUTE_CLIMB)
        assert guidance.throttle_limit is not None
        self.assertEqual(guidance.throttle_limit[1], 1.0)

    def test_normal_downwind_slight_sag_does_not_trigger_climb_capture(self) -> None:
        # Regression for a chatter bug found during task #8 development:
        # the first version used a 150 ft capture band, which a C172 on
        # downwind routinely hits during steady-state TECS settling.
        # The climb-capture then fired every ~5 seconds, accelerating the
        # aircraft during downwind and pushing touchdown ~350 ft further
        # along the runway. Must only fire on big deficits, not normal
        # pattern altitude wobble.
        state = self._pattern_state(
            alt_msl_ft=self.config.pattern_altitude_msl_ft - 150.0,
            ias_kt=78.0,
        )
        guidance = self.profile._guidance_for_phase(state, FlightPhase.DOWNWIND)
        self.assertIsNone(guidance.tecs_phase_override)
        assert guidance.throttle_limit is not None
        # Normal DOWNWIND ceiling is 0.55, not the climb-capture 1.0.
        self.assertLess(guidance.throttle_limit[1], 0.7)

    def test_pattern_entry_at_target_altitude_uses_normal_ceiling(self) -> None:
        state = self._pattern_state(
            alt_msl_ft=self.config.pattern_altitude_msl_ft,
            ias_kt=80.0,
        )
        guidance = self.profile._guidance_for_phase(state, FlightPhase.PATTERN_ENTRY)
        self.assertIsNone(guidance.tecs_phase_override)
        assert guidance.throttle_limit is not None
        self.assertLess(guidance.throttle_limit[1], 0.7)

    def test_manual_go_around_records_manual_trigger_reason(self) -> None:
        # Regression for task #15: the reason is captured on the tick a
        # GA fires, distinguishing manual (LLM go_around tool) from
        # safety-monitor triggers.
        self.profile.phase = FlightPhase.FINAL
        self.profile.go_around()  # set trigger
        state = self._airborne_state()
        import dataclasses
        state = dataclasses.replace(state, alt_agl_ft=300.0)
        self.profile.contribute(state, 0.2, self.pilot)
        self.assertEqual(self.profile.phase, FlightPhase.GO_AROUND)
        self.assertEqual(self.profile.last_go_around_reason, "manual_trigger")

    def test_safety_triggered_go_around_records_safety_reason(self) -> None:
        # The SafetyMonitor fires unstable_lateral on FINAL below 200 ft
        # when centerline error exceeds the altitude-scaled limit. The
        # reason string that comes back must be what the profile stores.
        self.profile.phase = FlightPhase.FINAL
        state = self._airborne_state()
        import dataclasses
        state = dataclasses.replace(
            state,
            alt_agl_ft=150.0,
            alt_msl_ft=self.config.airport.field_elevation_ft + 150.0,
            centerline_error_ft=-3500.0,
            runway_x_ft=-500.0,
            runway_y_ft=-3500.0,
        )
        self.profile.contribute(state, 0.2, self.pilot)
        self.assertEqual(self.profile.phase, FlightPhase.GO_AROUND)
        assert self.profile.last_go_around_reason is not None
        self.assertIn("unstable_lateral", self.profile.last_go_around_reason)
        self.assertIn("cle=3500ft", self.profile.last_go_around_reason)

    def test_subsequent_go_around_tick_does_not_overwrite_reason(self) -> None:
        # Defensive: once GO_AROUND is latched, the reason string should
        # NOT be overwritten by later ticks (the profile can't easily
        # distinguish "still in GA from earlier trip" vs. "newly fired").
        # The transition-only capture path means the reason reflects the
        # cause of the flip, not later state.
        self.profile.phase = FlightPhase.GO_AROUND
        self.profile.last_go_around_reason = "manual_trigger"
        state = self._airborne_state()
        import dataclasses
        state = dataclasses.replace(state, alt_agl_ft=500.0)
        self.profile.contribute(state, 0.2, self.pilot)
        self.assertEqual(self.profile.last_go_around_reason, "manual_trigger")

    def test_pattern_entry_uses_direct_to_join_point_not_entry_leg(self) -> None:
        # Regression for task #9: in the KWHP log, the LLM engaged
        # pattern_fly from a position SE of the airport. The old
        # PATTERN_ENTRY guidance used follow_leg on a fixed entry_leg
        # that started NE of the join point, so L1 dragged the aircraft
        # NE AWAY from the runway to reach the leg start. With the fix,
        # PATTERN_ENTRY computes a direct-to toward the downwind join
        # point regardless of the entry_leg's position.
        from sim_pilot.core.types import AircraftState, Vec2, course_between
        import dataclasses
        # Put the aircraft somewhere NOT on the entry_leg: past the
        # threshold along the runway axis but on the correct (downwind)
        # side for left traffic.
        state = self._airborne_state()
        # Override runway-frame coords to put the aircraft explicitly
        # near the join point side but ahead of the entry_leg start.
        join_rf = self.profile.pattern.join_point_runway_ft
        # Position the aircraft ~1000 ft short of the join point but
        # laterally on the downwind corridor. In runway frame that's
        # (join_x + 1000, downwind_y).
        probe_runway_x = join_rf.x + 1000.0
        probe_runway_y = self.profile.pattern.downwind_y_ft
        probe_world = self.profile.runway_frame.to_world_frame(Vec2(probe_runway_x, probe_runway_y))
        state = dataclasses.replace(
            state,
            position_ft=probe_world,
            runway_x_ft=probe_runway_x,
            runway_y_ft=probe_runway_y,
        )
        guidance = self.profile._guidance_for_phase(state, FlightPhase.PATTERN_ENTRY)
        # The expected course is direct-to the join point from probe_world.
        join_world = self.profile.runway_frame.to_world_frame(join_rf)
        expected_course = course_between(probe_world, join_world)
        self.assertIsNotNone(guidance.target_track_deg)
        assert guidance.target_track_deg is not None
        self.assertAlmostEqual(guidance.target_track_deg, expected_course, places=0)

    def test_base_phase_does_not_trigger_climb_capture_even_if_low(self) -> None:
        # BASE is a descent phase; climb-capture should never fire there
        # even if the aircraft is below the base target altitude.
        base_target_ft = self.config.airport.field_elevation_ft + 600.0
        state = self._pattern_state(
            alt_msl_ft=base_target_ft - 800.0,  # way below target
            ias_kt=55.0,
        )
        guidance = self.profile._guidance_for_phase(state, FlightPhase.BASE)
        self.assertIsNone(guidance.tecs_phase_override)


class IdleProfileShapeTests(unittest.TestCase):
    def test_idle_profiles_own_exactly_one_axis_each(self) -> None:
        self.assertEqual(IdleLateralProfile.owns, frozenset({Axis.LATERAL}))
        self.assertEqual(IdleVerticalProfile.owns, frozenset({Axis.VERTICAL}))
        self.assertEqual(IdleSpeedProfile.owns, frozenset({Axis.SPEED}))


class AltitudeHoldCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_default_config_bundle()
        self.pilot = PilotCore(self.config)

    def _make_state(self, alt_msl_ft: float):
        from sim_pilot.core.types import AircraftState, Vec2
        return AircraftState(
            t_sim=0.0,
            dt=0.2,
            position_ft=Vec2(0.0, 0.0),
            alt_msl_ft=alt_msl_ft,
            alt_agl_ft=max(0.0, alt_msl_ft - 425.0),
            pitch_deg=0.0,
            roll_deg=0.0,
            heading_deg=0.0,
            track_deg=0.0,
            p_rad_s=0.0,
            q_rad_s=0.0,
            r_rad_s=0.0,
            ias_kt=80.0,
            tas_kt=80.0,
            gs_kt=80.0,
            vs_fpm=0.0,
            ground_velocity_ft_s=Vec2(0.0, 0.0),
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
            stall_margin=2.0,
        )

    def test_large_climb_error_picks_enroute_climb_phase(self) -> None:
        profile = AltitudeHoldProfile(altitude_ft=2000.0)
        # 613 ft below target → should switch to climb-capture regime
        contribution = profile.contribute(self._make_state(1387.0), 0.2, self.pilot)
        self.assertEqual(contribution.tecs_phase_override, FlightPhase.ENROUTE_CLIMB)
        self.assertEqual(contribution.throttle_limit, (0.7, 1.0))

    def test_large_descent_error_picks_descent_phase(self) -> None:
        profile = AltitudeHoldProfile(altitude_ft=2000.0)
        # 1000 ft above target → descent-capture regime
        contribution = profile.contribute(self._make_state(3000.0), 0.2, self.pilot)
        self.assertEqual(contribution.tecs_phase_override, FlightPhase.DESCENT)
        self.assertEqual(contribution.throttle_limit, (0.1, 0.5))

    def test_small_error_uses_default_cruise_tuning(self) -> None:
        profile = AltitudeHoldProfile(altitude_ft=2000.0)
        # Within 150 ft → cruise hold, no override
        contribution = profile.contribute(self._make_state(1950.0), 0.2, self.pilot)
        self.assertIsNone(contribution.tecs_phase_override)
        self.assertEqual(contribution.throttle_limit, (0.1, 0.9))

    def test_altitude_hold_regime_flows_through_pilotcore_into_commands(self) -> None:
        """End-to-end: PilotCore.update with AltitudeHold + SpeedHold, at a
        state 613 ft below target, should command high throttle because the
        capture regime picks ENROUTE_CLIMB trim (0.85) and clamps throttle
        into (0.7, 1.0). Under the old CRUISE-only path it would sit at ~0.59."""
        self.pilot.engage_profile(AltitudeHoldProfile(altitude_ft=2000.0))
        self.pilot.engage_profile(SpeedHoldProfile(speed_kt=80.0))
        from sim_pilot.sim.simple_dynamics import DynamicsState
        from sim_pilot.core.types import Vec2
        raw = DynamicsState(
            position_ft=Vec2(0.0, 0.0),
            altitude_ft=1387.0,  # 613 ft below target
            heading_deg=0.0,
            roll_deg=0.0,
            pitch_deg=0.0,
            ias_kt=80.0,
            throttle_pos=0.5,
            on_ground=False,
            time_s=10.0,
            p_rad_s=0.0,
            q_rad_s=0.0,
            r_rad_s=0.0,
            ground_velocity_ft_s=Vec2(0.0, 0.0),
            vertical_speed_ft_s=0.0,
            flap_index=0,
            gear_down=True,
        )
        _state, commands = self.pilot.update(raw, 0.2)
        # With ENROUTE_CLIMB trim (0.85) and throttle_limit clamped to (0.7, 1.0),
        # commanded throttle must be at or above the 0.7 lower clamp.
        self.assertGreaterEqual(commands.throttle, 0.7)
        # Elevator should be pulling back (positive) to achieve climb pitch
        self.assertGreater(commands.elevator, 0.0)


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

    def test_rotate_guidance_on_ground_uses_rollout_centerline(self) -> None:
        # While the wheels are still on the runway, rotation guidance
        # MUST use ROLLOUT_CENTERLINE (rudder + nosewheel) so the aircraft
        # keeps tracking the centerline. PATH_FOLLOW (bank-based) is
        # useless here — ailerons don't steer a three-wheel aircraft.
        state = self._rotate_state(on_ground=True)
        gt = build_rotate_guidance(
            self.config,
            self.pilot.runway_frame,
            state,
            bank_limit_deg=self.config.limits.max_bank_enroute_deg,
        )
        self.assertEqual(gt.throttle_limit, (1.0, 1.0))
        self.assertEqual(gt.target_pitch_deg, 8.0)
        self.assertEqual(gt.target_speed_kt, self.config.performance.vy_kt)
        self.assertEqual(gt.lateral_mode, LateralMode.ROLLOUT_CENTERLINE)
        self.assertEqual(
            gt.target_track_deg,
            self.config.airport.runway.course_deg,
        )

    def test_rotate_guidance_airborne_banks_toward_runway_course(self) -> None:
        # Regression for the KWHP log: during ROTATE the guidance bus
        # showed tgt_hdg=— (the old build_rotate_guidance never set a
        # target_track_deg). After liftoff the profile must emit a
        # TRACK_HOLD toward runway course so the aircraft rolls back onto
        # the extended centerline.
        state = self._rotate_state(on_ground=False, track_deg_override=30.0)
        gt = build_rotate_guidance(
            self.config,
            self.pilot.runway_frame,
            state,
            bank_limit_deg=self.config.limits.max_bank_enroute_deg,
        )
        self.assertEqual(gt.lateral_mode, LateralMode.TRACK_HOLD)
        self.assertEqual(
            gt.target_track_deg,
            self.config.airport.runway.course_deg,
        )
        # Track 30 vs course 0 means we need to roll LEFT (negative bank)
        # to turn back onto runway heading.
        self.assertIsNotNone(gt.target_bank_deg)
        assert gt.target_bank_deg is not None
        self.assertLess(gt.target_bank_deg, 0.0)

    def _rotate_state(self, *, on_ground: bool, track_deg_override: float | None = None):
        from sim_pilot.core.types import AircraftState, Vec2
        course = self.config.airport.runway.course_deg
        track = track_deg_override if track_deg_override is not None else course
        return AircraftState(
            t_sim=0.0,
            dt=0.2,
            position_ft=Vec2(0.0, 0.0),
            alt_msl_ft=self.config.airport.field_elevation_ft + (0.0 if on_ground else 20.0),
            alt_agl_ft=0.0 if on_ground else 20.0,
            pitch_deg=3.0 if on_ground else 8.0,
            roll_deg=0.0,
            heading_deg=track,
            track_deg=track,
            p_rad_s=0.0,
            q_rad_s=0.0,
            r_rad_s=0.0,
            ias_kt=self.config.performance.vr_kt + 2.0,
            tas_kt=self.config.performance.vr_kt + 2.0,
            gs_kt=self.config.performance.vr_kt + 2.0,
            vs_fpm=0.0 if on_ground else 200.0,
            ground_velocity_ft_s=Vec2(0.0, 0.0),
            flap_index=0,
            gear_down=True,
            on_ground=on_ground,
            throttle_pos=1.0,
            runway_id=self.config.airport.runway.id,
            runway_dist_remaining_ft=None,
            runway_x_ft=1800.0,
            runway_y_ft=10.0,
            centerline_error_ft=10.0,
            threshold_abeam=False,
            distance_to_touchdown_ft=None,
            stall_margin=1.5,
        )


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

    def test_airborne_phase_commands_bank_toward_runway_course(self) -> None:
        """Regression: the previous TakeoffProfile hard-coded
        target_bank_deg=0.0 in INITIAL_CLIMB/ENROUTE_CLIMB/CRUISE, so after
        liftoff the bank controller held wings level and never corrected
        any post-takeoff heading drift. Now the profile should compute a
        bank command proportional to the track error."""
        from sim_pilot.core.types import AircraftState, Vec2

        takeoff = TakeoffProfile(self.config, self.pilot.runway_frame)
        # Force phase to INITIAL_CLIMB so we exercise the airborne branch
        takeoff.phase = FlightPhase.INITIAL_CLIMB

        # Aircraft airborne, 40° right of runway course (course=0 in KTEST,
        # current track=40 means we need to turn LEFT, negative bank)
        state = AircraftState(
            t_sim=0.0,
            dt=0.2,
            position_ft=Vec2(0.0, 500.0),
            alt_msl_ft=500.0,
            alt_agl_ft=500.0,
            pitch_deg=5.0,
            roll_deg=0.0,
            heading_deg=40.0,
            track_deg=40.0,
            p_rad_s=0.0,
            q_rad_s=0.0,
            r_rad_s=0.0,
            ias_kt=80.0,
            tas_kt=80.0,
            gs_kt=80.0,
            vs_fpm=500.0,
            ground_velocity_ft_s=Vec2(0.0, 0.0),
            flap_index=0,
            gear_down=True,
            on_ground=False,
            throttle_pos=0.9,
            runway_id=None,
            runway_dist_remaining_ft=None,
            runway_x_ft=None,
            runway_y_ft=None,
            centerline_error_ft=None,
            threshold_abeam=False,
            distance_to_touchdown_ft=None,
            stall_margin=2.0,
        )
        contribution = takeoff.contribute(state, 0.2, self.pilot)
        # Course 0, track 40 → track_error = wrap(0 - 40) = -40
        # target_bank = -40 * 0.35 = -14 (negative = left roll)
        self.assertIsNotNone(contribution.target_bank_deg)
        assert contribution.target_bank_deg is not None  # for type checker
        self.assertLess(contribution.target_bank_deg, 0.0)
        self.assertEqual(contribution.target_track_deg, self.config.airport.runway.course_deg)


if __name__ == "__main__":
    unittest.main()
