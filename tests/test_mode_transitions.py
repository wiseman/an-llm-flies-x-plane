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

    def test_slow_downwind_can_turn_base_soon_after_abeam(self) -> None:
        phase = self.mode_manager.update(
            FlightPhase.DOWNWIND,
            make_state(
                runway_x_ft=500.0,
                runway_y_ft=self.pattern.downwind_y_ft,
                threshold_abeam=True,
                gs_kt=35.0,
                ias_kt=65.0,
            ),
            self.route_manager,
            self.pattern,
            self.safe,
        )
        self.assertEqual(phase, FlightPhase.BASE)

    def test_normal_speed_downwind_does_not_turn_base_at_abeam(self) -> None:
        phase = self.mode_manager.update(
            FlightPhase.DOWNWIND,
            make_state(
                runway_x_ft=500.0,
                runway_y_ft=self.pattern.downwind_y_ft,
                threshold_abeam=True,
                gs_kt=80.0,
                ias_kt=80.0,
            ),
            self.route_manager,
            self.pattern,
            self.safe,
        )
        self.assertEqual(phase, FlightPhase.DOWNWIND)


class TakeoffRollToRotateGuardTests(unittest.TestCase):
    """Regression tests for TAKEOFF_ROLL → ROTATE transition guard."""

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
        self.vr_kt = self.config.performance.vr_kt

    def _rolling_state(self, **overrides):
        defaults = dict(
            runway_x_ft=2000.0,
            runway_y_ft=0.0,
            centerline_error_ft=0.0,
            on_ground=True,
            ias_kt=self.vr_kt + 1.0,
            gs_kt=self.vr_kt + 1.0,
            alt_agl_ft=0.0,
        )
        defaults.update(overrides)
        return make_state(**defaults)

    def test_rotates_at_vr_when_on_centerline(self) -> None:
        phase = self.mode_manager.update(
            FlightPhase.TAKEOFF_ROLL,
            self._rolling_state(centerline_error_ft=5.0),
            self.route_manager,
            self.pattern,
            self.safe,
        )
        self.assertEqual(phase, FlightPhase.ROTATE)

    def test_does_not_rotate_before_vr(self) -> None:
        phase = self.mode_manager.update(
            FlightPhase.TAKEOFF_ROLL,
            self._rolling_state(ias_kt=self.vr_kt - 5.0, gs_kt=self.vr_kt - 5.0),
            self.route_manager,
            self.pattern,
            self.safe,
        )
        self.assertEqual(phase, FlightPhase.TAKEOFF_ROLL)

    def test_rotates_with_moderate_crosswind_centerline_error(self) -> None:
        # Regression for the old 25 ft guard: with any real crosswind the
        # aircraft sits 30-45 ft off centerline at Vr, and the old guard
        # refused to transition until centerline_error_ft<=25 — which in
        # the KWHP log never happened, so the transition eventually fired
        # via the airborne-bailout path at 100 kt and 2800 ft off axis.
        # The new guard relaxes to half unstable_centerline_error_ft (=50 ft).
        phase = self.mode_manager.update(
            FlightPhase.TAKEOFF_ROLL,
            self._rolling_state(centerline_error_ft=40.0),
            self.route_manager,
            self.pattern,
            self.safe,
        )
        self.assertEqual(phase, FlightPhase.ROTATE)

    def test_does_not_rotate_when_way_off_centerline_on_ground(self) -> None:
        phase = self.mode_manager.update(
            FlightPhase.TAKEOFF_ROLL,
            self._rolling_state(centerline_error_ft=200.0),
            self.route_manager,
            self.pattern,
            self.safe,
        )
        self.assertEqual(phase, FlightPhase.TAKEOFF_ROLL)

    def test_airborne_bailout_rotates_even_when_way_off_centerline(self) -> None:
        # Airborne bailout: once the wheels leave the ground at or above
        # Vr, the state machine MUST advance — rollout can no longer act
        # and rotate/climb guidance is safer than a stuck TAKEOFF_ROLL.
        phase = self.mode_manager.update(
            FlightPhase.TAKEOFF_ROLL,
            self._rolling_state(
                on_ground=False,
                alt_agl_ft=5.0,
                centerline_error_ft=500.0,
                ias_kt=self.vr_kt + 20.0,
                gs_kt=self.vr_kt + 20.0,
            ),
            self.route_manager,
            self.pattern,
            self.safe,
        )
        self.assertEqual(phase, FlightPhase.ROTATE)

    def test_airborne_below_vr_still_blocks(self) -> None:
        # Airborne bailout requires ias>=vr — we don't want to transition
        # during a momentary bounce at 40 kt.
        phase = self.mode_manager.update(
            FlightPhase.TAKEOFF_ROLL,
            self._rolling_state(
                on_ground=False,
                alt_agl_ft=2.0,
                ias_kt=self.vr_kt - 10.0,
                gs_kt=self.vr_kt - 10.0,
            ),
            self.route_manager,
            self.pattern,
            self.safe,
        )
        self.assertEqual(phase, FlightPhase.TAKEOFF_ROLL)


class PreflightAirborneEngagementTests(unittest.TestCase):
    """Regression: a profile (TakeoffProfile or PatternFlyProfile) engaged
    while already airborne should skip TAKEOFF_ROLL entirely instead of
    emitting one tick of takeoff-roll guidance at Vr target speed."""

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
        self.vr_kt = self.config.performance.vr_kt

    def test_preflight_on_ground_goes_to_takeoff_roll(self) -> None:
        phase = self.mode_manager.update(
            FlightPhase.PREFLIGHT,
            make_state(
                on_ground=True,
                ias_kt=0.0,
                gs_kt=0.0,
                alt_agl_ft=0.0,
            ),
            self.route_manager,
            self.pattern,
            self.safe,
        )
        self.assertEqual(phase, FlightPhase.TAKEOFF_ROLL)

    def test_preflight_airborne_above_vr_skips_straight_to_initial_climb(self) -> None:
        phase = self.mode_manager.update(
            FlightPhase.PREFLIGHT,
            make_state(
                on_ground=False,
                ias_kt=self.vr_kt + 40.0,
                gs_kt=self.vr_kt + 40.0,
                alt_agl_ft=300.0,
            ),
            self.route_manager,
            self.pattern,
            self.safe,
        )
        self.assertEqual(phase, FlightPhase.INITIAL_CLIMB)

    def test_preflight_airborne_below_vr_still_goes_to_takeoff_roll(self) -> None:
        # Weird edge case: somehow airborne but below Vr (bounce, slow
        # stall, whatever). Stay in the ground branch so the airborne
        # bailout in TAKEOFF_ROLL handles the rest defensively.
        phase = self.mode_manager.update(
            FlightPhase.PREFLIGHT,
            make_state(
                on_ground=False,
                ias_kt=self.vr_kt - 5.0,
                gs_kt=self.vr_kt - 5.0,
                alt_agl_ft=5.0,
            ),
            self.route_manager,
            self.pattern,
            self.safe,
        )
        self.assertEqual(phase, FlightPhase.TAKEOFF_ROLL)


class StayInPatternClimbTests(unittest.TestCase):
    """Regression tests for the INITIAL_CLIMB → CROSSWIND → DOWNWIND flow.

    Stay-in-pattern missions (PatternFlyProfile engaged from takeoff_roll)
    must climb to ~700 AGL straight on runway course, turn crosswind,
    cross over to the downwind offset, turn downwind, and never visit
    ENROUTE_CLIMB / CRUISE / DESCENT / PATTERN_ENTRY — those are reserved
    for mid-flight rejoin engagements.
    """

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

    def test_initial_climb_below_crosswind_altitude_stays_in_initial_climb(self) -> None:
        phase = self.mode_manager.update(
            FlightPhase.INITIAL_CLIMB,
            make_state(alt_agl_ft=500.0, vs_fpm=800.0, on_ground=False),
            self.route_manager,
            self.pattern,
            self.safe,
            stay_in_pattern=True,
        )
        self.assertEqual(phase, FlightPhase.INITIAL_CLIMB)

    def test_initial_climb_above_crosswind_altitude_turns_crosswind(self) -> None:
        # Pattern altitude is 1000 AGL by default, so the crosswind-turn
        # altitude is ~700 AGL (300 below TPA per AC 90-66C).
        phase = self.mode_manager.update(
            FlightPhase.INITIAL_CLIMB,
            make_state(alt_agl_ft=750.0, vs_fpm=800.0, on_ground=False),
            self.route_manager,
            self.pattern,
            self.safe,
            stay_in_pattern=True,
        )
        self.assertEqual(phase, FlightPhase.CROSSWIND)

    def test_initial_climb_without_stay_in_pattern_uses_legacy_enroute_flow(self) -> None:
        # TakeoffProfile climbs to cruise — it passes stay_in_pattern=False
        # (the default) and must still see the old INITIAL_CLIMB →
        # ENROUTE_CLIMB transition at 400 ft AGL.
        phase = self.mode_manager.update(
            FlightPhase.INITIAL_CLIMB,
            make_state(alt_agl_ft=450.0, vs_fpm=800.0, on_ground=False),
            self.route_manager,
            self.pattern,
            self.safe,
        )
        self.assertEqual(phase, FlightPhase.ENROUTE_CLIMB)

    def test_crosswind_holds_until_heading_captured_and_at_offset(self) -> None:
        # Just after turning crosswind, the aircraft is still near the
        # runway centerline in the y axis — it must stay in CROSSWIND.
        # Crosswind course for runway 36 (course 0) left traffic is 270°.
        phase = self.mode_manager.update(
            FlightPhase.CROSSWIND,
            make_state(
                alt_agl_ft=900.0,
                runway_x_ft=5000.0,
                runway_y_ft=-500.0,  # only 500 ft toward the -3500 offset
                heading_deg=270.0,
                track_deg=270.0,
                on_ground=False,
            ),
            self.route_manager,
            self.pattern,
            self.safe,
            stay_in_pattern=True,
        )
        self.assertEqual(phase, FlightPhase.CROSSWIND)

    def test_crosswind_holds_when_at_offset_but_still_turning(self) -> None:
        # Regression: the old trigger fired at 80% offset even if the
        # aircraft was still mid-turn from upwind to crosswind heading.
        # Now we require the crosswind heading to be captured (within
        # 15° of course - 90) before DOWNWIND can fire.
        phase = self.mode_manager.update(
            FlightPhase.CROSSWIND,
            make_state(
                alt_agl_ft=950.0,
                runway_x_ft=5000.0,
                runway_y_ft=-3500.0,  # at the full offset
                heading_deg=330.0,  # still mostly on upwind heading
                track_deg=330.0,
                on_ground=False,
            ),
            self.route_manager,
            self.pattern,
            self.safe,
            stay_in_pattern=True,
        )
        self.assertEqual(phase, FlightPhase.CROSSWIND)

    def test_crosswind_transitions_to_downwind_when_captured_and_at_offset(self) -> None:
        # Runway 36 course 0°, left traffic: crosswind course is 270°.
        # Both conditions satisfied → transition fires.
        phase = self.mode_manager.update(
            FlightPhase.CROSSWIND,
            make_state(
                alt_agl_ft=950.0,
                runway_x_ft=5000.0,
                runway_y_ft=-3500.0,
                heading_deg=270.0,
                track_deg=270.0,
                on_ground=False,
            ),
            self.route_manager,
            self.pattern,
            self.safe,
            stay_in_pattern=True,
        )
        self.assertEqual(phase, FlightPhase.DOWNWIND)


class GoAroundHoldsPatternAltitudeTests(unittest.TestCase):
    """Regression: after a go-around the state machine should hold the
    aircraft in GO_AROUND phase — it must NOT transition to ENROUTE_CLIMB
    (which would target cruise altitude 3000 MSL instead of pattern
    altitude). Observed in sim_pilot-20260415-130505.log: go-around from
    a bad final approach immediately flipped to ENROUTE_CLIMB and the
    aircraft started climbing toward cruise altitude instead of holding
    pattern altitude for the next pattern entry."""

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

    def test_go_around_persists_below_pattern_altitude(self) -> None:
        phase = self.mode_manager.update(
            FlightPhase.GO_AROUND,
            make_state(alt_agl_ft=300.0, vs_fpm=700.0, on_ground=False),
            self.route_manager,
            self.pattern,
            self.safe,
        )
        self.assertEqual(phase, FlightPhase.GO_AROUND)

    def test_go_around_persists_even_above_400_agl(self) -> None:
        # Regression: used to transition to ENROUTE_CLIMB at 400 AGL,
        # which then targeted cruise altitude (3000 MSL) instead of
        # pattern altitude.
        phase = self.mode_manager.update(
            FlightPhase.GO_AROUND,
            make_state(alt_agl_ft=600.0, vs_fpm=500.0, on_ground=False),
            self.route_manager,
            self.pattern,
            self.safe,
        )
        self.assertEqual(phase, FlightPhase.GO_AROUND)

    def test_go_around_persists_at_pattern_altitude(self) -> None:
        phase = self.mode_manager.update(
            FlightPhase.GO_AROUND,
            make_state(
                alt_agl_ft=self.config.pattern.altitude_agl_ft,
                alt_msl_ft=self.config.pattern_altitude_msl_ft,
                vs_fpm=0.0,
                on_ground=False,
            ),
            self.route_manager,
            self.pattern,
            self.safe,
        )
        self.assertEqual(phase, FlightPhase.GO_AROUND)


class BaseToFinalAlongTrackTests(unittest.TestCase):
    """Regression tests for BASE → FINAL gating. The base leg is a
    diagonal, so the aircraft's track during base is ~130° off runway
    course — we can't gate on "track within 30° of runway course" without
    locking the aircraft in BASE forever. Instead the guard is: proximity
    to base_end_ft AND along-track ≥ 70% of the leg length (so the
    aircraft has actually flown most of the base leg before the
    transition fires)."""

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

    def test_does_not_transition_when_only_proximity_satisfied(self) -> None:
        # Aircraft is within Euclidean distance of base_end but near the
        # START of the base leg (along-track ≈ 0) — must stay in BASE.
        base_start = self.pattern.base_leg.start_ft
        phase = self.mode_manager.update(
            FlightPhase.BASE,
            make_state(
                position_ft=base_start,
                runway_x_ft=self.pattern.base_leg.start_ft.x,
                runway_y_ft=self.pattern.base_leg.start_ft.y,
                track_deg=180.0,
                alt_agl_ft=900.0,
            ),
            self.route_manager,
            self.pattern,
            self.safe,
        )
        # base_start is ~4600 ft from base_end, well outside the 1400 ft
        # proximity trigger, so BASE holds regardless of along-track.
        self.assertEqual(phase, FlightPhase.BASE)

    def test_transitions_near_end_of_base_leg(self) -> None:
        # Fly the aircraft to 90% of the way along the base leg — it
        # should be within proximity AND past the 70% along-track
        # threshold, so BASE → FINAL fires.
        leg = self.pattern.base_leg
        delta = leg.end_ft - leg.start_ft
        near_end = leg.start_ft + delta * 0.9
        phase = self.mode_manager.update(
            FlightPhase.BASE,
            make_state(
                position_ft=near_end,
                runway_x_ft=near_end.x,
                runway_y_ft=near_end.y,
                track_deg=130.0,  # diagonal base leg course, NOT runway course
                alt_agl_ft=600.0,
            ),
            self.route_manager,
            self.pattern,
            self.safe,
        )
        self.assertEqual(phase, FlightPhase.FINAL)


class DownwindBaseTurnFormulaTests(unittest.TestCase):
    """Regression tests for task #11: the base turn trigger was tied to
    a very-deep nominal point (downwind_offset + 1500 = -5000) and had
    adaptive groundspeed relief that could push the turn far past the
    threshold. The new formula uses a fixed -1500 nominal with relief
    bounded by the abeam-numbers window (≈750 ft)."""

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

    def test_base_turn_nominal_point_matches_downwind_offset(self) -> None:
        # base_turn_x_ft is now -downwind_offset_ft so the perpendicular
        # base leg and final leg both get reasonable lengths. For the
        # default 3500 ft offset that means -3500 ft past the threshold.
        self.assertEqual(
            self.pattern.base_turn_x_ft, -self.config.pattern.downwind_offset_ft
        )

    def test_normal_speed_turns_base_at_nominal_point(self) -> None:
        phase = self.mode_manager.update(
            FlightPhase.DOWNWIND,
            make_state(
                runway_x_ft=self.pattern.base_turn_x_ft - 100.0,
                runway_y_ft=self.pattern.downwind_y_ft,
                gs_kt=self.config.performance.downwind_speed_kt,
                ias_kt=self.config.performance.downwind_speed_kt,
            ),
            self.route_manager,
            self.pattern,
            self.safe,
        )
        self.assertEqual(phase, FlightPhase.BASE)

    def test_normal_speed_does_not_turn_base_before_nominal_point(self) -> None:
        phase = self.mode_manager.update(
            FlightPhase.DOWNWIND,
            make_state(
                runway_x_ft=self.pattern.base_turn_x_ft + 500.0,
                runway_y_ft=self.pattern.downwind_y_ft,
                gs_kt=self.config.performance.downwind_speed_kt,
                ias_kt=self.config.performance.downwind_speed_kt,
            ),
            self.route_manager,
            self.pattern,
            self.safe,
        )
        self.assertEqual(phase, FlightPhase.DOWNWIND)

    def test_slow_speed_can_turn_base_earlier_but_capped_at_abeam(self) -> None:
        # 40 kt shortfall → 40*140 = 5600 ft of relief, clamped to
        # earliest - nominal = 750 - (-1500) = 2250 ft. Adaptive
        # turn point = -1500 + 2250 = 750 (== abeam_window_ft).
        phase = self.mode_manager.update(
            FlightPhase.DOWNWIND,
            make_state(
                runway_x_ft=700.0,  # just inside the earliest allowed
                runway_y_ft=self.pattern.downwind_y_ft,
                gs_kt=40.0,
                ias_kt=55.0,
            ),
            self.route_manager,
            self.pattern,
            self.safe,
        )
        self.assertEqual(phase, FlightPhase.BASE)

    def test_slow_speed_still_does_not_turn_base_before_abeam(self) -> None:
        # Even at standstill, the aircraft should wait until at least
        # the abeam-window cap before turning — otherwise a stuck-on-
        # downwind aircraft could turn base from miles away.
        phase = self.mode_manager.update(
            FlightPhase.DOWNWIND,
            make_state(
                runway_x_ft=2000.0,  # way short of threshold
                runway_y_ft=self.pattern.downwind_y_ft,
                gs_kt=30.0,
                ias_kt=45.0,
            ),
            self.route_manager,
            self.pattern,
            self.safe,
        )
        self.assertEqual(phase, FlightPhase.DOWNWIND)


if __name__ == "__main__":
    unittest.main()
