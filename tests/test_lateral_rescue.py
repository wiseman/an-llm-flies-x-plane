from __future__ import annotations

import unittest

from sim_pilot.core.types import AircraftState, KT_TO_FPS, StraightLeg, Vec2, heading_to_vector
from sim_pilot.guidance.lateral import L1PathFollower, _distance_to_leg_segment


def _make_state(position_ft: Vec2, track_deg: float, gs_kt: float = 80.0) -> AircraftState:
    return AircraftState(
        t_sim=0.0,
        dt=0.2,
        position_ft=position_ft,
        alt_msl_ft=3000.0,
        alt_agl_ft=2500.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        heading_deg=track_deg,
        track_deg=track_deg,
        p_rad_s=0.0,
        q_rad_s=0.0,
        r_rad_s=0.0,
        ias_kt=gs_kt,
        tas_kt=gs_kt,
        gs_kt=gs_kt,
        vs_fpm=0.0,
        ground_velocity_ft_s=heading_to_vector(track_deg, gs_kt * KT_TO_FPS),
        flap_index=0,
        gear_down=True,
        on_ground=False,
        throttle_pos=0.6,
        runway_id=None,
        runway_dist_remaining_ft=None,
        runway_x_ft=None,
        runway_y_ft=None,
        centerline_error_ft=None,
        threshold_abeam=False,
        distance_to_touchdown_ft=None,
        stall_margin=1.8,
    )


class DistanceToLegSegmentTests(unittest.TestCase):
    def test_distance_is_perpendicular_when_projection_is_inside_segment(self) -> None:
        leg = StraightLeg(start_ft=Vec2(0.0, 0.0), end_ft=Vec2(10000.0, 0.0))
        state = _make_state(position_ft=Vec2(5000.0, 500.0), track_deg=0.0)
        self.assertAlmostEqual(_distance_to_leg_segment(state, leg), 500.0, places=1)

    def test_distance_is_to_start_when_projection_is_behind(self) -> None:
        leg = StraightLeg(start_ft=Vec2(0.0, 0.0), end_ft=Vec2(10000.0, 0.0))
        state = _make_state(position_ft=Vec2(-3000.0, 4000.0), track_deg=0.0)
        # 3-4-5 triangle from origin: 5000 ft
        self.assertAlmostEqual(_distance_to_leg_segment(state, leg), 5000.0, places=1)

    def test_distance_is_to_end_when_projection_is_past(self) -> None:
        leg = StraightLeg(start_ft=Vec2(0.0, 0.0), end_ft=Vec2(10000.0, 0.0))
        state = _make_state(position_ft=Vec2(13000.0, 4000.0), track_deg=0.0)
        self.assertAlmostEqual(_distance_to_leg_segment(state, leg), 5000.0, places=1)


class L1RescuePathTests(unittest.TestCase):
    """Regression: when the LLM engages pattern_fly from several nautical
    miles out, the L1 path follower projected the plane's position onto
    the INFINITE extension of the entry leg and computed an intercept
    angle from cross_track alone. With cross_track >> lookahead, the
    desired track always came out roughly perpendicular to the leg's
    direction — not pointed at the leg itself. The plane would fly
    perpendicular forever and never actually reach the pattern.

    Now the follower keys off cross_track magnitude: when cross_track >
    max_cross_track_ft, it rescues by direct-to the segment's start.
    Crucially, when cross_track is small (e.g. rollout past touchdown on
    the runway centerline) the normal L1 math continues to run so the
    plane tracks straight down the runway."""

    def test_far_perpendicular_from_leg_commands_direct_to_start(self) -> None:
        # Entry leg from the failing log, in world frame
        leg = StraightLeg(
            start_ft=Vec2(-37534.0, 31607.0),
            end_ft=Vec2(-32596.0, 31262.0),
        )
        # Plane 5.8 nm southeast of the entry point, currently flying NW (320°)
        state = _make_state(position_ft=Vec2(-1911.0, 7507.0), track_deg=320.0, gs_kt=63.0)
        follower = L1PathFollower()
        desired_track_deg, _ = follower.follow_leg(state, leg, max_bank_deg=25.0)
        # Direct-to leg.start from the plane's position:
        #   delta = (-35623, 24100), heading = atan2(east=-35623, north=24100)
        #   ≈ -55.9° wrapped to 304°
        self.assertAlmostEqual(desired_track_deg, 304.1, delta=2.0)

    def test_close_to_leg_uses_normal_l1(self) -> None:
        # Simple horizontal leg, plane 500 ft north (within cross_track threshold)
        leg = StraightLeg(start_ft=Vec2(0.0, 0.0), end_ft=Vec2(10000.0, 0.0))
        state = _make_state(position_ft=Vec2(5000.0, 500.0), track_deg=90.0)
        follower = L1PathFollower()
        desired_track_deg, _ = follower.follow_leg(state, leg, max_bank_deg=25.0)
        # Leg direction is 90° (east). Plane 500 ft north of the midpoint
        # should command an intercept slightly south of east (90°-ish).
        # Definitely NOT a rescue direction (direct-to-origin ≈ 225°).
        self.assertLess(abs(_wrap_angle_diff(desired_track_deg, 90.0)), 80.0)

    def test_past_end_on_centerline_uses_normal_l1_not_rescue(self) -> None:
        """Regression: rollout past touchdown is ``cross_track ≈ 0``
        with along_track > path_length. The rescue must NOT fire — the
        plane should keep tracking straight down the runway, not do a
        180° U-turn back to the leg start.

        This is the failing scenario case from the commit-level test:
        final leg from (0, -10000) to (0, 600), plane at (0, 3611)
        heading 0° — 3011 ft past the end of the segment, ON the
        centerline."""
        leg = StraightLeg(start_ft=Vec2(0.0, -10000.0), end_ft=Vec2(0.0, 600.0))
        state = _make_state(position_ft=Vec2(0.0, 3611.0), track_deg=0.0, gs_kt=30.0)
        follower = L1PathFollower()
        desired_track_deg, _ = follower.follow_leg(state, leg, max_bank_deg=10.0)
        # Must stay pointed north (along the leg's course = 0°), not south
        self.assertLess(abs(_wrap_angle_diff(desired_track_deg, 0.0)), 10.0)

    def test_large_cross_track_rescue_threshold(self) -> None:
        leg = StraightLeg(start_ft=Vec2(0.0, 0.0), end_ft=Vec2(10000.0, 0.0))
        follower = L1PathFollower(max_cross_track_ft=2500.0)
        # Plane 2000 ft north of the midpoint — under threshold, use L1
        state_close = _make_state(position_ft=Vec2(5000.0, 2000.0), track_deg=90.0)
        close_track, _ = follower.follow_leg(state_close, leg, max_bank_deg=25.0)
        # Plane 5000 ft north — over threshold, rescue direct-to leg.start
        state_far = _make_state(position_ft=Vec2(5000.0, 5000.0), track_deg=90.0)
        far_track, _ = follower.follow_leg(state_far, leg, max_bank_deg=25.0)
        # Far: direct-to (0, 0) from (5000, 5000) → heading 225°
        self.assertAlmostEqual(far_track, 225.0, delta=1.0)
        self.assertNotAlmostEqual(close_track, far_track, places=0)


def _wrap_angle_diff(a: float, b: float) -> float:
    diff = (a - b + 180.0) % 360.0 - 180.0
    return diff


if __name__ == "__main__":
    unittest.main()
