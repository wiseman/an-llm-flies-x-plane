from __future__ import annotations

import unittest

from sim_pilot.core.config import load_default_config_bundle
from sim_pilot.core.mission_manager import PilotCore, StatusSnapshot
from sim_pilot.core.profiles import AltitudeHoldProfile, HeadingHoldProfile, SpeedHoldProfile
from sim_pilot.core.types import ActuatorCommands, AircraftState, GuidanceTargets, LateralMode, Vec2, VerticalMode
from sim_pilot.tui import format_snapshot_display


def _make_state(*, heading_deg: float = 0.0, ias_kt: float = 0.0, on_ground: bool = True) -> AircraftState:
    return AircraftState(
        t_sim=42.0,
        dt=0.2,
        position_ft=Vec2(0.0, 0.0),
        alt_msl_ft=500.0,
        alt_agl_ft=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        heading_deg=heading_deg,
        track_deg=heading_deg,
        p_rad_s=0.0,
        q_rad_s=0.0,
        r_rad_s=0.0,
        ias_kt=ias_kt,
        tas_kt=ias_kt,
        gs_kt=ias_kt,
        vs_fpm=0.0,
        ground_velocity_ft_s=Vec2(0.0, 0.0),
        flap_index=0,
        gear_down=True,
        on_ground=on_ground,
        throttle_pos=0.0,
        runway_id=None,
        runway_dist_remaining_ft=None,
        runway_x_ft=0.0,
        runway_y_ft=0.0,
        centerline_error_ft=0.0,
        threshold_abeam=False,
        distance_to_touchdown_ft=None,
        stall_margin=2.0,
    )


def _make_snapshot(
    *,
    state: AircraftState,
    profiles: tuple[str, ...],
    throttle: float,
    guidance: GuidanceTargets | None,
) -> StatusSnapshot:
    return StatusSnapshot(
        t_sim=42.0,
        active_profiles=profiles,
        phase=None,
        state=state,
        last_commands=ActuatorCommands(
            aileron=0.0,
            elevator=0.0,
            rudder=0.0,
            throttle=throttle,
            flaps=None,
            gear_down=True,
            brakes=0.0,
        ),
        last_guidance=guidance,
    )


class FormatSnapshotDisplayTests(unittest.TestCase):
    def test_none_snapshot_shows_waiting_placeholder(self) -> None:
        self.assertIn("waiting for first pilot tick", format_snapshot_display(None))

    def test_idle_has_no_target_heading(self) -> None:
        snapshot = _make_snapshot(
            state=_make_state(),
            profiles=("idle_lateral", "idle_vertical", "idle_speed"),
            throttle=0.0,
            guidance=GuidanceTargets(
                lateral_mode=LateralMode.BANK_HOLD,
                vertical_mode=VerticalMode.PITCH_HOLD,
                target_bank_deg=0.0,
                target_pitch_deg=0.0,
                throttle_limit=(0.0, 0.0),
            ),
        )
        out = format_snapshot_display(snapshot)
        self.assertIn("idle_lateral, idle_vertical, idle_speed", out)
        self.assertIn("throttle", out)
        self.assertIn("0.00", out)
        # No desired heading (both fields absent from guidance)
        self.assertIn("\u2014", out)  # some target field is "—"

    def test_heading_hold_shows_target_heading(self) -> None:
        guidance = GuidanceTargets(
            lateral_mode=LateralMode.TRACK_HOLD,
            vertical_mode=VerticalMode.TECS,
            target_bank_deg=5.0,
            target_heading_deg=270.0,
            target_track_deg=270.0,
            target_altitude_ft=3000.0,
            target_speed_kt=95.0,
            throttle_limit=(0.1, 0.9),
        )
        snapshot = _make_snapshot(
            state=_make_state(heading_deg=265.0, ias_kt=95.0, on_ground=False),
            profiles=("heading_hold", "altitude_hold", "speed_hold"),
            throttle=0.55,
            guidance=guidance,
        )
        out = format_snapshot_display(snapshot)
        self.assertIn("heading_hold, altitude_hold, speed_hold", out)
        self.assertIn("throttle", out)
        self.assertIn("0.55", out)
        self.assertIn("265\u00b0", out)  # current heading
        self.assertIn("270\u00b0", out)  # target heading
        self.assertIn("2500 AGL", out)  # 3000 MSL - 500 field elev = 2500 AGL
        self.assertIn(" 95 kt IAS", out)  # current speed
        self.assertIn(" 95 kt", out)  # target speed
        self.assertIn("airborne", out)

    def test_target_track_falls_back_when_heading_missing(self) -> None:
        guidance = GuidanceTargets(
            lateral_mode=LateralMode.TRACK_HOLD,
            vertical_mode=VerticalMode.TECS,
            target_bank_deg=0.0,
            target_heading_deg=None,
            target_track_deg=180.0,
            target_altitude_ft=2000.0,
            target_speed_kt=85.0,
            throttle_limit=(0.1, 0.9),
        )
        snapshot = _make_snapshot(
            state=_make_state(heading_deg=180.0, ias_kt=85.0, on_ground=False),
            profiles=("pattern_fly",),
            throttle=0.4,
            guidance=guidance,
        )
        out = format_snapshot_display(snapshot)
        # target_heading_deg is None, so the display falls back to
        # target_track_deg (180°) for the target column.
        target_line = [line for line in out.splitlines() if "heading" in line][0]
        self.assertIn("180°", target_line)

    def test_layout_has_two_column_current_target(self) -> None:
        snapshot = _make_snapshot(
            state=_make_state(),
            profiles=("idle_lateral",),
            throttle=0.0,
            guidance=None,
        )
        lines = format_snapshot_display(snapshot).splitlines()
        self.assertIn("\u25b8", lines[0])  # ▸ phase marker
        self.assertIn("idle_lateral", lines[0])
        # Header row announces the two columns
        header = next(line for line in lines if "current" in line and "target" in line)
        self.assertIsNotNone(header)
        # Secondary row shows throttle/flaps/gear/runway
        bottom = next(
            line for line in lines if "throttle" in line and "flaps" in line and "gear" in line
        )
        self.assertIn("rwy", bottom)

    def test_shows_flap_position(self) -> None:
        # Regression: flap position must appear on the bottom status row
        # so the operator can see what configuration the aircraft is in.
        state = _make_state()
        import dataclasses
        state = dataclasses.replace(state, flap_index=20)
        snapshot = _make_snapshot(
            state=state,
            profiles=("pattern_fly",),
            throttle=0.45,
            guidance=None,
        )
        out = format_snapshot_display(snapshot)
        self.assertIn("flaps 20°", out)

    def test_runway_line_shows_runway_x_and_y(self) -> None:
        state = _make_state()
        import dataclasses
        state = dataclasses.replace(state, runway_x_ft=1234.0, runway_y_ft=-567.0)
        snapshot = _make_snapshot(
            state=state,
            profiles=("pattern_fly",),
            throttle=0.5,
            guidance=None,
        )
        out = format_snapshot_display(snapshot)
        self.assertIn("rwy x+1234 y-567", out)

    def test_runway_line_handles_missing_runway_frame_coords(self) -> None:
        state = _make_state()
        import dataclasses
        state = dataclasses.replace(state, runway_x_ft=None, runway_y_ft=None)
        snapshot = _make_snapshot(
            state=state,
            profiles=("idle_lateral",),
            throttle=0.0,
            guidance=None,
        )
        out = format_snapshot_display(snapshot)
        self.assertIn("rwy —", out)

    def test_shows_ground_vs_airborne(self) -> None:
        on_ground = format_snapshot_display(
            _make_snapshot(
                state=_make_state(on_ground=True),
                profiles=("idle_lateral",),
                throttle=0.0,
                guidance=None,
            )
        )
        self.assertIn("on ground", on_ground)
        airborne = format_snapshot_display(
            _make_snapshot(
                state=_make_state(on_ground=False),
                profiles=("idle_lateral",),
                throttle=0.0,
                guidance=None,
            )
        )
        self.assertIn("airborne", airborne)


class LatestSnapshotWiringTests(unittest.TestCase):
    """Confirm the guidance is actually flowing through PilotCore.update()
    into latest_snapshot.last_guidance — not just when you construct a
    snapshot manually."""

    def setUp(self) -> None:
        self.config = load_default_config_bundle()
        self.pilot = PilotCore(self.config)

    def test_idle_pilot_produces_guidance_on_snapshot(self) -> None:
        from sim_pilot.sim.simple_dynamics import SimpleAircraftModel
        model = SimpleAircraftModel(self.config, Vec2(0.0, 0.0))
        raw = model.initial_state()
        self.pilot.update(raw, 0.2)
        snap = self.pilot.latest_snapshot
        self.assertIsNotNone(snap)
        self.assertIsNotNone(snap.last_guidance)
        # Idle profiles produce throttle_limit=(0,0), so commanded throttle is 0
        self.assertEqual(snap.last_commands.throttle, 0.0)

    def test_heading_hold_writes_target_heading_to_snapshot(self) -> None:
        self.pilot.engage_profile(HeadingHoldProfile(heading_deg=270.0))
        self.pilot.engage_profile(AltitudeHoldProfile(altitude_ft=3000.0))
        self.pilot.engage_profile(SpeedHoldProfile(speed_kt=90.0))
        from sim_pilot.sim.simple_dynamics import SimpleAircraftModel
        model = SimpleAircraftModel(self.config, Vec2(0.0, 0.0))
        raw = model.initial_state()
        self.pilot.update(raw, 0.2)
        snap = self.pilot.latest_snapshot
        assert snap is not None
        assert snap.last_guidance is not None
        self.assertEqual(snap.last_guidance.target_heading_deg, 270.0)
        self.assertEqual(snap.last_guidance.target_altitude_ft, 3000.0)
        self.assertEqual(snap.last_guidance.target_speed_kt, 90.0)


if __name__ == "__main__":
    unittest.main()
