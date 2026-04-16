from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sim_pilot.core.config import load_default_config_bundle
from sim_pilot.core.mission_manager import PilotCore
from sim_pilot.core.profiles import HeadingHoldProfile, PatternFlyProfile
from sim_pilot.llm.tools import ToolContext, dispatch_tool


class FakeBridge:
    def __init__(
        self,
        initial_values: dict[str, float] | None = None,
        *,
        georef_lat_deg: float = 47.4638,
        georef_lon_deg: float = -122.308,
    ) -> None:
        from sim_pilot.sim.xplane_bridge import GeoReference
        self.writes: list[dict[str, float | int]] = []
        self._values: dict[str, float] = dict(initial_values or {})
        self.georef = GeoReference(
            threshold_lat_deg=georef_lat_deg,
            threshold_lon_deg=georef_lon_deg,
        )

    def write_dataref_values(self, updates: dict[str, float | int]) -> None:
        self.writes.append(updates)
        for name, value in updates.items():
            self._values[name] = float(value)

    def get_dataref_value(self, name: str) -> float | None:
        return self._values.get(name)


def make_ctx(bridge: FakeBridge | None = None, runway_csv_path: Path | None = None) -> ToolContext:
    config = load_default_config_bundle()
    pilot = PilotCore(config)
    return ToolContext(
        pilot=pilot,
        bridge=bridge,
        config=config,
        recent_broadcasts=[],
        runway_csv_path=runway_csv_path,
    )


_RUNWAY_CSV_HEADER = (
    "id,airport_ref,airport_ident,length_ft,width_ft,surface,lighted,closed,"
    "le_ident,le_latitude_deg,le_longitude_deg,le_elevation_ft,le_heading_degT,le_displaced_threshold_ft,"
    "he_ident,he_latitude_deg,he_longitude_deg,he_elevation_ft,he_heading_degT,he_displaced_threshold_ft"
)


def _build_fake_runway_csv(path: Path, extra_rows: list[str] | None = None) -> None:
    rows = [
        # KSEA — three real runways
        "1,1,KSEA,11900,150,ASP,1,0,16L,47.4638,-122.308,432,180.0,,34R,47.4312,-122.308,347,360.0,",
        "2,1,KSEA,9426,150,CONC-F,1,0,16C,47.4638,-122.311,430,180.0,,34C,47.438,-122.311,363,360.0,",
        "3,1,KSEA,9426,150,CON,1,0,16R,47.4638,-122.318,430,180.0,,34L,47.4405,-122.318,363,360.0,",
        # KBFI — Boeing Field, ~5 km north
        "4,2,KBFI,10007,200,ASP,1,0,14R,47.5301,-122.3015,21,142.0,,32L,47.5111,-122.2876,21,322.0,",
        # KJFK — far away, used to confirm spatial sort puts it last
        "5,3,KJFK,12079,150,ASP,1,0,04L,40.6235,-73.7944,12,40.0,,22R,40.6432,-73.7821,12,220.0,",
    ]
    if extra_rows:
        rows.extend(extra_rows)
    path.write_text(_RUNWAY_CSV_HEADER + "\n" + "\n".join(rows) + "\n")


def make_call(tool_name: str, /, **args: object) -> dict[str, object]:
    return {"name": tool_name, "call_id": "call_test", "arguments": json.dumps(args)}


class DispatchBasicsTests(unittest.TestCase):
    def test_unknown_tool_returns_error(self) -> None:
        ctx = make_ctx()
        result = dispatch_tool({"name": "not_a_tool", "arguments": "{}"}, ctx)
        self.assertTrue(result.startswith("error:"))
        self.assertIn("unknown tool", result)

    def test_invalid_arguments_json_returns_error(self) -> None:
        ctx = make_ctx()
        result = dispatch_tool({"name": "engage_heading_hold", "arguments": "not-json"}, ctx)
        self.assertTrue(result.startswith("error:"))

    def test_missing_required_arg_returns_error(self) -> None:
        ctx = make_ctx()
        result = dispatch_tool(make_call("engage_heading_hold"), ctx)
        self.assertTrue(result.startswith("error:"))


class ProfileToolsTests(unittest.TestCase):
    def test_engage_heading_hold_engages_profile(self) -> None:
        ctx = make_ctx()
        result = dispatch_tool(make_call("engage_heading_hold", heading_deg=270.0), ctx)
        self.assertIn("heading_hold", result)
        self.assertIn("heading_hold", ctx.pilot.list_profile_names())
        self.assertNotIn("idle_lateral", ctx.pilot.list_profile_names())

    def test_engage_cruise_installs_three_holds_atomically(self) -> None:
        ctx = make_ctx()
        result = dispatch_tool(
            make_call("engage_cruise", heading_deg=130.0, altitude_ft=2500.0, speed_kt=95.0),
            ctx,
        )
        self.assertIn("engaged cruise", result)
        names = set(ctx.pilot.list_profile_names())
        self.assertEqual(names, {"heading_hold", "altitude_hold", "speed_hold"})

    def test_engage_cruise_from_takeoff_displaces_three_axis_profile(self) -> None:
        ctx = make_ctx()
        dispatch_tool(make_call("engage_takeoff"), ctx)
        self.assertEqual(ctx.pilot.list_profile_names(), ["takeoff"])
        # Engage cruise atomically: the single tool call should
        # displace takeoff and install all three holds, never leaving
        # the vertical or speed axes uncovered mid-swap.
        result = dispatch_tool(
            make_call("engage_cruise", heading_deg=180.0, altitude_ft=3000.0, speed_kt=100.0),
            ctx,
        )
        self.assertIn("displaced", result)
        self.assertIn("takeoff", result)
        names = set(ctx.pilot.list_profile_names())
        self.assertEqual(names, {"heading_hold", "altitude_hold", "speed_hold"})
        self.assertNotIn("takeoff", names)

    def test_engage_cruise_carries_target_values(self) -> None:
        from sim_pilot.core.profiles import AltitudeHoldProfile, HeadingHoldProfile, SpeedHoldProfile
        ctx = make_ctx()
        dispatch_tool(
            make_call("engage_cruise", heading_deg=270.0, altitude_ft=3500.0, speed_kt=110.0),
            ctx,
        )
        hh = ctx.pilot.find_profile("heading_hold")
        ah = ctx.pilot.find_profile("altitude_hold")
        sh = ctx.pilot.find_profile("speed_hold")
        assert isinstance(hh, HeadingHoldProfile)
        assert isinstance(ah, AltitudeHoldProfile)
        assert isinstance(sh, SpeedHoldProfile)
        self.assertAlmostEqual(hh.heading_deg, 270.0)
        self.assertAlmostEqual(ah.altitude_ft, 3500.0)
        self.assertAlmostEqual(sh.speed_kt, 110.0)

    def test_engage_pattern_fly_without_required_args_errors(self) -> None:
        ctx = make_ctx()
        result = dispatch_tool(make_call("engage_pattern_fly"), ctx)
        self.assertTrue(result.startswith("error:"))
        # Profile must not be engaged when the call fails
        self.assertNotIn("pattern_fly", ctx.pilot.list_profile_names())

    def test_engage_takeoff_displaces_idle_profiles(self) -> None:
        ctx = make_ctx()
        result = dispatch_tool(make_call("engage_takeoff"), ctx)
        self.assertIn("takeoff", result)
        self.assertEqual(ctx.pilot.list_profile_names(), ["takeoff"])

    def test_engage_takeoff_refuses_when_parking_brake_set(self) -> None:
        bridge = FakeBridge(initial_values={"sim/cockpit2/controls/parking_brake_ratio": 1.0})
        ctx = make_ctx(bridge=bridge)
        result = dispatch_tool(make_call("engage_takeoff"), ctx)
        self.assertTrue(result.startswith("error:"))
        self.assertIn("parking brake", result)
        self.assertIn("set_parking_brake", result)
        # Confirm the profile was NOT engaged
        self.assertNotIn("takeoff", ctx.pilot.list_profile_names())

    def test_engage_takeoff_succeeds_after_parking_brake_released(self) -> None:
        bridge = FakeBridge(initial_values={"sim/cockpit2/controls/parking_brake_ratio": 1.0})
        ctx = make_ctx(bridge=bridge)
        # First attempt: refused
        refused = dispatch_tool(make_call("engage_takeoff"), ctx)
        self.assertTrue(refused.startswith("error:"))
        # Release the brake via the tool
        dispatch_tool(make_call("set_parking_brake", engaged=False), ctx)
        # Second attempt: succeeds
        result = dispatch_tool(make_call("engage_takeoff"), ctx)
        self.assertIn("takeoff", result)
        self.assertIn("takeoff", ctx.pilot.list_profile_names())

    def test_engage_takeoff_allows_when_no_bridge(self) -> None:
        # Simple backend (no X-Plane bridge) has no parking brake to check,
        # so engage_takeoff should still work as before.
        ctx = make_ctx()
        self.assertIsNone(ctx.bridge)
        result = dispatch_tool(make_call("engage_takeoff"), ctx)
        self.assertIn("takeoff", result)
        self.assertIn("takeoff", ctx.pilot.list_profile_names())

    def test_disengage_profile_readds_idles(self) -> None:
        ctx = make_ctx()
        ctx.pilot.engage_profile(HeadingHoldProfile(heading_deg=270.0))
        result = dispatch_tool(make_call("disengage_profile", name="heading_hold"), ctx)
        self.assertIn("heading_hold", result)
        self.assertIn("idle_lateral", ctx.pilot.list_profile_names())

    def test_list_profiles_returns_comma_separated(self) -> None:
        ctx = make_ctx()
        result = dispatch_tool(make_call("list_profiles"), ctx)
        self.assertIn("idle_lateral", result)
        self.assertIn("idle_vertical", result)
        self.assertIn("idle_speed", result)

    def test_engage_approach_returns_not_implemented(self) -> None:
        ctx = make_ctx()
        result = dispatch_tool(make_call("engage_approach", runway_id="16L"), ctx)
        self.assertIn("not yet implemented", result)


class PatternEventTests(unittest.TestCase):
    def test_extend_downwind_without_pattern_fly_errors(self) -> None:
        ctx = make_ctx()
        result = dispatch_tool(make_call("extend_downwind", extension_ft=1000.0), ctx)
        self.assertTrue(result.startswith("error:"))
        self.assertIn("pattern_fly", result)

    def test_extend_downwind_with_pattern_fly_succeeds(self) -> None:
        ctx = make_ctx()
        ctx.pilot.engage_profile(PatternFlyProfile(ctx.config, ctx.pilot.runway_frame))
        result = dispatch_tool(make_call("extend_downwind", extension_ft=1500.0), ctx)
        self.assertIn("extended downwind", result)
        profile = ctx.pilot.find_profile("pattern_fly")
        assert isinstance(profile, PatternFlyProfile)
        self.assertEqual(profile.pattern_extension_ft, 1500.0)

    def test_turn_base_now_without_pattern_fly_errors(self) -> None:
        ctx = make_ctx()
        result = dispatch_tool(make_call("turn_base_now"), ctx)
        self.assertTrue(result.startswith("error:"))

    def test_turn_base_now_sets_profile_trigger(self) -> None:
        ctx = make_ctx()
        ctx.pilot.engage_profile(PatternFlyProfile(ctx.config, ctx.pilot.runway_frame))
        dispatch_tool(make_call("turn_base_now"), ctx)
        profile = ctx.pilot.find_profile("pattern_fly")
        assert isinstance(profile, PatternFlyProfile)
        self.assertTrue(profile._turn_base_trigger)

    def test_go_around_without_pattern_fly_errors(self) -> None:
        ctx = make_ctx()
        result = dispatch_tool(make_call("go_around"), ctx)
        self.assertTrue(result.startswith("error:"))


class PatternFlyRunwayLookupTests(unittest.TestCase):
    """Tests for engage_pattern_fly + join_pattern against a real runway
    looked up from the DuckDB-backed runway CSV."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.csv_path = Path(self._tmp.name) / "runways.csv"
        _build_fake_runway_csv(self.csv_path)

    def _make_ctx_with_bridge(self) -> ToolContext:
        bridge = FakeBridge(
            georef_lat_deg=47.4638,
            georef_lon_deg=-122.308,
        )
        ctx = make_ctx(bridge=bridge, runway_csv_path=self.csv_path)
        return ctx

    def test_engage_pattern_fly_requires_all_four_args(self) -> None:
        ctx = self._make_ctx_with_bridge()
        result = dispatch_tool(make_call("engage_pattern_fly"), ctx)
        self.assertTrue(result.startswith("error:"))

    def test_engage_pattern_fly_for_takeoff_roll(self) -> None:
        ctx = self._make_ctx_with_bridge()
        result = dispatch_tool(
            make_call(
                "engage_pattern_fly",
                airport_ident="KSEA",
                runway_ident="16L",
                side="left",
                start_phase="takeoff_roll",
            ),
            ctx,
        )
        self.assertNotIn("error", result)
        self.assertEqual(ctx.pilot.list_profile_names(), ["pattern_fly"])
        from sim_pilot.core.types import FlightPhase
        profile = ctx.pilot.find_profile("pattern_fly")
        assert isinstance(profile, PatternFlyProfile)
        self.assertEqual(profile.phase, FlightPhase.TAKEOFF_ROLL)

    def test_engage_pattern_fly_with_ksea_16l_installs_runway_frame(self) -> None:
        ctx = self._make_ctx_with_bridge()
        result = dispatch_tool(
            make_call(
                "engage_pattern_fly",
                airport_ident="KSEA",
                runway_ident="16L",
                side="left",
                start_phase="pattern_entry",
            ),
            ctx,
        )
        self.assertIn("pattern_fly", result)
        self.assertNotIn("error", result)
        # The pilot's runway_frame should now be anchored at KSEA 16L
        self.assertEqual(ctx.pilot.runway_frame.runway.id, "16L")
        self.assertEqual(ctx.pilot.config.airport.airport, "KSEA")
        # KSEA 16L course is 180° (southbound)
        self.assertAlmostEqual(ctx.pilot.runway_frame.runway.course_deg, 180.0, delta=5.0)
        # Field elevation comes from the CSV's le_elevation_ft (432 ft for the fixture)
        self.assertAlmostEqual(ctx.pilot.config.airport.field_elevation_ft, 432.0, delta=1.0)
        # Profile should be pre-positioned at pattern_entry (not preflight)
        profile = ctx.pilot.find_profile("pattern_fly")
        assert isinstance(profile, PatternFlyProfile)
        from sim_pilot.core.types import FlightPhase
        self.assertEqual(profile.phase, FlightPhase.PATTERN_ENTRY)

    def test_engage_pattern_fly_sets_aim_point_from_runway_length(self) -> None:
        # Regression: the old implementation hardcoded touchdown_zone_ft
        # to 1000, which pushed ``touchdown_runway_x_ft`` to the 500 ft
        # floor on every runway regardless of length. The fix synthesizes
        # touchdown_zone_ft from runway length (2000 ft TDZ for any
        # runway >= 4000 ft), which puts the aim point at ~1000 ft past
        # the threshold — the conventional aim-point marker location.
        ctx = self._make_ctx_with_bridge()
        result = dispatch_tool(
            make_call(
                "engage_pattern_fly",
                airport_ident="KSEA",
                runway_ident="16L",  # 11900 ft long
                side="left",
                start_phase="pattern_entry",
            ),
            ctx,
        )
        self.assertNotIn("error", result)
        self.assertAlmostEqual(
            ctx.pilot.runway_frame.touchdown_runway_x_ft, 1000.0, delta=1.0
        )

    def test_engage_pattern_fly_with_34r_picks_high_end_threshold(self) -> None:
        ctx = self._make_ctx_with_bridge()
        result = dispatch_tool(
            make_call(
                "engage_pattern_fly",
                airport_ident="KSEA",
                runway_ident="34R",
                side="left",
                start_phase="pattern_entry",
            ),
            ctx,
        )
        self.assertNotIn("error", result)
        self.assertEqual(ctx.pilot.runway_frame.runway.id, "34R")
        # 34R course is 360° (northbound), from the CSV fixture's he_heading_degT
        self.assertAlmostEqual(ctx.pilot.runway_frame.runway.course_deg, 360.0, delta=5.0)

    def test_engage_pattern_fly_with_unknown_airport_returns_error(self) -> None:
        ctx = self._make_ctx_with_bridge()
        result = dispatch_tool(
            make_call(
                "engage_pattern_fly",
                airport_ident="NOWHERE",
                runway_ident="16L",
                side="left",
                start_phase="pattern_entry",
            ),
            ctx,
        )
        self.assertTrue(result.startswith("error:"))
        self.assertIn("not found", result)
        # Profile should NOT have been engaged
        self.assertNotIn("pattern_fly", ctx.pilot.list_profile_names())

    def test_engage_pattern_fly_with_missing_runway_ident_errors(self) -> None:
        ctx = self._make_ctx_with_bridge()
        # Only airport_ident — missing required runway_ident/side/start_phase
        result = dispatch_tool(
            make_call("engage_pattern_fly", airport_ident="KSEA"),
            ctx,
        )
        self.assertTrue(result.startswith("error:"))

    def test_engage_pattern_fly_invalid_start_phase(self) -> None:
        ctx = self._make_ctx_with_bridge()
        result = dispatch_tool(
            make_call(
                "engage_pattern_fly",
                airport_ident="KSEA",
                runway_ident="16L",
                side="left",
                start_phase="barrel_roll",
            ),
            ctx,
        )
        self.assertTrue(result.startswith("error:"))
        self.assertIn("unknown start_phase", result)

    def test_engage_pattern_fly_without_bridge_errors_for_runway_lookup(self) -> None:
        # Simple-backend context has no bridge → can't compute world-frame threshold
        config = load_default_config_bundle()
        pilot = PilotCore(config)
        ctx = ToolContext(
            pilot=pilot,
            bridge=None,
            config=config,
            recent_broadcasts=[],
            runway_csv_path=self.csv_path,
        )
        result = dispatch_tool(
            make_call(
                "engage_pattern_fly",
                airport_ident="KSEA",
                runway_ident="16L",
                side="left",
                start_phase="pattern_entry",
            ),
            ctx,
        )
        self.assertTrue(result.startswith("error:"))
        self.assertIn("bridge", result.lower())

    def test_join_pattern_acks_with_pattern_fly_active(self) -> None:
        """Regression: the previous implementation compared runway_id against
        profile.runway_frame.runway.id and rejected the call whenever the id
        was None (which was the case after bootstrap). It now succeeds as
        a pure acknowledgment regardless of the active runway's id."""
        ctx = self._make_ctx_with_bridge()
        # Engage pattern_fly with a specific runway so pattern_fly is active
        dispatch_tool(
            make_call(
                "engage_pattern_fly",
                airport_ident="KSEA",
                runway_ident="16L",
                side="left",
                start_phase="pattern_entry",
            ),
            ctx,
        )
        result = dispatch_tool(make_call("join_pattern", runway_id="30"), ctx)
        self.assertNotIn("error", result)
        self.assertIn("runway=30", result)

    def test_join_pattern_without_pattern_fly_active_errors(self) -> None:
        ctx = self._make_ctx_with_bridge()
        result = dispatch_tool(make_call("join_pattern", runway_id="30"), ctx)
        self.assertTrue(result.startswith("error:"))
        self.assertIn("engage_pattern_fly", result)


class RadioToolsTests(unittest.TestCase):
    def test_tune_radio_without_bridge_errors(self) -> None:
        ctx = make_ctx(bridge=None)
        result = dispatch_tool(make_call("tune_radio", radio="com1", frequency_mhz=118.30), ctx)
        self.assertTrue(result.startswith("error:"))

    def test_tune_radio_writes_dataref(self) -> None:
        bridge = FakeBridge()
        ctx = make_ctx(bridge=bridge)
        result = dispatch_tool(make_call("tune_radio", radio="com1", frequency_mhz=118.30), ctx)
        self.assertIn("118.300 MHz", result)
        self.assertEqual(len(bridge.writes), 1)
        (update,) = bridge.writes
        name, value = next(iter(update.items()))
        self.assertIn("com1_frequency_hz_833", name)
        self.assertEqual(value, 118_300)

    def test_tune_radio_rejects_unknown_radio(self) -> None:
        bridge = FakeBridge()
        ctx = make_ctx(bridge=bridge)
        result = dispatch_tool(make_call("tune_radio", radio="com9", frequency_mhz=118.30), ctx)
        self.assertTrue(result.startswith("error:"))
        self.assertEqual(bridge.writes, [])

    def test_broadcast_appends_to_recent_and_prints(self) -> None:
        ctx = make_ctx()
        buf = io.StringIO()
        with patch("sys.stdout", new=buf):
            result = dispatch_tool(make_call("broadcast_on_radio", radio="com1", message="hello"), ctx)
        self.assertIn("hello", result)
        self.assertEqual(len(ctx.recent_broadcasts), 1)
        self.assertIn("[BROADCAST com1] hello", ctx.recent_broadcasts[0])
        self.assertIn("[BROADCAST com1] hello", buf.getvalue())

    def test_broadcast_pushes_to_bus_when_provided(self) -> None:
        from sim_pilot.bus import SimBus
        bus = SimBus(echo=False)
        ctx = make_ctx()
        ctx.bus = bus
        dispatch_tool(make_call("broadcast_on_radio", radio="com1", message="hello bus"), ctx)
        _, _, radio = bus.snapshot()
        self.assertEqual(len(radio), 1)
        self.assertIn("[BROADCAST com1] hello bus", radio[0])

    def test_set_parking_brake_engage_writes_1(self) -> None:
        bridge = FakeBridge()
        ctx = make_ctx(bridge=bridge)
        result = dispatch_tool(make_call("set_parking_brake", engaged=True), ctx)
        self.assertIn("engaged", result)
        self.assertEqual(len(bridge.writes), 1)
        (update,) = bridge.writes
        name, value = next(iter(update.items()))
        self.assertEqual(name, "sim/cockpit2/controls/parking_brake_ratio")
        self.assertEqual(value, 1.0)

    def test_set_parking_brake_release_writes_0(self) -> None:
        bridge = FakeBridge()
        ctx = make_ctx(bridge=bridge)
        result = dispatch_tool(make_call("set_parking_brake", engaged=False), ctx)
        self.assertIn("released", result)
        (update,) = bridge.writes
        self.assertEqual(update["sim/cockpit2/controls/parking_brake_ratio"], 0.0)

    def test_set_parking_brake_without_bridge_errors(self) -> None:
        ctx = make_ctx(bridge=None)
        result = dispatch_tool(make_call("set_parking_brake", engaged=True), ctx)
        self.assertTrue(result.startswith("error:"))


class TakeoffChecklistTests(unittest.TestCase):
    def _seed_snapshot(self, ctx: ToolContext, *, flap_index: int = 0, gear_down: bool = True,
                       on_ground: bool = True, ias_kt: float = 0.0) -> None:
        from sim_pilot.core.mission_manager import StatusSnapshot
        from sim_pilot.core.types import ActuatorCommands, AircraftState, Vec2
        state = AircraftState(
            t_sim=0.0,
            dt=0.2,
            position_ft=Vec2(0.0, 0.0),
            alt_msl_ft=314.0,
            alt_agl_ft=0.0,
            pitch_deg=0.0,
            roll_deg=0.0,
            heading_deg=0.0,
            track_deg=0.0,
            p_rad_s=0.0,
            q_rad_s=0.0,
            r_rad_s=0.0,
            ias_kt=ias_kt,
            tas_kt=ias_kt,
            gs_kt=ias_kt,
            vs_fpm=0.0,
            ground_velocity_ft_s=Vec2(0.0, 0.0),
            flap_index=flap_index,
            gear_down=gear_down,
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
        ctx.pilot.latest_snapshot = StatusSnapshot(
            t_sim=0.0,
            active_profiles=tuple(ctx.pilot.list_profile_names()),
            phase=None,
            state=state,
            last_commands=ActuatorCommands(
                aileron=0.0, elevator=0.0, rudder=0.0, throttle=0.0,
                flaps=None, gear_down=True, brakes=0.0,
            ),
        )

    def test_checklist_flags_set_parking_brake(self) -> None:
        bridge = FakeBridge(initial_values={"sim/cockpit2/controls/parking_brake_ratio": 1.0})
        ctx = make_ctx(bridge=bridge)
        self._seed_snapshot(ctx)
        result = dispatch_tool(make_call("takeoff_checklist"), ctx)
        self.assertIn("[ACTION] parking brake: SET", result)
        self.assertIn("set_parking_brake(engaged=False)", result)
        self.assertIn("need action", result)

    def test_checklist_passes_when_brake_released_and_flaps_ok(self) -> None:
        bridge = FakeBridge(initial_values={"sim/cockpit2/controls/parking_brake_ratio": 0.0})
        ctx = make_ctx(bridge=bridge)
        self._seed_snapshot(ctx, flap_index=0)
        result = dispatch_tool(make_call("takeoff_checklist"), ctx)
        self.assertIn("[OK]     parking brake: released", result)
        self.assertIn("[OK]     flaps: 0 deg", result)
        self.assertIn("[OK]     on ground", result)
        self.assertIn("All items OK", result)

    def test_checklist_flags_too_many_flaps(self) -> None:
        bridge = FakeBridge(initial_values={"sim/cockpit2/controls/parking_brake_ratio": 0.0})
        ctx = make_ctx(bridge=bridge)
        self._seed_snapshot(ctx, flap_index=20)
        result = dispatch_tool(make_call("takeoff_checklist"), ctx)
        self.assertIn("[ACTION] flaps: 20 deg", result)

    def test_checklist_flags_not_on_ground(self) -> None:
        bridge = FakeBridge(initial_values={"sim/cockpit2/controls/parking_brake_ratio": 0.0})
        ctx = make_ctx(bridge=bridge)
        self._seed_snapshot(ctx, on_ground=False)
        result = dispatch_tool(make_call("takeoff_checklist"), ctx)
        self.assertIn("[ERROR]  not on ground", result)

    def test_checklist_no_snapshot_returns_error(self) -> None:
        ctx = make_ctx()
        result = dispatch_tool(make_call("takeoff_checklist"), ctx)
        self.assertTrue(result.startswith("error:"))

    def test_checklist_brake_state_unavailable_still_flags_action(self) -> None:
        # Bridge has no parking brake value cached → mark as unknown and treat as action needed
        bridge = FakeBridge(initial_values={})
        ctx = make_ctx(bridge=bridge)
        self._seed_snapshot(ctx)
        result = dispatch_tool(make_call("takeoff_checklist"), ctx)
        self.assertIn("parking brake: state unavailable", result)
        self.assertIn("need action", result)


class StatusToolTests(unittest.TestCase):
    def test_get_status_before_any_update_returns_uninitialized(self) -> None:
        ctx = make_ctx()
        result = dispatch_tool(make_call("get_status"), ctx)
        payload = json.loads(result)
        self.assertEqual(payload, {"status": "uninitialized"})

    def test_sleep_returns_string(self) -> None:
        ctx = make_ctx()
        result = dispatch_tool(make_call("sleep"), ctx)
        self.assertIsInstance(result, str)

    def test_get_status_payload_omits_pilot_reference_and_runway_frame(self) -> None:
        from sim_pilot.core.mission_manager import StatusSnapshot
        from sim_pilot.core.types import ActuatorCommands, AircraftState, Vec2

        ctx = make_ctx()
        state = AircraftState(
            t_sim=42.0,
            dt=0.2,
            position_ft=Vec2(100.0, 200.0),
            alt_msl_ft=3000.0,
            alt_agl_ft=2500.0,
            pitch_deg=1.0,
            roll_deg=0.0,
            heading_deg=90.0,
            track_deg=88.0,
            p_rad_s=0.0,
            q_rad_s=0.0,
            r_rad_s=0.0,
            ias_kt=95.0,
            tas_kt=95.0,
            gs_kt=92.0,
            vs_fpm=0.0,
            ground_velocity_ft_s=Vec2(0.0, 0.0),
            flap_index=0,
            gear_down=True,
            on_ground=False,
            throttle_pos=0.6,
            runway_id=ctx.config.airport.runway.id,
            runway_dist_remaining_ft=None,
            runway_x_ft=10.0,
            runway_y_ft=-3.0,
            centerline_error_ft=-3.0,
            threshold_abeam=False,
            distance_to_touchdown_ft=None,
            stall_margin=2.0,
        )
        ctx.pilot.latest_snapshot = StatusSnapshot(
            t_sim=42.0,
            active_profiles=("idle_lateral", "idle_vertical", "idle_speed"),
            phase=None,
            state=state,
            last_commands=ActuatorCommands(
                aileron=0.0, elevator=0.0, rudder=0.0, throttle=0.6,
                flaps=None, gear_down=True, brakes=0.0,
            ),
        )

        result = dispatch_tool(make_call("get_status"), ctx)
        payload = json.loads(result)

        # The pilot reference, world-frame position, and runway-frame position
        # are all hidden from the agent — it must check the sim itself.
        self.assertNotIn("pilot_reference", payload)
        self.assertNotIn("runway_id", payload)
        self.assertNotIn("position", payload)
        self.assertNotIn("world_frame_ft", payload)
        self.assertNotIn("runway_frame_ft", payload)
        # Lat/lon are the only spatial facts exposed; null without a bridge.
        self.assertIn("lat_deg", payload)
        self.assertIn("lon_deg", payload)
        self.assertIsNone(payload["lat_deg"])
        self.assertIsNone(payload["lon_deg"])
        # Universal flight-state fields remain.
        self.assertEqual(payload["heading_deg"], 90.0)
        self.assertEqual(payload["alt_msl_ft"], 3000.0)
        self.assertEqual(payload["ias_kt"], 95.0)


class SqlQueryToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.csv_path = Path(self._tmp.name) / "runways.csv"
        _build_fake_runway_csv(self.csv_path)

    def test_select_returns_tab_separated_rows_with_header(self) -> None:
        ctx = make_ctx(runway_csv_path=self.csv_path)
        result = dispatch_tool(
            make_call(
                "sql_query",
                query="SELECT airport_ident, le_ident, length_ft FROM runways WHERE airport_ident='KSEA' ORDER BY length_ft DESC",
            ),
            ctx,
        )
        lines = result.splitlines()
        self.assertEqual(lines[0], "airport_ident\tle_ident\tlength_ft")
        self.assertEqual(len(lines), 4)  # header + 3 rows
        self.assertEqual(lines[1].split("\t"), ["KSEA", "16L", "11900"])

    def test_empty_result_returns_zero_rows_message(self) -> None:
        ctx = make_ctx(runway_csv_path=self.csv_path)
        result = dispatch_tool(
            make_call("sql_query", query="SELECT * FROM runways WHERE airport_ident='NOWHERE'"),
            ctx,
        )
        self.assertEqual(result, "0 rows")

    def test_invalid_sql_returns_error(self) -> None:
        ctx = make_ctx(runway_csv_path=self.csv_path)
        result = dispatch_tool(make_call("sql_query", query="SELECT * FROM does_not_exist"), ctx)
        self.assertTrue(result.startswith("error:"))

    def test_write_attempt_against_view_returns_error(self) -> None:
        ctx = make_ctx(runway_csv_path=self.csv_path)
        result = dispatch_tool(
            make_call("sql_query", query="DELETE FROM runways WHERE airport_ident='KSEA'"),
            ctx,
        )
        self.assertTrue(result.startswith("error:"))
        # Subsequent SELECT should still see all KSEA rows because the view's
        # underlying table was never touched.
        verify = dispatch_tool(
            make_call("sql_query", query="SELECT COUNT(*) FROM runways WHERE airport_ident='KSEA'"),
            ctx,
        )
        # Header + one row with the count
        self.assertEqual(verify.splitlines()[1], "3")

    def test_missing_csv_path_returns_error(self) -> None:
        ctx = make_ctx(runway_csv_path=None)
        result = dispatch_tool(make_call("sql_query", query="SELECT 1"), ctx)
        self.assertTrue(result.startswith("error:"))
        self.assertIn("not configured", result)

    def test_nonexistent_csv_file_returns_error(self) -> None:
        ctx = make_ctx(runway_csv_path=self.csv_path.with_name("nope.csv"))
        result = dispatch_tool(make_call("sql_query", query="SELECT 1"), ctx)
        self.assertTrue(result.startswith("error:"))
        self.assertIn("not found", result)

    def test_row_cap_truncates_large_results(self) -> None:
        # Build a CSV with 60+ extra rows to exceed the 50-row cap.
        extra = [
            f"{100+i},{i},K{i:03d},5000,100,ASP,1,0,09,40.0,-100.0,500,90.0,,27,40.0,-100.0,500,270.0,"
            for i in range(60)
        ]
        _build_fake_runway_csv(self.csv_path, extra_rows=extra)
        ctx = make_ctx(runway_csv_path=self.csv_path)
        result = dispatch_tool(
            make_call("sql_query", query="SELECT airport_ident FROM runways"),
            ctx,
        )
        lines = result.splitlines()
        self.assertEqual(len(lines), 52)  # header + 50 rows + truncation notice
        self.assertIn("truncated", lines[-1])

    def test_spatial_function_finds_nearest_runways(self) -> None:
        ctx = make_ctx(runway_csv_path=self.csv_path)
        result = dispatch_tool(
            make_call(
                "sql_query",
                query=(
                    "SELECT airport_ident, le_ident "
                    "FROM runways "
                    "WHERE closed = 0 AND le_latitude_deg IS NOT NULL "
                    "ORDER BY ST_Distance_Sphere("
                    "  ST_Point(le_longitude_deg, le_latitude_deg), "
                    "  ST_Point(-122.31, 47.46)) "
                    "LIMIT 3"
                ),
            ),
            ctx,
        )
        lines = result.splitlines()
        self.assertEqual(lines[0], "airport_ident\tle_ident")
        # The three KSEA runways must come back first, sorted by spherical distance
        # from (47.46N, -122.31W). KBFI and KJFK are farther away.
        airports = [line.split("\t")[0] for line in lines[1:4]]
        self.assertEqual(airports, ["KSEA", "KSEA", "KSEA"])

    def test_least_of_both_ends_finds_closer_high_numbered_threshold(self) -> None:
        """Regression: aircraft parked at 34R's threshold must resolve to the
        16L/34R row even though 16L's le_* threshold is ~2 nm away. This is
        the failing query the agent wrote with only le_* in the ORDER BY."""
        ctx = make_ctx(runway_csv_path=self.csv_path)
        # 34R threshold in the KSEA fixture row: 47.4312, -122.308
        nearest_result = dispatch_tool(
            make_call(
                "sql_query",
                query=(
                    "SELECT airport_ident, le_ident, he_ident, "
                    "  LEAST("
                    "    ST_Distance_Sphere(ST_Point(le_longitude_deg, le_latitude_deg), ST_Point(-122.308039, 47.431388)), "
                    "    ST_Distance_Sphere(ST_Point(he_longitude_deg, he_latitude_deg), ST_Point(-122.308039, 47.431388))"
                    "  ) AS dist_m "
                    "FROM runways "
                    "WHERE closed = 0 AND le_latitude_deg IS NOT NULL AND he_latitude_deg IS NOT NULL "
                    "ORDER BY dist_m LIMIT 1"
                ),
            ),
            ctx,
        )
        lines = nearest_result.splitlines()
        self.assertEqual(lines[0], "airport_ident\tle_ident\the_ident\tdist_m")
        airport, le_ident, he_ident, dist_m = lines[1].split("\t")
        self.assertEqual(airport, "KSEA")
        self.assertEqual(le_ident, "16L")
        self.assertEqual(he_ident, "34R")
        # ~25 m between (47.4312, -122.308) and (47.431388, -122.308039)
        self.assertLess(float(dist_m), 100.0)

    def test_active_ident_cosine_case_picks_correct_end_with_angular_wrap(self) -> None:
        """Regression: aircraft at the 34R threshold pointing north (heading
        0.6°) should resolve to active_ident = '34R', not '16L'. The raw
        |hdg - end_hdg| comparison the LLM did produced |0.6 - 180|=179.4 vs
        |0.6 - 360|=359.4 and picked 180° (16L) — which is backwards. The
        cos(radians(...)) trick in the canonical example handles wraparound
        correctly."""
        ctx = make_ctx(runway_csv_path=self.csv_path)
        result = dispatch_tool(
            make_call(
                "sql_query",
                query=(
                    "SELECT airport_ident, le_ident, he_ident, "
                    "  LEAST("
                    "    ST_Distance_Sphere(ST_Point(le_longitude_deg, le_latitude_deg), ST_Point(-122.308039, 47.431388)), "
                    "    ST_Distance_Sphere(ST_Point(he_longitude_deg, he_latitude_deg), ST_Point(-122.308039, 47.431388))"
                    "  ) AS dist_m, "
                    "  CASE "
                    "    WHEN cos(radians(le_heading_degT - 0.6)) > cos(radians(he_heading_degT - 0.6)) "
                    "    THEN le_ident "
                    "    ELSE he_ident "
                    "  END AS active_ident "
                    "FROM runways "
                    "WHERE closed = 0 AND le_latitude_deg IS NOT NULL AND he_latitude_deg IS NOT NULL "
                    "ORDER BY dist_m LIMIT 1"
                ),
            ),
            ctx,
        )
        lines = result.splitlines()
        self.assertEqual(lines[0], "airport_ident\tle_ident\the_ident\tdist_m\tactive_ident")
        airport, le_ident, he_ident, dist_m, active_ident = lines[1].split("\t")
        self.assertEqual(airport, "KSEA")
        self.assertEqual(active_ident, "34R")

    def test_active_ident_picks_le_when_heading_matches_le_direction(self) -> None:
        """Symmetric regression: at the 16L threshold pointing south (180°),
        active_ident should be '16L' — not '34R'."""
        ctx = make_ctx(runway_csv_path=self.csv_path)
        result = dispatch_tool(
            make_call(
                "sql_query",
                query=(
                    "SELECT le_ident, he_ident, "
                    "  CASE "
                    "    WHEN cos(radians(le_heading_degT - 180.0)) > cos(radians(he_heading_degT - 180.0)) "
                    "    THEN le_ident "
                    "    ELSE he_ident "
                    "  END AS active_ident "
                    "FROM runways "
                    "WHERE airport_ident = 'KSEA' AND le_ident = '16L' LIMIT 1"
                ),
            ),
            ctx,
        )
        active_ident = result.splitlines()[1].split("\t")[2]
        self.assertEqual(active_ident, "16L")

    def test_connection_is_cached_across_queries(self) -> None:
        ctx = make_ctx(runway_csv_path=self.csv_path)
        self.assertIsNone(ctx._runway_conn)
        dispatch_tool(make_call("sql_query", query="SELECT 1"), ctx)
        first_conn = ctx._runway_conn
        self.assertIsNotNone(first_conn)
        dispatch_tool(make_call("sql_query", query="SELECT 2"), ctx)
        self.assertIs(ctx._runway_conn, first_conn)


if __name__ == "__main__":
    unittest.main()
