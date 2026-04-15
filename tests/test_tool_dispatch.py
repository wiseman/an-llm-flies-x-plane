from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch

from sim_pilot.core.config import load_default_config_bundle
from sim_pilot.core.mission_manager import PilotCore
from sim_pilot.core.profiles import HeadingHoldProfile, PatternFlyProfile
from sim_pilot.llm.tools import ToolContext, dispatch_tool


class FakeBridge:
    def __init__(self) -> None:
        self.writes: list[dict[str, float | int]] = []

    def write_dataref_values(self, updates: dict[str, float | int]) -> None:
        self.writes.append(updates)


def make_ctx(bridge: FakeBridge | None = None) -> ToolContext:
    config = load_default_config_bundle()
    pilot = PilotCore(config)
    return ToolContext(pilot=pilot, bridge=bridge, config=config, recent_broadcasts=[])


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

    def test_engage_pattern_fly_displaces_idle_profiles(self) -> None:
        ctx = make_ctx()
        result = dispatch_tool(make_call("engage_pattern_fly"), ctx)
        self.assertIn("pattern_fly", result)
        self.assertEqual(ctx.pilot.list_profile_names(), ["pattern_fly"])

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


if __name__ == "__main__":
    unittest.main()
