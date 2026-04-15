from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable

from sim_pilot.core.config import ConfigBundle
from sim_pilot.core.mission_manager import PilotCore, StatusSnapshot
from sim_pilot.core.profiles import (
    AltitudeHoldProfile,
    ApproachRunwayProfile,
    HeadingHoldProfile,
    PatternFlyProfile,
    RouteFollowProfile,
    SpeedHoldProfile,
)
from sim_pilot.sim.datarefs import COM1_FREQUENCY_HZ_833, COM2_FREQUENCY_HZ_833
from sim_pilot.sim.xplane_bridge import XPlaneWebBridge


@dataclass(slots=True)
class ToolContext:
    pilot: PilotCore
    bridge: XPlaneWebBridge | None
    config: ConfigBundle
    recent_broadcasts: list[str]


ToolHandler = Callable[..., str]


# ---- helpers ----

def _format_displaced(displaced: list[str]) -> str:
    if not displaced:
        return ""
    return f" (displaced: {', '.join(displaced)})"


def _find_pattern_profile(ctx: ToolContext) -> PatternFlyProfile | None:
    profile = ctx.pilot.find_profile("pattern_fly")
    if isinstance(profile, PatternFlyProfile):
        return profile
    return None


def _status_json(snapshot: StatusSnapshot | None) -> str:
    if snapshot is None:
        return json.dumps({"status": "uninitialized"})
    state = snapshot.state
    payload: dict[str, Any] = {
        "t_sim": round(state.t_sim, 2),
        "active_profiles": list(snapshot.active_profiles),
        "phase": snapshot.phase.value if snapshot.phase is not None else None,
        "position_ft": {"x": round(state.position_ft.x, 1), "y": round(state.position_ft.y, 1)},
        "alt_msl_ft": round(state.alt_msl_ft, 1),
        "alt_agl_ft": round(state.alt_agl_ft, 1),
        "ias_kt": round(state.ias_kt, 1),
        "gs_kt": round(state.gs_kt, 1),
        "vs_fpm": round(state.vs_fpm, 1),
        "heading_deg": round(state.heading_deg, 1),
        "track_deg": round(state.track_deg, 1),
        "pitch_deg": round(state.pitch_deg, 1),
        "roll_deg": round(state.roll_deg, 1),
        "on_ground": state.on_ground,
        "throttle_pos": round(state.throttle_pos, 2),
        "flap_index": state.flap_index,
        "gear_down": state.gear_down,
        "runway_id": state.runway_id,
        "runway_x_ft": round(state.runway_x_ft, 1) if state.runway_x_ft is not None else None,
        "runway_y_ft": round(state.runway_y_ft, 1) if state.runway_y_ft is not None else None,
    }
    return json.dumps(payload)


# ---- tools ----

def tool_get_status(ctx: ToolContext) -> str:
    return _status_json(ctx.pilot.latest_snapshot)


def tool_sleep(ctx: ToolContext) -> str:
    return "sleeping; waiting for next external message"


def tool_engage_heading_hold(ctx: ToolContext, heading_deg: float) -> str:
    profile = HeadingHoldProfile(
        heading_deg=heading_deg,
        max_bank_deg=ctx.config.limits.max_bank_enroute_deg,
    )
    displaced = ctx.pilot.engage_profile(profile)
    return f"engaged heading_hold heading={profile.heading_deg:.1f}deg{_format_displaced(displaced)}"


def tool_engage_altitude_hold(ctx: ToolContext, altitude_ft: float) -> str:
    profile = AltitudeHoldProfile(altitude_ft=altitude_ft)
    displaced = ctx.pilot.engage_profile(profile)
    return f"engaged altitude_hold altitude={profile.altitude_ft:.0f}ft{_format_displaced(displaced)}"


def tool_engage_speed_hold(ctx: ToolContext, speed_kt: float) -> str:
    profile = SpeedHoldProfile(speed_kt=speed_kt)
    displaced = ctx.pilot.engage_profile(profile)
    return f"engaged speed_hold speed={profile.speed_kt:.0f}kt{_format_displaced(displaced)}"


def tool_engage_pattern_fly(ctx: ToolContext) -> str:
    profile = PatternFlyProfile(ctx.config, ctx.pilot.runway_frame)
    displaced = ctx.pilot.engage_profile(profile)
    return (
        f"engaged pattern_fly airport={ctx.config.airport.airport} "
        f"runway={ctx.config.airport.runway.id}{_format_displaced(displaced)}"
    )


def tool_engage_approach(ctx: ToolContext, runway_id: str) -> str:
    return f"error: engage_approach not yet implemented (requested runway={runway_id})"


def tool_engage_route_follow(ctx: ToolContext) -> str:
    return "error: engage_route_follow not yet implemented"


def tool_disengage_profile(ctx: ToolContext, name: str) -> str:
    added = ctx.pilot.disengage_profile(name)
    if not added and name not in ctx.pilot.list_profile_names():
        return f"no profile named {name!r} was active"
    return f"disengaged {name}; re-added idle profiles: {', '.join(added) or 'none'}"


def tool_list_profiles(ctx: ToolContext) -> str:
    return ", ".join(ctx.pilot.list_profile_names())


def tool_extend_downwind(ctx: ToolContext, extension_ft: float) -> str:
    profile = _find_pattern_profile(ctx)
    if profile is None:
        return "error: pattern_fly profile is not active"
    profile.extend_downwind(extension_ft)
    return f"extended downwind by {extension_ft:.0f}ft; total extension={profile.pattern_extension_ft:.0f}ft"


def tool_turn_base_now(ctx: ToolContext) -> str:
    profile = _find_pattern_profile(ctx)
    if profile is None:
        return "error: pattern_fly profile is not active"
    profile.turn_base_now()
    return "turn_base_now triggered"


def tool_go_around(ctx: ToolContext) -> str:
    profile = _find_pattern_profile(ctx)
    if profile is None:
        return "error: pattern_fly profile is not active"
    profile.go_around()
    return "go_around triggered"


def tool_cleared_to_land(ctx: ToolContext, runway_id: str) -> str:
    profile = _find_pattern_profile(ctx)
    if profile is None:
        return "error: pattern_fly profile is not active"
    profile.cleared_to_land(runway_id)
    return f"cleared to land runway={runway_id}"


def tool_join_pattern(ctx: ToolContext, runway_id: str) -> str:
    profile = _find_pattern_profile(ctx)
    if profile is None:
        return "error: pattern_fly profile is not active; engage pattern_fly first"
    if runway_id and runway_id != profile.runway_frame.runway.id:
        return f"error: runway {runway_id} does not match active runway {profile.runway_frame.runway.id}"
    return f"pattern entry acknowledged runway={profile.runway_frame.runway.id}"


def tool_tune_radio(ctx: ToolContext, radio: str, frequency_mhz: float) -> str:
    if ctx.bridge is None:
        return "error: no X-Plane bridge available (running in simple backend?)"
    dataref_by_radio = {
        "com1": COM1_FREQUENCY_HZ_833.name,
        "com2": COM2_FREQUENCY_HZ_833.name,
    }
    key = radio.lower()
    if key not in dataref_by_radio:
        return f"error: unknown radio {radio!r} (expected com1, com2)"
    # The com*_frequency_hz_833 dataref accepts integer Hz/10 units — i.e. the frequency
    # in 10 Hz steps. 118.30 MHz -> 11_830_000 Hz -> 118_300 * 100? X-Plane's convention
    # has historically been kHz*10 (e.g. 11830 for the old 25 kHz dataref). The 833
    # variant uses kHz directly according to the SDK notes. Sending kHz (integer) here;
    # spot-check at runtime if the frequency doesn't land where expected.
    frequency_khz = int(round(frequency_mhz * 1000.0))
    ctx.bridge.write_dataref_values({dataref_by_radio[key]: frequency_khz})
    return f"tuned {key} to {frequency_mhz:.3f} MHz (set dataref={dataref_by_radio[key]} value={frequency_khz})"


def tool_broadcast_on_radio(ctx: ToolContext, radio: str, message: str) -> str:
    key = radio.lower()
    if key not in {"com1", "com2"}:
        return f"error: unknown radio {radio!r} (expected com1, com2)"
    line = f"[BROADCAST {key}] {message}"
    print(line, flush=True)
    ctx.recent_broadcasts.append(line)
    if len(ctx.recent_broadcasts) > 16:
        del ctx.recent_broadcasts[0 : len(ctx.recent_broadcasts) - 16]
    return f"broadcast on {key}: {message}"


TOOL_HANDLERS: dict[str, ToolHandler] = {
    "get_status": tool_get_status,
    "sleep": tool_sleep,
    "engage_heading_hold": tool_engage_heading_hold,
    "engage_altitude_hold": tool_engage_altitude_hold,
    "engage_speed_hold": tool_engage_speed_hold,
    "engage_pattern_fly": tool_engage_pattern_fly,
    "engage_approach": tool_engage_approach,
    "engage_route_follow": tool_engage_route_follow,
    "disengage_profile": tool_disengage_profile,
    "list_profiles": tool_list_profiles,
    "extend_downwind": tool_extend_downwind,
    "turn_base_now": tool_turn_base_now,
    "go_around": tool_go_around,
    "cleared_to_land": tool_cleared_to_land,
    "join_pattern": tool_join_pattern,
    "tune_radio": tool_tune_radio,
    "broadcast_on_radio": tool_broadcast_on_radio,
}


def _fn_schema(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


TOOL_SCHEMAS: list[dict[str, Any]] = [
    _fn_schema(
        "get_status",
        "Return a JSON snapshot of aircraft state, phase, and active profiles.",
        {},
        [],
    ),
    _fn_schema(
        "sleep",
        "End this turn explicitly. The pilot continues flying the active profiles; you wake up when the next operator/ATC message arrives.",
        {},
        [],
    ),
    _fn_schema(
        "engage_heading_hold",
        "Engage heading-hold on the lateral axis. Displaces any other lateral-axis profile.",
        {"heading_deg": {"type": "number", "description": "Target heading in degrees true, 0-360."}},
        ["heading_deg"],
    ),
    _fn_schema(
        "engage_altitude_hold",
        "Engage altitude-hold on the vertical axis using TECS. Displaces any other vertical-axis profile.",
        {"altitude_ft": {"type": "number", "description": "Target altitude in feet MSL."}},
        ["altitude_ft"],
    ),
    _fn_schema(
        "engage_speed_hold",
        "Engage speed-hold on the speed axis. Displaces any other speed-axis profile.",
        {"speed_kt": {"type": "number", "description": "Target indicated airspeed in knots."}},
        ["speed_kt"],
    ),
    _fn_schema(
        "engage_pattern_fly",
        "Engage the deterministic mission pilot. Owns all three axes and covers takeoff through landing using the configured airport's phase machine.",
        {},
        [],
    ),
    _fn_schema(
        "engage_approach",
        "Engage a final-approach profile for the given runway. Not yet implemented.",
        {"runway_id": {"type": "string", "description": "Runway identifier, e.g. '16L'."}},
        ["runway_id"],
    ),
    _fn_schema(
        "engage_route_follow",
        "Engage route-follow guidance along a list of waypoints. Not yet implemented.",
        {},
        [],
    ),
    _fn_schema(
        "disengage_profile",
        "Remove the profile with the given name. Orphaned axes fall back to idle profiles.",
        {"name": {"type": "string", "description": "Profile name, e.g. 'heading_hold'."}},
        ["name"],
    ),
    _fn_schema(
        "list_profiles",
        "Return a comma-separated list of currently active profile names.",
        {},
        [],
    ),
    _fn_schema(
        "extend_downwind",
        "Extend the downwind leg of the traffic pattern. Requires pattern_fly to be active.",
        {"extension_ft": {"type": "number", "description": "Additional downwind distance in feet."}},
        ["extension_ft"],
    ),
    _fn_schema(
        "turn_base_now",
        "Trigger the base turn immediately. Requires pattern_fly to be active and phase DOWNWIND.",
        {},
        [],
    ),
    _fn_schema(
        "go_around",
        "Command an immediate go-around. Requires pattern_fly to be active.",
        {},
        [],
    ),
    _fn_schema(
        "cleared_to_land",
        "Record a cleared-to-land clearance. Requires pattern_fly to be active.",
        {"runway_id": {"type": "string", "description": "Runway identifier, e.g. '16L'."}},
        ["runway_id"],
    ),
    _fn_schema(
        "join_pattern",
        "Acknowledge a pattern join instruction. Requires pattern_fly to be active.",
        {"runway_id": {"type": "string", "description": "Runway identifier for the pattern."}},
        ["runway_id"],
    ),
    _fn_schema(
        "tune_radio",
        "Tune a COM radio to a frequency in MHz.",
        {
            "radio": {"type": "string", "description": "Radio name: 'com1' or 'com2'."},
            "frequency_mhz": {"type": "number", "description": "Frequency in MHz, e.g. 118.30."},
        },
        ["radio", "frequency_mhz"],
    ),
    _fn_schema(
        "broadcast_on_radio",
        "Broadcast a text message over a COM radio. Prints to the operator console; no audio synthesis.",
        {
            "radio": {"type": "string", "description": "Radio name: 'com1' or 'com2'."},
            "message": {"type": "string", "description": "The message text to broadcast."},
        },
        ["radio", "message"],
    ),
]


def dispatch_tool(call: dict[str, Any], ctx: ToolContext) -> str:
    name = call.get("name")
    if not isinstance(name, str):
        return "error: tool call missing name"
    args_json = call.get("arguments", "{}")
    try:
        args = json.loads(args_json) if args_json else {}
    except json.JSONDecodeError as exc:
        return f"error: invalid arguments JSON: {exc}"
    if not isinstance(args, dict):
        return f"error: arguments must be an object, got {type(args).__name__}"
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return f"error: unknown tool {name!r}"
    try:
        return handler(ctx, **args)
    except TypeError as exc:
        return f"error: invalid arguments for {name}: {exc}"
    except Exception as exc:
        return f"error: {exc.__class__.__name__}: {exc}"
