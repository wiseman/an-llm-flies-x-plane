from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Callable

import duckdb

from sim_pilot.bus import SimBus
from sim_pilot.core.config import ConfigBundle
from sim_pilot.core.mission_manager import PilotCore, StatusSnapshot
from sim_pilot.core.profiles import (
    AltitudeHoldProfile,
    ApproachRunwayProfile,
    HeadingHoldProfile,
    PatternFlyProfile,
    RouteFollowProfile,
    SpeedHoldProfile,
    TakeoffProfile,
)
from sim_pilot.core.types import FlightPhase, Runway, TrafficSide
from sim_pilot.guidance.runway_geometry import RunwayFrame
from sim_pilot.sim.datarefs import (
    COM1_FREQUENCY_HZ_833,
    COM2_FREQUENCY_HZ_833,
    FLAP_HANDLE_REQUEST_RATIO,
    LATITUDE_DEG,
    LONGITUDE_DEG,
    PARKING_BRAKE_RATIO,
)
from sim_pilot.sim.xplane_bridge import XPlaneWebBridge, _geodetic_offset_ft


@dataclass(slots=True)
class ToolContext:
    pilot: PilotCore
    bridge: XPlaneWebBridge | None
    config: ConfigBundle
    recent_broadcasts: list[str]
    runway_csv_path: Path | None = None
    bus: SimBus | None = None
    _runway_conn: duckdb.DuckDBPyConnection | None = field(default=None, repr=False)


SQL_QUERY_MAX_ROWS = 50


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


def build_status_payload(
    snapshot: StatusSnapshot | None,
    bridge: XPlaneWebBridge | None,
) -> dict[str, Any]:
    """Build the status dict that ``get_status`` returns and that heartbeats
    embed. Pure function; no ToolContext needed. When ``bridge`` is set,
    the payload includes lat/lon read from the bridge's cached datarefs."""
    if snapshot is None:
        return {"status": "uninitialized"}
    state = snapshot.state
    lat_deg: float | None = None
    lon_deg: float | None = None
    if bridge is not None:
        lat_deg = bridge.get_dataref_value(LATITUDE_DEG.name)
        lon_deg = bridge.get_dataref_value(LONGITUDE_DEG.name)
    return {
        "t_sim": round(state.t_sim, 2),
        "active_profiles": list(snapshot.active_profiles),
        "phase": snapshot.phase.value if snapshot.phase is not None else None,
        "lat_deg": round(lat_deg, 6) if lat_deg is not None else None,
        "lon_deg": round(lon_deg, 6) if lon_deg is not None else None,
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
    }


def _status_json(snapshot: StatusSnapshot | None, ctx: "ToolContext") -> str:
    return json.dumps(build_status_payload(snapshot, ctx.bridge))


# ---- tools ----

def tool_get_status(ctx: ToolContext) -> str:
    return _status_json(ctx.pilot.latest_snapshot, ctx)


def tool_sleep(ctx: ToolContext) -> str:
    return "sleeping; waiting for next external message"


def tool_engage_heading_hold(
    ctx: ToolContext,
    heading_deg: float,
    turn_direction: str | None = None,
) -> str:
    try:
        profile = HeadingHoldProfile(
            heading_deg=heading_deg,
            max_bank_deg=ctx.config.limits.max_bank_enroute_deg,
            turn_direction=turn_direction,
        )
    except ValueError as exc:
        return f"error: {exc}"
    displaced = ctx.pilot.engage_profile(profile)
    direction_note = f" via {turn_direction}" if turn_direction else ""
    return f"engaged heading_hold heading={profile.heading_deg:.1f}deg{direction_note}{_format_displaced(displaced)}"


def tool_engage_altitude_hold(ctx: ToolContext, altitude_ft: float) -> str:
    profile = AltitudeHoldProfile(altitude_ft=altitude_ft)
    displaced = ctx.pilot.engage_profile(profile)
    return f"engaged altitude_hold altitude={profile.altitude_ft:.0f}ft{_format_displaced(displaced)}"


def tool_engage_speed_hold(ctx: ToolContext, speed_kt: float) -> str:
    profile = SpeedHoldProfile(speed_kt=speed_kt)
    displaced = ctx.pilot.engage_profile(profile)
    return f"engaged speed_hold speed={profile.speed_kt:.0f}kt{_format_displaced(displaced)}"


def tool_engage_cruise(
    ctx: ToolContext,
    heading_deg: float,
    altitude_ft: float,
    speed_kt: float,
) -> str:
    """Atomically install heading_hold + altitude_hold + speed_hold.

    Convenience for the cross-country cruise leg: rather than calling
    the three single-axis tools in sequence (which briefly leaves the
    vertical and speed axes uncovered when you're transitioning out of
    a three-axis profile like pattern_fly or takeoff), this engages all
    three holds under one lock acquisition so the control loop never
    sees an intermediate state with orphaned axes.
    """
    profiles = [
        HeadingHoldProfile(
            heading_deg=heading_deg,
            max_bank_deg=ctx.config.limits.max_bank_enroute_deg,
        ),
        AltitudeHoldProfile(altitude_ft=altitude_ft),
        SpeedHoldProfile(speed_kt=speed_kt),
    ]
    displaced = ctx.pilot.engage_profiles(profiles)
    return (
        f"engaged cruise heading={heading_deg:.0f}deg alt={altitude_ft:.0f}ft "
        f"speed={speed_kt:.0f}kt{_format_displaced(displaced)}"
    )


def _pilot_reference_label(ctx: ToolContext) -> str:
    parts = []
    if ctx.config.airport.airport:
        parts.append(f"airport={ctx.config.airport.airport}")
    if ctx.config.airport.runway.id:
        parts.append(f"runway={ctx.config.airport.runway.id}")
    parts.append(f"course={ctx.config.airport.runway.course_deg:.0f}deg")
    return " ".join(parts)


def _ensure_runway_conn(ctx: ToolContext) -> duckdb.DuckDBPyConnection:
    if ctx.runway_csv_path is None:
        raise RuntimeError("runway CSV path is not configured (pass --runway-csv-path)")
    if not ctx.runway_csv_path.exists():
        raise RuntimeError(f"runway CSV not found at {ctx.runway_csv_path}")
    if ctx._runway_conn is None:
        ctx._runway_conn = _open_runway_duckdb(ctx.runway_csv_path)
    return ctx._runway_conn


def _lookup_runway_for_pattern(
    ctx: ToolContext,
    airport_ident: str,
    runway_ident: str,
    side: str,
) -> tuple[Runway, float]:
    """Query the runway DB for a specific runway end and return a Runway anchored
    in the bridge's world frame plus the runway's field elevation."""
    if ctx.bridge is None:
        raise RuntimeError("no X-Plane bridge available; cannot compute world-frame threshold")
    conn = _ensure_runway_conn(ctx)
    query = (
        "SELECT le_ident, he_ident, "
        "le_latitude_deg, le_longitude_deg, le_heading_degT, le_elevation_ft, "
        "he_latitude_deg, he_longitude_deg, he_heading_degT, he_elevation_ft, "
        "length_ft "
        "FROM runways "
        "WHERE airport_ident = ? AND (le_ident = ? OR he_ident = ?) AND closed = 0 "
        "LIMIT 1"
    )
    row = conn.execute(query, [airport_ident, runway_ident, runway_ident]).fetchone()
    if row is None:
        raise RuntimeError(f"runway {runway_ident!r} at {airport_ident!r} not found in database")
    (
        le_ident,
        he_ident,
        le_lat,
        le_lon,
        le_hdg,
        le_elev,
        he_lat,
        he_lon,
        he_hdg,
        he_elev,
        length_ft,
    ) = row
    if runway_ident == le_ident:
        threshold_lat, threshold_lon = le_lat, le_lon
        course_hdg = le_hdg
        runway_elev = le_elev
    elif runway_ident == he_ident:
        threshold_lat, threshold_lon = he_lat, he_lon
        course_hdg = he_hdg
        runway_elev = he_elev
    else:
        raise RuntimeError(
            f"runway lookup returned {le_ident}/{he_ident} for query {runway_ident}; internal mismatch"
        )
    if threshold_lat is None or threshold_lon is None or course_hdg is None:
        raise RuntimeError(
            f"runway {airport_ident}/{runway_ident} has no threshold coordinates or course in the database"
        )
    try:
        traffic_side = TrafficSide(side.lower())
    except ValueError as exc:
        raise RuntimeError(f"invalid side {side!r}; expected 'left' or 'right'") from exc
    threshold_ft = _geodetic_offset_ft(
        lat_deg=float(threshold_lat),
        lon_deg=float(threshold_lon),
        georef=ctx.bridge.georef,
    )
    field_elevation_ft = float(runway_elev) if runway_elev is not None else 0.0
    resolved_length_ft = float(length_ft) if length_ft is not None else 5000.0
    runway = Runway(
        id=runway_ident,
        threshold_ft=threshold_ft,
        course_deg=float(course_hdg),
        length_ft=resolved_length_ft,
        # The runways CSV doesn't carry a touchdown-zone length. We synthesize
        # one from the runway length so ``RunwayFrame.touchdown_runway_x_ft``
        # (which is half of this value, clamped to [500, length/3]) lands
        # the aim point ~1000 ft past the threshold on normal runways and
        # scales down for short fields. The old hardcoded 1000.0 put the
        # aim point at the 500-ft floor regardless of runway length.
        touchdown_zone_ft=_synthesize_touchdown_zone_ft(resolved_length_ft),
        traffic_side=traffic_side,
    )
    return runway, field_elevation_ft


def _synthesize_touchdown_zone_ft(length_ft: float) -> float:
    return min(2000.0, max(500.0, length_ft * 0.5))


def _install_runway_in_pilot_core(
    ctx: ToolContext,
    airport_ident: str,
    runway: Runway,
    field_elevation_ft: float,
) -> None:
    """Anchor the pilot core at a new runway.

    Updates ``pilot.runway_frame`` and ``pilot.config`` so the state
    estimator computes runway-relative coordinates against the new runway
    and AGL is relative to the new field elevation. The tool context's
    ``config`` is updated to stay in sync.
    """
    new_airport_config = replace(
        ctx.config.airport,
        airport=airport_ident,
        field_elevation_ft=field_elevation_ft,
        runway=runway,
    )
    new_config = replace(ctx.config, airport=new_airport_config)
    ctx.config = new_config
    ctx.pilot.config = new_config
    ctx.pilot.runway_frame = RunwayFrame(runway)


def tool_engage_pattern_fly(
    ctx: ToolContext,
    airport_ident: str,
    runway_ident: str,
    side: str,
    start_phase: str,
) -> str:
    """Engage the full mission pilot profile anchored at a specific runway.

    Looks up the runway in the DuckDB database, anchors the pilot core at its
    real threshold, builds the pattern geometry from the DB-reported course and
    length, and positions the phase machine at ``start_phase``. The LLM is
    required to provide all four arguments so it is always explicit about
    which runway the pattern is for — use 'left' for standard US traffic
    pattern and 'takeoff_roll' as start_phase when starting on the runway
    from the ground, or 'pattern_entry' when joining mid-flight.
    """
    try:
        runway, field_elev = _lookup_runway_for_pattern(
            ctx, airport_ident, runway_ident, side
        )
    except Exception as exc:
        return f"error: {exc}"
    _install_runway_in_pilot_core(ctx, airport_ident, runway, field_elev)

    profile = PatternFlyProfile(ctx.config, ctx.pilot.runway_frame)
    try:
        profile.phase = FlightPhase(start_phase.lower())
    except ValueError:
        return (
            f"error: unknown start_phase {start_phase!r}; valid values are "
            f"{[p.value for p in FlightPhase]}"
        )
    displaced = ctx.pilot.engage_profile(profile)
    return f"engaged pattern_fly {_pilot_reference_label(ctx)}{_format_displaced(displaced)}"


def _parking_brake_ratio(ctx: ToolContext) -> float | None:
    if ctx.bridge is None:
        return None
    value = ctx.bridge.get_dataref_value(PARKING_BRAKE_RATIO.name)
    return None if value is None else float(value)


def tool_takeoff_checklist(ctx: ToolContext) -> str:
    snapshot = ctx.pilot.latest_snapshot
    if snapshot is None:
        return (
            "error: no aircraft state yet — call get_status first so the control "
            "loop publishes a snapshot."
        )
    state = snapshot.state
    lines: list[str] = ["TAKEOFF CHECKLIST — address every [ACTION] item before engage_takeoff:"]
    action_needed = False

    pb_value = _parking_brake_ratio(ctx)
    if pb_value is None:
        lines.append(
            "  [?]      parking brake: state unavailable — call "
            "set_parking_brake(engaged=False) to be safe"
        )
        action_needed = True
    elif pb_value >= 0.5:
        lines.append(
            f"  [ACTION] parking brake: SET (ratio={pb_value:.2f}) — call "
            f"set_parking_brake(engaged=False)"
        )
        action_needed = True
    else:
        lines.append(f"  [OK]     parking brake: released (ratio={pb_value:.2f})")

    if state.flap_index <= 10:
        lines.append(f"  [OK]     flaps: {state.flap_index} deg")
    else:
        lines.append(
            f"  [ACTION] flaps: {state.flap_index} deg — call set_flaps(degrees=10) or "
            f"set_flaps(degrees=0) for normal takeoff"
        )
        action_needed = True

    if state.gear_down:
        lines.append("  [OK]     gear: down")
    else:
        lines.append("  [ACTION] gear: up — extend the gear before takeoff")
        action_needed = True

    if state.on_ground:
        lines.append("  [OK]     on ground")
    else:
        lines.append("  [ERROR]  not on ground — you cannot engage_takeoff from the air")
        action_needed = True

    if state.ias_kt > 5.0:
        lines.append(
            f"  [?]      already rolling at {state.ias_kt:.1f} kt IAS — double-check you "
            f"meant to call this"
        )

    already_running = [p for p in snapshot.active_profiles if p in {"takeoff", "pattern_fly"}]
    if already_running:
        lines.append(f"  [INFO]   already engaged: {', '.join(already_running)}")

    lines.append(
        "  [REMINDER] identify the runway you are on (get_status for lat/lon + heading, then "
        "sql_query with the 'What runway am I on?' example) before committing to the roll"
    )
    lines.append(
        "  [REMINDER] acknowledge takeoff clearance on the radio (broadcast_on_radio) if ATC "
        "has issued one"
    )

    lines.append("")
    if action_needed:
        lines.append(
            "One or more items need action. Fix them, then re-run this checklist or proceed "
            "to engage_takeoff."
        )
    else:
        lines.append("All items OK. Call engage_takeoff() to begin the roll.")
    return "\n".join(lines)


def tool_engage_takeoff(ctx: ToolContext) -> str:
    pb_value = _parking_brake_ratio(ctx)
    if pb_value is not None and pb_value >= 0.5:
        return (
            f"error: parking brake is SET (ratio={pb_value:.2f}) — call "
            f"set_parking_brake(engaged=False) first, then retry engage_takeoff"
        )
    profile = TakeoffProfile(ctx.config, ctx.pilot.runway_frame)
    displaced = ctx.pilot.engage_profile(profile)
    return f"engaged takeoff {_pilot_reference_label(ctx)}{_format_displaced(displaced)}"


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


def tool_execute_touch_and_go(ctx: ToolContext) -> str:
    """Declare that the upcoming landing is a touch-and-go. Must be
    called during BASE or FINAL (or at the latest on the ROUNDOUT
    heartbeat) before the wheels touch. Once the aircraft touches
    down, the phase machine skips ROLLOUT (which applies brakes) and
    transitions directly to TAKEOFF_ROLL, so the aircraft accelerates
    straight back into another pattern instead of braking to a stop.
    """
    profile = _find_pattern_profile(ctx)
    if profile is None:
        return "error: pattern_fly profile is not active"
    profile.execute_touch_and_go()
    return "touch-and-go armed; landing will transition to takeoff_roll instead of rollout"


def tool_cleared_to_land(ctx: ToolContext, runway_id: str) -> str:
    profile = _find_pattern_profile(ctx)
    if profile is None:
        return "error: pattern_fly profile is not active"
    profile.cleared_to_land(runway_id)
    return f"cleared to land runway={runway_id}"


def tool_join_pattern(ctx: ToolContext, runway_id: str) -> str:
    """Record a 'join pattern' clearance. This tool is informational only —
    to actually reconfigure the pilot core for a specific runway, call
    engage_pattern_fly(airport_ident=..., runway_ident=..., side=...) which
    looks up the runway in the database and anchors the pattern geometry
    there. If pattern_fly is not engaged, this returns an error so the LLM
    knows to engage_pattern_fly first."""
    profile = _find_pattern_profile(ctx)
    if profile is None:
        return (
            "error: pattern_fly profile is not active; engage_pattern_fly(airport_ident, runway_ident) first"
        )
    return f"pattern entry acknowledged runway={runway_id}"


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


def tool_set_parking_brake(ctx: ToolContext, engaged: bool) -> str:
    if ctx.bridge is None:
        return "error: no X-Plane bridge available (running in simple backend?)"
    value = 1.0 if engaged else 0.0
    ctx.bridge.write_dataref_values({PARKING_BRAKE_RATIO.name: value})
    return "parking brake engaged" if engaged else "parking brake released"


_VALID_FLAP_SETTINGS = (0, 10, 20, 30)


def tool_set_flaps(ctx: ToolContext, degrees: int) -> str:
    if ctx.bridge is None:
        return "error: no X-Plane bridge available (running in simple backend?)"
    degrees = int(degrees)
    if degrees not in _VALID_FLAP_SETTINGS:
        return f"error: invalid flap setting {degrees} — valid settings are {', '.join(str(s) for s in _VALID_FLAP_SETTINGS)}"
    ratio = degrees / 30.0
    ctx.bridge.write_dataref_values({FLAP_HANDLE_REQUEST_RATIO.name: ratio})
    warning = ""
    pattern = _find_pattern_profile(ctx)
    if pattern is not None:
        warning = " (note: pattern_fly is active and manages flaps per phase — it may override this setting on the next tick)"
    return f"flaps set to {degrees}\u00b0{warning}"


def tool_broadcast_on_radio(ctx: ToolContext, radio: str, message: str) -> str:
    key = radio.lower()
    if key not in {"com1", "com2"}:
        return f"error: unknown radio {radio!r} (expected com1, com2)"
    line = f"[BROADCAST {key}] {message}"
    if ctx.bus is not None:
        ctx.bus.push_radio(line)
    else:
        print(line, flush=True)
    ctx.recent_broadcasts.append(line)
    if len(ctx.recent_broadcasts) > 16:
        del ctx.recent_broadcasts[0 : len(ctx.recent_broadcasts) - 16]
    return f"broadcast on {key}: {message}"


def _open_runway_duckdb(csv_path: Path) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection backed by the runways CSV.

    Loads the spatial extension (best-effort — non-spatial queries still work
    if the install fails), creates a private base table from the CSV, and
    exposes a read-only ``runways`` view. Writes against the view are rejected
    by DuckDB at bind time, which is how we keep the LLM from mutating the
    dataset.
    """
    conn = duckdb.connect(":memory:")
    try:
        conn.execute("INSTALL spatial; LOAD spatial;")
    except Exception:
        # Spatial unavailable (no network on first install, etc.). Non-spatial
        # queries still work. Spatial queries will fail with a clear error.
        pass
    conn.execute(
        "CREATE TABLE _runways_data AS SELECT * FROM read_csv_auto(?, header=true, sample_size=-1)",
        [str(csv_path)],
    )
    conn.execute("CREATE VIEW runways AS SELECT * FROM _runways_data")
    return conn


def tool_sql_query(ctx: ToolContext, query: str) -> str:
    if ctx.runway_csv_path is None:
        return "error: runway CSV path is not configured (pass --runway-csv-path)"
    csv_path = ctx.runway_csv_path
    if not csv_path.exists():
        return f"error: runway CSV not found at {csv_path}"
    if ctx._runway_conn is None:
        try:
            ctx._runway_conn = _open_runway_duckdb(csv_path)
        except Exception as exc:
            return f"error: could not open runway CSV: {exc}"
    try:
        cursor = ctx._runway_conn.execute(query)
        column_names = [desc[0] for desc in (cursor.description or [])]
        rows = cursor.fetchmany(SQL_QUERY_MAX_ROWS)
        has_more = cursor.fetchone() is not None
    except duckdb.Error as exc:
        return f"error: {exc}"
    if not rows:
        return "0 rows"
    lines = ["\t".join(column_names)]
    for row in rows:
        lines.append("\t".join("" if value is None else str(value) for value in row))
    if has_more:
        lines.append(f"(truncated at {SQL_QUERY_MAX_ROWS} rows; add LIMIT/WHERE to narrow)")
    return "\n".join(lines)


TOOL_HANDLERS: dict[str, ToolHandler] = {
    "get_status": tool_get_status,
    "sleep": tool_sleep,
    "engage_heading_hold": tool_engage_heading_hold,
    "engage_altitude_hold": tool_engage_altitude_hold,
    "engage_speed_hold": tool_engage_speed_hold,
    "engage_cruise": tool_engage_cruise,
    "engage_pattern_fly": tool_engage_pattern_fly,
    "engage_takeoff": tool_engage_takeoff,
    "takeoff_checklist": tool_takeoff_checklist,
    "engage_approach": tool_engage_approach,
    "engage_route_follow": tool_engage_route_follow,
    "disengage_profile": tool_disengage_profile,
    "list_profiles": tool_list_profiles,
    "extend_downwind": tool_extend_downwind,
    "turn_base_now": tool_turn_base_now,
    "go_around": tool_go_around,
    "execute_touch_and_go": tool_execute_touch_and_go,
    "cleared_to_land": tool_cleared_to_land,
    "join_pattern": tool_join_pattern,
    "tune_radio": tool_tune_radio,
    "broadcast_on_radio": tool_broadcast_on_radio,
    "set_parking_brake": tool_set_parking_brake,
    "set_flaps": tool_set_flaps,
    "sql_query": tool_sql_query,
}


SQL_QUERY_DESCRIPTION = """\
Run an arbitrary read-only SQL query against the runway/airport database. This is
the AUTHORITATIVE source for runway and airport facts — never guess a runway
identifier, airport code, course, length, or elevation; query for it. The backend
is DuckDB with the spatial extension loaded, so you have full ST_* geospatial
functions available. Results are tab-separated with a header row, truncated to 50
rows.

View: runways (read-only; one row per physical runway, worldwide)
  id BIGINT
  airport_ref BIGINT
  airport_ident VARCHAR        -- airport ICAO code, e.g. 'KSEA', 'EGLL'
  length_ft BIGINT
  width_ft BIGINT
  surface VARCHAR              -- e.g. 'ASP', 'CONC', 'CON', 'GRVL', 'TURF', 'ASPH-G'
  lighted BIGINT               -- 0 or 1
  closed BIGINT                -- 0 or 1
  le_ident VARCHAR             -- low-numbered-end runway identifier, e.g. '16L'
  le_latitude_deg DOUBLE       -- threshold latitude at the low end
  le_longitude_deg DOUBLE
  le_elevation_ft BIGINT
  le_heading_degT DOUBLE       -- runway course (true heading) from low-end threshold
  le_displaced_threshold_ft BIGINT
  he_ident VARCHAR             -- high-numbered-end identifier, e.g. '34R'
  he_latitude_deg DOUBLE
  he_longitude_deg DOUBLE
  he_elevation_ft BIGINT
  he_heading_degT DOUBLE
  he_displaced_threshold_ft BIGINT

Spatial functions (DuckDB spatial extension): ST_Point(lon, lat) — note that
LONGITUDE comes first — plus ST_Distance_Sphere(p1, p2) which returns meters
along the great circle. Use these for "nearest runway", "within N nm", etc.
Other ST_* functions (ST_Distance, ST_DWithin, ST_AsGeoJSON) are also available.

Common queries:

  -- "What runway am I on?" / "Where am I?" — call get_status first to get
  -- your lat/lon AND heading, then run this with all three substituted in.
  -- IMPORTANT details baked into this query (do not skip any of them):
  --   * each runway row stores BOTH ends (le_* and he_*); the LEAST() of the
  --     two spatial distances is your distance to the nearest threshold on
  --     that runway, which matters when you're sitting at the high-numbered
  --     end (e.g. 34R, whose threshold is in the he_* columns).
  --   * active_ident is computed in SQL from cos(heading - end_heading).
  --     cos handles angular wraparound automatically: heading 0.6 is next
  --     to 360 (cos ~ +1), not across from it. Do NOT compute this column
  --     in your head — read it from the query result.
  --   * no bounding box. ORDER BY the spatial distance over the whole
  --     table. DuckDB scans 43k rows in milliseconds.
  SELECT airport_ident, le_ident, he_ident, length_ft, surface,
         le_heading_degT, he_heading_degT,
         LEAST(
           ST_Distance_Sphere(ST_Point(le_longitude_deg, le_latitude_deg),
                              ST_Point(<lon>, <lat>)),
           ST_Distance_Sphere(ST_Point(he_longitude_deg, he_latitude_deg),
                              ST_Point(<lon>, <lat>))
         ) AS dist_m,
         CASE
           WHEN cos(radians(le_heading_degT - <hdg>)) >
                cos(radians(he_heading_degT - <hdg>))
           THEN le_ident
           ELSE he_ident
         END AS active_ident
  FROM runways
  WHERE closed = 0
    AND le_latitude_deg IS NOT NULL
    AND he_latitude_deg IS NOT NULL
  ORDER BY dist_m
  LIMIT 5;

  -- The runway you are on is the top row's active_ident column, provided
  -- dist_m is small (< ~100 m if you're actually on a runway). Do not pick
  -- le_ident or he_ident yourself — always read active_ident.

  -- All active runways at a known airport
  SELECT airport_ident, le_ident, he_ident, length_ft, surface,
         le_heading_degT, he_heading_degT
  FROM runways WHERE airport_ident = 'KSEA' AND closed = 0;

  -- Nearest paved runway at least 5000 ft long (both ends checked)
  SELECT airport_ident, le_ident, he_ident, length_ft, surface,
         LEAST(
           ST_Distance_Sphere(ST_Point(le_longitude_deg, le_latitude_deg),
                              ST_Point(<lon>, <lat>)),
           ST_Distance_Sphere(ST_Point(he_longitude_deg, he_latitude_deg),
                              ST_Point(<lon>, <lat>))
         ) / 1852.0 AS dist_nm
  FROM runways
  WHERE closed = 0 AND length_ft >= 5000 AND surface IN ('ASP', 'CONC', 'CON')
    AND le_latitude_deg IS NOT NULL AND he_latitude_deg IS NOT NULL
  ORDER BY dist_nm
  LIMIT 5;
"""


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
        "Engage heading-hold on the lateral axis. Displaces any other lateral-axis profile. By default takes the shortest-path turn to the target heading. If the operator or ATC specifies a turn direction (e.g. 'turn right to 290'), pass turn_direction='right' or 'left' to force that direction even when the other way is shorter; the direction lock clears automatically once within 5 degrees of target so the autopilot will not overshoot.",
        {
            "heading_deg": {"type": "number", "description": "Target heading in degrees true, 0-360."},
            "turn_direction": {
                "type": ["string", "null"],
                "enum": ["left", "right", None],
                "description": "Optional forced turn direction: 'left', 'right', or null for shortest-path (default).",
            },
        },
        ["heading_deg", "turn_direction"],
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
        "engage_cruise",
        "Atomically install heading_hold + altitude_hold + speed_hold in a single "
        "tool call. Use this when transitioning out of a three-axis profile "
        "(takeoff, pattern_fly) into a steady cross-country leg — engaging the "
        "three single-axis holds separately briefly leaves the vertical and speed "
        "axes uncovered between calls, while this installs them under one lock so "
        "the control loop never sees an intermediate state. Displaces any "
        "lateral-, vertical-, or speed-axis profile currently engaged (including "
        "takeoff and pattern_fly, which own all three).",
        {
            "heading_deg": {"type": "number", "description": "Target true heading, 0-360."},
            "altitude_ft": {"type": "number", "description": "Target altitude MSL in feet."},
            "speed_kt": {"type": "number", "description": "Target indicated airspeed in knots."},
        },
        ["heading_deg", "altitude_ft", "speed_kt"],
    ),
    _fn_schema(
        "engage_pattern_fly",
        "Engage the deterministic mission pilot anchored at a specific runway. "
        "Owns all three axes. The tool looks up the runway in the database, "
        "anchors the pattern geometry at its real threshold, and positions the "
        "phase machine at start_phase. All four arguments are REQUIRED — if you "
        "don't know which runway you're on, call get_status + sql_query first. "
        "Use start_phase='takeoff_roll' to start on the ground for takeoff, or "
        "start_phase='pattern_entry' to join an existing pattern from cruise. "
        "Typical takeoff call: engage_pattern_fly(airport_ident='KSEA', "
        "runway_ident='16L', side='left', start_phase='takeoff_roll'). Typical "
        "join-from-cruise call: engage_pattern_fly(airport_ident='KSEA', "
        "runway_ident='16L', side='left', start_phase='pattern_entry').",
        {
            "airport_ident": {
                "type": "string",
                "description": "ICAO airport code (e.g. 'KSEA').",
            },
            "runway_ident": {
                "type": "string",
                "description": "Runway end identifier (e.g. '16L', '34R').",
            },
            "side": {
                "type": "string",
                "description": "Traffic pattern side: 'left' (standard US) or 'right'.",
            },
            "start_phase": {
                "type": "string",
                "description": "Initial phase for the phase machine. 'takeoff_roll' if starting on the ground, 'pattern_entry' if joining from cruise, 'downwind' if already on downwind, etc.",
            },
        },
        ["airport_ident", "runway_ident", "side", "start_phase"],
    ),
    _fn_schema(
        "engage_takeoff",
        "Start the takeoff sequence: full power, accelerate on the runway centerline, rotate at Vr, then hold a straight-ahead climb at Vy along the runway track. Owns all three axes. Does NOT auto-disengage — once you're safely airborne, transition by engaging another profile (engage_heading_hold, engage_altitude_hold, engage_pattern_fly, etc.), which will displace this one via axis-ownership conflict. REFUSES to engage if the parking brake is set; call takeoff_checklist first to see what needs to be fixed.",
        {},
        [],
    ),
    _fn_schema(
        "takeoff_checklist",
        "Return a takeoff-readiness checklist with each item marked [OK], [ACTION], [ERROR], or [REMINDER]. Reads live state (parking brake, flaps, gear, on-ground, active profiles). Call this before engage_takeoff and address every [ACTION] item — the most common miss is a set parking brake, which will also cause engage_takeoff to refuse.",
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
        "execute_touch_and_go",
        "Declare that the upcoming landing is a touch-and-go. Must be called "
        "during BASE or FINAL (before the wheels touch). On touchdown the phase "
        "machine will skip ROLLOUT (braking) and transition directly to "
        "TAKEOFF_ROLL: full throttle, flaps retract to 10°, no brakes. The "
        "aircraft re-accelerates, rotates, and flies another pattern. The flag "
        "auto-clears on TAKEOFF_ROLL → ROTATE so the next approach defaults to "
        "a normal full-stop landing unless you call execute_touch_and_go again. "
        "Requires pattern_fly to be active.",
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
        "Transmit a text message over a COM radio. This is the ONLY way your words reach ATC or anyone outside the cockpit — plain-text replies are visible to the operator only and are not transmitted. Always use this tool to acknowledge clearances, read back ATC instructions, make position calls, or make any external radio call. Use standard aviation phraseology. Typically com1 is the active comm radio.",
        {
            "radio": {"type": "string", "description": "Radio name: 'com1' or 'com2'."},
            "message": {"type": "string", "description": "The exact words to transmit, e.g. 'Seattle Tower, Cessna 123AB, runway 16L cleared for takeoff'."},
        },
        ["radio", "message"],
    ),
    _fn_schema(
        "set_parking_brake",
        "Engage or release the parking brake. Unlike the toe brakes, the parking brake holds its state without continuous input, so this is the right tool for 'set the brake and hold it'.",
        {"engaged": {"type": "boolean", "description": "True to engage (set) the parking brake, false to release it."}},
        ["engaged"],
    ),
    _fn_schema(
        "set_flaps",
        "Set the flap handle position. Valid settings for the C172 are 0, 10, 20, or 30 degrees. Note: when pattern_fly is active, it manages flaps automatically per flight phase and will override this setting on the next tick. Use this tool when flying with single-axis profiles (heading_hold, altitude_hold, speed_hold) or during ground ops.",
        {"degrees": {"type": "integer", "description": "Flap setting in degrees: 0, 10, 20, or 30."}},
        ["degrees"],
    ),
    _fn_schema(
        "sql_query",
        SQL_QUERY_DESCRIPTION,
        {"query": {"type": "string", "description": "A single SQL statement to execute."}},
        ["query"],
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
