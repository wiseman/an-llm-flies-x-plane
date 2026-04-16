"""Terminal UI and status-panel formatting for the live X-Plane pilot."""

from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any, TYPE_CHECKING

from typing import Callable

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import D
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame

from sim_pilot.bus import SimBus
from sim_pilot.core.types import wrap_degrees_180
from sim_pilot.llm.conversation import IncomingMessage

if TYPE_CHECKING:
    from sim_pilot.core.mission_manager import PilotCore, StatusSnapshot
    from sim_pilot.live_runner import HeartbeatPump
    from sim_pilot.llm.responses_client import CacheStats


STATUS_PANE_HEIGHT = 11
RADIO_PANE_HEIGHT = 8

_THR_BAR_WIDTH = 8
_THR_FILL = "\u2588"  # █
_THR_EMPTY = "\u2591"  # ░
_SEP = "\u2500" * 64  # ─

# ---- phase → style class mapping ----

_PHASE_STYLES: dict[str, str] = {
    "preflight": "class:ph-idle",
    "takeoff_roll": "class:ph-active",
    "rotate": "class:ph-active",
    "initial_climb": "class:ph-climb",
    "crosswind": "class:ph-climb",
    "enroute_climb": "class:ph-climb",
    "cruise": "class:ph-cruise",
    "descent": "class:ph-appr",
    "pattern_entry": "class:ph-appr",
    "downwind": "class:ph-pattern",
    "base": "class:ph-pattern",
    "final": "class:ph-crit",
    "roundout": "class:ph-crit",
    "flare": "class:ph-crit",
    "rollout": "class:ph-active",
    "taxi_clear": "class:ph-idle",
    "go_around": "class:ph-alert",
}

# ---- TUI color scheme ----

TUI_STYLE = Style.from_dict(
    {
        # Frame chrome
        "frame.border": "#555555",
        "frame.label": "#5fd7ff bold",
        # Phase indicator
        "ph-idle": "#666666 bold",
        "ph-active": "#ffaf00 bold",
        "ph-climb": "#00d787 bold",
        "ph-cruise": "#00d787 bold",
        "ph-appr": "#5fd7ff bold",
        "ph-pattern": "#00d787 bold",
        "ph-crit": "#ffaf00 bold",
        "ph-alert": "#ff5f5f bold",
        # Status content
        "marker": "#00d787",
        "profiles": "#666666",
        "sep": "#333333",
        "hdr": "#555555",
        "lbl": "#555555",
        "val": "#ffffff bold",
        "val-ok": "#00d787 bold",
        "val-warn": "#ffaf00 bold",
        "val-bad": "#ff5f5f bold",
        "tgt": "#5fd7ff",
        "dim": "#555555",
        "thr-fill": "#00d787",
        "thr-bg": "#444444",
        "gear-dn": "#00d787",
        "gear-up": "#ffaf00",
        "gnd": "#ffaf00",
        "air": "#5fd7ff",
        "rwy-info": "#999999",
        # Log pane
        "log-error": "#ff5f5f bold",
        "log-warn": "#ffaf00",
        "log-llm": "#5fd7ff",
        "log-dim": "#555555",
    }
)


class _LogLexer(Lexer):
    """Line-level syntax highlighter for the log pane.

    Colors entire lines based on their prefix/content so errors, LLM
    responses, heartbeats, and safety events are visually distinct.
    """

    def lex_document(self, document: Document) -> Callable[[int], list[tuple[str, str]]]:
        lines = document.lines

        def get_line(lineno: int) -> list[tuple[str, str]]:
            if lineno >= len(lines):
                return [("", "")]
            line = lines[lineno]
            lower = line.lower()
            if "error" in lower:
                return [("class:log-error", line)]
            if "[safety]" in lower:
                return [("class:log-warn", line)]
            if "[heartbeat]" in lower:
                return [("class:log-dim", line)]
            if "[llm]" in lower:
                return [("class:log-llm", line)]
            return [("", line)]

        return get_line


def _dev_style(deviation: float, near: float, far: float) -> str:
    """Return a style class based on how far a value is from its target.

    *near* and *far* are the thresholds for green/yellow/red coloring.
    """
    if deviation <= near:
        return "class:val-ok"
    if deviation <= far:
        return "class:val-warn"
    return "class:val-bad"


def format_snapshot_styled(
    snapshot: "StatusSnapshot | None",
    cache_stats: "CacheStats | None" = None,
) -> list[tuple[str, str]]:
    """Render the multi-line status panel as styled text fragments.

    Returns a list of ``(style_class, text)`` tuples suitable for
    prompt_toolkit's :class:`FormattedTextControl`.  The companion
    :func:`format_snapshot_display` strips styles for headless output.
    """
    if snapshot is None:
        return [("class:dim", "  (waiting for first pilot tick)")]

    state = snapshot.state
    guidance = snapshot.last_guidance
    commands = snapshot.last_commands

    profiles = ", ".join(snapshot.active_profiles) if snapshot.active_profiles else "(none)"
    phase = snapshot.phase.value if snapshot.phase is not None else "\u2014"
    phase_cls = _PHASE_STYLES.get(phase, "class:ph-idle")

    # ---- target values ----
    tgt_hdg: float | None = None
    tgt_alt_msl: float | None = None
    tgt_spd: float | None = None
    if guidance is not None:
        tgt_hdg = guidance.target_heading_deg
        if tgt_hdg is None:
            tgt_hdg = guidance.target_track_deg
        tgt_alt_msl = guidance.target_altitude_ft
        tgt_spd = guidance.target_speed_kt

    field_elev = state.alt_msl_ft - state.alt_agl_ft
    tgt_alt_agl: float | None = None
    if tgt_alt_msl is not None:
        tgt_alt_agl = tgt_alt_msl - field_elev

    f: list[tuple[str, str]] = []

    # ---- runway info (when pattern_fly is active) ----
    rwy_info = ""
    if snapshot.airport_ident or snapshot.runway_id:
        parts: list[str] = []
        if snapshot.airport_ident:
            parts.append(snapshot.airport_ident)
        if snapshot.runway_id:
            parts.append(f"rwy {snapshot.runway_id}")
        rwy_info = " ".join(parts)
        if snapshot.field_elevation_ft is not None:
            rwy_info += f" \u00b7 {snapshot.field_elevation_ft:.0f} ft"

    # ---- line 1: phase + runway info + profiles ----
    f.append(("class:marker", " \u25b8 "))
    if rwy_info:
        f.append((phase_cls, phase.upper()))
        f.append(("", "  "))
        f.append(("class:rwy-info", rwy_info))
        f.append(("", "  "))
    else:
        f.append((phase_cls, f"{phase.upper():<24}"))
    f.append(("class:profiles", profiles))
    f.append(("", "\n"))

    # ---- line 2: separator ----
    f.append(("class:sep", f" {_SEP}\n"))

    # ---- line 3: column headers (right-aligned over number fields) ----
    f.append(("class:hdr", f"{'current':>27}{'target':>15}\n"))

    # ---- lines 4-7: instruments ----
    # Layout: 17-char label, then number right-aligned in 10 chars,
    # unit left-aligned in 8 chars (18 total for current column),
    # target number right-aligned in 7 chars, unit left-aligned in 5 chars.

    # heading
    if tgt_hdg is not None:
        hdg_cls = _dev_style(abs(wrap_degrees_180(state.heading_deg - tgt_hdg)), 3.0, 15.0)
    else:
        hdg_cls = "class:val"
    f.append(("class:lbl", "  heading        "))
    f.append((hdg_cls, f"{state.heading_deg:10.0f}"))
    f.append((hdg_cls, f"{'\u00b0':<8}"))
    if tgt_hdg is not None:
        f.append(("class:tgt", f"{tgt_hdg:7.0f}"))
        f.append(("class:tgt", f"{'\u00b0':<5}"))
    else:
        f.append(("class:dim", f"{'\u2014':>7}"))
    f.append(("", "\n"))

    # altitude
    if tgt_alt_agl is not None:
        alt_cls = _dev_style(abs(state.alt_agl_ft - tgt_alt_agl), 50.0, 200.0)
    else:
        alt_cls = "class:val"
    f.append(("class:lbl", "  altitude       "))
    f.append((alt_cls, f"{state.alt_agl_ft:10.0f}"))
    f.append((alt_cls, f"{' AGL':<8}"))
    if tgt_alt_agl is not None:
        f.append(("class:tgt", f"{tgt_alt_agl:7.0f}"))
        f.append(("class:tgt", f"{' AGL':<5}"))
    else:
        f.append(("class:dim", f"{'\u2014':>7}"))
    f.append(("", "\n"))

    # airspeed
    if tgt_spd is not None:
        spd_cls = _dev_style(abs(state.ias_kt - tgt_spd), 3.0, 10.0)
    else:
        spd_cls = "class:val"
    f.append(("class:lbl", "  airspeed       "))
    f.append((spd_cls, f"{state.ias_kt:10.0f}"))
    f.append((spd_cls, f"{' kt IAS':<8}"))
    if tgt_spd is not None:
        f.append(("class:tgt", f"{tgt_spd:7.0f}"))
        f.append(("class:tgt", f"{' kt':<5}"))
    else:
        f.append(("class:dim", f"{'\u2014':>7}"))
    f.append(("", "\n"))

    # vertical speed
    f.append(("class:lbl", "  vertical       "))
    f.append(("class:val", f"{state.vs_fpm:+10.0f}"))
    f.append(("class:val", f"{' fpm':<8}"))
    f.append(("", "\n"))

    # ---- line 8: separator ----
    f.append(("class:sep", f" {_SEP}\n"))

    # ---- line 9: config bar ----
    thr = commands.throttle
    filled = round(thr * _THR_BAR_WIDTH)
    flap_str = f"{state.flap_index}\u00b0"
    gear_str = "dn" if state.gear_down else "up"

    f.append(("class:lbl", "  throttle "))
    f.append(("class:thr-fill", _THR_FILL * filled))
    f.append(("class:thr-bg", _THR_EMPTY * (_THR_BAR_WIDTH - filled)))
    f.append(("class:val", f" {thr:4.2f}"))
    f.append(("class:lbl", "   flaps "))
    f.append(("class:val", f"{flap_str:<5}"))
    f.append(("class:lbl", "   gear "))
    f.append(("class:gear-dn" if state.gear_down else "class:gear-up", gear_str))
    f.append(("", "   "))
    f.append(
        ("class:gnd" if state.on_ground else "class:air", "on ground" if state.on_ground else "airborne")
    )
    f.append(("", "   "))
    if state.runway_x_ft is not None and state.runway_y_ft is not None:
        f.append(("class:dim", f"rwy x{state.runway_x_ft:+.0f} y{state.runway_y_ft:+.0f}"))
    else:
        f.append(("class:dim", "rwy \u2014"))

    # Cache hit rate (when available)
    if cache_stats is not None:
        input_tok, cached_tok, n_req = cache_stats.snapshot()
        if n_req > 0:
            rate = cached_tok / input_tok if input_tok > 0 else 0.0
            rate_cls = "class:val-ok" if rate >= 0.5 else "class:val-warn" if rate >= 0.2 else "class:dim"
            f.append(("", "   "))
            f.append(("class:dim", "cache "))
            f.append((rate_cls, f"{rate:.0%}"))

    return f


def format_snapshot_display(snapshot: "StatusSnapshot | None") -> str:
    """Render the status panel as plain text for the headless log.

    Two-column "current vs target" layout for the primary flight references
    (heading, altitude, airspeed, vertical speed) with a single-line bottom
    row for the secondary state (throttle, flaps, gear, runway position).
    Used by both the headless control loop (pushes to SimBus every
    status_interval_s) and the test suite.
    """
    return "".join(text for _, text in format_snapshot_styled(snapshot)).rstrip("\n")


def parse_input_source(raw: str) -> tuple[str, str]:
    """Parse an optional source prefix from an input line.

    Accepts ``atc: ...``, ``[atc] ...``, ``operator: ...``, ``[operator] ...``
    (case-insensitive). Without a prefix the message is tagged as ``operator``.
    Lets the user role-play ATC from the stdin prompt so the LLM sees the
    correct source tag and routes its response to the radio path.
    """
    text = raw.strip()
    lowered = text.lower()
    for prefix in ("atc:", "[atc]"):
        if lowered.startswith(prefix):
            return "atc", text[len(prefix) :].strip()
    for prefix in ("operator:", "[operator]"):
        if lowered.startswith(prefix):
            return "operator", text[len(prefix) :].strip()
    return "operator", text


def run_tui(
    *,
    bus: SimBus,
    input_queue: "queue.Queue[IncomingMessage]",
    stop_event: threading.Event,
    pilot: "PilotCore",
    heartbeat_pump: "HeartbeatPump | None" = None,
    cache_stats: "CacheStats | None" = None,
) -> None:
    """Run the prompt_toolkit TUI on the calling thread until exit.

    This blocks until the user exits (Ctrl-C / Ctrl-D), at which point
    ``stop_event`` is set so background threads can shut down.

    Log and radio panes are backed by read-only prompt_toolkit Buffers so
    prompt_toolkit's own cursor-tracking keeps the view scrolled to the most
    recent line. A small async background task polls the bus every 100 ms
    and copies new content into the buffers, moving the cursor to the end —
    that's what makes the auto-scroll work reliably.

    The status pane reads ``pilot.latest_snapshot`` directly on every
    render (~10 Hz) so active profiles, throttle, and desired heading are
    always current — it does NOT wait for the control loop to push via
    the bus.
    """
    input_buffer = Buffer(multiline=False)
    log_buffer = Buffer(multiline=True, read_only=True)
    radio_buffer = Buffer(multiline=True, read_only=True)
    log_detail = [True]  # mutable flag; toggled by Ctrl-T

    def _status_fragments() -> list[tuple[str, str]]:
        return format_snapshot_styled(pilot.latest_snapshot, cache_stats=cache_stats)

    def _log_title() -> str:
        return " LOG " if log_detail[0] else " LOG (compact) "

    status_frame = Frame(
        Window(FormattedTextControl(_status_fragments), wrap_lines=True),
        title=" FLIGHT ",
        height=D.exact(STATUS_PANE_HEIGHT),
    )
    log_frame = Frame(
        Window(BufferControl(buffer=log_buffer, focusable=False, lexer=_LogLexer()), wrap_lines=True),
        title=_log_title,
    )
    radio_frame = Frame(
        Window(BufferControl(buffer=radio_buffer, focusable=False), wrap_lines=True),
        title=" RADIO ",
        height=D.exact(RADIO_PANE_HEIGHT),
    )
    input_frame = Frame(
        Window(BufferControl(buffer=input_buffer), height=D.exact(1)),
        title=" INPUT \u2014 enter to send \u00b7 ctrl-t log detail \u00b7 ctrl-c to exit ",
    )

    layout = Layout(HSplit([status_frame, log_frame, radio_frame, input_frame]))
    layout.focus(input_buffer)

    bindings = KeyBindings()

    @bindings.add("enter")
    def _on_enter(event: Any) -> None:
        raw = input_buffer.text.strip()
        if raw:
            source, payload = parse_input_source(raw)
            input_queue.put(IncomingMessage(source=source, text=payload))
            if heartbeat_pump is not None:
                heartbeat_pump.record_user_input()
            bus.push_log(f"[{source}] {payload}")
        input_buffer.reset()

    @bindings.add("c-t")
    def _on_toggle_detail(event: Any) -> None:
        log_detail[0] = not log_detail[0]

    @bindings.add("c-c")
    @bindings.add("c-d")
    def _on_exit(event: Any) -> None:
        stop_event.set()
        event.app.exit()

    app = Application(
        layout=layout,
        key_bindings=bindings,
        full_screen=True,
        refresh_interval=0.1,
        style=TUI_STYLE,
    )

    def _set_buffer_text_pin_end(target: Buffer, text: str) -> None:
        if target.text == text:
            return
        target.set_document(
            Document(text=text, cursor_position=len(text)),
            bypass_readonly=True,
        )

    def _is_verbose_line(line: str) -> bool:
        """Return True for log lines that should be hidden in compact mode."""
        if "[llm-worker] tool " in line:
            return True
        if "[heartbeat]" in line:
            return True
        return False

    async def _refresh_scroll_buffers() -> None:
        while True:
            try:
                raw_lines = bus.log_tail(500)
                if log_detail[0]:
                    log_lines = raw_lines
                else:
                    log_lines = [ln for ln in raw_lines if not _is_verbose_line(ln)]
                log_text = "\n".join(log_lines)
                radio_text = "\n".join(bus.radio_tail(50))
                _set_buffer_text_pin_end(log_buffer, log_text)
                _set_buffer_text_pin_end(radio_buffer, radio_text)
            except Exception:
                pass
            try:
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                return

    async def _run_app_with_refresh() -> None:
        refresh_task = asyncio.create_task(_refresh_scroll_buffers())
        try:
            await app.run_async()
        finally:
            refresh_task.cancel()
            try:
                await refresh_task
            except (asyncio.CancelledError, Exception):
                pass

    try:
        asyncio.run(_run_app_with_refresh())
    finally:
        stop_event.set()
