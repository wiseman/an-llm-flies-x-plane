from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any, TYPE_CHECKING

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import D
from prompt_toolkit.widgets import Frame

from sim_pilot.bus import SimBus
from sim_pilot.llm.conversation import IncomingMessage

if TYPE_CHECKING:
    from sim_pilot.core.mission_manager import PilotCore, StatusSnapshot
    from sim_pilot.live_runner import HeartbeatPump


STATUS_PANE_HEIGHT = 11
RADIO_PANE_HEIGHT = 8


def format_snapshot_display(snapshot: "StatusSnapshot | None") -> str:
    """Render the multi-line status panel text for the TUI and headless log.

    Two-column "current vs target" layout for the primary flight references
    (heading, altitude, airspeed, vertical speed) with a single-line bottom
    row for the secondary state (throttle, flaps, gear, runway position).
    Used by both the TUI (reads pilot.latest_snapshot directly at ~10 Hz)
    and the headless control loop (pushes to SimBus every status_interval_s).
    """
    if snapshot is None:
        return "(waiting for first pilot tick)"
    state = snapshot.state
    guidance = snapshot.last_guidance
    commands = snapshot.last_commands

    profiles = ", ".join(snapshot.active_profiles) if snapshot.active_profiles else "(none)"
    phase = snapshot.phase.value if snapshot.phase is not None else "—"

    # --- target values (from the last guidance) ---
    target_heading_deg: float | None = None
    target_alt_msl_ft: float | None = None
    target_spd_kt: float | None = None
    if guidance is not None:
        target_heading_deg = guidance.target_heading_deg
        if target_heading_deg is None:
            target_heading_deg = guidance.target_track_deg
        target_alt_msl_ft = guidance.target_altitude_ft
        target_spd_kt = guidance.target_speed_kt

    # alt_msl - alt_agl gives the ground elevation directly beneath the
    # aircraft, which is the right reference for converting a target MSL
    # altitude into target AGL. For pattern flying AGL is the more
    # meaningful number for a reader.
    field_elev_ft = state.alt_msl_ft - state.alt_agl_ft
    target_alt_agl_ft: float | None = None
    if target_alt_msl_ft is not None:
        target_alt_agl_ft = target_alt_msl_ft - field_elev_ft

    current_hdg = f"{state.heading_deg:3.0f}°"
    target_hdg = f"{target_heading_deg:3.0f}°" if target_heading_deg is not None else "—"

    current_alt = f"{state.alt_agl_ft:5.0f} AGL"
    target_alt = f"{target_alt_agl_ft:5.0f} AGL" if target_alt_agl_ft is not None else "—"

    current_spd = f"{state.ias_kt:3.0f} kt IAS"
    target_spd = f"{target_spd_kt:3.0f} kt" if target_spd_kt is not None else "—"

    # Secondary row: throttle (actual), flap degrees, gear, runway x/y.
    flap_str = f"{state.flap_index}°"
    gear_str = "dn" if state.gear_down else "up"
    ground_str = "on ground" if state.on_ground else "airborne"
    if state.runway_x_ft is not None and state.runway_y_ft is not None:
        runway_str = f"rwy x{state.runway_x_ft:+.0f} y{state.runway_y_ft:+.0f}"
    else:
        runway_str = "rwy —"

    lines = [
        f"phase:       {phase:<18} profiles:   {profiles}",
        "",
        f"                 current           target",
        f"  heading        {current_hdg:<18}{target_hdg}",
        f"  altitude       {current_alt:<18}{target_alt}",
        f"  airspeed       {current_spd:<18}{target_spd}",
        f"  vertical       {state.vs_fpm:+5.0f} fpm",
        "",
        f"  throttle {commands.throttle:4.2f}   flaps {flap_str:<5}   gear {gear_str}   {ground_str}   {runway_str}",
    ]
    return "\n".join(lines)


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
            return "atc", text[len(prefix):].strip()
    for prefix in ("operator:", "[operator]"):
        if lowered.startswith(prefix):
            return "operator", text[len(prefix):].strip()
    return "operator", text


def run_tui(
    *,
    bus: SimBus,
    input_queue: "queue.Queue[IncomingMessage]",
    stop_event: threading.Event,
    pilot: "PilotCore",
    heartbeat_pump: "HeartbeatPump | None" = None,
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

    def _status_text() -> str:
        return format_snapshot_display(pilot.latest_snapshot)

    status_frame = Frame(
        Window(FormattedTextControl(_status_text), wrap_lines=True),
        title="status",
        height=D.exact(STATUS_PANE_HEIGHT),
    )
    log_frame = Frame(
        Window(BufferControl(buffer=log_buffer, focusable=False), wrap_lines=True),
        title="log",
    )
    radio_frame = Frame(
        Window(BufferControl(buffer=radio_buffer, focusable=False), wrap_lines=True),
        title="radio",
        height=D.exact(RADIO_PANE_HEIGHT),
    )
    input_frame = Frame(
        Window(BufferControl(buffer=input_buffer), height=D.exact(1)),
        title="input (Enter to send, Ctrl-C to exit)",
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
    )

    def _set_buffer_text_pin_end(target: Buffer, text: str) -> None:
        if target.text == text:
            return
        target.set_document(
            Document(text=text, cursor_position=len(text)),
            bypass_readonly=True,
        )

    async def _refresh_scroll_buffers() -> None:
        while True:
            try:
                log_text = "\n".join(bus.log_tail(500))
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
