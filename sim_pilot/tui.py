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


STATUS_PANE_HEIGHT = 7
RADIO_PANE_HEIGHT = 8


def format_snapshot_display(snapshot: "StatusSnapshot | None") -> str:
    """Render the multi-line status panel text for the TUI and headless log.

    Shows active profiles, phase, current throttle (commanded), desired
    heading (from the last guidance), and basic flight state. Used by both
    the TUI (reads pilot.latest_snapshot directly at ~10 Hz) and the headless
    control loop (pushes to SimBus every status_interval_s).
    """
    if snapshot is None:
        return "(waiting for first pilot tick)"
    state = snapshot.state
    guidance = snapshot.last_guidance
    commands = snapshot.last_commands

    profiles = ", ".join(snapshot.active_profiles) if snapshot.active_profiles else "(none)"
    phase = snapshot.phase.value if snapshot.phase is not None else "—"

    desired_heading: float | None = None
    if guidance is not None:
        desired_heading = guidance.target_heading_deg
        if desired_heading is None:
            desired_heading = guidance.target_track_deg
    heading_str = f"{desired_heading:.0f}°" if desired_heading is not None else "—"

    target_alt_ft = guidance.target_altitude_ft if guidance is not None else None
    target_spd_kt = guidance.target_speed_kt if guidance is not None else None
    target_alt_str = f"{target_alt_ft:.0f}ft" if target_alt_ft is not None else "—"
    target_spd_str = f"{target_spd_kt:.0f}kt" if target_spd_kt is not None else "—"

    # position_ft is the world-frame offset from the georef origin — x is
    # east, y is north (see _geodetic_offset_ft in xplane_bridge.py).
    # Historically labeled "lat/lon" here, which was misleading since they
    # are NOT geographic coordinates.
    runway_x_str = (
        f"{state.runway_x_ft:+.0f}ft" if state.runway_x_ft is not None else "—"
    )
    runway_y_str = (
        f"{state.runway_y_ft:+.0f}ft" if state.runway_y_ft is not None else "—"
    )
    lines = [
        f"profiles:   {profiles}",
        f"phase:      {phase}",
        f"cmd:        throttle={commands.throttle:.2f}  tgt_hdg={heading_str}  tgt_alt={target_alt_str}  tgt_spd={target_spd_str}",
        f"state:      hdg={state.heading_deg:.0f}°  ias={state.ias_kt:.0f}kt  gs={state.gs_kt:.0f}kt  alt_agl={state.alt_agl_ft:.0f}ft  vs={state.vs_fpm:+.0f}fpm",
        f"world:      east={state.position_ft.x:+.0f}ft  north={state.position_ft.y:+.0f}ft  on_ground={state.on_ground}",
        f"runway:     x={runway_x_str}  y={runway_y_str}",
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
