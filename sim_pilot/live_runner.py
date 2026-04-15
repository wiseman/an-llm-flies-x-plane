from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import queue
import threading
import time

from sim_pilot.bus import SimBus
from sim_pilot.core.config import ConfigBundle
from sim_pilot.core.mission_manager import PilotCore
from sim_pilot.core.profiles import PatternFlyProfile
from sim_pilot.core.types import clamp
from sim_pilot.llm.conversation import IncomingMessage, run_conversation_loop
from sim_pilot.llm.responses_client import ResponsesClient
from sim_pilot.llm.tools import ToolContext
from sim_pilot.sim.xplane_bridge import (
    BootstrapSample,
    DEFAULT_WEB_PORT,
    GeoReference,
    M_TO_FT,
    XPlaneWebBridge,
    probe_bootstrap_sample,
)
from sim_pilot.tui import run_tui


@dataclass(slots=True, frozen=True)
class LiveRunConfig:
    xplane_host: str
    xplane_port: int
    llm_model: str
    atc_messages: tuple[str, ...]
    interactive_atc: bool
    control_hz: float
    status_interval_s: float
    engage_profile: str
    runway_csv_path: Path | None = None


def run_live_xplane(config: ConfigBundle, runtime: LiveRunConfig) -> None:
    # Probe the sim ONCE up front to learn where the aircraft is. The agent
    # starts knowing nothing about runways/airports — this just anchors the
    # pilot core's runway frame at the current aircraft state so the math
    # works. The LLM is responsible for discovering the actual airport via
    # sql_query when it matters.
    sample = probe_bootstrap_sample(host=runtime.xplane_host, port=runtime.xplane_port)
    live_config = bootstrap_config_from_sample(config, sample)
    bridge = XPlaneWebBridge(
        georef=GeoReference(
            threshold_lat_deg=sample.posi.lat_deg,
            threshold_lon_deg=sample.posi.lon_deg,
        ),
        host=runtime.xplane_host,
        port=runtime.xplane_port,
    )
    config = live_config
    pilot = PilotCore(config)
    _engage_startup_profile(pilot, config, runtime.engage_profile)

    bus = SimBus(echo=not runtime.interactive_atc)
    bus.push_log(f"bridge connected host={runtime.xplane_host} port={runtime.xplane_port}")
    airport_label = config.airport.airport or "(unset)"
    runway_label = config.airport.runway.id or "(unset)"
    bus.push_log(
        f"pilot_reference airport={airport_label} runway={runway_label} "
        f"course={config.airport.runway.course_deg:.0f} field_elev={config.airport.field_elevation_ft:.0f}ft"
    )
    if runtime.engage_profile != "idle":
        bus.push_log(f"startup profile engaged: {runtime.engage_profile}")

    input_queue: "queue.Queue[IncomingMessage]" = queue.Queue()
    for message in runtime.atc_messages:
        input_queue.put(IncomingMessage(source="atc", text=message))

    tool_context = ToolContext(
        pilot=pilot,
        bridge=bridge,
        config=config,
        recent_broadcasts=[],
        runway_csv_path=runtime.runway_csv_path,
        bus=bus,
    )
    llm_client = ResponsesClient(model=runtime.llm_model)
    llm_stop = threading.Event()
    llm_thread = threading.Thread(
        target=run_conversation_loop,
        kwargs={
            "client": llm_client,
            "tool_context": tool_context,
            "input_queue": input_queue,
            "stop_event": llm_stop,
            "bus": bus,
        },
        name="llm-worker",
        daemon=True,
    )
    llm_thread.start()

    control_stop = threading.Event()

    if runtime.interactive_atc:
        control_thread = threading.Thread(
            target=_run_control_loop,
            kwargs={
                "bridge": bridge,
                "pilot": pilot,
                "bus": bus,
                "runtime": runtime,
                "stop_event": control_stop,
            },
            name="control-loop",
            daemon=True,
        )
        control_thread.start()
        try:
            run_tui(bus=bus, input_queue=input_queue, stop_event=control_stop)
        finally:
            control_stop.set()
            llm_stop.set()
            bridge.close()
    else:
        try:
            _run_control_loop(
                bridge=bridge,
                pilot=pilot,
                bus=bus,
                runtime=runtime,
                stop_event=control_stop,
            )
        finally:
            llm_stop.set()
            bridge.close()


def _run_control_loop(
    *,
    bridge: XPlaneWebBridge,
    pilot: PilotCore,
    bus: SimBus,
    runtime: LiveRunConfig,
    stop_event: threading.Event,
) -> None:
    last_state_time_s: float | None = None
    next_status_wall_s = time.monotonic() + runtime.status_interval_s
    target_period_s = 1.0 / max(1.0, runtime.control_hz)
    while not stop_event.is_set():
        loop_start_s = time.monotonic()
        try:
            raw_state = bridge.read_state()
        except Exception as exc:
            bus.push_log(f"[bridge] read error: {exc!r}")
            time.sleep(target_period_s)
            continue
        dt = _resolve_dt(raw_state.time_s, last_state_time_s, fallback_dt=target_period_s)
        last_state_time_s = raw_state.time_s
        try:
            estimated_state, commands = pilot.update(raw_state, dt)
            bridge.write_commands(commands)
        except Exception as exc:
            bus.push_log(f"[pilot] tick error: {exc!r}")
            time.sleep(target_period_s)
            continue

        if time.monotonic() >= next_status_wall_s:
            status_line = " ".join(
                [
                    f"profiles={','.join(pilot.list_profile_names())}",
                    f"phase={pilot.phase.value}",
                    f"alt_agl_ft={estimated_state.alt_agl_ft:.0f}",
                    f"ias_kt={estimated_state.ias_kt:.1f}",
                    f"gs_kt={estimated_state.gs_kt:.1f}",
                    f"heading_deg={estimated_state.heading_deg:.1f}",
                    f"runway_x_ft={0.0 if estimated_state.runway_x_ft is None else estimated_state.runway_x_ft:.0f}",
                    f"runway_y_ft={0.0 if estimated_state.runway_y_ft is None else estimated_state.runway_y_ft:.0f}",
                ]
            )
            bus.push_status(status_line)
            next_status_wall_s = time.monotonic() + runtime.status_interval_s

        elapsed_s = time.monotonic() - loop_start_s
        if elapsed_s < target_period_s:
            time.sleep(target_period_s - elapsed_s)


def _engage_startup_profile(pilot: PilotCore, config: ConfigBundle, name: str) -> None:
    key = (name or "idle").lower()
    if key == "idle":
        return
    if key == "pattern_fly":
        pilot.engage_profile(PatternFlyProfile(config, pilot.runway_frame))
        return
    raise SystemExit(f"Unknown --engage-profile value {name!r}; expected one of: idle, pattern_fly")


def bootstrap_config_from_sample(config: ConfigBundle, sample: BootstrapSample) -> ConfigBundle:
    """Build a live ConfigBundle from a one-shot probe of the running sim.

    Anchors the pilot core's runway frame to wherever the aircraft currently
    is — on a runway, in cruise, or over the ocean. The derivation:

    - ``course_deg`` = the aircraft's current true heading. When parked on a
      runway this matches the runway course; in flight it's just a starting
      orientation for the runway frame, which no profile cares about until
      the LLM engages takeoff/pattern_fly against a real runway.
    - ``field_elevation_ft`` = ``MSL - AGL`` at the aircraft's current
      position. Over the ocean that's ~0 (sea level). On a parked aircraft
      it's the airport's field elevation. In cruise over terrain it's the
      ground elevation below the aircraft. In every case it's the right
      reference for AGL computation at the bootstrap moment.
    - ``airport`` and ``runway.id`` are None — the agent looks up the real
      airport via sql_query when it needs to.

    Works at any altitude; no runway-proximity check, no explicit overrides.
    """
    course_deg = sample.posi.heading_deg
    field_elevation_ft = (sample.posi.altitude_msl_m * M_TO_FT) - sample.alt_agl_ft
    runway = replace(config.airport.runway, id=None, course_deg=course_deg)
    airport_config = replace(
        config.airport,
        airport=None,
        field_elevation_ft=field_elevation_ft,
        runway=runway,
    )
    return replace(config, airport=airport_config)


def _resolve_dt(current_time_s: float, last_time_s: float | None, fallback_dt: float) -> float:
    if last_time_s is None:
        return fallback_dt
    dt = current_time_s - last_time_s
    if dt <= 1e-6:
        return fallback_dt
    return clamp(dt, 0.02, 0.5)
