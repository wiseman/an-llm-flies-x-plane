from __future__ import annotations

from dataclasses import dataclass, replace
import queue
import threading
import time

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


@dataclass(slots=True, frozen=True)
class LiveRunConfig:
    xplane_host: str
    xplane_port: int
    threshold_lat_deg: float
    threshold_lon_deg: float
    llm_model: str
    atc_messages: tuple[str, ...]
    interactive_atc: bool
    control_hz: float
    status_interval_s: float
    engage_profile: str


@dataclass(slots=True)
class OperatorInputPump:
    input_queue: "queue.Queue[IncomingMessage]"
    prompt: str = "ATC> "

    def __post_init__(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="operator-input", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                line = input(self.prompt).strip()
            except EOFError:
                return
            if line:
                self.input_queue.put(IncomingMessage(source="operator", text=line))


def run_live_xplane(config: ConfigBundle, runtime: LiveRunConfig) -> None:
    bridge = XPlaneWebBridge(
        georef=GeoReference(
            threshold_lat_deg=runtime.threshold_lat_deg,
            threshold_lon_deg=runtime.threshold_lon_deg,
        ),
        host=runtime.xplane_host,
        port=runtime.xplane_port,
    )
    pilot = PilotCore(config)
    _engage_startup_profile(pilot, config, runtime.engage_profile)

    input_queue: "queue.Queue[IncomingMessage]" = queue.Queue()
    for message in runtime.atc_messages:
        input_queue.put(IncomingMessage(source="atc", text=message))

    tool_context = ToolContext(
        pilot=pilot,
        bridge=bridge,
        config=config,
        recent_broadcasts=[],
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
        },
        name="llm-worker",
        daemon=True,
    )
    llm_thread.start()

    input_pump: OperatorInputPump | None = None
    if runtime.interactive_atc:
        input_pump = OperatorInputPump(input_queue=input_queue)
        input_pump.start()
        print("interactive_atc=true")
        print("type operator messages and press enter to send them to the LLM")

    last_state_time_s: float | None = None
    next_status_wall_s = time.monotonic() + runtime.status_interval_s
    target_period_s = 1.0 / max(1.0, runtime.control_hz)
    try:
        while True:
            loop_start_s = time.monotonic()
            raw_state = bridge.read_state()
            dt = _resolve_dt(raw_state.time_s, last_state_time_s, fallback_dt=target_period_s)
            last_state_time_s = raw_state.time_s
            estimated_state, commands = pilot.update(raw_state, dt)
            bridge.write_commands(commands)

            if time.monotonic() >= next_status_wall_s:
                print(
                    " ".join(
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
                    ),
                    flush=True,
                )
                next_status_wall_s = time.monotonic() + runtime.status_interval_s

            elapsed_s = time.monotonic() - loop_start_s
            if elapsed_s < target_period_s:
                time.sleep(target_period_s - elapsed_s)
    finally:
        llm_stop.set()
        if input_pump is not None:
            input_pump.stop()
        bridge.close()


def _engage_startup_profile(pilot: PilotCore, config: ConfigBundle, name: str) -> None:
    key = (name or "idle").lower()
    if key == "idle":
        return
    if key == "pattern_fly":
        pilot.engage_profile(PatternFlyProfile(config, pilot.runway_frame))
        return
    raise SystemExit(f"Unknown --engage-profile value {name!r}; expected one of: idle, pattern_fly")


def apply_live_config_overrides(
    config: ConfigBundle,
    *,
    airport: str | None,
    runway_id: str | None,
    runway_course_deg: float | None,
    field_elevation_ft: float | None,
) -> ConfigBundle:
    runway = replace(
        config.airport.runway,
        id=runway_id or config.airport.runway.id,
        course_deg=config.airport.runway.course_deg if runway_course_deg is None else runway_course_deg,
    )
    airport_config = replace(
        config.airport,
        airport=airport or config.airport.airport,
        field_elevation_ft=config.airport.field_elevation_ft if field_elevation_ft is None else field_elevation_ft,
        runway=runway,
    )
    return replace(config, airport=airport_config)


def bootstrap_live_config_from_sim(
    config: ConfigBundle,
    *,
    host: str,
    xplane_port: int,
    airport: str | None,
    runway_id: str | None,
    runway_course_deg: float | None,
    field_elevation_ft: float | None,
) -> tuple[ConfigBundle, float, float]:
    sample = probe_bootstrap_sample(
        host=host,
        port=xplane_port,
    )
    bootstrapped = apply_bootstrap_sample_to_config(
        config,
        sample=sample,
        airport=airport,
        runway_id=runway_id,
        runway_course_deg=runway_course_deg,
        field_elevation_ft=field_elevation_ft,
    )
    return bootstrapped, sample.posi.lat_deg, sample.posi.lon_deg


def apply_bootstrap_sample_to_config(
    config: ConfigBundle,
    *,
    sample: BootstrapSample,
    airport: str | None,
    runway_id: str | None,
    runway_course_deg: float | None,
    field_elevation_ft: float | None,
) -> ConfigBundle:
    derived_course_deg = _snap_runway_course_deg(sample.posi.heading_deg) if runway_course_deg is None else runway_course_deg
    derived_runway_id = _runway_id_from_course_deg(derived_course_deg) if runway_id is None else runway_id
    if field_elevation_ft is None:
        if not sample.on_ground and sample.alt_agl_ft > 20.0:
            raise ValueError("Bootstrap from sim requires the aircraft to be on or near the runway, or an explicit --field-elevation-ft.")
        derived_field_elevation_ft = (sample.posi.altitude_msl_m * M_TO_FT) - sample.alt_agl_ft
    else:
        derived_field_elevation_ft = field_elevation_ft
    return apply_live_config_overrides(
        config,
        airport=airport,
        runway_id=derived_runway_id,
        runway_course_deg=derived_course_deg,
        field_elevation_ft=derived_field_elevation_ft,
    )


def _resolve_dt(current_time_s: float, last_time_s: float | None, fallback_dt: float) -> float:
    if last_time_s is None:
        return fallback_dt
    dt = current_time_s - last_time_s
    if dt <= 1e-6:
        return fallback_dt
    return clamp(dt, 0.02, 0.5)


def _snap_runway_course_deg(heading_deg: float) -> float:
    snapped = (round(heading_deg / 10.0) * 10.0) % 360.0
    if abs(snapped - 360.0) <= 1e-6:
        return 0.0
    return snapped


def _runway_id_from_course_deg(course_deg: float) -> str:
    runway_number = int(round(course_deg / 10.0)) % 36
    if runway_number == 0:
        runway_number = 36
    return f"{runway_number:02d}"
