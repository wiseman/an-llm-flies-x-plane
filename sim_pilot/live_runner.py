from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable
import queue
import threading
import time

from sim_pilot.bus import FileLog, SimBus
from sim_pilot.core.config import ConfigBundle
from sim_pilot.core.mission_manager import PilotCore
from sim_pilot.core.profiles import PatternFlyProfile
from sim_pilot.core.types import FlightPhase, clamp
from sim_pilot.llm.conversation import IncomingMessage, run_conversation_loop
from sim_pilot.llm.responses_client import ResponsesClient
from sim_pilot.llm.tools import ToolContext, build_status_payload
from sim_pilot.sim.xplane_bridge import (
    BootstrapSample,
    DEFAULT_WEB_PORT,
    GeoReference,
    M_TO_FT,
    XPlaneWebBridge,
    probe_bootstrap_sample,
)
from sim_pilot.tui import format_snapshot_display, run_tui


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
    log_file_path: Path | None = None
    heartbeat_interval_s: float = 30.0
    heartbeat_enabled: bool = True


class HeartbeatPump:
    """Proactively wakes the LLM worker.

    Pushes an ``IncomingMessage(source="heartbeat", ...)`` into the shared
    input queue under three conditions:

    * the aircraft's flight phase changes (e.g. DOWNWIND → BASE inside a
      PatternFlyProfile), observed via ``pilot.latest_snapshot.phase``;
    * the set of active profiles changes (engaged, disengaged, displaced),
      observed via ``pilot.latest_snapshot.active_profiles``;
    * ``heartbeat_interval_s`` seconds have elapsed since the last
      user-facing input and since the last heartbeat.

    Callers push real user/ATC messages into the queue via the usual path
    and call :meth:`record_user_input` right after doing so — that resets
    the idle timer so heartbeats don't fire immediately on top of real
    conversation.

    The pump never emits back-to-back heartbeats faster than
    ``min_interval_s`` (default 2 s) to debounce rapid change events.

    Thread-safe: the polling loop runs on its own daemon thread; the
    observed ``pilot.latest_snapshot`` is a single-attribute read which
    is atomic in CPython.
    """

    def __init__(
        self,
        *,
        input_queue: "queue.Queue[IncomingMessage]",
        pilot: PilotCore,
        bridge: XPlaneWebBridge | None = None,
        bus: SimBus | None = None,
        heartbeat_interval_s: float = 30.0,
        poll_interval_s: float = 0.5,
        min_interval_s: float = 2.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.input_queue = input_queue
        self.pilot = pilot
        self.bridge = bridge
        self.bus = bus
        self.heartbeat_interval_s = heartbeat_interval_s
        self.poll_interval_s = poll_interval_s
        self.min_interval_s = min_interval_s
        self._clock = clock or time.monotonic
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="heartbeat-pump", daemon=True)
        self._last_user_input_time_s: float = self._clock()
        self._last_heartbeat_time_s: float = self._clock()
        self._last_seen_phase: FlightPhase | None = None
        self._last_seen_profiles: tuple[str, ...] | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def record_user_input(self) -> None:
        """Reset the periodic idle timer. Call this right after putting a
        real user/ATC message into the input queue so the next heartbeat
        fires ``heartbeat_interval_s`` seconds from now, not from the
        previous heartbeat."""
        self._last_user_input_time_s = self._clock()

    def check_and_emit(self) -> None:
        """Run one polling iteration. Exposed for tests that don't want to
        spin up the background thread."""
        now = self._clock()
        snapshot = self.pilot.latest_snapshot
        if snapshot is None:
            return
        current_phase = snapshot.phase
        current_profiles = tuple(snapshot.active_profiles)

        # First call: seed the "previously seen" state without firing.
        if self._last_seen_profiles is None:
            self._last_seen_phase = current_phase
            self._last_seen_profiles = current_profiles
            return

        phase_changed = current_phase != self._last_seen_phase
        profiles_changed = current_profiles != self._last_seen_profiles

        if phase_changed or profiles_changed:
            # Push a dedicated bus log line on go-around transitions so
            # the reason is visible in the log timeline separately from
            # the heartbeat message. Fires even when debounced below.
            if (
                phase_changed
                and current_phase is FlightPhase.GO_AROUND
                and self._last_seen_phase is not FlightPhase.GO_AROUND
                and self.bus is not None
            ):
                ga_reason = snapshot.go_around_reason or "unknown"
                self.bus.push_log(f"[safety] go_around triggered: {ga_reason}")
            if now - self._last_heartbeat_time_s >= self.min_interval_s:
                reason = self._describe_change(
                    old_phase=self._last_seen_phase,
                    new_phase=current_phase,
                    old_profiles=self._last_seen_profiles,
                    new_profiles=current_profiles,
                    go_around_reason=snapshot.go_around_reason,
                )
                self._emit_heartbeat(reason, now)
            # Always update the "last seen" state so we don't re-fire the
            # same change on the next tick.
            self._last_seen_phase = current_phase
            self._last_seen_profiles = current_profiles
            return

        # Periodic idle heartbeat
        last_input = max(self._last_user_input_time_s, self._last_heartbeat_time_s)
        if now - last_input >= self.heartbeat_interval_s:
            self._emit_heartbeat("periodic check-in", now)

    def _emit_heartbeat(self, reason: str, now: float) -> None:
        import json as _json
        self._last_heartbeat_time_s = now
        status = build_status_payload(self.pilot.latest_snapshot, self.bridge)
        text = f"{reason} | status={_json.dumps(status)}"
        self.input_queue.put(IncomingMessage(source="heartbeat", text=text))

    @staticmethod
    def _describe_change(
        *,
        old_phase: FlightPhase | None,
        new_phase: FlightPhase | None,
        old_profiles: tuple[str, ...],
        new_profiles: tuple[str, ...],
        go_around_reason: str | None = None,
    ) -> str:
        parts: list[str] = []
        if old_phase != new_phase:
            old = old_phase.value if old_phase is not None else "none"
            new = new_phase.value if new_phase is not None else "none"
            if new_phase is FlightPhase.GO_AROUND and go_around_reason:
                parts.append(f"phase changed: {old} -> {new} ({go_around_reason})")
            else:
                parts.append(f"phase changed: {old} -> {new}")
        if tuple(old_profiles) != tuple(new_profiles):
            old_set = set(old_profiles)
            new_set = set(new_profiles)
            added = sorted(new_set - old_set)
            removed = sorted(old_set - new_set)
            profile_parts: list[str] = []
            if added:
                profile_parts.append(f"engaged: {', '.join(added)}")
            if removed:
                profile_parts.append(f"disengaged: {', '.join(removed)}")
            if profile_parts:
                parts.append("profiles " + "; ".join(profile_parts))
        return "; ".join(parts) if parts else "state changed"

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.check_and_emit()
            except Exception:
                pass
            if self._stop.wait(self.poll_interval_s):
                return


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

    file_log = FileLog(runtime.log_file_path) if runtime.log_file_path is not None else None
    bus = SimBus(echo=not runtime.interactive_atc, file_log=file_log)
    if runtime.log_file_path is not None:
        bus.push_log(f"log_file={runtime.log_file_path}")
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

    heartbeat_pump: HeartbeatPump | None = None
    if runtime.heartbeat_enabled:
        heartbeat_pump = HeartbeatPump(
            input_queue=input_queue,
            pilot=pilot,
            bridge=bridge,
            bus=bus,
            heartbeat_interval_s=runtime.heartbeat_interval_s,
        )
        heartbeat_pump.start()
        bus.push_log(
            f"heartbeat pump started interval={runtime.heartbeat_interval_s:.0f}s"
        )

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
            run_tui(
                bus=bus,
                input_queue=input_queue,
                stop_event=control_stop,
                pilot=pilot,
                heartbeat_pump=heartbeat_pump,
            )
        finally:
            control_stop.set()
            llm_stop.set()
            if heartbeat_pump is not None:
                heartbeat_pump.stop()
            bridge.close()
            bus.close()
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
            if heartbeat_pump is not None:
                heartbeat_pump.stop()
            bridge.close()
            bus.close()


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
            bus.push_status(format_snapshot_display(pilot.latest_snapshot))
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
