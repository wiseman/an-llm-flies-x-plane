# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Python is managed with `uv`; there is no separate venv activation step.

- Run the full test suite: `uv run python -m unittest discover -s tests -v`
- Run a single test module: `uv run python -m unittest tests.test_mode_transitions -v`
- Run a single test method: `uv run python -m unittest tests.test_mode_transitions.TestModeTransitions.test_downwind_to_base -v`
- Run the deterministic simulator demo: `uv run python -m sim_pilot`
- Run the simulator with a CSV log and SVG plots: `uv run python -m sim_pilot --crosswind-kt 10 --log-csv output/flight_log.csv --plots-dir output/plots`
- Run against a live X-Plane instance: `uv run python -m sim_pilot --backend xplane --interactive-atc`. `--bootstrap-from-sim` (alias `--takeoff-from-here`) probes the current aircraft position from X-Plane and uses it as the runway reference.
- Run with an initial ATC instruction: `uv run python -m sim_pilot --backend xplane --interactive-atc --atc-message "take off, fly one lap in the pattern, then land"`

Requires Python 3.12+. Live X-Plane runs require an OpenAI API key (`OPENAI_API_KEY`) set in `.env` or exported. The offline simulator (`--backend simple`, the default) does not require an API key or X-Plane.

Third-party dependencies are fine — add them to `pyproject.toml` when they earn their keep. Historically the code has leaned on stdlib for several things (hand-rolled YAML parser in `core/config.py`, `urllib` for the OpenAI Responses API, hand-packed UDP in the legacy X-Plane bridge), but that's a starting point, not a rule.

## Architecture

This is a deterministic fixed-wing autopilot targeting X-Plane 12 via its built-in web API. The design philosophy — spelled out in `SPEC.md` — is that **the flight control loop is deterministic**. The LLM's role in the flight-control path is to interpret ATC/operator messages into high-level pilot intents (extend downwind, turn base, go around, fly heading) that feed the state machine — not to command elevator/aileron/rudder/throttle directly. Cockpit systems outside the flight control loop (transponder, lights, radios, etc.) are fair game for direct LLM control; the "don't touch the actuators" rule is scoped to the pilot's flight-control loop, not the whole cockpit.

### Layered control stack

Flow is top-down. The LLM runs on its own thread and never sits inside the 10 Hz control loop; tool calls mutate `PilotCore` state under a lock.

1. **LLM tool-call conversation** (`sim_pilot/llm/`) — `run_conversation_loop` in `conversation.py` consumes a `queue.Queue[IncomingMessage]` (tagged `operator` / `atc`). It drives the OpenAI Responses API with `tools=TOOL_SCHEMAS` (`sim_pilot/llm/tools.py`) in a multi-turn loop via `ResponsesClient` (`responses_client.py`). Each tool is a Python handler in `TOOL_HANDLERS` that receives a `ToolContext(pilot, bridge, config, recent_broadcasts)` and mutates pilot/bridge state under their locks. The conversation is persistent across messages with character-count compaction that drops oldest full user→assistant turns; pinned system prompt and a per-turn "active profiles" summary system message are never dropped. The `sleep` tool explicitly ends a turn.
2. **Profile composer** (`sim_pilot/core/mission_manager.py::PilotCore`) — owns `active_profiles: list[GuidanceProfile]` guarded by `_lock: threading.RLock`. Each tick, `update()` calls `contribute()` on every active profile and merges their `ProfileContribution`s into a single `GuidanceTargets` by axis ownership (`Axis.LATERAL` / `VERTICAL` / `SPEED`). `engage_profile` auto-disengages any existing profile that owns a conflicting axis; `disengage_profile` re-adds `IdleLateralProfile`/`IdleVerticalProfile`/`IdleSpeedProfile` for orphaned axes. Each tick also publishes a `StatusSnapshot` read by the `get_status` tool.
3. **Guidance profiles** (`sim_pilot/core/profiles.py`) — `IdleLateralProfile` / `IdleVerticalProfile` / `IdleSpeedProfile` are the default floor. `HeadingHoldProfile`, `AltitudeHoldProfile`, `SpeedHoldProfile` are narrow single-axis profiles. `TakeoffProfile` owns all three axes for the takeoff roll → rotate → initial climb sequence. `PatternFlyProfile` wraps the full deterministic mission phase machine (PREFLIGHT → TAKEOFF_ROLL → … → DOWNWIND → BASE → FINAL → ROUNDOUT → FLARE → ROLLOUT, plus GO_AROUND) — it owns all three axes and holds the `ModeManager`, `SafetyMonitor`, `RouteManager`, `PatternGeometry`, `L1PathFollower`, plus discrete trigger methods (`turn_base_now`, `extend_downwind`, `go_around`, `cleared_to_land`). `ApproachRunwayProfile` and `RouteFollowProfile` are stubs. The old `PilotDirectives` override struct is gone; single-axis holds are expressed as composed profiles.
4. **Low-level control** (`sim_pilot/control/`, still on `PilotCore`) — `BankController` / `PitchController` PID loops with rate damping, `TECSLite` maps (altitude_err, speed_err) → (pitch_cmd, throttle_cmd), `CenterlineRolloutController` handles takeoff/rollout lateral. `PilotCore._commands_from_guidance` assembles the final `ActuatorCommands` from the composed `GuidanceTargets`.

**Key invariants:**
- `PilotCore.update(raw_state, dt)` is the single per-tick entry point, is backend-agnostic, and holds `_lock` for its whole duration.
- The control loop never blocks on the LLM — tool handlers acquire `_lock` briefly, while the OpenAI HTTP call happens on the worker thread outside the lock.
- Every profile must own every axis it touches; `engage_profile` enforces unique per-axis ownership so there are no silent priority races.

### Two execution backends

- **`simple`** (default) — `sim_pilot/sim/scenario.py::ScenarioRunner` drives `SimpleAircraftModel` (a toy point-mass dynamics model in `simple_dynamics.py`) with a fixed 0.2 s step. Used by tests and for offline logging/plotting (`sim/logging.py`, `sim/plotting.py`). Wind is injected via `wind_vector_kt`.
- **`xplane`** — `sim_pilot/live_runner.py::run_live_xplane` drives a real X-Plane 12.1.1+ instance via the built-in web API on port 8086 (see https://developer.x-plane.com/article/x-plane-web-api/). `XPlaneWebBridge` (`sim/xplane_bridge.py`) does a one-shot REST call to `/api/capabilities` and `/datarefs?filter[name]=...` to resolve session-local dataref IDs, then opens a WebSocket to `/api/v3` and subscribes to the state datarefs. Incoming `dataref_update_values` messages are **deltas** — only changed fields are included — so the bridge keeps a `_cached_values` dict and merges each update into it. Actuator writes go out as `dataref_set_values` messages on the same WebSocket. State dataref specs live in `sim/datarefs.py` (`STATE_DATAREFS` for reads, `COMMAND_DATAREFS` for writes — note flaps split between the read-only `flap_handle_deploy_ratio` and the writable `flap_handle_request_ratio`). An `OperatorMessagePump` thread reads stdin when `--interactive-atc` is set; startup `--atc-message` strings are also routed through the LLM before the first control tick.

Both backends share `load_default_config_bundle()`, `PilotCore`, `RunwayFrame`, and `estimate_aircraft_state` — this is what makes the live bridge thin.

### Runtime plumbing (live backend)

- **`SimBus`** (`bus.py`) — thread-safe output bus with three channels: `status` (single-line overwrite), `log` (scrolling event log), and `radio` (ATC/pilot transmissions). Both the headless status printer and the TUI read from it. An optional `FileLog` backing writes timestamped transcripts to `output/sim_pilot-*.log`.
- **TUI** (`tui.py`) — a `prompt_toolkit` terminal UI with status/log/radio/input panes, launched when `--interactive-atc` is set. Operator messages typed in the input pane are pushed as `IncomingMessage(source="operator")` into the LLM's input queue.
- **`HeartbeatPump`** (`live_runner.py`) — proactively wakes the LLM conversation thread on flight-phase changes, profile changes, or after an idle timeout (`--heartbeat-interval`, default 30 s). Embeds a status snapshot in the heartbeat message so the LLM can react to state changes without waiting for an operator message.
- **Runway database** — `data/runways.csv` (sourced from OurAirports) is loaded into DuckDB with the spatial extension by the LLM's `sql_query` tool for airport/runway lookups. Override the path with `--runway-csv-path`.

### Runway-frame coordinates

Everything inside the pilot core reasons in a **runway frame** where `x` points down the runway centerline and the threshold is at the origin. `guidance/runway_geometry.py::RunwayFrame` converts between this frame and the world frame used by the dynamics/X-Plane bridge. Pattern geometry (`guidance/pattern_manager.py`) is defined in runway-frame coordinates and only converted out when feeding `L1PathFollower`. For live X-Plane runs, `GeoReference(threshold_lat_deg, threshold_lon_deg)` anchors the transform via a flat-earth approximation (`_geodetic_offset_ft`).

### Configuration

All aircraft/airport/gain/limit values live in YAML files under `sim_pilot/config/` and are loaded into a frozen `ConfigBundle` by `core/config.py`. The hand-rolled YAML parser only supports nested maps of scalars — no lists, anchors, or flow syntax. The default `KTEST` airport is synthetic; for live X-Plane runs, override the airport/runway metadata via `apply_live_config_overrides` (driven from `--airport`/`--runway-id`/`--runway-course-deg`/`--field-elevation-ft` CLI flags) or let `bootstrap_live_config_from_sim` derive them from the current X-Plane aircraft state.
