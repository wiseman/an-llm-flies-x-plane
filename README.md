# xplane-pilot

An LLM-driven autopilot for X-Plane 12 that flies a deterministic traffic pattern while an LLM handles ATC communication, mission decisions, and high-level pilot intent.

## Design

The flight control loop is deterministic and runs at 10 Hz -- the LLM never touches elevator, aileron, rudder, or throttle directly. Instead, the LLM interprets operator and ATC messages into high-level actions (take off, fly a heading, enter the pattern at a specific runway, extend downwind, go around, execute a touch-and-go) by calling tools that mutate the pilot core's profile stack. Composable guidance profiles own axes (lateral, vertical, speed) and are merged each tick into a single set of actuator commands. A `PatternFlyProfile` wraps the full phase machine (TAKEOFF_ROLL through TAXI_CLEAR, plus GO_AROUND) and handles the entire traffic pattern autonomously once engaged. Single-axis profiles (`HeadingHoldProfile`, `AltitudeHoldProfile`, `SpeedHoldProfile`) can be composed for cross-country cruise legs. The X-Plane bridge communicates via the built-in web API on port 8086 (REST for setup, WebSocket for real-time dataref reads and writes).

## Tools

| Tool | Description |
|------|-------------|
| `get_status` | JSON snapshot of aircraft state, phase, and active profiles |
| `sleep` | End the LLM's turn; control loop keeps flying active profiles |
| `engage_heading_hold` | Lateral heading hold (optional forced turn direction) |
| `engage_altitude_hold` | Vertical altitude hold via TECS (Total Energy Control System) |
| `engage_speed_hold` | Airspeed target hold |
| `engage_cruise` | Atomic combo: heading + altitude + speed hold in one call |
| `engage_pattern_fly` | Full deterministic pattern pilot anchored at a specific runway |
| `engage_takeoff` | Takeoff sequence: full power, rotate at Vr, climb at Vy |
| `takeoff_checklist` | Pre-takeoff readiness check (parking brake, flaps, gear, etc.) |
| `disengage_profile` | Remove a named profile; idle profiles fill orphaned axes |
| `list_profiles` | List currently active profile names |
| `extend_downwind` | Push the base-turn point further out on the downwind leg |
| `turn_base_now` | Force an immediate base turn (rebuilds base leg at current position) |
| `go_around` | Command an immediate go-around |
| `execute_touch_and_go` | Arm a touch-and-go: next touchdown skips braking and re-takes off |
| `cleared_to_land` | Record a landing clearance |
| `join_pattern` | Acknowledge a pattern-join instruction |
| `tune_radio` | Set a COM radio frequency |
| `broadcast_on_radio` | Transmit a message on a COM radio |
| `set_parking_brake` | Engage or release the parking brake |
| `sql_query` | Read-only SQL against the worldwide runway/airport database (DuckDB) |

## Running

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
# Run the offline deterministic simulator (no X-Plane needed)
uv run python -m sim_pilot

# Run with a crosswind and write CSV + SVG plots
uv run python -m sim_pilot --crosswind-kt 10 --log-csv output/flight.csv --plots-dir output/plots

# Connect to a live X-Plane 12 instance with the interactive TUI
uv run python -m sim_pilot --backend xplane --interactive-atc

# Send an initial instruction to the LLM at startup
uv run python -m sim_pilot --backend xplane --interactive-atc \
  --atc-message "take off, fly one lap in the pattern, then land"

# Run tests
uv run python -m unittest discover -s tests -v
```

## Configuration

The live backend requires:

- **X-Plane 12.1.1+** with the web API enabled on port 8086 (Settings > Data Output > Web Server)
- **An OpenAI API key** for the LLM worker that interprets ATC/operator messages
- **A runways CSV** from [ourairports](https://ourairports.com/data/) (defaults to `~/data/runways.csv`; override with `--runway-csv-path`)

Create a `.env` file in the project root (it is gitignored):

```
OPENAI_API_KEY=sk-...
```

Or export the variable directly:

```bash
export OPENAI_API_KEY=sk-...
```

The offline deterministic simulator (`--backend simple`, the default) does not require an API key or X-Plane.
