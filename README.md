# xplane-pilot

The repo now has two runnable paths:

- the original deterministic simple simulator for fast iteration and tests
- a live X-Plane MVP loop that reads aircraft state from X-Plane 12's built-in web API, runs the deterministic pilot core, writes actuator commands back to X-Plane, and sends ATC/operator text through an OpenAI model as structured pilot intents

The control stack remains deterministic. The LLM only interprets high-level instructions such as `extend_downwind`, `turn_base_now`, `maintain_speed`, `join_pattern`, and `go_around`.

## Test suite

Run the unit tests with:

```bash
uv run python -m unittest discover -s tests -v
```

## Simple simulator

Run the demo scenario with:

```bash
uv run python -m sim_pilot
```

Run a 10 kt crosswind case and write a CSV log with pitch, altitude, throttle, speed, heading, and bank:

```bash
uv run python -m sim_pilot --crosswind-kt 10 --log-csv output/flight_log.csv
```

Generate SVG plots for altitude, speed, bank, pitch, throttle, heading, a phase-marked time axis, and a phase-colored runway-frame ground path:

```bash
uv run python -m sim_pilot --crosswind-kt 10 --plots-dir output/plots
```

The ground-path SVG is now north-up world coordinates rather than runway-frame coordinates.

You can also label the run explicitly:

```bash
uv run python -m sim_pilot --scenario-name pattern_debug --log-csv output/flight_log.csv
```

## Live X-Plane MVP

Requirements:

- X-Plane 12.1.1 or later running on the host you point the bridge at (the built-in web server listens on port 8086 by default)
- an OpenAI API key in `OPENAI_API_KEY`
- a real runway threshold latitude/longitude for the runway you want to fly

Important:

- the default `KTEST` config is synthetic and meant for the local simulator
- for a live X-Plane run, override the airport/runway metadata to match the real runway you are using
- the live bridge does not yet emit CSV logs or SVG plots

Example live run:

```bash
uv run python -m sim_pilot \
  --backend xplane \
  --airport KXYZ \
  --runway-id 36 \
  --runway-course-deg 0 \
  --field-elevation-ft 500 \
  --threshold-lat-deg 34.0000 \
  --threshold-lon-deg -118.0000 \
  --llm-model gpt-5-mini \
  --interactive-atc
```

If the aircraft is already sitting on the runway and you want the pilot to bootstrap from the current X-Plane state, use:

```bash
uv run python -m sim_pilot \
  --backend xplane \
  --takeoff-from-here \
  --airport KSEA
```

`--takeoff-from-here` is an alias for `--bootstrap-from-sim`. It probes the current aircraft latitude, longitude, heading, and field elevation from X-Plane, treats the current aircraft position as the runway reference, and starts the deterministic pilot from there.

You can also inject startup messages through the LLM without using stdin:

```bash
uv run python -m sim_pilot \
  --backend xplane \
  --airport KXYZ \
  --runway-id 36 \
  --runway-course-deg 0 \
  --field-elevation-ft 500 \
  --threshold-lat-deg 34.0000 \
  --threshold-lon-deg -118.0000 \
  --atc-message "Join left traffic runway 36" \
  --atc-message "Extend downwind, I'll call your base"
```

While the live runner is active it will print periodic status lines with the current phase, altitude AGL, indicated airspeed, groundspeed, heading, and runway-frame position.
