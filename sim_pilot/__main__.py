from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

from sim_pilot.core.config import load_default_config_bundle
from sim_pilot.core.types import Vec2
from sim_pilot.live_runner import LiveRunConfig, run_live_xplane
from sim_pilot.sim.logging import write_scenario_log_csv
from sim_pilot.sim.plotting import write_scenario_plots
from sim_pilot.sim.scenario import ScenarioRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the deterministic simulator or the live X-Plane MVP bridge.")
    parser.add_argument(
        "--backend",
        choices=("simple", "xplane"),
        default="simple",
        help="Execution backend. `simple` runs the local deterministic sim, `xplane` connects to a live X-Plane instance via the X-Plane 12 web API.",
    )
    parser.add_argument(
        "--scenario-name",
        type=str,
        default=None,
        help="Optional label to print alongside the run summary.",
    )
    parser.add_argument(
        "--crosswind-kt",
        type=float,
        default=0.0,
        help="Adds a lateral wind component on the world X axis. Useful for the simple runway-36 crosswind case.",
    )
    parser.add_argument(
        "--wind-x-kt",
        type=float,
        default=0.0,
        help="World X wind component in knots.",
    )
    parser.add_argument(
        "--wind-y-kt",
        type=float,
        default=0.0,
        help="World Y wind component in knots.",
    )
    parser.add_argument(
        "--log-csv",
        type=Path,
        default=None,
        help="Write a CSV flight log with pitch, altitude, throttle, speed, heading, and bank.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Write SVG plots for the flight overview and north-up ground path.",
    )
    parser.add_argument(
        "--xplane-host",
        type=str,
        default="127.0.0.1",
        help="Host running the X-Plane 12 web API when using `--backend xplane`.",
    )
    parser.add_argument(
        "--xplane-port",
        type=int,
        default=8086,
        help="TCP port that the X-Plane 12 web server is listening on.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-5.4-2026-03-05",
        help="OpenAI model used to interpret ATC/operator messages for live X-Plane runs. Pinned to the 2026-03-05 snapshot so behavior is stable; override with e.g. --llm-model gpt-5.4-mini-2026-03-17 for faster/cheaper inference or --llm-model gpt-5.4-pro-2026-03-05 for the smarter variant.",
    )
    parser.add_argument(
        "--atc-message",
        action="append",
        default=[],
        help="ATC/operator message to send through the LLM at startup. Repeat for multiple messages.",
    )
    parser.add_argument(
        "--interactive-atc",
        action="store_true",
        help="Read ATC/operator messages from stdin during a live X-Plane run and send them through the LLM.",
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=10.0,
        help="Approximate control-loop frequency for live X-Plane runs.",
    )
    parser.add_argument(
        "--status-interval-s",
        type=float,
        default=2.0,
        help="Console status print cadence for live X-Plane runs.",
    )
    parser.add_argument(
        "--engage-profile",
        type=str,
        default="idle",
        choices=("idle", "pattern_fly"),
        help="Guidance profile to engage at startup. `idle` leaves the pilot in wings-level / idle-throttle until the LLM engages a profile; `pattern_fly` starts the deterministic mission pilot immediately.",
    )
    parser.add_argument(
        "--runway-csv-path",
        type=Path,
        default=Path.home() / "data" / "runways.csv",
        help="Path to the runway/airport CSV that the LLM's sql_query tool reads via DuckDB (with the spatial extension loaded). The CSV is read directly — no separate build step.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to a file that receives a timestamped transcript of every status/log/radio line. If omitted, a file is auto-generated under output/ with a timestamped name; pass an empty string or a specific path to override. Pass --no-log-file to disable.",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable the default file logging entirely.",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=30.0,
        help="Seconds of idle time before the heartbeat pump wakes the LLM with a 'nothing has happened, do you want to act?' check-in. The timer resets whenever a user/ATC message arrives, and a heartbeat is emitted immediately on any phase or profile change regardless of the idle timer. Default 30.",
    )
    parser.add_argument(
        "--no-heartbeat",
        action="store_true",
        help="Disable the heartbeat pump entirely. The LLM will only be invoked when the user or ATC sends a message.",
    )
    return parser


def _resolve_log_file_path(explicit_path: Path | None, disabled: bool) -> Path | None:
    if disabled:
        return None
    if explicit_path is not None:
        return explicit_path
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return Path("output") / f"sim_pilot-{timestamp}.log"


def resolve_scenario_name(explicit_name: str | None, wind_vector: Vec2) -> str:
    if explicit_name:
        return explicit_name
    if abs(wind_vector.x) <= 1e-9 and abs(wind_vector.y) <= 1e-9:
        return "takeoff_to_pattern_landing"
    if abs(wind_vector.x) > 1e-9 and abs(wind_vector.y) <= 1e-9:
        return f"takeoff_to_pattern_landing_crosswind_{wind_vector.x:.1f}kt"
    return f"takeoff_to_pattern_landing_wind_x_{wind_vector.x:.1f}_y_{wind_vector.y:.1f}_kt"


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_default_config_bundle()
    if args.backend == "xplane":
        if args.log_csv is not None or args.plots_dir is not None:
            raise SystemExit("Live X-Plane runs do not yet support --log-csv or --plots-dir.")
        log_file_path = _resolve_log_file_path(args.log_file, args.no_log_file)
        print("backend=xplane")
        print(f"xplane_host={args.xplane_host}")
        print(f"xplane_port={args.xplane_port}")
        print(f"llm_model={args.llm_model}")
        if log_file_path is not None:
            print(f"log_file={log_file_path}")
        run_live_xplane(
            config,
            LiveRunConfig(
                xplane_host=args.xplane_host,
                xplane_port=args.xplane_port,
                llm_model=args.llm_model,
                atc_messages=tuple(args.atc_message),
                interactive_atc=args.interactive_atc,
                control_hz=args.control_hz,
                status_interval_s=args.status_interval_s,
                engage_profile=args.engage_profile,
                runway_csv_path=args.runway_csv_path,
                log_file_path=log_file_path,
                heartbeat_interval_s=args.heartbeat_interval,
                heartbeat_enabled=not args.no_heartbeat,
            ),
        )
        return
    wind_vector = Vec2(args.wind_x_kt + args.crosswind_kt, args.wind_y_kt)
    scenario_name = resolve_scenario_name(args.scenario_name, wind_vector)
    result = ScenarioRunner(config, wind_vector_kt=wind_vector).run()
    print(f"scenario_name={scenario_name}")
    print(f"success={result.success}")
    print(f"final_phase={result.final_phase.value}")
    print(f"duration_s={result.duration_s:.1f}")
    print(f"wind_x_kt={wind_vector.x:.1f}")
    print(f"wind_y_kt={wind_vector.y:.1f}")
    if result.touchdown_runway_x_ft is not None:
        print(f"touchdown_runway_x_ft={result.touchdown_runway_x_ft:.1f}")
    if result.touchdown_centerline_ft is not None:
        print(f"touchdown_centerline_ft={result.touchdown_centerline_ft:.1f}")
    if result.touchdown_sink_fpm is not None:
        print(f"touchdown_sink_fpm={result.touchdown_sink_fpm:.1f}")
    if args.log_csv is not None:
        output_path = write_scenario_log_csv(result, args.log_csv)
        print(f"log_csv={output_path}")
    if args.plots_dir is not None:
        plots = write_scenario_plots(result, config, scenario_name, args.plots_dir)
        print(f"plot_overview_svg={plots.overview_svg}")
        print(f"plot_ground_path_svg={plots.ground_path_svg}")


if __name__ == "__main__":
    main()
