from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from sim_pilot.core.config import load_default_config_bundle
from sim_pilot.core.types import Vec2
from sim_pilot.live_runner import LiveRunConfig, apply_live_config_overrides, bootstrap_live_config_from_sim, run_live_xplane
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
        "--threshold-lat-deg",
        type=float,
        default=None,
        help="Runway threshold latitude for live X-Plane runs.",
    )
    parser.add_argument(
        "--threshold-lon-deg",
        type=float,
        default=None,
        help="Runway threshold longitude for live X-Plane runs.",
    )
    parser.add_argument(
        "--bootstrap-from-sim",
        "--takeoff-from-here",
        dest="bootstrap_from_sim",
        action="store_true",
        help="Probe the current X-Plane aircraft position and heading, treat that as the runway reference, and start the pilot from there.",
    )
    parser.add_argument(
        "--airport",
        type=str,
        default=None,
        help="Override the airport code used by the live pilot core.",
    )
    parser.add_argument(
        "--runway-id",
        type=str,
        default=None,
        help="Override the runway identifier used by the live pilot core.",
    )
    parser.add_argument(
        "--runway-course-deg",
        type=float,
        default=None,
        help="Override the runway magnetic/true course in degrees used by the live pilot core.",
    )
    parser.add_argument(
        "--field-elevation-ft",
        type=float,
        default=None,
        help="Override the field elevation used by the live pilot core.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-5-mini",
        help="OpenAI model used to interpret ATC/operator messages for live X-Plane runs.",
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
    return parser


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
        if args.bootstrap_from_sim:
            live_config, threshold_lat_deg, threshold_lon_deg = bootstrap_live_config_from_sim(
                config,
                host=args.xplane_host,
                xplane_port=args.xplane_port,
                airport=args.airport,
                runway_id=args.runway_id,
                runway_course_deg=args.runway_course_deg,
                field_elevation_ft=args.field_elevation_ft,
            )
        else:
            if args.threshold_lat_deg is None or args.threshold_lon_deg is None:
                raise SystemExit("--backend xplane requires either --bootstrap-from-sim or both --threshold-lat-deg and --threshold-lon-deg.")
            threshold_lat_deg = args.threshold_lat_deg
            threshold_lon_deg = args.threshold_lon_deg
            live_config = apply_live_config_overrides(
                config,
                airport=args.airport,
                runway_id=args.runway_id,
                runway_course_deg=args.runway_course_deg,
                field_elevation_ft=args.field_elevation_ft,
            )
        print("backend=xplane")
        print(f"bootstrap_from_sim={str(args.bootstrap_from_sim).lower()}")
        print(f"airport={live_config.airport.airport}")
        print(f"runway_id={live_config.airport.runway.id}")
        print(f"runway_course_deg={live_config.airport.runway.course_deg:.1f}")
        print(f"field_elevation_ft={live_config.airport.field_elevation_ft:.1f}")
        print(f"threshold_lat_deg={threshold_lat_deg:.6f}")
        print(f"threshold_lon_deg={threshold_lon_deg:.6f}")
        print(f"xplane_host={args.xplane_host}")
        print(f"xplane_port={args.xplane_port}")
        print(f"llm_model={args.llm_model}")
        run_live_xplane(
            live_config,
            LiveRunConfig(
                xplane_host=args.xplane_host,
                xplane_port=args.xplane_port,
                threshold_lat_deg=threshold_lat_deg,
                threshold_lon_deg=threshold_lon_deg,
                llm_model=args.llm_model,
                atc_messages=tuple(args.atc_message),
                interactive_atc=args.interactive_atc,
                control_hz=args.control_hz,
                status_interval_s=args.status_interval_s,
                engage_profile=args.engage_profile,
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
