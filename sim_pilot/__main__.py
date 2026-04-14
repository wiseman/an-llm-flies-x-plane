from __future__ import annotations

from sim_pilot.core.config import load_default_config_bundle
from sim_pilot.sim.scenario import ScenarioRunner


def main() -> None:
    config = load_default_config_bundle()
    result = ScenarioRunner(config).run()
    print(f"success={result.success}")
    print(f"final_phase={result.final_phase.value}")
    print(f"duration_s={result.duration_s:.1f}")
    if result.touchdown_runway_x_ft is not None:
        print(f"touchdown_runway_x_ft={result.touchdown_runway_x_ft:.1f}")
    if result.touchdown_centerline_ft is not None:
        print(f"touchdown_centerline_ft={result.touchdown_centerline_ft:.1f}")
    if result.touchdown_sink_fpm is not None:
        print(f"touchdown_sink_fpm={result.touchdown_sink_fpm:.1f}")


if __name__ == "__main__":
    main()
