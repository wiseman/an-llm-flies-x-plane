from __future__ import annotations

import csv
from pathlib import Path

from sim_pilot.sim.scenario import ScenarioResult


def write_scenario_log_csv(result: ScenarioResult, path: str | Path) -> Path:
    output_path = Path(path)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time_s",
                "phase",
                "position_x_ft",
                "position_y_ft",
                "runway_x_ft",
                "runway_y_ft",
                "pitch_deg",
                "altitude_msl_ft",
                "altitude_agl_ft",
                "throttle_pos",
                "throttle_cmd",
                "ias_kt",
                "gs_kt",
                "heading_deg",
                "bank_deg",
            ]
        )
        for row in result.history:
            writer.writerow(
                [
                    f"{row.time_s:.1f}",
                    row.phase.value,
                    f"{row.position_x_ft:.3f}",
                    f"{row.position_y_ft:.3f}",
                    f"{row.runway_x_ft:.3f}",
                    f"{row.runway_y_ft:.3f}",
                    f"{row.pitch_deg:.3f}",
                    f"{row.altitude_msl_ft:.3f}",
                    f"{row.altitude_agl_ft:.3f}",
                    f"{row.throttle_pos:.3f}",
                    f"{row.throttle_cmd:.3f}",
                    f"{row.ias_kt:.3f}",
                    f"{row.gs_kt:.3f}",
                    f"{row.heading_deg:.3f}",
                    f"{row.bank_deg:.3f}",
                ]
            )
    return output_path
