from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sim_pilot.core.config import load_default_config_bundle
from sim_pilot.sim.logging import write_scenario_log_csv
from sim_pilot.sim.scenario import ScenarioRunner


class ScenarioLoggingTests(unittest.TestCase):
    def test_log_writer_emits_requested_columns(self) -> None:
        result = ScenarioRunner(load_default_config_bundle()).run()
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "flight_log.csv"
            written_path = write_scenario_log_csv(result, output_path)
            self.assertEqual(written_path, output_path)
            with written_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertIsNotNone(reader.fieldnames)
                assert reader.fieldnames is not None
                self.assertEqual(
                    reader.fieldnames,
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
                    ],
                )
                first_row = next(reader)
                self.assertIn(first_row["phase"], {"takeoff_roll", "rotate"})
                self.assertNotEqual(first_row["position_x_ft"], "")
                self.assertNotEqual(first_row["runway_x_ft"], "")
                self.assertNotEqual(first_row["pitch_deg"], "")
                self.assertNotEqual(first_row["altitude_msl_ft"], "")
                self.assertNotEqual(first_row["throttle_pos"], "")
                self.assertNotEqual(first_row["ias_kt"], "")
                self.assertNotEqual(first_row["heading_deg"], "")
                self.assertNotEqual(first_row["bank_deg"], "")


if __name__ == "__main__":
    unittest.main()
