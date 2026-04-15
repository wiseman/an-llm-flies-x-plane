from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sim_pilot.core.config import load_default_config_bundle
from sim_pilot.sim.plotting import write_scenario_plots
from sim_pilot.sim.scenario import ScenarioRunner


class ScenarioPlottingTests(unittest.TestCase):
    def test_plot_writer_emits_svg_files(self) -> None:
        config = load_default_config_bundle()
        result = ScenarioRunner(config).run()
        with TemporaryDirectory() as directory:
            plots = write_scenario_plots(
                result=result,
                config=config,
                scenario_name="takeoff_to_pattern_landing",
                output_dir=directory,
            )
            self.assertTrue(plots.overview_svg.exists())
            self.assertTrue(plots.ground_path_svg.exists())
            overview_text = plots.overview_svg.read_text(encoding="utf-8")
            ground_text = plots.ground_path_svg.read_text(encoding="utf-8")
            self.assertIn("<svg", overview_text)
            self.assertIn("Scenario: takeoff_to_pattern_landing", overview_text)
            self.assertIn("phase by time", overview_text)
            # downwind is long enough to be labeled in the phase bar; cruise
            # used to be labeled too but no longer spans meaningful time now
            # that the mission routes straight from climb to descent.
            self.assertIn("downwind", overview_text)
            self.assertIn("<svg", ground_text)
            self.assertIn(f"Runway {config.airport.runway.id} threshold", ground_text)
            self.assertIn("north-up world frame", ground_text)
            self.assertIn("north up", ground_text)
            self.assertNotIn("runway_frame_x_ft vs runway_frame_y_ft", ground_text)
            self.assertIn("takeoff roll", ground_text)


if __name__ == "__main__":
    unittest.main()
