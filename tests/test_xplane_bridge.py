from __future__ import annotations

import unittest

from sim_pilot.sim.xplane_bridge import (
    GeoReference,
    _coerce_scalar,
    _flap_ratio_to_setting,
    _flap_setting_to_ratio,
    _geodetic_offset_ft,
    _select_index,
)


class GeodeticOffsetTests(unittest.TestCase):
    def test_positive_lat_and_lon_delta_produces_east_north_feet(self) -> None:
        georef = GeoReference(threshold_lat_deg=34.0, threshold_lon_deg=-118.0)
        position = _geodetic_offset_ft(lat_deg=34.001, lon_deg=-117.999, georef=georef)
        self.assertGreater(position.x, 0.0)
        self.assertGreater(position.y, 0.0)

    def test_same_point_is_origin(self) -> None:
        georef = GeoReference(threshold_lat_deg=47.449, threshold_lon_deg=-122.309)
        position = _geodetic_offset_ft(lat_deg=47.449, lon_deg=-122.309, georef=georef)
        self.assertAlmostEqual(position.x, 0.0)
        self.assertAlmostEqual(position.y, 0.0)


class FlapConversionTests(unittest.TestCase):
    def test_round_trip_matches_nearest_setting(self) -> None:
        for setting in (0, 10, 20, 30):
            ratio = _flap_setting_to_ratio(setting)
            self.assertEqual(_flap_ratio_to_setting(ratio), setting)

    def test_ratio_outside_bounds_is_clamped_to_nearest_setting(self) -> None:
        self.assertEqual(_flap_ratio_to_setting(-0.1), 0)
        self.assertEqual(_flap_ratio_to_setting(1.5), 30)


class ValueCoercionTests(unittest.TestCase):
    def test_scalar_coerces_to_float(self) -> None:
        self.assertEqual(_coerce_scalar(1), 1.0)
        self.assertEqual(_coerce_scalar(0.5), 0.5)

    def test_list_coerces_first_element(self) -> None:
        self.assertEqual(_coerce_scalar([1.0, 2.0]), 1.0)
        self.assertEqual(_coerce_scalar([]), 0.0)

    def test_select_index_returns_requested_element(self) -> None:
        self.assertEqual(_select_index([10.0, 20.0, 30.0], 1), 20.0)
        self.assertEqual(_select_index(42.0, None), 42.0)

    def test_select_index_raises_on_out_of_range(self) -> None:
        with self.assertRaises(RuntimeError):
            _select_index([1.0], 5)


if __name__ == "__main__":
    unittest.main()
