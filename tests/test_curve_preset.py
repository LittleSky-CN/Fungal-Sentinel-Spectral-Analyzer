import json
import tempfile
import unittest
from pathlib import Path

from core.curve_preset import load_curve_preset, save_curve_preset, validate_curve_preset
from core.quantification import calculate_concentration


class CurvePresetTests(unittest.TestCase):
    def _cache(self):
        concentrations = [0.0, 1.0, 2.0]
        means = [1.0, 3.0, 5.0]
        fit = calculate_concentration(concentrations, means, 0.0, mode="legacy")
        cache = {
            **fit,
            "concentrations": concentrations,
            "means": means,
            "stds": [0.1, 0.2, 0.3],
            "metric_mode": "unmixing",
            "unmix": {
                "wls": [500.0, 510.0, 520.0],
                "E": [0.0, 2.0, 0.0],
                "auc_E": 20.0,
                "window": (500.0, 520.0),
                "ref_conc": 2.0,
            },
        }
        return cache

    def test_round_trip_stores_curve_and_wavelength_reference_without_pixel_coordinates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            preset_path = Path(temp_dir) / "curve.json"
            save_curve_preset(
                preset_path, self._cache(), "legacy", "Ypet (Venus variant)")
            preset = load_curve_preset(preset_path, "legacy")

        self.assertEqual(preset["concentrations"], [0.0, 1.0, 2.0])
        self.assertEqual(preset["means"], [1.0, 3.0, 5.0])
        self.assertEqual(preset["reference_emission"]["wavelengths"], [500.0, 510.0, 520.0])
        serialized = json.dumps(preset)
        self.assertNotIn("pixel", serialized.lower())

    def test_rejects_preset_for_a_different_analysis_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            preset_path = Path(temp_dir) / "curve.json"
            save_curve_preset(
                preset_path, self._cache(), "legacy", "Ypet (Venus variant)")
            with self.assertRaisesRegex(ValueError, "different analysis mode"):
                load_curve_preset(preset_path, "lod_iupac")

    def test_rejects_non_monotonic_reference_wavelengths(self):
        cache = self._cache()
        save_data = {
            "schema_version": 1,
            "analysis_mode": "legacy",
            "fluorophore": "Ypet (Venus variant)",
            "metric_mode": "unmixing",
            "regression_method": "ols",
            "concentrations": cache["concentrations"],
            "means": cache["means"],
            "stds": cache["stds"],
            "fit": {key: cache[key] for key in ("slope", "intercept", "r_value", "r_squared")},
            "reference_emission": {
                "wavelengths": [500.0, 520.0, 510.0],
                "emission": [0.0, 2.0, 0.0],
                "auc": 20.0,
                "window_nm": [505.0, 520.0],
                "ref_conc": 2.0,
            },
        }
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_curve_preset(save_data, expected_mode="legacy")


if __name__ == "__main__":
    unittest.main()
