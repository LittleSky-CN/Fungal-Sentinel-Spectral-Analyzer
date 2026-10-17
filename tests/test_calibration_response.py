import unittest

import numpy as np

from core.calibration import (
    analyze_blank_subtracted_replicates,
    correct_by_response,
    remove_isolated_spikes,
)


class ResponseCorrectionTests(unittest.TestCase):
    def test_does_not_amplify_signal_below_response_floor(self):
        signal = np.array([100.0, 100.0, 10.0])
        response = np.array([0.0, 0.01, 1.0])

        corrected = correct_by_response(signal, response)

        np.testing.assert_array_equal(corrected, [0.0, 0.0, 10.0])

    def test_replaces_single_pixel_spike_without_shaping_a_broad_peak(self):
        wavelengths = np.arange(31, dtype=np.float64)
        peak = np.exp(-0.5 * ((wavelengths - 15.0) / 4.0) ** 2)
        spectrum = peak.copy()
        spectrum[4] = 1.0

        filtered = remove_isolated_spikes(spectrum)

        self.assertLess(filtered[4], 0.03)
        self.assertAlmostEqual(float(filtered[15]), 1.0)
        self.assertGreater(filtered[11], filtered[8])
        self.assertGreater(filtered[19], filtered[22])

    def test_blank_subtraction_clips_each_replicate_before_averaging(self):
        wavelengths = np.arange(10, dtype=np.float64)
        areas, mean_spectrum = analyze_blank_subtracted_replicates(
            blank_profile=np.full(10, 2.0),
            sample_profiles=np.array([np.full(10, 1.0), np.full(10, 5.0)]),
            wavelengths=wavelengths,
            area_mask=np.ones(10, dtype=bool),
        )

        np.testing.assert_allclose(areas, [0.0, 27.0])
        np.testing.assert_allclose(mean_spectrum, np.full(10, 1.5))


if __name__ == "__main__":
    unittest.main()
