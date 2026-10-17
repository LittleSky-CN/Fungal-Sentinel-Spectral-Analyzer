import unittest

import numpy as np

from core.unmixing import select_reference_emission, unmix_spectrum


class ReferenceEmissionTests(unittest.TestCase):
    def test_unmixing_recovers_reference_scale_and_offset(self):
        emission = np.array([0.0, 1.0, 0.5, 2.0, 1.5, 0.3, 0.1, 0.0])
        blank = np.array([1.0, 0.0, 2.0, 0.5, 3.0, 1.5, 0.2, 2.5])
        sample = 1.7 * emission + 0.4 * blank + 3.0

        result = unmix_spectrum(
            sample,
            emission,
            blank,
            np.ones(emission.size, dtype=bool),
        )

        self.assertAlmostEqual(result["alpha"], 1.7)
        self.assertAlmostEqual(result["scatter_scale"], 0.4)
        self.assertAlmostEqual(result["offset"], 3.0)
        self.assertAlmostEqual(result["r2"], 1.0)

    def test_selects_largest_positive_emission_not_first_standard(self):
        wavelengths = np.array([500.0, 510.0, 520.0])
        blank = np.zeros(3)
        no_target_emission = np.zeros(3)
        weaker_emission = np.array([0.0, 1.0, 0.0])
        stronger_emission = np.array([0.0, 2.0, 0.0])

        index, reference, auc = select_reference_emission(
            wavelengths,
            [
                ("#1 (20)", no_target_emission, blank),
                ("#2 (60)", weaker_emission, blank),
                ("#3 (100)", stronger_emission, blank),
            ],
            np.array([True, True, True]),
        )

        self.assertEqual(index, 2)
        np.testing.assert_array_equal(reference, stronger_emission)
        self.assertEqual(auc, 20.0)

    def test_reports_standards_with_no_target_emission(self):
        wavelengths = np.array([500.0, 510.0, 520.0])
        zeros = np.zeros(3)

        with self.assertRaisesRegex(ValueError, "No standard has positive reference emission.*#1 \\(20\\): 0"):
            select_reference_emission(
                wavelengths,
                [("#1 (20)", zeros, zeros)],
                np.array([True, True, True]),
            )

    def test_reports_positive_peak_outside_selected_window(self):
        wavelengths = np.array([500.0, 510.0, 520.0, 530.0])
        blank = np.zeros(4)
        sample = np.array([0.0, 0.0, 1.0, 0.0])

        with self.assertRaisesRegex(
            ValueError,
            "Strongest positive blank-subtracted peak is at 520.0 nm.*outside the selected emission window",
        ):
            select_reference_emission(
                wavelengths,
                [("#1 (20)", sample, blank)],
                np.array([True, True, False, False]),
            )

    def test_reports_when_no_visible_positive_signal_exists(self):
        wavelengths = np.array([500.0, 510.0, 520.0])
        zeros = np.zeros(3)

        with self.assertRaisesRegex(ValueError, "No positive blank-subtracted signal was found"):
            select_reference_emission(
                wavelengths,
                [("#1 (20)", zeros, zeros)],
                np.array([True, True, True]),
            )


if __name__ == "__main__":
    unittest.main()
