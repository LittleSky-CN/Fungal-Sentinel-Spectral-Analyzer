"""Fusion-weight parity tests.

These lock the Step 2 channel-fusion rule shared by Python FSSA v1.4.5 and
Android MycoSpec v2.2.6:

    w_c(λ) = resp_c(λ) / (resp_R + resp_G + resp_B)(λ)      Σ_c w_c(λ) = 1

The reference values below are computed in float32-friendly form so they match
the Kotlin implementation bit-for-bit within 1e-6.
"""
import unittest

import numpy as np

from core.fusion import channel_weights, fuse_pointwise

EPS = 1e-6


class ResponseProportionalWeightTests(unittest.TestCase):
    def setUp(self):
        # Arbitrary three-channel response set; index 0 is an unmasked pixel
        # where every channel reads zero.
        self.resp_b = np.array([0.0, 1.0, 2.0, 3.0])
        self.resp_g = np.array([0.0, 2.0, 2.0, 1.0])
        self.resp_r = np.array([0.0, 3.0, 0.0, 0.5])

    def test_weights_are_response_proportions(self):
        w_b, w_g, w_r = channel_weights(self.resp_b, self.resp_g, self.resp_r)

        for i, (b, g, r) in enumerate(zip(self.resp_b, self.resp_g, self.resp_r)):
            total = b + g + r + EPS
            self.assertAlmostEqual(w_b[i], b / total, places=9)
            self.assertAlmostEqual(w_g[i], g / total, places=9)
            self.assertAlmostEqual(w_r[i], r / total, places=9)

    def test_weights_sum_to_one_where_signal_exists(self):
        w_b, w_g, w_r = channel_weights(self.resp_b, self.resp_g, self.resp_r)
        total = w_b + w_g + w_r

        # Unmasked / all-zero pixel contributes nothing.
        self.assertAlmostEqual(total[0], 0.0, places=9)
        # Within the SPD mask the weights form a partition of unity (to eps).
        for value in total[1:]:
            self.assertAlmostEqual(value, 1.0, delta=1e-5)

    def test_most_sensitive_channel_dominates(self):
        w_b, w_g, w_r = channel_weights(self.resp_b, self.resp_g, self.resp_r)

        self.assertGreater(w_r[1], w_g[1])
        self.assertGreater(w_g[1], w_b[1])
        self.assertGreater(w_b[3], w_g[3])
        self.assertGreater(w_g[3], w_r[3])

    def test_fusion_is_a_partition_of_unity_average(self):
        """Identical corrected spectra must fuse back to that same spectrum."""
        w_b, w_g, w_r = channel_weights(self.resp_b, self.resp_g, self.resp_r)
        curve = np.array([0.0, 1.0, 1.0, 1.0])

        fused = fuse_pointwise((w_b, w_g, w_r), (curve, curve, curve))

        self.assertAlmostEqual(fused[0], 0.0, places=9)
        for value in fused[1:]:
            self.assertAlmostEqual(value, 1.0, delta=1e-5)

    def test_reference_values_match_android_test_vector(self):
        """Same numeric expectation as AlgorithmParityTest on the Android side."""
        w_b, w_g, w_r = channel_weights(
            np.array([1.0]), np.array([2.0]), np.array([3.0])
        )

        self.assertAlmostEqual(w_b[0], 1.0 / (6.0 + EPS), places=9)
        self.assertAlmostEqual(w_g[0], 2.0 / (6.0 + EPS), places=9)
        self.assertAlmostEqual(w_r[0], 3.0 / (6.0 + EPS), places=9)


if __name__ == "__main__":
    unittest.main()
