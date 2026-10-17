# core/fusion.py
"""Pointwise channel-weight fusion (NO cross-wavelength smoothing).

Fusion rule (identical on Python FSSA and Android MycoSpec)
-----------------------------------------------------------
Channel weights use RESPONSE-PROPORTIONAL NORMALISATION:

    S_c(λ)   = resp_R(λ) + resp_G(λ) + resp_B(λ)
    w_c(λ)   = resp_c(λ) / (S_c(λ) + eps)          c in {B, G, R}
    Fused(λ) = w_B·Corrected_B + w_G·Corrected_G + w_R·Corrected_R

so that sum_c w_c(λ) = 1 at every wavelength, and each wavelength is carried
predominantly by whichever channel is genuinely most sensitive there. This is
the same expression implemented in Android
(app/src/main/java/org/fungalsentinel/app/analysis/SpectralResponseCalibrator.kt),
so both front-ends return numerically identical fused spectra for the same DNG.

Note on auto_handoffs(): the B→G / G→R crossing wavelengths are NOT used to
build the weights any more. They are retained purely as a diagnostic readout of
where two adjacent channels have equal absolute responsivity (drawn on
Figure 4d and quoted in the parameter panel).
"""
import numpy as np

EPS = 1e-6


def channel_weights(resp_B, resp_G, resp_R, eps=EPS):
    """Response-proportional weights, w_B + w_G + w_R = 1 pointwise.

    w_c(λ) = resp_c(λ) / (resp_R(λ) + resp_G(λ) + resp_B(λ) + eps)

    Wavelengths where every channel response is zero (i.e. outside the
    SPD/range mask) collapse to w = 0 for all three channels, which leaves the
    fused curve at zero there — exactly what the corrected spectra already are.
    """
    resp_B = np.asarray(resp_B, dtype=float)
    resp_G = np.asarray(resp_G, dtype=float)
    resp_R = np.asarray(resp_R, dtype=float)
    total = resp_B + resp_G + resp_R + eps
    return resp_B / total, resp_G / total, resp_R / total


def auto_handoffs(wavelengths, resp_B, resp_G, resp_R,
                  default_bg=483.0, default_gr=577.0):
    """DIAGNOSTIC ONLY: B→G / G→R responsivity crossings (pointwise search).

    Returns the two wavelengths where adjacent channel responses are closest to
    each other. These are reported as figures of merit; they no longer drive
    the fusion weights.
    """
    wavelengths = np.asarray(wavelengths, dtype=float)
    resp_B = np.asarray(resp_B, dtype=float)
    resp_G = np.asarray(resp_G, dtype=float)
    resp_R = np.asarray(resp_R, dtype=float)
    h_bg, h_gr = default_bg, default_gr
    m = (resp_B > 0) & (resp_G > 0)
    if np.any(m):
        d = np.where(m, np.abs(resp_B - resp_G), np.inf)
        h_bg = float(wavelengths[int(np.argmin(d))])
    m2 = (resp_G > 0) & (resp_R > 0) & (wavelengths > h_bg + 20.0)
    if np.any(m2):
        d2 = np.where(m2, np.abs(resp_G - resp_R), np.inf)
        h_gr = float(wavelengths[int(np.argmin(d2))])
    return h_bg, h_gr


def fuse_pointwise(weights, curves):
    """S(λ) = w_B·B + w_G·G + w_R·R  (pure per-wavelength scalar multiplication)."""
    (w_B, w_G, w_R), (c_B, c_G, c_R) = weights, curves
    return w_B * c_B + w_G * c_G + w_R * c_R
