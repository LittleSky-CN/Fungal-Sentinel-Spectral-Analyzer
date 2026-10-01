# core/spectral_features.py
"""Spectral feature extraction (Microplate Reader style parameters)."""
import numpy as np
from typing import Dict

def calculate_spectral_features(wavelengths: np.ndarray, intensity: np.ndarray, mask: np.ndarray = None) -> Dict:
    if wavelengths is None or intensity is None or len(wavelengths) == 0:
        return {"peak_wl": 0, "peak_intensity": 0, "fwhm": 0, "auc": 0, "centroid": 0}

    if mask is not None and np.any(mask):
        wls = wavelengths[mask]
        ints = intensity[mask]
    else:
        wls = wavelengths
        ints = intensity

    peak_idx = np.argmax(ints)
    peak_wl = float(wls[peak_idx])
    peak_intensity = float(ints[peak_idx])

    half_max = peak_intensity / 2.0
    above_half = ints >= half_max
    if np.any(above_half):
        indices = np.where(above_half)[0]
        fwhm = float(wls[indices[-1]] - wls[indices[0]])
    else:
        fwhm = 0.0

    auc = float(np.trapz(ints, wls)) if hasattr(np, 'trapz') else float(np.trapezoid(ints, wls))

    total_int = np.sum(ints)
    if total_int > 0:
        centroid = float(np.sum(wls * ints) / total_int)
    else:
        centroid = 0.0

    return {
        "peak_wl": peak_wl, "peak_intensity": peak_intensity,
        "fwhm": fwhm, "auc": auc, "centroid": centroid
    }