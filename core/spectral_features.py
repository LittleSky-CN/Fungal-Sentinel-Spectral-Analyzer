# core/spectral_features.py
"""Microplate-style spectral feature extraction (Peak, FWHM, Centroid, AUC).
Aligned with Android SampleAnalyzer.kt logic. NumPy 1.x/2.x compatible.
"""
import numpy as np

# ---- NumPy 2.0+ compatibility shim ----
if hasattr(np, "trapezoid"):
    np_trapz = np.trapezoid
else:
    np_trapz = np.trapz


def calculate_spectral_features(wavelengths: np.ndarray, intensity: np.ndarray, mask: np.ndarray) -> dict:
    """
    Calculate spectral features within the fluorophore integration window.
    Returns dict with keys: peak_wl, peak_intensity, fwhm, auc, centroid.
    """
    wls = wavelengths[mask]
    ints = intensity[mask]

    if len(wls) == 0:
        return {"peak_wl": 0.0, "peak_intensity": 0.0, "fwhm": 0.0, "auc": 0.0, "centroid": 0.0}

    # make sure wavelengths increase (compatible with negative-slope mappings)
    if wls[0] > wls[-1]:
        wls = wls[::-1]
        ints = ints[::-1]

    # 1. Peak (initial 0.0: stays 0 for an all-negative spectrum, matching Kotlin)
    peak_intensity = 0.0
    peak_wl = 0.0
    peak_idx = 0
    for i in range(len(wls)):
        if ints[i] > peak_intensity:
            peak_intensity = float(ints[i])
            peak_wl = float(wls[i])
            peak_idx = i

    # 2. Centroid
    sum_i = float(np.sum(ints))
    centroid = float(np.sum(ints * wls) / sum_i) if sum_i > 0.0 else peak_wl

    # 3. AUC (trapezoidal integration, using the compatibility shim)
    auc = float(np_trapz(ints, wls)) if len(wls) > 1 else 0.0

    # 4. FWHM (crossing detection + linear interpolation, the right boundary is searched after the peak)
    fwhm = 0.0
    if peak_intensity > 0.0:
        half = peak_intensity / 2.0
        left_nm = None
        right_nm = None

        for i in range(len(wls) - 1):
            v1, v2 = floats(ints[i]), floats(ints[i + 1])
            if v1 < half <= v2:
                t = (half - v1) / (v2 - v1 + 1e-12)
                left_nm = wls[i] + t * (wls[i + 1] - wls[i])
                break

        if left_nm is not None:
            for i in range(peak_idx, len(wls) - 1):
                v1, v2 = floats(ints[i]), floats(ints[i + 1])
                if v1 >= half > v2:
                    t = (half - v1) / (v2 - v1 + 1e-12)
                    right_nm = wls[i] + t * (wls[i + 1] - wls[i])
                    break

        if left_nm is not None and right_nm is not None:
            fwhm = float(right_nm - left_nm)
        else:
            fwhm = float(abs(wls[-1] - wls[0]))
    else:
        fwhm = float(abs(wls[-1] - wls[0])) if len(wls) > 1 else 0.0

    return {
        "peak_wl": peak_wl,
        "peak_intensity": peak_intensity,
        "fwhm": fwhm,
        "auc": auc,
        "centroid": centroid,
    }


def floats(x) -> float:
    return float(x)