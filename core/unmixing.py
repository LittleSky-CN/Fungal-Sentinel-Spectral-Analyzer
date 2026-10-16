# core/unmixing.py
"""Spectral unmixing for low-LoD quantification (v1.4.4).
Model (from LOD_report_A_20260905, mentor-approved):
    sample(λ) = α·E(λ) + s·B(λ) + k
  E(λ) : reference fluorophore emission shape = (highest standard − its blank)
  B(λ) : measured blank spectrum of the same row (scatter / Raman / cuvette autofluorescence)
  k    : constant offset (dark residual / stray light)
  α    : fluorophore scale; quantitative metric = α · AUC_emission(E)
Linear least squares in (α, s, k), pointwise over the unmixing window.
NO smoothing, NO cross-wavelength averaging (resolution-preservation rule).
"""
import numpy as np
from typing import Dict, Tuple

np_trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

def default_unmix_window(peak_wl: float, valid_range: Tuple[float, float] = (420.0, 680.0)) -> Tuple[float, float]:
    """Emission window starting just beyond the scatter peak (report: 490-600 nm for EGFP@509)."""
    lo = max(valid_range[0], peak_wl - 20.0)
    hi = min(valid_range[1], peak_wl + 90.0)
    return float(lo), float(hi)

def build_reference_emission(sample_mean: np.ndarray, blank_mean: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(sample_mean, float) - np.asarray(blank_mean, float), 0.0, None)

def emission_auc(wavelengths: np.ndarray, ref_emission: np.ndarray, emission_mask: np.ndarray) -> float:
    w = np.asarray(wavelengths, float)[emission_mask]
    e = np.asarray(ref_emission, float)[emission_mask]
    if len(w) < 2: return 0.0
    order = np.argsort(w)
    return float(np_trapz(e[order], w[order]))

def select_reference_emission(
    wavelengths: np.ndarray,
    candidates: list[tuple[str, np.ndarray, np.ndarray]],
    emission_mask: np.ndarray,
) -> tuple[int, np.ndarray, float]:
    """Select the standard with the strongest positive measured emission in-band."""
    wavelengths = np.asarray(wavelengths, dtype=float)
    emission_mask = np.asarray(emission_mask, dtype=bool)
    if wavelengths.ndim != 1 or emission_mask.shape != wavelengths.shape:
        raise ValueError("Wavelengths and emission mask must be matching one-dimensional arrays")

    best: tuple[int, np.ndarray, float] | None = None
    aucs = []
    strongest_peak: tuple[str, float, float] | None = None
    candidate_details = []
    visible_mask = (wavelengths >= 400.0) & (wavelengths <= 700.0)
    for index, (label, sample_mean, blank_mean) in enumerate(candidates):
        sample_mean = np.asarray(sample_mean, dtype=float)
        blank_mean = np.asarray(blank_mean, dtype=float)
        if sample_mean.shape != wavelengths.shape or blank_mean.shape != wavelengths.shape:
            raise ValueError(f"Standard {label} spectrum does not match the wavelength array")
        net = sample_mean - blank_mean
        emission = np.clip(net, 0.0, None)
        auc = emission_auc(wavelengths, emission, emission_mask)
        positive_count = int(np.count_nonzero(net[emission_mask] > 0))
        band_count = int(np.count_nonzero(emission_mask))
        sample_level = float(np.mean(sample_mean[emission_mask])) if band_count else 0.0
        blank_level = float(np.mean(blank_mean[emission_mask])) if band_count else 0.0
        aucs.append(f"{label}: {auc:.6g}")
        candidate_details.append(
            f"{label}: sample/blank mean={sample_level:.6g}/{blank_level:.6g}, "
            f"sample>blank={positive_count}/{band_count} points"
        )
        if auc > 0 and (best is None or auc > best[2]):
            best = (index, emission, auc)
        if np.any(visible_mask):
            visible_indices = np.flatnonzero(visible_mask)
            peak_index = visible_indices[np.argmax(emission[visible_mask])]
            peak = (label, float(wavelengths[peak_index]), float(emission[peak_index]))
            if strongest_peak is None or peak[2] > strongest_peak[2]:
                strongest_peak = peak

    if best is None:
        band = wavelengths[emission_mask]
        if band.size:
            window = f"{band.min():.1f}-{band.max():.1f} nm"
        else:
            window = "empty wavelength window"
        if strongest_peak is not None and strongest_peak[2] > 0:
            peak_label, peak_wavelength, peak_signal = strongest_peak
            peak_diagnostic = (
                f" Strongest positive blank-subtracted peak is at {peak_wavelength:.1f} nm "
                f"(standard {peak_label}, signal {peak_signal:.6g})."
            )
            if not emission_mask.any() or not np.any(
                (wavelengths == peak_wavelength) & emission_mask
            ):
                peak_diagnostic += " It is outside the selected emission window."
        else:
            peak_diagnostic = " No positive blank-subtracted signal was found from 400 to 700 nm."
        raise ValueError(
            f"No standard has positive reference emission in {window}."
            f"{peak_diagnostic} Check the selected fluorophore/window, "
            f"standard sample/blank assignments, and concentration labels. "
            f"Measured AUCs: {', '.join(aucs) or 'none'}. "
            f"In-band levels: {'; '.join(candidate_details) or 'none'}."
        )
    return best

def unmix_spectrum(sample: np.ndarray, ref_emission: np.ndarray, blank: np.ndarray,
                   window_mask: np.ndarray) -> Dict[str, float]:
    """Solve sample = α·E + s·B + k over window_mask by linear least squares."""
    y = np.asarray(sample, float)[window_mask]
    E = np.asarray(ref_emission, float)[window_mask]
    B = np.asarray(blank, float)[window_mask]
    X = np.column_stack([E, B, np.ones_like(E)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, s, k = (float(c) for c in coef)
    y_hat = X @ coef
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"alpha": alpha, "scatter_scale": s, "offset": k, "r2": r2}

def slope_t_statistic(r_value: float, n: int) -> float:
    """t = |r|·sqrt(n-2)/sqrt(1-r²); report rule requires t >= 3 for a usable calibration."""
    if n <= 2: return 0.0
    r2 = float(r_value) ** 2
    if r2 >= 1.0: return float('inf')
    return abs(float(r_value)) * np.sqrt(n - 2) / np.sqrt(1.0 - r2)