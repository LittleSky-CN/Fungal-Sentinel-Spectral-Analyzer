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