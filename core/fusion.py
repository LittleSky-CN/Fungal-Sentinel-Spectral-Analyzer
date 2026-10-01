# core/fusion.py
"""Per-wavelength channel weighting & fusion (v1.4.3).
Design rules (per mentor's spec):
  1. Weights w_c(λ) >= 0 and w_R + w_G + w_B = 1 at every wavelength.
  2. Continuous handoff between channels (no hard switching, no extra smoothing).
  3. Pointwise scalar-per-wavelength operations ONLY:
     no Savitzky-Golay, no moving average, no averaging across adjacent wavelengths.
     (ROI column averaging is kept upstream: it averages REPEATS at identical λ,
      which reduces shot noise without lowering spectral resolution.)
"""
import numpy as np
from typing import Dict, Optional, Tuple

def compute_channel_weights(resp_R: np.ndarray, resp_G: np.ndarray, resp_B: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """w_c(λ) = resp_c(λ) / Σresp(λ). Continuous, data-driven, sums to 1."""
    r = np.asarray(resp_R, dtype=np.float64)
    g = np.asarray(resp_G, dtype=np.float64)
    b = np.asarray(resp_B, dtype=np.float64)
    total = r + g + b
    safe = np.where(total > 0, total, 1.0)
    w_R = np.where(total > 0, r / safe, 0.0)
    w_G = np.where(total > 0, g / safe, 0.0)
    w_B = np.where(total > 0, b / safe, 0.0)
    return w_R, w_G, w_B

def fuse_spectra(corr_R: np.ndarray, corr_G: np.ndarray, corr_B: np.ndarray,
                 w_R: np.ndarray, w_G: np.ndarray, w_B: np.ndarray,
                 valid: Optional[np.ndarray] = None) -> np.ndarray:
    """F(λ) = w_R*C_R + w_G*C_G + w_B*C_B  (pointwise scalar ops only)."""
    F = (w_R * np.nan_to_num(np.asarray(corr_R, float))
         + w_G * np.nan_to_num(np.asarray(corr_G, float))
         + w_B * np.nan_to_num(np.asarray(corr_B, float)))
    if valid is not None:
        F = np.where(valid, F, np.nan)
    return F

def find_handoff_wavelength(wls: np.ndarray, w_dom: np.ndarray, w_next: np.ndarray
                            ) -> Optional[float]:
    """Wavelength where the dominant channel's weight is overtaken (first +→− crossing)."""
    d = np.asarray(w_dom, float) - np.asarray(w_next, float)
    sign = np.sign(d)
    idx = np.where(np.diff(sign) < 0)[0]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    denom = d[i] - d[i + 1]
    t = d[i] / denom if denom != 0 else 0.5
    return float(wls[i] + t * (wls[i + 1] - wls[i]))

def robustness_stats(wls: np.ndarray, fused: np.ndarray,
                     corr: Dict[str, np.ndarray], weights: Dict[str, np.ndarray],
                     w_thresh: float = 0.5) -> Dict[str, Tuple[int, float]]:
    """Mean |C_c − Fused| / max|Fused| inside each channel's dominant region (w_c > 0.5).
    Small values prove the three channels agree → fusion is robust."""
    out = {}
    for c in ("R", "G", "B"):
        m = (weights[c] > w_thresh) & np.isfinite(fused) & np.isfinite(np.asarray(corr[c], float))
        n = int(np.count_nonzero(m))
        if n == 0:
            out[c] = (0, float('nan'))
            continue
        denom = float(np.nanmax(np.abs(fused[m])))
        rel = float(np.nanmean(np.abs(np.asarray(corr[c], float)[m] - fused[m])) / denom) if denom > 0 else float('nan')
        out[c] = (n, rel * 100.0)
    return out

def weight_at(wls: np.ndarray, w: np.ndarray, target_nm: float) -> float:
    if len(wls) == 0: return 0.0
    i = int(np.argmin(np.abs(np.asarray(wls, float) - target_nm)))
    return float(w[i])