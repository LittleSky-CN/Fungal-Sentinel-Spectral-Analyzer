# core/fusion.py
"""Pointwise channel-weight fusion (NO cross-wavelength smoothing)."""
import numpy as np

def _logistic(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))

def auto_handoffs(wavelengths, resp_B, resp_G, resp_R,
                  default_bg=483.0, default_gr=577.0):
    """B→G / G→R handoff = response-curve crossing points (pointwise search)."""
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

def channel_weights(wavelengths, h_bg, h_gr, width=15.0):
    """Continuous weights, sum=1, computed per wavelength (no smoothing)."""
    t1 = _logistic((wavelengths - h_bg) / width)   # B→G handoff
    t2 = _logistic((wavelengths - h_gr) / width)   # G→R handoff
    w_B = 1.0 - t1
    w_G = t1 * (1.0 - t2)
    w_R = t2
    return w_B, w_G, w_R

def fuse_pointwise(weights, curves):
    """S(λ) = w_B·B + w_G·G + w_R·R  (pure per-wavelength scalar multiplication)."""
    (w_B, w_G, w_R), (c_B, c_G, c_R) = weights, curves
    return w_B * c_B + w_G * c_G + w_R * c_R