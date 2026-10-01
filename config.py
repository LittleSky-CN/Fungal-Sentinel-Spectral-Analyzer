# config.py
"""Global configurations and fluorophore database for FSSA."""

FLUOROPHORES = {
    "Ypet (Venus variant)": {"peak_wl": 527, "channel": "green", "fwhm": 20, "color": "green"},
    "EGFP": {"peak_wl": 509, "channel": "green", "fwhm": 20, "color": "green"},
    "mCherry": {"peak_wl": 610, "channel": "red", "fwhm": 25, "color": "red"},
    "CFP": {"peak_wl": 475, "channel": "blue", "fwhm": 20, "color": "blue"},
    "mTurquoise2": {"peak_wl": 482, "channel": "blue", "fwhm": 20, "color": "cyan"}
}

DEFAULT_PARAMS = {
    "black_level": 64.0,
    "display_percentile": 99.7,
    "spd_threshold_ratio": 0.1,  # 【修复】提高阈值以过滤边缘噪声
    "valid_wavelength_range": (430, 660), # 【修复】重新引入波长范围保护
    "smooth_window": 21
}