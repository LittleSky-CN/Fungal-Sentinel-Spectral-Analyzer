# config.py
"""FSSA v1.4.6 Global Configuration."""

# =============================================================================
# 荧光蛋白/目标物数据库 (Fluorophore Database)
# =============================================================================
FLUOROPHORES = {
    "Ypet (Venus variant)": {"peak_wl": 527, "channel": "green", "fwhm": 20, "color": "green"},
    "EGFP":                 {"peak_wl": 509, "channel": "green", "fwhm": 20, "color": "green"},
    "mCherry":              {"peak_wl": 610, "channel": "red",   "fwhm": 25, "color": "red"},
    "CFP":                  {"peak_wl": 475, "channel": "blue",  "fwhm": 20, "color": "blue"},
    "mTurquoise2":          {"peak_wl": 482, "channel": "blue",  "fwhm": 20, "color": "cyan"},
}

# =============================================================================
# 核心算法默认参数 (供 core/calibration.py 等读取)
# =============================================================================
DEFAULT_PARAMS = {
    "black_level": 64.0,                 # DNG 未提供黑电平元数据时的回退值
    "display_percentile": 99.7,          # 预览归一化百分位
    "spd_threshold_ratio": 0.1,          # SPD 可信阈值比例 (与 full_report 一致)
    "valid_wavelength_range": (430.0, 660.0),  # 有效波段硬掩码 (与报告 Effective Band 一致)
    "smooth_window": 21,                 # 平滑窗 (v1.4.5 流程中设为 1 可完全关闭平滑)
}

# =============================================================================
# UI 主题配置 (Theme Configuration)
# =============================================================================
THEME_ENABLED = True

PIXEL_FONT = {
    "path": "assets/pixel.ttf",
    "family": "Press Start 2P",
}