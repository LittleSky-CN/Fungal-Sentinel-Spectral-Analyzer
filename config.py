# config.py
"""FSSA v1.5.0 Global Configuration.

【v1.5.0】新增 Fluorescein (sodium salt) 荧光素选项（发射 520 nm / 激发 494 nm /
green 通道 / FWHM 25）；荧光素标识统一为 canonical ID（FLUOROPHORES 字典键改为
YPET/EGFP/FLUORESCEIN/MCHERRY/CFP/MTURQUOISE2，新增 canonical_fluorophore()
归一化解析），以便与 Android 端曲线预设互通。
"""

# =============================================================================
# 荧光蛋白/目标物数据库 (Fluorophore Database)
# =============================================================================
# key 是 canonical ID（与 Android MycoSpec 的 enum name 一致），display_name 是
# Python 端显示给用户、且写进存量 preset 的历史写法。preset JSON 一律存 canonical。
FLUOROPHORES = {
    "YPET":         {"display_name": "Ypet (Venus variant)",      "peak_wl": 527, "channel": "green", "fwhm": 20, "color": "green"},
    "EGFP":         {"display_name": "EGFP",                      "peak_wl": 509, "channel": "green", "fwhm": 20, "color": "green"},
    # 荧光素钠：发射 ~520 nm（绿通道），激发 ~494 nm。fwhm 是积分窗全宽
    # （peak ± fwhm/2）；实际发射峰较宽（~40 nm），25 为先与其他选项保持
    # 同档的保守值，需要更宽积分窗时调这一个数字即可。
    "FLUORESCEIN": {
        "display_name": "Fluorescein (sodium salt)",
        "peak_wl": 520, "channel": "green", "fwhm": 25, "color": "green",
        "excitation_nm": 494,   # 元数据：界面无激发波长字段，先随配置记录
    },
    "MCHERRY":      {"display_name": "mCherry",                   "peak_wl": 610, "channel": "red",   "fwhm": 25, "color": "red"},
    "CFP":          {"display_name": "CFP",                       "peak_wl": 475, "channel": "blue",  "fwhm": 20, "color": "blue"},
    "MTURQUOISE2":  {"display_name": "mTurquoise2",               "peak_wl": 482, "channel": "blue",  "fwhm": 20, "color": "cyan"},
}

# 显式别名表：canonical ID -> 所有已知写法（大小写/空格/连字符差异由归一化吸收，
# 但"哪些字符串算同一个荧光素"必须逐条列全，不做模糊匹配，避免误判）。
_FLUOROPHORE_ALIASES = {
    "YPET": (
        "YPET", "ypet", "Ypet", "YPet",
        "Ypet (Venus variant)", "Ypet (venus variant)", "YPET (Venus variant)",
    ),
    "EGFP": (
        "EGFP", "egfp", "EGFP ", "EGFP (enhanced GFP)", "enhanced GFP", "Enhanced GFP",
    ),
    "FLUORESCEIN": (
        "FLUORESCEIN", "fluorescein", "Fluorescein",
        "Fluorescein (sodium salt)", "fluorescein (sodium salt)",
        "Fluorescein sodium salt", "fluorescein sodium salt",
        "sodium fluorescein", "Sodium fluorescein", "Fluorescein Na",
    ),
    "MCHERRY": (
        "MCHERRY", "mCherry", "mcherry", "m-cherry", "m-Cherry", "m Cherry", "mCherry ",
    ),
    "CFP": (
        "CFP", "cfp", "Cfp", "CFP (cyan fluorescent protein)", "cyan fluorescent protein",
    ),
    "MTURQUOISE2": (
        "MTURQUOISE2", "mturquoise2", "mTurquoise2", "MTurquoise2",
        "m-turquoise2", "mTurquoise-2", "mTurquoise 2", "mTurquoise2 ",
    ),
}


def _normalize_fluorophore_name(name: str) -> str:
    """归一化：小写 + 去掉空格/连字符/下划线，用于大小写与分隔符不敏感的比对。"""
    return "".join(ch for ch in str(name).strip().lower() if ch not in " \t-_")


# 归一化后的别名 -> canonical ID。canonical key 与 display_name 一并登记，
# 这样"旧 Python 显示名"这类存量 preset 里的写法也能解析。
_FLUOROPHORE_LOOKUP = {}
for _cid, _aliases in _FLUOROPHORE_ALIASES.items():
    for _alias in (*_aliases, _cid, FLUOROPHORES[_cid]["display_name"]):
        _FLUOROPHORE_LOOKUP[_normalize_fluorophore_name(_alias)] = _cid


def canonical_fluorophore(name: str | None) -> str | None:
    """把任意写法解析成 canonical ID；无法识别返回 None。

    识别顺序：canonical 精确匹配 -> 归一化匹配（别名表 + display_name）。
    大小写、空格、连字符/下划线差异一律不敏感。
    """
    if name is None or not isinstance(name, str):
        return None
    if name in FLUOROPHORES:
        return name
    key = _normalize_fluorophore_name(name)
    return _FLUOROPHORE_LOOKUP.get(key) if key else None


def fluorophore_display_name(canonical_id: str | None) -> str | None:
    """canonical ID -> 本地化显示名；未知 ID 原样返回（界面宁可显示原始串也不崩）。"""
    if canonical_id is None:
        return None
    return FLUOROPHORES.get(canonical_id, {}).get("display_name", canonical_id)

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