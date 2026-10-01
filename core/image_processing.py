# core/image_processing.py
import re
import cv2
import numpy as np
import tifffile as tiff
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 【新增】导入类型定义以消除 Pylance 警告
from .calibration import KnownWavelength, PeakResult

np_trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

np_trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

def get_exposure_time(path: str | Path) -> float:
    try:
        with tiff.TiffFile(str(path)) as tif:
            page = tif.pages[0]
            if 33434 in page.tags:
                val = page.tags[33434].value
                if isinstance(val, tuple) and len(val) == 2:
                    return float(val[0]) / float(val[1]) if val[1] != 0 else 1.0
                return float(val)
    except Exception:
        pass
    return 1.0

def read_dng_linear_rgb(path: str | Path, black_level: float = 64.0) -> Tuple[np.ndarray, float]:
    path = Path(path)
    raw = tiff.imread(str(path)).astype(np.float32)
    raw = np.clip(raw - black_level, 0, None)
    exp_time = get_exposure_time(path)
    if exp_time <= 0: exp_time = 1.0
    raw = raw / exp_time
    max_val = np.max(raw)
    scale = 65535.0 / max_val if max_val > 0 else 1.0
    raw16 = np.clip(raw * scale, 0, 65535).astype(np.uint16)
    rgb = cv2.cvtColor(raw16, cv2.COLOR_BayerRG2RGB).astype(np.float32) / scale
    return rgb, exp_time

def normalize_for_display(rgb: np.ndarray, percentile: float = 99.7) -> np.ndarray:
    upper = float(np.percentile(rgb, percentile))
    if upper <= 0: upper = 1.0
    return np.clip(rgb / upper, 0, 1)

def moving_average(x: np.ndarray, window: int = 9) -> np.ndarray:
    if window <= 1: return x.astype(np.float64)
    if window % 2 == 0: window += 1
    padded = np.pad(x.astype(np.float64), pad_width=window//2, mode="edge")
    return np.convolve(padded, np.ones(window)/window, mode="valid")

def auto_detect_x_roi(rgb: np.ndarray) -> Tuple[int, int]:
    width = rgb.shape[1]
    x_profile = np.percentile(rgb.sum(axis=2), 99.5, axis=0)
    x_smooth = moving_average(x_profile, window=max(11, width // 300 * 2 + 1))
    base, peak = float(np.percentile(x_smooth, 20)), float(np.max(x_smooth))
    if peak <= base: return int(width * 0.35), int(width * 0.65)
    mask = x_smooth > (base + 0.35 * (peak - base))
    segments, start = [], None
    for i, val in enumerate(mask):
        if val and start is None: start = i
        if (not val or i == len(mask) - 1) and start is not None:
            segments.append((start, i if not val else i + 1)); start = None
    if not segments: return int(width * 0.35), int(width * 0.65)
    x0, x1 = max(segments, key=lambda s: s[1] - s[0])
    return int(max(0, x0 - 80)), int(min(width, x1 + 80))

def make_channel_profiles(rgb: np.ndarray, x_roi: Tuple[int, int]) -> Dict[str, np.ndarray]:
    x0, x1 = x_roi
    return {
        "red": rgb[:, x0:x1, 0].mean(axis=1),
        "green": rgb[:, x0:x1, 1].mean(axis=1),
        "blue": rgb[:, x0:x1, 2].mean(axis=1)
    }

def get_average_profiles(dng_paths: List[str], x_roi: Tuple[int, int]) -> Dict[str, np.ndarray]:
    if not dng_paths: return {"red": np.zeros(1), "green": np.zeros(1), "blue": np.zeros(1)}
    sum_profiles = {"red": None, "green": None, "blue": None}
    for path in dng_paths:
        rgb, _ = read_dng_linear_rgb(path)
        profs = make_channel_profiles(rgb, x_roi)
        for k in sum_profiles:
            if sum_profiles[k] is None: sum_profiles[k] = profs[k].copy()
            else: sum_profiles[k] += profs[k]
    n = len(dng_paths)
    return {k: v / n for k, v in sum_profiles.items()}

def find_peak_auto(profile: np.ndarray, known) -> 'PeakResult':
    from .calibration import PeakResult
    n = len(profile)
    smooth = moving_average(profile, 11)
    valid_lo, valid_hi = 50, n - 50
    baseline = float(np.percentile(smooth[valid_lo:valid_hi], 10))
    corrected = np.clip(smooth - baseline, 0, None)
    y_max = int(np.argmax(corrected[valid_lo:valid_hi]) + valid_lo)
    peak_height = float(corrected[y_max])
    if peak_height <= 0: raise RuntimeError(f"No positive peak for {known.name}")
    half = 0.5 * peak_height
    left, right = y_max, y_max
    while left > valid_lo and corrected[left] > half: left -= 1
    while right < valid_hi - 1 and corrected[right] > half: right += 1
    lo, hi = max(valid_lo, left - 45), min(valid_hi, right + 46)
    ys = np.arange(lo, hi, dtype=np.float64)
    weights = np.clip(profile[lo:hi] - baseline, 0, None)
    if weights.sum() <= 0: raise RuntimeError(f"Centroid failed for {known.name}")
    y_centroid = float((ys * weights).sum() / weights.sum())
    return PeakResult(known.name, known.wavelength_nm, known.channel_name, y_centroid, y_max, baseline, lo, hi, peak_height)

def parse_known_wavelengths(file_path: Optional[str | Path]) -> Dict[str, 'KnownWavelength']:
    from .calibration import KnownWavelength
    defaults = {
        "R": KnownWavelength("R", 622.5, 2.5, 0, "red"),
        "G": KnownWavelength("G", 522.5, 2.5, 1, "green"),
        "B": KnownWavelength("B", 462.5, 2.5, 2, "blue")
    }
    if file_path is None: return defaults
    text = Path(file_path).read_text("utf-8", errors="ignore")
    pattern = re.compile(r"^\s*([RGB])\s*:\s*([0-9]+\.?[0-9]*)\s*nm\s*[+-±/]+\s*([0-9]+\.?[0-9]*)\s*nm", re.IGNORECASE | re.MULTILINE)
    found = {}
    for match in pattern.finditer(text):
        c = match.group(1).upper()
        found[c] = KnownWavelength(c, float(match.group(2)), float(match.group(3)), {"R":0, "G":1, "B":2}[c], {"R":"red", "G":"green", "B":"blue"}[c])
    for c in ("R", "G", "B"):
        if c not in found: found[c] = defaults[c]
    return found

def load_true_spd_data(file_path: Optional[str | Path]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if file_path is None: return None, None
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1, encoding='utf-8')
    data = data[~np.isnan(data).any(axis=1)]
    if data.ndim == 2 and data.shape[1] >= 2: return data[:, 0], data[:, 1]
    raise ValueError("SPD CSV format error.")