# core/image_processing.py
import re
from dataclasses import dataclass
import numpy as np
import tifffile as tiff
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 【新增】导入类型定义以消除 Pylance 警告
from .calibration import KnownWavelength, PeakResult
from config import DEFAULT_PARAMS

np_trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

np_trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

DEFAULT_BLACK_LEVEL = float(DEFAULT_PARAMS["black_level"])

# ---------------------------------------------------------------------------
# Android (MycoSpec v2.2.6) parity constants.
# Source of truth: app/src/main/java/org/fungalsentinel/app/analysis/
#     FrameProfileExtractor.kt  (DEFAULT_CFA :39, crosstalk :304-305)
#     WavelengthCalibrator.kt   (quality bands :99)
# ---------------------------------------------------------------------------
# 0=R, 1=G, 2=B -- identical convention on both ends.
DEFAULT_CFA = (0, 1, 1, 2)          # RGGB fallback, == Android DEFAULT_CFA
CROSSTALK_G_FROM_R = 0.15           # g = max(0, g - 0.15 * r)
CROSSTALK_R_FROM_G = 0.12           # r = max(0, r - 0.12 * g)  (uses the *corrected* g)
ORTHOGONAL_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAGONAL_OFFSETS = ((-1, -1), (1, -1), (-1, 1), (1, 1))
ROI_MARGIN_PX = 80                  # autoDetectXRoi :138
QUALITY_PASS_NM = 2.5               # WavelengthCalibrator :99
QUALITY_WARNING_NM = 10.0


def _rational_to_float(value) -> float:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        numerator, denominator = value
        denominator = float(denominator)
        if denominator == 0:
            raise ValueError("Invalid TIFF rational with a zero denominator")
        return float(numerator) / denominator
    return float(value)


def _black_level_from_page(page) -> float:
    tag = page.tags.get(50714)
    if tag is None:
        return DEFAULT_BLACK_LEVEL

    values = tag.value
    if tag.dtype == 5:
        flattened = np.asarray(values, dtype=np.float64).reshape(-1)
        if flattened.size >= 2 and flattened.size % 2 == 0:
            denominators = flattened[1::2]
            valid = denominators != 0
            if np.any(valid):
                return float(np.mean(flattened[::2][valid] / denominators[valid]))
        raise ValueError("Invalid DNG BlackLevel rational values")

    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    if not flattened.size:
        raise ValueError("DNG BlackLevel tag is empty")
    return float(np.mean(flattened))


def get_exposure_time(path: str | Path) -> float:
    with tiff.TiffFile(str(path)) as tif:
        page = tif.pages[0]
        exposure_tag = page.tags.get(33434)
        exposure = exposure_tag.value if exposure_tag is not None else None
        if exposure is None:
            exif_tag = page.tags.get(34665)
            exif = exif_tag.value if exif_tag is not None else {}
            if isinstance(exif, dict):
                exposure = exif.get("ExposureTime")
        if exposure is None:
            return 1.0
        seconds = _rational_to_float(exposure)
        if seconds <= 0:
            raise ValueError(f"Invalid exposure time in {path}")
        return seconds

def _cfa_pattern_from_page(page) -> Tuple[int, int, int, int]:
    """Read DNG tag 33422 (CFAPattern) with Android's normalisation rules.

    Mirrors ``FrameProfileExtractor.TiffReader.cfaPattern`` :483-500: the raw
    bytes are accepted when they are a 0..2 or a 1..3 permutation of one R, two
    G and one B; anything else falls back to ``DEFAULT_CFA`` (RGGB).
    """
    tag = page.tags.get(33422)
    if tag is None:
        return DEFAULT_CFA
    value = tag.value
    try:
        if isinstance(value, (bytes, bytearray)):
            # Android reads the four tag bytes verbatim (:489)
            raw = np.frombuffer(bytes(value[:4]), dtype=np.uint8).astype(np.int64)
        elif isinstance(value, str):
            raw = np.frombuffer(value.encode("latin-1")[:4], dtype=np.uint8).astype(np.int64)
        else:
            raw = np.asarray(value, dtype=np.int64).reshape(-1)
    except (TypeError, ValueError):
        return DEFAULT_CFA
    if raw.size < 4:
        return DEFAULT_CFA
    raw = raw[:4]
    low, high = int(raw.min()), int(raw.max())
    if low == 0 and high == 2:
        norm = raw
    elif low == 1 and high == 3:
        norm = raw - 1
    else:
        return DEFAULT_CFA
    if sorted(int(v) for v in norm) != [0, 1, 1, 2]:
        return DEFAULT_CFA
    return tuple(int(v) for v in norm)


def _normalise_samples(samples: np.ndarray, black_level: float,
                       exposure: float) -> np.ndarray:
    """``max(0, (sample & 0xFFFF) - black) / exposure`` exactly as Android does."""
    values = np.asarray(samples).astype(np.uint32) & 0xFFFF
    return np.clip(values.astype(np.float64) - float(black_level), 0.0, None) / float(exposure)


def _cfa_index_map(height: int, width: int,
                   cfa: Tuple[int, int, int, int]) -> np.ndarray:
    """``cfa[(y & 1) * 2 + (x & 1)]`` for every pixel."""
    pattern = np.array([[cfa[0], cfa[1]], [cfa[2], cfa[3]]], dtype=np.int8)
    return np.tile(pattern, (height // 2 + 1, width // 2 + 1))[:height, :width]


def _neighbor_stats(values: np.ndarray, cfa_map: np.ndarray, target: int,
                    offsets) -> Tuple[np.ndarray, np.ndarray]:
    """Sum / count of the neighbours whose CFA colour equals ``target``."""
    height, width = values.shape
    total = np.zeros((height, width), dtype=np.float64)
    count = np.zeros((height, width), dtype=np.int32)
    for dx, dy in offsets:
        # out[y, x] <- in[y + dy, x + dx], out-of-bounds neighbours skipped
        ys_dst0, ys_dst1 = max(0, -dy), min(height, height - dy)
        xs_dst0, xs_dst1 = max(0, -dx), min(width, width - dx)
        if ys_dst1 <= ys_dst0 or xs_dst1 <= xs_dst0:
            continue
        source = values[ys_dst0 + dy:ys_dst1 + dy, xs_dst0 + dx:xs_dst1 + dx]
        matches = cfa_map[ys_dst0 + dy:ys_dst1 + dy, xs_dst0 + dx:xs_dst1 + dx] == target
        total[ys_dst0:ys_dst1, xs_dst0:xs_dst1] += np.where(matches, source, 0.0)
        count[ys_dst0:ys_dst1, xs_dst0:xs_dst1] += matches.astype(np.int32)
    return total, count


def _demosaic_channel(values: np.ndarray, cfa_map: np.ndarray, target: int) -> np.ndarray:
    """``FrameProfileExtractor.bilinearInterpolate`` :317-372, vectorised.

    * the pixel already carries the target colour -> use it directly;
    * green -> mean of the four orthogonal neighbours that are green;
    * red / blue -> mean of the four diagonal neighbours of that colour, and
      only when none of them matches, the orthogonal ones.
    """
    total_o, count_o = _neighbor_stats(values, cfa_map, target, ORTHOGONAL_OFFSETS)
    mean_o = np.where(count_o > 0, total_o / np.maximum(count_o, 1), 0.0)
    if target == 1:
        estimate = mean_o
    else:
        total_d, count_d = _neighbor_stats(values, cfa_map, target, DIAGONAL_OFFSETS)
        mean_d = np.where(count_d > 0, total_d / np.maximum(count_d, 1), 0.0)
        estimate = np.where(count_d > 0, mean_d, mean_o)
    return np.where(cfa_map == target, values, estimate)


def demosaic_bayer(samples: np.ndarray, black_level: float, exposure: float,
                   cfa: Tuple[int, int, int, int] = DEFAULT_CFA) -> np.ndarray:
    """Full-frame demosaic + the two-step channel-crosstalk correction."""
    values = _normalise_samples(samples, black_level, exposure)
    height, width = values.shape
    cfa_map = _cfa_index_map(height, width, cfa)
    red = _demosaic_channel(values, cfa_map, 0)
    green = _demosaic_channel(values, cfa_map, 1)
    blue = _demosaic_channel(values, cfa_map, 2)
    # 【Android 对齐】顺序敏感：先扣 g，再用**已校正的 g** 去扣 r
    green = np.maximum(green - CROSSTALK_G_FROM_R * red, 0.0)
    red = np.maximum(red - CROSSTALK_R_FROM_G * green, 0.0)
    return np.stack((red, green, blue), axis=-1).astype(np.float32)


def column_energy(samples: np.ndarray, black_level: float,
                  exposure: float) -> np.ndarray:
    """``colE``: per-column TopK 99.5th percentile computed on the RAW grid.

    Mirrors ``profileFromDng`` :258-268 -- the percentiles are taken **before**
    demosaicing, on ``max(0, sample - black) / exposure``.
    """
    values = _normalise_samples(samples, black_level, exposure)
    height, width = values.shape
    k_top = max(4, int(height * 0.005) + 4)
    index = min(k_top - 1, int(0.005 * height))
    index = min(index, max(0, height - 1))
    if k_top >= height:
        ordered = np.sort(values, axis=0)[::-1, :]
    else:
        top = np.partition(values, -k_top, axis=0)[-k_top:, :]
        ordered = np.sort(top, axis=0)[::-1, :]
    return ordered[index].astype(np.float32)


def auto_detect_x_roi(col_profile: np.ndarray, width: int) -> Tuple[int, int]:
    """``autoDetectXRoi`` :117-139.  Returns a half-open ``(start, end)`` pair.

    Android works on an inclusive ``(rs, re)`` pair; this module keeps the
    half-open slice convention, so the caller gets ``(rs, re + 1)``.
    """
    width = int(width)
    if width <= 0:
        return 0, 0
    profile = np.asarray(col_profile, dtype=np.float64).reshape(-1)
    if profile.size == 0 or profile.size != width:
        return _center_roi(width)
    window = max(11, width // 300 * 2 + 1)
    smoothed = moving_average(profile, window=window)
    ordered = np.sort(smoothed)
    base = float(ordered[min(int(smoothed.size * 0.20), smoothed.size - 1)])
    peak = float(np.max(smoothed))
    if peak <= base:
        return _center_roi(width)
    threshold = base + 0.35 * (peak - base)
    segments, start = [], None
    for i, value in enumerate(smoothed):
        if value > threshold and start is None:
            start = i
        if (value <= threshold or i == smoothed.size - 1) and start is not None:
            segments.append((start, i if value <= threshold else i + 1))
            start = None
    if not segments:
        return _center_roi(width)
    longest = max(segments, key=lambda segment: segment[1] - segment[0])
    roi_start = int(max(0, longest[0] - ROI_MARGIN_PX))
    roi_end = int(min(width - 1, longest[1] - 1 + ROI_MARGIN_PX)) + 1
    return roi_start, max(roi_start + 1, roi_end)


def _center_roi(width: int) -> Tuple[int, int]:
    """``centerRoi`` :100-102, expressed as a half-open slice pair."""
    start = int(min(max(int(width * 0.35), 0), max(0, width - 1)))
    end = int(min(max(int(width * 0.65), 0), max(0, width - 1)))
    if end <= start:
        end = min(start + 1, width)
    return start, end


def resolve_x_roi(col_profile: Optional[np.ndarray], width: int,
                  fixed_x_roi: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    """``resolveXRoi`` :141-152 -- an explicit ROI always wins."""
    if fixed_x_roi is not None and width > 0:
        start = int(min(max(int(fixed_x_roi[0]), 0), width - 1))
        end = int(min(max(int(fixed_x_roi[1]), 0), width - 1))
        if start <= end:
            return start, end + 1      # Android's pair is inclusive
    return auto_detect_x_roi(col_profile if col_profile is not None else np.zeros(0), width)


def read_dng_frame(path: str | Path, black_level: Optional[float] = None,
                   fixed_x_roi: Optional[Tuple[int, int]] = None) -> Dict:
    """Android ``FrameProfileExtractor.profileFromDng`` :242-315 equivalent.

    Returns a dict with ``rgb`` (H, W, 3 float32), ``exposure``, ``colE``,
    ``x_roi`` (half-open), ``cfa``, ``black_level``, ``width``, ``height``.
    When ``fixed_x_roi`` is supplied the column-energy scan is skipped entirely,
    exactly like ``resolveXRoi``.
    """
    path = Path(path)
    with tiff.TiffFile(str(path)) as tif:
        page = tif.pages[0]
        if black_level is None:
            black_level = _black_level_from_page(page)
        cfa = _cfa_pattern_from_page(page)
    exposure = get_exposure_time(path)
    if exposure <= 0:
        exposure = 1.0

    samples = np.asarray(tiff.imread(str(path)))
    if samples.ndim == 3:                       # already demosaiced -- no CFA to honour
        samples = samples.mean(axis=2)
    height, width = samples.shape[:2]

    col_energy = None
    if fixed_x_roi is None:
        col_energy = column_energy(samples, black_level, exposure)
    x_roi = resolve_x_roi(col_energy, width, fixed_x_roi)

    return {
        "rgb": demosaic_bayer(samples, black_level, exposure, cfa),
        "exposure": float(exposure),
        "colE": col_energy,
        "x_roi": x_roi,
        "cfa": cfa,
        "black_level": float(black_level),
        "width": int(width),
        "height": int(height),
    }


def read_dng_linear_rgb(path: str | Path, black_level: Optional[float] = None,
                        x_roi: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, float]:
    """Decode a DNG into a linear (black-subtracted, exposure-normalised) RGB image.

    This is the single decode path used for both the preview and every
    calculation, so the picture on screen and the numbers in the report can
    never disagree.  Passing ``x_roi`` skips the column-energy scan.
    """
    frame = read_dng_frame(path, black_level=black_level, fixed_x_roi=x_roi)
    return frame["rgb"], frame["exposure"]

def normalize_for_display(rgb: np.ndarray, percentile: float = 99.7) -> np.ndarray:
    upper = float(np.percentile(rgb, percentile))
    if upper <= 0: upper = 1.0
    return np.clip(rgb / upper, 0, 1)

def moving_average(x: np.ndarray, window: int = 9) -> np.ndarray:
    if window <= 1: return x.astype(np.float64)
    if window % 2 == 0: window += 1
    padded = np.pad(x.astype(np.float64), pad_width=window//2, mode="edge")
    return np.convolve(padded, np.ones(window)/window, mode="valid")

def make_channel_profiles(rgb: np.ndarray, x_roi: Tuple[int, int]) -> Dict[str, np.ndarray]:
    x0, x1 = x_roi
    return {
        "red": rgb[:, x0:x1, 0].mean(axis=1),
        "green": rgb[:, x0:x1, 1].mean(axis=1),
        "blue": rgb[:, x0:x1, 2].mean(axis=1)
    }

def apply_rb_swap(profiles: Dict[str, np.ndarray],
                  swapped: bool) -> Dict[str, np.ndarray]:
    """Android ``forceRbSwap`` (:82-83): relabel R and B on every later frame.

    Once Step 1 has decided the channel labels are inverted, every subsequent
    decode has to be relabelled too -- otherwise Step 2 would bake the response
    curves under the wrong channel names.
    """
    if not swapped:
        return profiles
    return {"red": profiles["blue"], "green": profiles["green"],
            "blue": profiles["red"]}


def get_average_profiles(dng_paths: List[str], x_roi: Optional[Tuple[int, int]]) -> Dict[str, np.ndarray]:
    """Average the R/G/B row profiles of several DNGs over ``x_roi``.

    ``x_roi`` is honoured verbatim (``resolveXRoi``'s ``fixedXRegion`` branch) --
    the ROI is never re-detected per frame, which is what makes a preset
    reusable across sessions.
    """
    if not dng_paths: return {"red": np.zeros(1), "green": np.zeros(1), "blue": np.zeros(1)}
    sum_profiles = {"red": None, "green": None, "blue": None}
    for path in dng_paths:
        rgb, _ = read_dng_linear_rgb(path, x_roi=x_roi)
        x0, x1 = resolve_x_roi(None, rgb.shape[1], x_roi)
        profs = make_channel_profiles(rgb, (x0, x1))
        for k in sum_profiles:
            if sum_profiles[k] is None: sum_profiles[k] = profs[k].copy()
            else: sum_profiles[k] += profs[k]
    n = len(dng_paths)
    return {k: v / n for k, v in sum_profiles.items()}


# ---------------------------------------------------------------------------
# Step 1: two-point fit + the R/B swap self-check (Android WavelengthCalibrator
# :42-62).  The R/B labels of a Bayer sensor are only knowable from the CFA tag,
# and a wrong tag shows up as R and B sitting on the wrong sides of G.
# ---------------------------------------------------------------------------
def _fit_from_peaks(peak_r, peak_g, peak_b) -> Dict[str, float]:
    k = (peak_r.y_centroid - peak_b.y_centroid) / (
        peak_r.wavelength_nm - peak_b.wavelength_nm)
    b = peak_b.y_centroid - k * peak_b.wavelength_nm
    predicted = k * peak_g.wavelength_nm + b
    residual = peak_g.y_centroid - predicted
    error_nm = residual / abs(k) if k != 0 else float("nan")
    quality = ("PASS" if abs(error_nm) <= QUALITY_PASS_NM
               else "WARNING" if abs(error_nm) <= QUALITY_WARNING_NM else "FAILED")
    return {"k": k, "b": b, "predicted": predicted, "residual": residual,
            "error_nm": error_nm, "quality": quality}


def calibrate_wavelength(profiles: Dict[str, np.ndarray],
                         knowns: Dict[str, 'KnownWavelength']) -> Optional[Dict]:
    """Fit px = k*λ + b from the B/R peaks, validate on G, and self-check R/B order.

    Returns ``None`` when no usable peak triple exists in either swap state.
    """
    def _attempt(swapped: bool):
        red = profiles["blue"] if swapped else profiles["red"]
        blue = profiles["red"] if swapped else profiles["blue"]
        try:
            peak_r = find_peak_auto(red, knowns["R"])
            peak_g = find_peak_auto(profiles["green"], knowns["G"])
            peak_b = find_peak_auto(blue, knowns["B"])
        except RuntimeError:
            return None
        fit = _fit_from_peaks(peak_r, peak_g, peak_b)
        fit.update({"swapped": swapped, "peaks": (peak_r, peak_g, peak_b)})
        return fit

    swapped = False
    try:
        peak_r = find_peak_auto(profiles["red"], knowns["R"])
        peak_g = find_peak_auto(profiles["green"], knowns["G"])
        peak_b = find_peak_auto(profiles["blue"], knowns["B"])
        # R and B on the wrong sides of G -> the channel labels are swapped
        swapped = bool(peak_r.y_centroid < peak_g.y_centroid
                       and peak_b.y_centroid > peak_g.y_centroid)
    except RuntimeError:
        pass

    result = _attempt(swapped)
    if result is None or result["quality"] == "FAILED":
        alternative = _attempt(not swapped)
        if alternative is not None and (
                result is None or abs(alternative["error_nm"]) < abs(result["error_nm"])):
            result = alternative
    return result

@dataclass
class PeakResult:
    name: str
    wavelength_nm: float
    channel_name: str
    y_centroid: float
    y_max: int
    baseline: float
    lo: int
    hi: int
    peak_height: float
    # 【v1.4.5 新增，带默认值，旧调用不受影响】
    estimator: str = "gauss"          # "gauss" | "edges"
    n_maxima: int = 1
    quality: str = "good"             # "good" | "suspect"
    centroid_legacy: float = 0.0      # 旧质心，仅诊断用


def find_peak_auto(profile: np.ndarray, known) -> PeakResult:
    n = len(profile)
    smooth = moving_average(profile, 11)
    valid_lo, valid_hi = 50, n - 50
    baseline = float(np.percentile(smooth[valid_lo:valid_hi], 10))
    corrected = np.clip(smooth - baseline, 0, None)
    y_max = int(np.argmax(corrected[valid_lo:valid_hi]) + valid_lo)
    peak_height = float(corrected[y_max])
    if peak_height <= 0:
        raise RuntimeError(f"No positive peak for {known.name}")
    half = 0.5 * peak_height

    # 1) 半高边（峰核边界）
    left, right = y_max, y_max
    while left > valid_lo and corrected[left] > half: left -= 1
    while right < valid_hi - 1 and corrected[right] > half: right += 1
    fwhm_lo, fwhm_hi = left, right
    fw = max(1, fwhm_hi - fwhm_lo)

    # 2) 【改进】谷界窗口：向外走到"重新上升"处即停，上限 1.5*FWHM+10
    def valley_bound(start, step):
        i, limit = start, start + step * int(1.5 * fw + 10)
        while valid_lo < i < valid_hi - 1 and i != limit:
            if corrected[i + step] >= corrected[i]:   # 剖面开始回升 => i 是谷
                break
            i += step
        return i
    lo = max(valid_lo, valley_bound(fwhm_lo, -1))
    hi = min(valid_hi, valley_bound(fwhm_hi, +1) + 1)

    # 3) 单峰性检查（窗内 >20% 峰高的局部极大个数）
    win = corrected[lo:hi]
    thr = 0.2 * peak_height
    n_max = sum(1 for i in range(1, len(win) - 1)
                if win[i] >= thr and win[i] >= win[i - 1] and win[i] > win[i + 1])

    # 4) 估计量：半高区 log-抛物线(高斯)顶点；失败/出界则用半高边中点
    g = np.arange(fwhm_lo, fwhm_hi + 1, dtype=np.float64)
    v = np.clip(corrected[fwhm_lo:fwhm_hi + 1], 1e-9, None)
    gauss_c = edge_c = 0.5 * (fwhm_lo + fwhm_hi)
    estimator = "edges"
    if len(g) >= 5:
        coef = np.polyfit(g, np.log(v), 2)
        if coef[0] < -1e-12:
            cand = float(-coef[1] / (2 * coef[0]))
            if fwhm_lo <= cand <= fwhm_hi:
                gauss_c, estimator = cand, "gauss"
    y_center = gauss_c if estimator == "gauss" else edge_c

    # 5) 一致性校验 + 旧质心(诊断)
    ys = np.arange(lo, hi, dtype=np.float64)
    w = np.clip(profile[lo:hi] - baseline, 0, None)
    centroid_legacy = float((ys * w).sum() / w.sum()) if w.sum() > 0 else float(y_max)
    suspect = (n_max > 1) or (abs(y_center - edge_c) > 0.25 * fw)

    return PeakResult(
        known.name, known.wavelength_nm, known.channel_name,
        float(y_center), y_max, baseline, lo, hi, peak_height,
        estimator=estimator, n_maxima=int(n_max),
        quality="suspect" if suspect else "good",
        centroid_legacy=centroid_legacy,
    )

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