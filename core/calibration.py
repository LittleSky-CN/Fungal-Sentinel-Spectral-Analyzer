# core/calibration.py
"""Wavelength mapping and Spectral Power Distribution (SPD) calibration.
【v1.5.0】荧光素标识统一为 canonical ID，APP_VERSION 升至 1.5.0，与 Android 端曲线预设互通。
v1.4.3 simplification: per-channel absolute response, NO Savitzky-Golay, NO self-normalization.

【SPD payload v3】响应曲线不再以像素为索引轴：``calibrate()`` 仍在逐像素域做除法，
但输出一律重采样到规范波长网格（380-780 nm，1 nm 步长，401 点）。因此 preset 与
任何 mapping 完全无关，Mode B（按当前 mapping 重算）与其依赖的像素域载荷
``lamp_*`` / ``spd_reference`` 一并删除。``slope`` / ``intercept`` /
``sensor_height`` 降级为纯诊断信息，绝不参与任何曲线重建。
"""
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple
import numpy as np
from scipy.interpolate import interp1d
from config import DEFAULT_PARAMS

# ---------------------------------------------------------------------------
# Tiered checks applied when an SPD preset loads (L0 version gate / L1 structure
# / L2 coverage / L3 drift).  The grid constants and the L3 thresholds are
# duplicated on the Android side (MycoSpec
# analysis/SpectralResponseCalibrator.kt); keep both in sync.
# ---------------------------------------------------------------------------
APP_VERSION = "1.5.0"
SPD_SCHEMA_VERSION = 3
PRESET_TYPE_SPD = "SPD"
SPD_GRID_START_NM = 380.0
SPD_GRID_STOP_NM = 780.0
SPD_GRID_STEP_NM = 1.0
AXIS_KIND_UNIFORM_WAVELENGTH = "uniform_wavelength"
DRIFT_WARN_NM = 2.0          # below this: silent.  Real Step 1 G-residual is ~5.5 nm.
DRIFT_SEVERE_NM = 10.0       # above this: orange warning + confirm.  Never a rejection.
DRIFT_FRACS = (0.10, 0.30, 0.50, 0.70, 0.90)
MIN_VALID_POINTS = 20        # L1: a response curve with fewer live points is unusable
COVERAGE_TOL_NM = 0.5        # L2: slack for the grid endpoints
# L1: 可选数组键（Android 会写，Python 不写）；存在则长度必须 == grid_points
SPD_OPTIONAL_ARRAYS = ("pos_r", "pos_g", "pos_b", "corr_r", "corr_g", "corr_b",
                       "weight_b", "weight_g", "weight_r", "fused")
# L1: 可选标量键（不校验长度）
SPD_OPTIONAL_SCALARS = ("crossing_bg", "crossing_gr", "agreement_pct")
# L1: 可选二元组键 [value, wavelength_nm]
SPD_OPTIONAL_PEAKS = ("peak_r", "peak_g", "peak_b", "fused_peak")


def drift_level(drift_nm_value) -> str:
    """Classify a mapping drift: ``OK`` / ``WARN`` / ``SEVERE`` / ``UNKNOWN``."""
    if drift_nm_value is None:
        return "UNKNOWN"
    try:
        value = float(drift_nm_value)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if not math.isfinite(value):
        return "UNKNOWN"
    if value <= DRIFT_WARN_NM:
        return "OK"
    if value <= DRIFT_SEVERE_NM:
        return "WARN"
    return "SEVERE"


def drift_nm(mapping_a, mapping_b, n_px, fracs=DRIFT_FRACS) -> Optional[float]:
    """``max |lambda_b(px) - lambda_a(px)|`` in nm over ``n_px`` sensor rows.

    ``None`` when either mapping is missing or the profile length is unknown.
    """
    if mapping_a is None or mapping_b is None or n_px is None:
        return None
    n_px = int(n_px)
    if n_px < 2:
        return None
    span = float(n_px - 1)
    worst = 0.0
    for frac in fracs:
        pixel = frac * span
        delta = mapping_b.pixel_to_wavelength(pixel) - mapping_a.pixel_to_wavelength(pixel)
        worst = max(worst, abs(float(delta)))
    return worst


def canonical_grid() -> np.ndarray:
    """The canonical wavelength axis an SPD preset is stored on.

    ``np.linspace`` (not ``np.arange``): ``arange`` accumulates float error and
    can miss the 780.0 endpoint, which would make the grid disagree with the
    ``axis`` block and fail L1 on a round trip.
    """
    count = int(round((SPD_GRID_STOP_NM - SPD_GRID_START_NM) / SPD_GRID_STEP_NM)) + 1
    return np.linspace(SPD_GRID_START_NM, SPD_GRID_STOP_NM, count, dtype=np.float64)


def resample_to_grid(wl_src: np.ndarray, values: np.ndarray,
                     grid: Optional[np.ndarray] = None) -> np.ndarray:
    """Resample pixel-domain values onto the canonical wavelength grid.

    Linear interpolation, matching the ``interp1d(bounds_error=False,
    fill_value=0.0)`` convention already used for the reference SPD: anything
    outside the source wavelength span becomes exactly 0.
    """
    if grid is None:
        grid = canonical_grid()
    wl_src = np.asarray(wl_src, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    return np.interp(grid, wl_src, values, left=0.0, right=0.0)


def _opt_float(value):
    """JSON-safe optional float: ``None`` when missing or non-finite."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _opt_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validate_spd_structure(data: dict, wavelengths: np.ndarray,
                            responses: dict) -> None:
    """L1 -- structural checks.  Failure here means the file is unusable."""
    if wavelengths.ndim != 1 or wavelengths.size < 2:
        raise ValueError("SPD preset wavelengths must contain at least two points.")
    if not np.all(np.isfinite(wavelengths)):
        raise ValueError("SPD preset wavelengths must be finite.")
    # v3: strictly INCREASING only.  A descending grid used to be accepted here
    # and then silently mis-interpolated by np.interp in gui._interp_response.
    if not np.all(np.diff(wavelengths) > 0):
        raise ValueError("SPD preset wavelengths must be strictly monotonic.")

    axis = data.get("axis")
    if not isinstance(axis, dict):
        raise ValueError("SPD preset is missing its 'axis' block.")
    if axis.get("kind") != AXIS_KIND_UNIFORM_WAVELENGTH:
        raise ValueError(
            f"SPD preset axis kind must be '{AXIS_KIND_UNIFORM_WAVELENGTH}' "
            f"(got {axis.get('kind')!r}).")
    start, stop = _opt_float(axis.get("start_nm")), _opt_float(axis.get("stop_nm"))
    step = _opt_float(axis.get("step_nm"))
    if start is None or stop is None or step is None or step <= 0.0:
        raise ValueError("SPD preset axis must carry finite start_nm/stop_nm/step_nm.")
    expected = canonical_grid() if (math.isclose(start, SPD_GRID_START_NM)
                                    and math.isclose(stop, SPD_GRID_STOP_NM)
                                    and math.isclose(step, SPD_GRID_STEP_NM)) \
        else np.linspace(start, stop, int(round((stop - start) / step)) + 1)
    if expected.size != wavelengths.size or not np.allclose(
            wavelengths, expected, rtol=0.0, atol=1e-6):
        raise ValueError(
            "SPD preset wavelengths do not match the axis block "
            f"({wavelengths.size} points vs {expected.size} expected from "
            f"{start}-{stop} nm @ {step} nm).")

    grid_points = data.get("grid_points")
    if _opt_int(grid_points) != wavelengths.size:
        raise ValueError(
            f"SPD preset grid_points ({grid_points!r}) does not match "
            f"len(wavelengths) ({wavelengths.size}).")
    for channel, response in responses.items():
        if response.ndim != 1 or response.size != wavelengths.size:
            raise ValueError(
                f"SPD preset response_{channel} does not match its wavelength grid "
                f"({response.size} vs {wavelengths.size} grid points).")
        if not np.all(np.isfinite(response)):
            raise ValueError(f"SPD preset response_{channel} must be finite.")
    mask = data.get("mask")
    if mask is not None:
        mask = np.asarray(mask, dtype=np.float64)
        if mask.ndim != 1 or mask.size != wavelengths.size:
            raise ValueError(
                "SPD preset mask does not match its wavelength grid "
                f"({mask.size} vs {wavelengths.size} grid points).")
    for key in SPD_OPTIONAL_ARRAYS:
        optional = data.get(key)
        if optional is None:
            continue
        optional = np.asarray(optional, dtype=np.float64)
        if optional.ndim != 1 or optional.size != wavelengths.size:
            raise ValueError(
                f"SPD preset optional array '{key}' does not match its wavelength "
                f"grid ({optional.size} vs {wavelengths.size} grid points).")
    live = np.count_nonzero(
        (responses["r"] != 0) | (responses["g"] != 0) | (responses["b"] != 0))
    if live < MIN_VALID_POINTS:
        raise ValueError(
            f"SPD preset only has {live} calibrated point(s); "
            f"at least {MIN_VALID_POINTS} are required.")


def _validate_spd_coverage(mapping, n_px, wavelengths: np.ndarray, valid_range) -> None:
    """L2 -- coverage check.  ``np.interp`` clamps silently outside the grid."""
    if mapping is None or n_px is None or int(n_px) < 2:
        return
    pixels = np.arange(int(n_px), dtype=np.float64)
    current = np.asarray(mapping.pixel_to_wavelength(pixels), dtype=np.float64)
    if not np.all(np.isfinite(current)):
        raise ValueError("The active wavelength mapping produces non-finite wavelengths.")
    lo_now = max(float(np.min(current)), float(valid_range[0]))
    hi_now = min(float(np.max(current)), float(valid_range[1]))
    if lo_now > hi_now:
        return      # current axis never enters the calibrated band -> nothing to protect
    lo_preset, hi_preset = float(np.min(wavelengths)), float(np.max(wavelengths))
    if lo_now < lo_preset - COVERAGE_TOL_NM or hi_now > hi_preset + COVERAGE_TOL_NM:
        raise ValueError(
            "当前波长轴未被 preset 覆盖 "
            f"(preset {lo_preset:.1f}-{hi_preset:.1f} nm vs current "
            f"{lo_now:.1f}-{hi_now:.1f} nm); np.interp would clamp to the endpoint.")


def correct_by_response(signal: np.ndarray, response: np.ndarray,
                        floor_fraction: float = 0.20) -> np.ndarray:
    """Correct signal only where the calibrated channel response is reliable."""
    signal = np.asarray(signal, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    if signal.shape != response.shape:
        raise ValueError("Signal and response arrays must have matching shapes")
    if not np.all(np.isfinite(response)) or np.any(response < 0):
        raise ValueError("Response values must be finite and non-negative")
    if not 0.0 <= floor_fraction < 1.0:
        raise ValueError("Response floor fraction must be in [0, 1)")

    max_response = float(np.max(response)) if response.size else 0.0
    if max_response <= 0.0:
        raise ValueError("No positive calibrated response is available")

    floor = max(1e-9, max_response * floor_fraction)
    reliable = response > floor
    corrected = np.zeros_like(signal)
    np.divide(signal, response, out=corrected, where=reliable)
    return corrected


def remove_isolated_spikes(spectrum: np.ndarray, radius: int = 2,
                           sigma_threshold: float = 4.5,
                           minimum_fraction: float = 0.02) -> np.ndarray:
    """Replace isolated spectral outliers with the local neighbor median."""
    values = np.asarray(spectrum, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Spectrum must be one-dimensional")
    if radius < 1 or sigma_threshold <= 0 or minimum_fraction < 0:
        raise ValueError("Invalid isolated-spike filter parameters")
    if not np.all(np.isfinite(values)):
        raise ValueError("Spectrum values must be finite")
    if values.size < 2 * radius + 2:
        return values.copy()

    result = values.copy()
    amplitude_floor = float(np.max(np.abs(values))) * minimum_fraction
    for index in range(radius, values.size - radius):
        neighbors = np.concatenate((values[index - radius:index], values[index + 1:index + radius + 1]))
        local_median = float(np.median(neighbors))
        mad = float(np.median(np.abs(neighbors - local_median)))
        threshold = max(sigma_threshold * 1.4826 * mad, amplitude_floor)
        if abs(values[index] - local_median) > threshold:
            result[index] = local_median
    return result


def analyze_blank_subtracted_replicates(
    blank_profile: np.ndarray,
    sample_profiles: np.ndarray,
    wavelengths: np.ndarray,
    area_mask: np.ndarray,
    response: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Android Step 3 background, response-floor, spike, and area processing."""
    blank = np.asarray(blank_profile, dtype=np.float64)
    samples = np.asarray(sample_profiles, dtype=np.float64)
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    area_mask = np.asarray(area_mask, dtype=bool)
    if samples.ndim != 2 or samples.shape[0] == 0:
        raise ValueError("At least one one-dimensional sample profile is required")
    if blank.ndim != 1 or samples.shape[1] != blank.size:
        raise ValueError("Blank and sample profiles must have matching lengths")
    if wavelengths.shape != blank.shape or area_mask.shape != blank.shape:
        raise ValueError("Wavelengths and area mask must match the profile length")
    if not np.all(np.isfinite(blank)) or not np.all(np.isfinite(samples)):
        raise ValueError("Blank and sample profiles must be finite")
    if not np.all(np.isfinite(wavelengths)):
        raise ValueError("Wavelengths must be finite")
    if response is not None:
        response = np.asarray(response, dtype=np.float64)
        if response.shape != blank.shape:
            raise ValueError("Response must match the profile length")

    net_replicates = np.clip(samples - blank, 0.0, None)
    areas = np.zeros(net_replicates.shape[0], dtype=np.float64)
    for index, net in enumerate(net_replicates):
        corrected = correct_by_response(net, response) if response is not None else net
        cleaned = remove_isolated_spikes(corrected)
        if np.count_nonzero(area_mask) > 1:
            x = wavelengths[area_mask]
            y = cleaned[area_mask]
            order = np.argsort(x)
            areas[index] = np.trapezoid(y[order], x[order]) if hasattr(np, "trapezoid") else np.trapz(y[order], x[order])

    mean_net = np.mean(net_replicates, axis=0)
    mean_corrected = correct_by_response(mean_net, response) if response is not None else mean_net
    mean_spectrum = remove_isolated_spikes(mean_corrected)
    return areas, mean_spectrum


@dataclass
class KnownWavelength:
    name: str
    wavelength_nm: float
    tolerance_nm: float
    channel_index: int
    channel_name: str

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

@dataclass
class LinearMapping:
    slope_k: float
    intercept_b: float
    def wavelength_to_pixel(self, wl: float) -> float: return self.slope_k * wl + self.intercept_b
    def pixel_to_wavelength(self, px: float) -> float: return (px - self.intercept_b) / self.slope_k if self.slope_k != 0 else float('nan')

@dataclass
class TwoPointValidation:
    train_names: Tuple[str, str]
    validate_name: str
    slope_k: float
    intercept_b: float
    predicted_y: float
    measured_y: float
    residual_px: float
    wavelength_error_nm: float

class SpectralResponseCalibrator:
    def __init__(self, mapping: LinearMapping):
        self.mapping = mapping
        self.valid_range = DEFAULT_PARAMS["valid_wavelength_range"]
        self.smooth_window = DEFAULT_PARAMS["smooth_window"]  # 保留字段仅供 JSON 兼容，不再使用
        self.spd_threshold_ratio = DEFAULT_PARAMS["spd_threshold_ratio"]
        self.wavelengths = self.response_R = self.response_G = self.response_B = None
        self.final_mask = None
        # ---- v3 溯源 / 诊断字段（一切与像素有关的量都只作诊断） ----
        self.rb_swapped = False         # R/B 通道是否互换（桌面端 DNG 直通，恒 False）
        self.x_roi = None               # X 方向 ROI (start, end)，px
        self.sensor_height = None       # 标定当时的像素行数（诊断用，不参与重建）
        self.g_error_nm = None          # Step 1 G 点校验残差，nm
        self.spd_source = None          # 参考灯谱 CSV 文件名
        self.device_model = "unknown"   # 采集设备型号（桌面端无从得知，占位）
        self.exposure_seconds = None    # 标定当时 SPD 曝光，s
        self.frames = None              # 标定当时平均的 SPD 张数
        self.app_version = APP_VERSION
        self.created_at = None
        self.effective_band = None
        self.masked_count = None
        self.grid_points = None
        self.drift_nm = None            # 与 preset 内嵌映射的漂移，nm（None = 未知）
        self.drift_level = "UNKNOWN"    # "OK" | "WARN" | "SEVERE" | "UNKNOWN"

    def calibrate(self, pixel_coords: np.ndarray, measured_rgb: np.ndarray,
                  true_spd_wl: np.ndarray, true_spd_val: np.ndarray):
        # 1. 像素转波长（负斜率自动反转）—— 逐像素中间量，只为生成规范网格数据
        wavelengths = np.asarray(self.mapping.pixel_to_wavelength(pixel_coords),
                                 dtype=np.float64)
        if wavelengths[0] > wavelengths[-1]:
            wavelengths = wavelengths[::-1]
            measured_rgb = np.asarray(measured_rgb, dtype=np.float64)[::-1, :]
        self.sensor_height = int(wavelengths.size)

        # 2. 真实 SPD 插值到相机波长点
        f_interp = interp1d(true_spd_wl, true_spd_val, bounds_error=False, fill_value=0.0)
        true_spd_aligned = f_interp(wavelengths)

        # 3. 光源可信阈值掩码 + 波长范围掩码
        max_spd = np.max(true_spd_aligned)
        spd_threshold = max_spd * self.spd_threshold_ratio
        reliable_spd_mask = true_spd_aligned > spd_threshold
        range_mask = (wavelengths >= self.valid_range[0]) & (wavelengths <= self.valid_range[1])
        final_mask = range_mask & reliable_spd_mask

        # 4. 逐通道绝对响应度 = 灯谱测量 / 真实 SPD（不合并通道）
        # 【v1.4.3 简化】不做 SG 平滑、不做逐通道自归一化：保留绝对物理量
        epsilon = 1e-6
        raw = measured_rgb / (true_spd_aligned[:, None] + epsilon)
        raw[~final_mask, :] = 0.0

        # 5. 重采样到规范波长网格 —— 输出从此与像素彻底解耦
        grid = canonical_grid()
        # mask 用 "> 1 - 1e-9" 而不是 "> 0.5"：网格点若跨在掩码边界上，np.interp 会把
        # 掩码外的 0 混进来，该点响应最多被削掉一半。只保留完全落在掩码内的网格点。
        mask_grid = resample_to_grid(wavelengths, final_mask.astype(np.float64), grid)
        keep = mask_grid >= 1.0 - 1e-9
        responses = np.column_stack(
            [resample_to_grid(wavelengths, raw[:, i], grid) for i in range(3)])
        responses[~keep, :] = 0.0

        self.wavelengths = grid
        self.final_mask = keep
        self.grid_points = int(grid.size)
        self.masked_count = int(np.count_nonzero(keep))
        self.effective_band = ([float(grid[keep][0]), float(grid[keep][-1])]
                               if np.any(keep) else [0.0, 0.0])
        self.response_R = np.ascontiguousarray(responses[:, 0])
        self.response_G = np.ascontiguousarray(responses[:, 1])
        self.response_B = np.ascontiguousarray(responses[:, 2])
        return self.response_R, self.response_G, self.response_B

    # ================= 参数导出与导入（JSON） =================
    def save_params(self, filepath: str):
        """Write an SPD payload v3 preset -- indexed by wavelength, never by pixel."""
        grid = (np.asarray(self.wavelengths, dtype=np.float64)
                if self.wavelengths is not None else canonical_grid())
        mask = (np.asarray(self.final_mask, dtype=np.int64)
                if self.final_mask is not None else np.zeros(grid.size, dtype=np.int64))
        live = np.asarray(mask, dtype=bool)
        if np.any(live):
            effective_band = [float(grid[live][0]), float(grid[live][-1])]
        else:
            effective_band = [0.0, 0.0]
        step_nm = ((float(grid[-1]) - float(grid[0])) / (grid.size - 1)
                   if grid.size > 1 else SPD_GRID_STEP_NM)
        data = {
            "schema_version": SPD_SCHEMA_VERSION,
            "app_version": APP_VERSION,
            "preset_type": PRESET_TYPE_SPD,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "device_model": self.device_model or "unknown",
            "axis": {
                "kind": AXIS_KIND_UNIFORM_WAVELENGTH,
                "start_nm": float(grid[0]),
                "stop_nm": float(grid[-1]),
                "step_nm": float(step_nm),
            },
            "wavelengths": [float(v) for v in grid],
            "response_r": [float(v) for v in self.response_R],
            "response_g": [float(v) for v in self.response_G],
            "response_b": [float(v) for v in self.response_B],
            "mask": [int(v) for v in mask],
            "valid_range": [float(self.valid_range[0]), float(self.valid_range[1])],
            "effective_band": effective_band,
            "spd_threshold_ratio": float(self.spd_threshold_ratio),
            "masked_count": int(np.count_nonzero(live)),
            "grid_points": int(grid.size),
            # ---- 以下与像素有关的一律只作诊断，绝不用于重建任何曲线 ----
            "slope": _opt_float(self.mapping.slope_k) if self.mapping is not None else None,
            "intercept": _opt_float(self.mapping.intercept_b) if self.mapping is not None else None,
            "g_error_nm": _opt_float(self.g_error_nm),
            "x_roi": [int(v) for v in self.x_roi] if self.x_roi is not None else None,
            "sensor_height": _opt_int(self.sensor_height),
            "rb_swapped": bool(self.rb_swapped),
            "exposure_seconds": _opt_float(self.exposure_seconds),
            "frames": _opt_int(self.frames),
            "spd_source": self.spd_source or "",
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, allow_nan=False)

    def drift_against(self, other_mapping, n_px=None) -> Optional[float]:
        """Drift between the mapping this calibrator holds and ``other_mapping``."""
        if n_px is None:
            n_px = self.sensor_height
        if n_px is None and self.wavelengths is not None:
            n_px = len(self.wavelengths)
        return drift_nm(self.mapping, other_mapping, n_px)

    @classmethod
    def load_params(cls, filepath: str, mapping: LinearMapping = None, *,
                    strict: bool = True) -> 'SpectralResponseCalibrator':
        """Read an SPD payload v3 preset.

        Checks, in order:
          L0 version gate  -- ``schema_version`` must be exactly 3 (always, even
                              with ``strict=False``).  Older files are refused,
                              never silently migrated.
          L1 structure     -- strictly increasing wavelength axis, self-consistent
                              ``axis`` block, every array exactly ``grid_points``
                              long, at least ``MIN_VALID_POINTS`` live points.
          L2 coverage      -- the working wavelength axis must fall inside the
                              stored grid (``np.interp`` would clamp otherwise).
          L3 drift         -- measured only.  Lands in ``drift_nm`` /
                              ``drift_level`` and never refuses a file.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"SPD preset {filepath} is not a JSON object.")

        # ---- L0 版本闸门 ----
        schema = data.get("schema_version")
        if isinstance(schema, bool) or not isinstance(schema, int) \
                or schema != SPD_SCHEMA_VERSION:
            raise ValueError(
                f"无法加载该 SPD preset：schema_version={schema!r}，"
                f"当前版本要求 schema_version={SPD_SCHEMA_VERSION}。"
                "这是旧格式的 SPD preset，请用当前版本重新导出或重做 Step 2。")

        wavelengths = np.asarray(data.get("wavelengths", []), dtype=np.float64)
        responses = {channel: np.asarray(data.get(f"response_{channel}", []), dtype=np.float64)
                     for channel in ("r", "g", "b")}
        if strict:
            _validate_spd_structure(data, wavelengths, responses)      # L1

        instance = cls(mapping)
        instance.valid_range = tuple(data.get("valid_range", (430, 660)))
        instance.spd_threshold_ratio = data.get("spd_threshold_ratio", 0.1)
        instance.wavelengths = wavelengths
        instance.response_R, instance.response_G, instance.response_B = (
            responses["r"], responses["g"], responses["b"])
        stored_mask = data.get("mask")
        if stored_mask is not None:
            instance.final_mask = np.asarray(stored_mask, dtype=np.float64) > 0.5
        else:
            instance.final_mask = ((instance.response_R > 0) | (instance.response_G > 0)
                                   | (instance.response_B > 0))

        # ---- 溯源 / 诊断字段 ----
        instance.app_version = data.get("app_version")
        instance.created_at = data.get("created_at")
        instance.device_model = data.get("device_model") or "unknown"
        instance.effective_band = data.get("effective_band")
        instance.masked_count = data.get("masked_count")
        instance.grid_points = _opt_int(data.get("grid_points"))
        instance.sensor_height = _opt_int(data.get("sensor_height"))
        instance.rb_swapped = bool(data.get("rb_swapped", False))
        instance.spd_source = data.get("spd_source")
        instance.g_error_nm = _opt_float(data.get("g_error_nm"))
        instance.exposure_seconds = _opt_float(data.get("exposure_seconds"))
        instance.frames = _opt_int(data.get("frames"))
        x_roi = data.get("x_roi")
        instance.x_roi = tuple(int(v) for v in x_roi) if isinstance(x_roi, (list, tuple)) \
            and len(x_roi) == 2 else None

        # ---- L3 漂移（软提示，永不拒绝） ----
        preset_mapping = None
        slope, intercept = data.get("slope"), data.get("intercept")
        if slope is not None and intercept is not None:
            try:
                preset_mapping = LinearMapping(float(slope), float(intercept))
            except (TypeError, ValueError):
                preset_mapping = None
        n_px = instance.sensor_height if instance.sensor_height is not None else wavelengths.size
        instance.drift_nm = drift_nm(preset_mapping, mapping, n_px)
        instance.drift_level = drift_level(instance.drift_nm)

        # ---- L2 覆盖度 ----
        if strict:
            _validate_spd_coverage(mapping, n_px, wavelengths, instance.valid_range)
        return instance