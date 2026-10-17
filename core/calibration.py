# core/calibration.py
"""Wavelength mapping and Spectral Power Distribution (SPD) calibration.
v1.4.3 simplification: per-channel absolute response, NO Savitzky-Golay, NO self-normalization.
v1.4.6: SPD presets are self-contained -- they carry their wavelength mapping
(provenance) plus the raw lamp profiles and the raw reference SPD grid, so a
preset can be *recomputed* against whatever mapping is active at load time.
"""
import json
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.interpolate import interp1d
from config import DEFAULT_PARAMS

# ---------------------------------------------------------------------------
# v1.4.6 constants -- the three tiered checks applied when an SPD preset loads.
# L3 thresholds are duplicated on the Android side (MycoSpec
# analysis/SpectralResponseCalibrator.kt); keep both in sync.
# ---------------------------------------------------------------------------
APP_VERSION = "1.4.8"
SPD_SCHEMA_VERSION = 2
DRIFT_WARN_NM = 2.0          # below this: silent.  Real Step 1 G-residual is ~5.5 nm.
DRIFT_SEVERE_NM = 10.0       # above this: orange warning + confirm.  Never a rejection.
DRIFT_FRACS = (0.10, 0.30, 0.50, 0.70, 0.90)
MIN_VALID_POINTS = 20        # L1: a response curve with fewer live points is unusable
COVERAGE_TOL_NM = 0.5        # L2: slack for the grid endpoints
RECOMPUTE_MIN_DRIFT_NM = 1e-3   # Mode B: skip the recompute when nothing moved


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


def _sig6(values) -> list:
    """Serialise at 6 significant digits -- far below the measurement noise."""
    out = []
    for value in np.asarray(values, dtype=np.float64).ravel():
        number = float(value)
        if not math.isfinite(number):
            number = 0.0
        out.append(float(f"{number:.6g}"))
    return out


def _validate_spd_structure(wavelengths: np.ndarray, responses: dict) -> None:
    """L1 -- structural checks.  Failure here means the file is unusable."""
    if wavelengths.ndim != 1 or wavelengths.size < 2:
        raise ValueError("SPD preset wavelengths must contain at least two points.")
    if not np.all(np.isfinite(wavelengths)):
        raise ValueError("SPD preset wavelengths must be finite.")
    diffs = np.diff(wavelengths)
    if np.any(diffs == 0) or not (np.all(diffs > 0) or np.all(diffs < 0)):
        raise ValueError("SPD preset wavelengths must be strictly monotonic.")
    for channel, response in responses.items():
        if response.ndim != 1 or response.size != wavelengths.size:
            raise ValueError(
                f"SPD preset response_{channel} does not match its wavelength grid "
                f"({response.size} vs {wavelengths.size}).")
        if not np.all(np.isfinite(response)):
            raise ValueError(f"SPD preset response_{channel} must be finite.")
    live = np.count_nonzero(
        (responses["R"] != 0) | (responses["G"] != 0) | (responses["B"] != 0))
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
        self.spd_aligned = None
        self.final_mask = None
        # ---- v1.4.6 溯源 / Mode B 自持重算所需状态 ----
        self.lamp_rgb = None            # 原始灯谱剖面 (N, 3)，与 self.wavelengths 同序
        self.spd_reference = None       # (wavelengths, values) 原始参考灯谱 CSV 网格
        self.rb_swapped = False         # R/B 通道是否互换（桌面端 DNG 直通，恒 False）
        self.x_roi = None               # X 方向 ROI (start, end)，px
        self.sensor_height = None       # 剖面长度 N（决定像素网格 arange(N)）
        self.g_error_nm = None          # Step 1 G 点校验残差，nm
        self.spd_source = None          # 参考灯谱 CSV 文件名
        self.drift_nm = None            # 与 preset 内嵌映射的漂移，nm（None = 未知）
        self.drift_level = "UNKNOWN"    # "OK" | "WARN" | "SEVERE" | "UNKNOWN"
        self.recomputed = False         # 本次加载是否走了 Mode B 重算

    def calibrate(self, pixel_coords: np.ndarray, measured_rgb: np.ndarray,
                  true_spd_wl: np.ndarray, true_spd_val: np.ndarray):
        # 1. 像素转波长（负斜率自动反转）
        self.wavelengths = self.mapping.pixel_to_wavelength(pixel_coords)
        if self.wavelengths[0] > self.wavelengths[-1]:
            self.wavelengths = self.wavelengths[::-1]
            measured_rgb = measured_rgb[::-1, :]

        # 【v1.4.6】Mode B：保存未被污染的原始输入，供加载端按当前映射重算
        self.lamp_rgb = np.asarray(measured_rgb, dtype=np.float64)
        self.spd_reference = (np.asarray(true_spd_wl, dtype=np.float64),
                              np.asarray(true_spd_val, dtype=np.float64))
        self.sensor_height = int(self.wavelengths.size)

        # 2. 真实 SPD 插值到相机波长点
        f_interp = interp1d(true_spd_wl, true_spd_val, bounds_error=False, fill_value=0.0)
        true_spd_aligned = f_interp(self.wavelengths)
        self.spd_aligned = true_spd_aligned

        # 3. 光源可信阈值掩码 + 波长范围掩码
        max_spd = np.max(true_spd_aligned)
        spd_threshold = max_spd * self.spd_threshold_ratio
        reliable_spd_mask = true_spd_aligned > spd_threshold
        range_mask = (self.wavelengths >= self.valid_range[0]) & (self.wavelengths <= self.valid_range[1])
        final_mask = range_mask & reliable_spd_mask
        self.final_mask = final_mask

        # 4. 逐通道绝对响应度 = 灯谱测量 / 真实 SPD（不合并通道）
        epsilon = 1e-6
        raw_R = measured_rgb[:, 0] / (true_spd_aligned + epsilon)
        raw_G = measured_rgb[:, 1] / (true_spd_aligned + epsilon)
        raw_B = measured_rgb[:, 2] / (true_spd_aligned + epsilon)
        raw_R[~final_mask] = 0
        raw_G[~final_mask] = 0
        raw_B[~final_mask] = 0

        # 【v1.4.3 简化】不做 SG 平滑、不做逐通道自归一化：保留绝对物理量，便于排查与跨通道比较
        self.response_R, self.response_G, self.response_B = raw_R, raw_G, raw_B
        return self.response_R, self.response_G, self.response_B

    # ================= 参数导出与导入（JSON） =================
    def save_params(self, filepath: str):
        """Write the SPD preset.

        v1.4.6 keeps every legacy key untouched (bit-exact round trip for the
        baked arrays) and appends the provenance block plus the raw Mode B
        inputs (``lamp_*`` / ``spd_reference``) at 6 significant digits.
        """
        lamp = self.lamp_rgb
        reference = self.spd_reference
        data = {
            "version": APP_VERSION,
            "schema_version": SPD_SCHEMA_VERSION,
            "valid_range": list(self.valid_range),
            "smooth_window": self.smooth_window,
            "spd_threshold_ratio": self.spd_threshold_ratio,
            "wavelengths": self.wavelengths.tolist() if self.wavelengths is not None else [],
            "response_R": self.response_R.tolist() if self.response_R is not None else [],
            "response_G": self.response_G.tolist() if self.response_G is not None else [],
            "response_B": self.response_B.tolist() if self.response_B is not None else [],
            # ---- 溯源（与 Android SpdCalibrationResult 同名同量纲） ----
            "slope": float(self.mapping.slope_k) if self.mapping is not None else None,
            "intercept": float(self.mapping.intercept_b) if self.mapping is not None else None,
            "g_error_nm": float(self.g_error_nm) if self.g_error_nm is not None else None,
            "x_roi": [int(v) for v in self.x_roi] if self.x_roi is not None else None,
            "sensor_height": int(self.sensor_height) if self.sensor_height is not None else None,
            "rb_swapped": bool(self.rb_swapped),
            "spd_source": self.spd_source,
            # ---- Mode B 原始输入（可缺省：旧文件自动回退 Mode A） ----
            "lamp_R": _sig6(lamp[:, 0]) if lamp is not None else None,
            "lamp_G": _sig6(lamp[:, 1]) if lamp is not None else None,
            "lamp_B": _sig6(lamp[:, 2]) if lamp is not None else None,
            "spd_reference": (
                {"wavelengths": _sig6(reference[0]), "values": _sig6(reference[1])}
                if reference is not None else None),
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

    def _recompute_from_source(self, data: dict, n_px: int) -> bool:
        """Mode B: rebuild the responses on the *current* mapping.

        The stored ``response_*`` arrays were baked with the mapping that was
        active at save time, so they carry a multiplicative
        ``SPD(λ-δ)/SPD(λ)`` error once the mapping drifts.  Recomputing from the
        raw lamp profiles removes it entirely: the result is by construction
        identical to re-running Step 2 with the current mapping.

        Returns ``False`` (leaving the baked arrays in place) whenever the raw
        inputs are missing or the recomputation would degenerate -- that keeps
        every pre-v1.4.6 file working.
        """
        if self.mapping is None or n_px is None or int(n_px) < 2:
            return False
        # Nothing moved -> the baked arrays are already exact (and bit-exact).
        if self.drift_nm is None or self.drift_nm <= RECOMPUTE_MIN_DRIFT_NM:
            return False

        lamp_columns = [data.get(f"lamp_{channel}") for channel in ("R", "G", "B")]
        if not all(isinstance(column, list) for column in lamp_columns):
            return False
        if not all(len(column) == int(n_px) for column in lamp_columns):
            return False
        reference = data.get("spd_reference")
        if not isinstance(reference, dict):
            return False
        ref_wl = np.asarray(reference.get("wavelengths") or [], dtype=np.float64)
        ref_val = np.asarray(reference.get("values") or [], dtype=np.float64)
        if ref_wl.size < 2 or ref_val.size != ref_wl.size:
            return False
        if not np.all(np.isfinite(ref_wl)) or not np.all(np.isfinite(ref_val)):
            return False
        if np.any(np.diff(ref_wl) <= 0):        # np.interp needs a rising xp
            order = np.argsort(ref_wl)
            ref_wl, ref_val = ref_wl[order], ref_val[order]

        lamp = np.column_stack([np.asarray(column, dtype=np.float64)
                                for column in lamp_columns])
        n_px = int(n_px)
        pixels = np.arange(n_px, dtype=np.float64)
        wavelengths = np.asarray(self.mapping.pixel_to_wavelength(pixels), dtype=np.float64)
        if wavelengths[0] > wavelengths[-1]:
            wavelengths = wavelengths[::-1]
        # ``lamp_*`` was written by calibrate() already aligned to the ASCENDING
        # wavelength axis, so it must only be flipped when the stored grid itself
        # was descending (i.e. the writing build used a positive slope).
        stored_wl = np.asarray(data.get("wavelengths") or [], dtype=np.float64)
        if stored_wl.size >= 2 and stored_wl[0] > stored_wl[-1]:
            lamp = lamp[::-1, :]

        spd_aligned = np.interp(wavelengths, ref_wl, ref_val)
        # match calibrate()'s interp1d(bounds_error=False, fill_value=0.0)
        outside = (wavelengths < ref_wl[0]) | (wavelengths > ref_wl[-1])
        spd_aligned[outside] = 0.0
        threshold = float(np.max(spd_aligned)) * self.spd_threshold_ratio
        mask = ((wavelengths >= self.valid_range[0]) & (wavelengths <= self.valid_range[1])
                & (spd_aligned > threshold))
        if np.count_nonzero(mask) < MIN_VALID_POINTS:
            return False        # degenerates on this mapping -> keep Mode A

        epsilon = 1e-6
        responses = lamp / (spd_aligned[:, None] + epsilon)
        responses[~mask, :] = 0.0

        self.wavelengths = wavelengths
        self.response_R = np.ascontiguousarray(responses[:, 0])
        self.response_G = np.ascontiguousarray(responses[:, 1])
        self.response_B = np.ascontiguousarray(responses[:, 2])
        self.final_mask = mask
        self.spd_aligned = spd_aligned
        self.lamp_rgb = lamp
        self.spd_reference = (ref_wl, ref_val)
        self.recomputed = True
        return True

    @classmethod
    def load_params(cls, filepath: str, mapping: LinearMapping = None, *,
                    strict: bool = True, recompute: bool = True) -> 'SpectralResponseCalibrator':
        """Read an SPD preset.

        ``strict=True`` (the GUI default) runs L1 (structure) and L2 (coverage);
        both raise ``ValueError``.  L3 (mapping drift) is *only* measured -- it
        lands in ``instance.drift_nm`` / ``instance.drift_level`` and never
        refuses a file.  With ``recompute=True`` and a preset that carries
        ``lamp_*`` + ``spd_reference``, the responses are rebuilt on the current
        mapping (Mode B); older files fall back to the baked arrays (Mode A).
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        wavelengths = np.asarray(data.get("wavelengths", []), dtype=np.float64)
        responses = {channel: np.asarray(data.get(f"response_{channel}", []), dtype=np.float64)
                     for channel in ("R", "G", "B")}
        if strict:
            _validate_spd_structure(wavelengths, responses)      # L1

        instance = cls(mapping)
        instance.valid_range = tuple(data.get("valid_range", (430, 660)))
        instance.smooth_window = data.get("smooth_window", 21)
        instance.spd_threshold_ratio = data.get("spd_threshold_ratio", 0.1)
        instance.wavelengths = wavelengths
        instance.response_R, instance.response_G, instance.response_B = (
            responses["R"], responses["G"], responses["B"])
        instance.final_mask = ((instance.response_R > 0) | (instance.response_G > 0)
                               | (instance.response_B > 0))

        # ---- 溯源字段 ----
        instance.sensor_height = data.get("sensor_height")
        instance.rb_swapped = bool(data.get("rb_swapped", False))
        instance.spd_source = data.get("spd_source")
        instance.g_error_nm = data.get("g_error_nm")
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

        # ---- Mode B 自持重算 ----
        if recompute:
            instance._recompute_from_source(data, n_px)
        return instance