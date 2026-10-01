# core/calibration.py
"""Wavelength mapping and Spectral Power Distribution (SPD) calibration.
v1.4.3 simplification: per-channel absolute response, NO Savitzky-Golay, NO self-normalization."""
import json
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy.interpolate import interp1d
from config import DEFAULT_PARAMS

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

    def calibrate(self, pixel_coords: np.ndarray, measured_rgb: np.ndarray,
                  true_spd_wl: np.ndarray, true_spd_val: np.ndarray):
        # 1. 像素转波长（负斜率自动反转）
        self.wavelengths = self.mapping.pixel_to_wavelength(pixel_coords)
        if self.wavelengths[0] > self.wavelengths[-1]:
            self.wavelengths = self.wavelengths[::-1]
            measured_rgb = measured_rgb[::-1, :]

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
        data = {
            "version": "1.4.3",
            "valid_range": list(self.valid_range),
            "smooth_window": self.smooth_window,
            "spd_threshold_ratio": self.spd_threshold_ratio,
            "wavelengths": self.wavelengths.tolist() if self.wavelengths is not None else [],
            "response_R": self.response_R.tolist() if self.response_R is not None else [],
            "response_G": self.response_G.tolist() if self.response_G is not None else [],
            "response_B": self.response_B.tolist() if self.response_B is not None else [],
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load_params(cls, filepath: str, mapping: LinearMapping = None) -> 'SpectralResponseCalibrator':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        instance = cls(mapping)
        instance.valid_range = tuple(data.get("valid_range", (430, 660)))
        instance.smooth_window = data.get("smooth_window", 21)
        instance.spd_threshold_ratio = data.get("spd_threshold_ratio", 0.1)
        instance.wavelengths = np.array(data["wavelengths"])
        instance.response_R = np.array(data["response_R"])
        instance.response_G = np.array(data["response_G"])
        instance.response_B = np.array(data["response_B"])
        instance.final_mask = (instance.response_R > 0) | (instance.response_G > 0) | (instance.response_B > 0)
        return instance