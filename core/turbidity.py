# core/turbidity.py
"""Turbidity Measurement Algorithms for FSSA"""
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class TurbidityResult:
    turbidity_ntu: float
    scattering_ratio: float
    transmission_loss: float
    quality_flag: str
    raw_blue_intensity: float
    raw_green_intensity: float
    wavelength_range: Tuple[float, float]

class TurbidityAnalyzer:
    def __init__(self, mapping, calibrator=None):
        self.mapping = mapping
        self.calibrator = calibrator
        self.blue_wl_range = (420, 490)
        self.green_wl_range = (500, 570)
        
    def calculate_turbidity_from_spectrum(self, sample_spectrum: np.ndarray, wavelengths: np.ndarray, blank_spectrum: Optional[np.ndarray] = None, calibration_curve: Optional[Tuple[float, float]] = None) -> TurbidityResult:
        if wavelengths[0] > wavelengths[-1]:
            wavelengths = wavelengths[::-1]
            sample_spectrum = sample_spectrum[::-1]
            if blank_spectrum is not None: blank_spectrum = blank_spectrum[::-1]
        
        blue_mask = (wavelengths >= self.blue_wl_range[0]) & (wavelengths <= self.blue_wl_range[1])
        green_mask = (wavelengths >= self.green_wl_range[0]) & (wavelengths <= self.green_wl_range[1])
        
        if not np.any(blue_mask) or not np.any(green_mask):
            raise ValueError("Wavelength range does not cover required bands")
        
        blue_intensity = np.mean(sample_spectrum[blue_mask])
        green_intensity = np.mean(sample_spectrum[green_mask])
        
        if blank_spectrum is not None:
            blank_blue = np.mean(blank_spectrum[blue_mask])
            blank_green = np.mean(blank_spectrum[green_mask])
            if blank_blue <= 0: blank_blue = 1e-6
            if blank_green <= 0: blank_green = 1e-6
            transmission_loss = (1 - green_intensity / blank_green) * 100
        else:
            transmission_loss = 0.0
            
        scattering_ratio = blue_intensity / green_intensity if green_intensity > 0 else float('inf')
        
        if calibration_curve is not None:
            slope, intercept = calibration_curve
            turbidity_ntu = (scattering_ratio - intercept) / slope if slope != 0 else 0.0
        else:
            turbidity_ntu = max(0, (scattering_ratio - 0.8) * 100)
            
        if transmission_loss < 10 and scattering_ratio < 1.2: quality_flag = "Good"
        elif transmission_loss < 30 and scattering_ratio < 2.0: quality_flag = "Warning"
        else: quality_flag = "Poor"
            
        return TurbidityResult(turbidity_ntu, scattering_ratio, transmission_loss, quality_flag, blue_intensity, green_intensity, (float(wavelengths[0]), float(wavelengths[-1])))