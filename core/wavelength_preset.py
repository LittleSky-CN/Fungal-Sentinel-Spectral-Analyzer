"""Portable Step 1 wavelength-calibration presets (``WAVELENGTH``).

Three-stage API mirroring ``core.curve_preset.py``::

    validate_wavelength_preset()  ->  save_wavelength_preset()  ->  load_wavelength_preset()

The file carries two parallel views of the same Step 1 state:

* **flat keys** (``slope`` / ``intercept`` / ``gErrorNm`` / ``roiStart`` / ...) --
  named verbatim after the Android ``PresetEntity`` columns so a human (or a
  future import bridge) can line the two ends up field by field;
* **nested groups** (``device`` / ``mapping`` / ``roi`` / ``training`` /
  ``peaks`` / ``validation`` / ``capture``) -- the full desktop context,
  including the peak diagnostics that have no Android column.

Both views are populated from one assembly step, so they can never disagree.
The reader accepts either shape (and any mix of the two).
"""
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from core.calibration import LinearMapping

SCHEMA_VERSION = 1
PRESET_TYPE = "WAVELENGTH"
APP_VERSION = "1.4.8"

#: ``pixel_to_wavelength`` divides by the slope; anything flatter is unusable.
MIN_ABS_SLOPE = 1e-6

QUALITIES = ("GOOD", "SUSPECT", "FAILED", "PRESET")
PEAK_FIELDS = (
    "name", "centroid_px", "lo", "hi", "y_max", "baseline", "height",
    "estimator", "n_maxima", "quality", "centroid_legacy_px",
)
DEFAULT_DRIFT_FRACS = (0.10, 0.30, 0.50, 0.70, 0.90)

# peak field -> attribute / key aliases accepted on the way in
_PEAK_SOURCE_FIELDS = {
    "name": ("name",),
    "centroid_px": ("centroid_px", "y_centroid"),
    "lo": ("lo",),
    "hi": ("hi",),
    "y_max": ("y_max",),
    "baseline": ("baseline",),
    "height": ("height", "peak_height"),
    "estimator": ("estimator",),
    "n_maxima": ("n_maxima",),
    "quality": ("quality",),
    "centroid_legacy_px": ("centroid_legacy_px", "centroid_legacy"),
}


def _finite_number(value, name) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _optional_number(value, name) -> Optional[float]:
    """``None`` passes through untouched; everything else must be finite."""
    if value is None:
        return None
    return _finite_number(value, name)


def _optional_int(value, name) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _optional_str(value, name) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _norm_quality(value, default: str = "GOOD") -> str:
    """Normalise a quality tag to the upper-case vocabulary in ``QUALITIES``."""
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError("quality must be a string")
    candidate = value.strip().upper()
    return candidate if candidate in QUALITIES else default


def _first(*values):
    """First value that is neither ``None`` nor an empty string."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value:
            continue
        return value
    return None


def _normalize_peak(raw) -> Dict[str, Any]:
    """Accept a ``PeakResult``-like object or an already-serialised dict."""
    if raw is None:
        raise ValueError("Invalid peak entry in wavelength preset.")
    if isinstance(raw, dict):
        def _get(field):
            for source in _PEAK_SOURCE_FIELDS[field]:
                if source in raw:
                    return raw[source]
            return None
    else:
        def _get(field):
            for source in _PEAK_SOURCE_FIELDS[field]:
                if hasattr(raw, source):
                    return getattr(raw, source)
            return None

    name = _get("name")
    estimator = _get("estimator")
    return {
        "name": name if isinstance(name, str) else "",
        "centroid_px": _optional_number(_get("centroid_px"), "peak centroid"),
        "lo": _optional_int(_get("lo"), "peak lo"),
        "hi": _optional_int(_get("hi"), "peak hi"),
        "y_max": _optional_int(_get("y_max"), "peak y_max"),
        "baseline": _optional_number(_get("baseline"), "peak baseline"),
        "height": _optional_number(_get("height"), "peak height"),
        "estimator": estimator if isinstance(estimator, str) else "",
        "n_maxima": _optional_int(_get("n_maxima"), "peak n_maxima"),
        "quality": _norm_quality(_get("quality")),
        "centroid_legacy_px": _optional_number(
            _get("centroid_legacy_px"), "peak legacy centroid"),
    }


def validate_wavelength_preset(data, expected_mode=None) -> Dict[str, Any]:
    """Validate and normalise a Step 1 WAVELENGTH preset payload.

    Raises ``ValueError`` for anything that would silently produce a wrong
    wavelength axis later on (missing schema marker, missing mapping, a slope
    too flat to invert, a non-positive sensor height, an inverted ROI).
    """
    if not isinstance(data, dict):
        raise ValueError("Unsupported or invalid wavelength preset.")
    schema_version = data.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != SCHEMA_VERSION:
        raise ValueError("Unsupported or invalid wavelength preset.")
    if data.get("preset_type") != PRESET_TYPE:
        raise ValueError("Not a WAVELENGTH preset.")

    # ---- mapping -------------------------------------------------------
    nested_mapping = data.get("mapping")
    nested_mapping = nested_mapping if isinstance(nested_mapping, dict) else {}
    slope_raw = _first(data.get("slope"), nested_mapping.get("slope_k"))
    intercept_raw = _first(data.get("intercept"), nested_mapping.get("intercept_b"))
    if slope_raw is None:
        raise ValueError("Missing slope in wavelength preset.")
    if intercept_raw is None:
        raise ValueError("Missing intercept in wavelength preset.")
    slope = _finite_number(slope_raw, "slope")
    intercept = _finite_number(intercept_raw, "intercept")
    if abs(slope) < MIN_ABS_SLOPE:
        raise ValueError("Wavelength mapping slope is too small to be invertible.")

    # ---- device --------------------------------------------------------
    device = data.get("device")
    device = device if isinstance(device, dict) else {}
    device_model = _optional_str(_first(data.get("device_model"), device.get("model")),
                                 "device_model")
    device_make = _optional_str(_first(data.get("device_make"), device.get("make")),
                                "device_make")
    sensor_width = _optional_int(_first(data.get("sensorWidth"),
                                        device.get("image_width")), "sensorWidth")
    sensor_height = _optional_int(_first(data.get("sensorHeight"),
                                         device.get("image_height")), "sensorHeight")
    if sensor_width is not None and sensor_width <= 0:
        raise ValueError("sensorWidth must be positive.")
    if sensor_height is not None and sensor_height <= 0:
        raise ValueError("sensorHeight must be positive.")

    # ---- roi -----------------------------------------------------------
    nested_roi = data.get("roi")
    nested_roi = nested_roi if isinstance(nested_roi, dict) else {}
    roi_start = _optional_int(_first(data.get("roiStart"), nested_roi.get("start")),
                              "roiStart")
    roi_end = _optional_int(_first(data.get("roiEnd"), nested_roi.get("end")), "roiEnd")
    if roi_start is not None and roi_end is not None and roi_start > roi_end:
        raise ValueError("roiStart must not be greater than roiEnd.")

    # ---- training / known wavelengths ----------------------------------
    training = data.get("training")
    training = training if isinstance(training, dict) else {}
    wl_r = _optional_number(_first(data.get("wlR"), training.get("wl_r_nm")), "wlR")
    wl_g = _optional_number(_first(data.get("wlG"), training.get("wl_g_nm")), "wlG")
    wl_b = _optional_number(_first(data.get("wlB"), training.get("wl_b_nm")), "wlB")
    tolerance_nm = _optional_number(training.get("tolerance_nm"), "tolerance_nm")

    # ---- peaks ---------------------------------------------------------
    raw_peaks = data.get("peaks")
    peaks: Optional[List[Dict[str, Any]]] = None
    if isinstance(raw_peaks, (list, tuple)) and raw_peaks:
        peaks = [_normalize_peak(entry) for entry in raw_peaks]
    peak_r = _optional_number(data.get("peakR"), "peakR")
    peak_g = _optional_number(data.get("peakG"), "peakG")
    peak_b = _optional_number(data.get("peakB"), "peakB")
    if peaks:
        for peak in peaks:
            if peak["name"] == "R" and peak_r is None:
                peak_r = peak["centroid_px"]
            elif peak["name"] == "G" and peak_g is None:
                peak_g = peak["centroid_px"]
            elif peak["name"] == "B" and peak_b is None:
                peak_b = peak["centroid_px"]

    # ---- validation / capture ------------------------------------------
    nested_validation = data.get("validation")
    nested_validation = nested_validation if isinstance(nested_validation, dict) else {}
    g_error_nm = _optional_number(_first(data.get("gErrorNm"),
                                         nested_validation.get("error_nm")), "gErrorNm")
    validation = {
        "predicted_px": _optional_number(nested_validation.get("predicted_px"),
                                         "validation predicted_px"),
        "measured_px": _optional_number(nested_validation.get("measured_px"),
                                        "validation measured_px"),
        "residual_px": _optional_number(nested_validation.get("residual_px"),
                                        "validation residual_px"),
        "error_nm": g_error_nm,
        "quality": _norm_quality(nested_validation.get("quality")),
    }

    capture = data.get("capture")
    capture = capture if isinstance(capture, dict) else {}
    exposure_seconds = _optional_number(
        _first(data.get("exposureSeconds"), capture.get("exposure_seconds")),
        "exposureSeconds")
    frames = _optional_int(_first(data.get("frames"), capture.get("frames")), "frames")
    rb_swapped_raw = _first(data.get("rbSwapped"), capture.get("rb_swapped"))
    rb_swapped = bool(rb_swapped_raw) if rb_swapped_raw is not None else False

    # ---- mode ----------------------------------------------------------
    mode = _optional_str(_first(data.get("analysisMode"), training.get("mode")),
                         "analysisMode")
    if expected_mode is not None and mode and mode != expected_mode:
        raise ValueError("This wavelength preset uses a different analysis mode.")

    quality = _norm_quality(_first(data.get("quality"), nested_validation.get("quality")))

    return {
        "preset_type": PRESET_TYPE,
        "schema_version": SCHEMA_VERSION,
        "app_version": _optional_str(data.get("app_version"), "app_version") or APP_VERSION,
        "created_at": _optional_str(data.get("created_at"), "created_at"),
        "name": _optional_str(data.get("name"), "name"),
        "device_model": device_model,
        "device_make": device_make,
        "slope": slope,
        "intercept": intercept,
        "gErrorNm": g_error_nm,
        "quality": quality,
        "roiStart": roi_start,
        "roiEnd": roi_end,
        "sensorWidth": sensor_width,
        "sensorHeight": sensor_height,
        "peakR": peak_r,
        "peakG": peak_g,
        "peakB": peak_b,
        "wlR": wl_r,
        "wlG": wl_g,
        "wlB": wl_b,
        "exposureSeconds": exposure_seconds,
        "frames": frames,
        "rbSwapped": rb_swapped,
        "analysisMode": mode or None,
        # ---- nested mirror (same values, grouped for readability) ----
        "device": {
            "model": device_model,
            "make": device_make,
            "image_width": sensor_width,
            "image_height": sensor_height,
        },
        "mapping": {"slope_k": slope, "intercept_b": intercept},
        "roi": {"start": roi_start, "end": roi_end},
        "training": {
            "wl_r_nm": wl_r,
            "wl_g_nm": wl_g,
            "wl_b_nm": wl_b,
            "tolerance_nm": tolerance_nm,
            "mode": mode or None,
        },
        "peaks": peaks,
        "validation": validation,
        "capture": {
            "exposure_seconds": exposure_seconds,
            "frames": frames,
            "rb_swapped": rb_swapped,
        },
    }


def save_wavelength_preset(path, mapping, *, name: str = "",
                           validation=None, peaks: Optional[Sequence] = None,
                           x_roi=None, sensor_size=None, known_wls=None,
                           exposure_seconds=None, frames=None,
                           rb_swapped: bool = False, device_model: str = "",
                           device_make: str = "", analysis_mode: str = "",
                           created_at: Optional[str] = None) -> None:
    """Assemble, validate and write a Step 1 WAVELENGTH preset to ``path``."""
    if mapping is None:
        raise ValueError("Run Step 1 wavelength calibration before saving a preset.")
    slope = _finite_number(getattr(mapping, "slope_k", None), "slope")
    intercept = _finite_number(getattr(mapping, "intercept_b", None), "intercept")

    roi_start = roi_end = None
    if x_roi is not None:
        roi_start = _optional_int(x_roi[0], "roiStart")
        roi_end = _optional_int(x_roi[1], "roiEnd")

    width = height = None
    if sensor_size is not None:
        # sensor_size is (width, height) in pixels
        width = _optional_int(sensor_size[0], "sensorWidth")
        height = _optional_int(sensor_size[1], "sensorHeight")

    wl_r = wl_g = wl_b = tolerance_nm = None
    if known_wls:
        def _wl(key):
            value = known_wls.get(key) if isinstance(known_wls, dict) else None
            if value is None:
                return None
            return float(getattr(value, "wavelength_nm", value))
        wl_r, wl_g, wl_b = _wl("R"), _wl("G"), _wl("B")
        probe = known_wls.get("R") if isinstance(known_wls, dict) else None
        if probe is not None:
            tolerance_nm = _optional_number(getattr(probe, "tolerance_nm", None),
                                            "tolerance_nm")

    peak_list = None
    if peaks:
        peak_list = [_normalize_peak(entry) for entry in peaks]

    predicted_px = measured_px = residual_px = error_nm = None
    if validation is not None:
        predicted_px = _optional_number(getattr(validation, "predicted_y", None),
                                        "validation predicted_px")
        measured_px = _optional_number(getattr(validation, "measured_y", None),
                                       "validation measured_px")
        residual_px = _optional_number(getattr(validation, "residual_px", None),
                                       "validation residual_px")
        error_nm = _optional_number(getattr(validation, "wavelength_error_nm", None),
                                    "validation error_nm")
        quality = "GOOD" if (error_nm is None or abs(error_nm) <= 2.5) else "SUSPECT"
    else:
        quality = "GOOD"

    payload = validate_wavelength_preset({
        "preset_type": PRESET_TYPE,
        "schema_version": SCHEMA_VERSION,
        "app_version": APP_VERSION,
        "created_at": created_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "name": name,
        "device_model": device_model,
        "device_make": device_make,
        "slope": slope,
        "intercept": intercept,
        "gErrorNm": error_nm,
        "quality": quality,
        "roiStart": roi_start,
        "roiEnd": roi_end,
        "sensorWidth": width,
        "sensorHeight": height,
        "wlR": wl_r,
        "wlG": wl_g,
        "wlB": wl_b,
        "exposureSeconds": exposure_seconds,
        "frames": frames,
        "rbSwapped": rb_swapped,
        "analysisMode": analysis_mode,
        "peaks": peak_list,
        "validation": {
            "predicted_px": predicted_px,
            "measured_px": measured_px,
            "residual_px": residual_px,
            "error_nm": error_nm,
            "quality": quality,
        },
        "training": {
            "wl_r_nm": wl_r,
            "wl_g_nm": wl_g,
            "wl_b_nm": wl_b,
            "tolerance_nm": tolerance_nm,
            "mode": analysis_mode,
        },
    }, expected_mode=analysis_mode or None)

    with Path(path).open("w", encoding="utf-8") as preset_file:
        json.dump(payload, preset_file, ensure_ascii=False, indent=2, allow_nan=False)


def load_wavelength_preset(path, expected_mode=None) -> Dict[str, Any]:
    """Read and validate a Step 1 WAVELENGTH preset from ``path``."""
    with Path(path).open("r", encoding="utf-8") as preset_file:
        data = json.load(preset_file)
    return validate_wavelength_preset(data, expected_mode=expected_mode)


def mapping_from_preset(preset) -> LinearMapping:
    """Rebuild the ``LinearMapping`` stored inside a WAVELENGTH preset."""
    if not isinstance(preset, dict):
        raise ValueError("Unsupported or invalid wavelength preset.")
    nested = preset.get("mapping")
    nested = nested if isinstance(nested, dict) else {}
    slope = _first(preset.get("slope"), nested.get("slope_k"))
    intercept = _first(preset.get("intercept"), nested.get("intercept_b"))
    return LinearMapping(_finite_number(slope, "slope"),
                         _finite_number(intercept, "intercept"))


def mapping_drift_nm(preset, mapping, fracs=DEFAULT_DRIFT_FRACS) -> Optional[float]:
    """``max |lambda_current(px) - lambda_preset(px)|`` in nm over the profile.

    ``None`` when either mapping is missing or the profile length is unknown --
    drift is only ever a *soft* hint, never a reason to reject a preset.
    """
    if mapping is None or not isinstance(preset, dict):
        return None
    preset_mapping = mapping_from_preset(preset)
    nested = preset.get("device")
    nested = nested if isinstance(nested, dict) else {}
    n_px = _first(preset.get("sensorHeight"), nested.get("image_height"))
    if n_px is None:
        wavelengths = preset.get("wavelengths") or []
        n_px = len(wavelengths)
    if not n_px or int(n_px) < 2:
        return None

    span = float(int(n_px) - 1)
    worst = 0.0
    for frac in fracs:
        pixel = frac * span
        delta = mapping.pixel_to_wavelength(pixel) - preset_mapping.pixel_to_wavelength(pixel)
        worst = max(worst, abs(float(delta)))
    return worst
