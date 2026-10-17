"""Portable Step 4 calibration-curve presets."""
import json
import math
from pathlib import Path

from core.quantification import calculate_concentration


_FIT_FIELDS = (
    "slope", "intercept", "r_value", "r_squared", "syx",
    "lod_conc", "loq_conc", "lod_signal", "loq_signal",
    "lod_fullrange", "loq_fullrange", "lod_source", "lod_source_exp",
)
_REGRESSION_METHODS = {"ols", "weighted_1y", "weighted_1y2"}
_METRIC_MODES = {"unmixing", "legacy"}


def _finite_number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def validate_curve_preset(data, expected_mode=None):
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Unsupported or invalid curve preset.")
    if expected_mode is not None and data.get("analysis_mode") != expected_mode:
        raise ValueError("This curve preset uses a different analysis mode.")

    metric_mode = data.get("metric_mode")
    regression_method = data.get("regression_method")
    if metric_mode not in _METRIC_MODES:
        raise ValueError("Invalid quantification metric in curve preset.")
    if regression_method not in _REGRESSION_METHODS:
        raise ValueError("Invalid regression method in curve preset.")
    if not isinstance(data.get("fluorophore"), str) or not data["fluorophore"]:
        raise ValueError("Missing fluorophore in curve preset.")

    concentrations = data.get("concentrations")
    means = data.get("means")
    stds = data.get("stds")
    if not all(isinstance(values, list) for values in (concentrations, means, stds)):
        raise ValueError("Curve points must be lists.")
    if len(concentrations) < 2 or len(means) != len(concentrations) or len(stds) != len(concentrations):
        raise ValueError("Curve preset must contain at least two matching standard points.")
    concentrations = [_finite_number(v, "Concentration") for v in concentrations]
    means = [_finite_number(v, "Signal") for v in means]
    stds = [_finite_number(v, "Signal deviation") for v in stds]
    if any(value < 0 for value in stds):
        raise ValueError("Signal deviations cannot be negative.")

    fit = data.get("fit")
    if not isinstance(fit, dict):
        raise ValueError("Missing fitted curve in preset.")
    fit = {key: fit[key] for key in _FIT_FIELDS if key in fit}
    for key in ("slope", "intercept", "r_value", "r_squared"):
        if key not in fit:
            raise ValueError(f"Missing {key} in fitted curve.")
        fit[key] = _finite_number(fit[key], key)
    if fit["slope"] == 0:
        raise ValueError("Curve slope cannot be zero.")
    for key in ("syx", "lod_conc", "loq_conc", "lod_signal", "loq_signal",
                "lod_fullrange", "loq_fullrange", "lod_source_exp"):
        if key in fit:
            fit[key] = _finite_number(fit[key], key)
    if "lod_source" in fit and not isinstance(fit["lod_source"], str):
        raise ValueError("Invalid LoD source in curve preset.")

    recalculated = calculate_concentration(
        concentrations, means, 0.0, mode=data.get("analysis_mode", ""),
        regression_method=regression_method,
    )
    if not math.isclose(fit["slope"], recalculated["slope"], rel_tol=1e-7, abs_tol=1e-10):
        raise ValueError("Stored curve slope does not match its standard points.")
    if not math.isclose(fit["intercept"], recalculated["intercept"], rel_tol=1e-7, abs_tol=1e-10):
        raise ValueError("Stored curve intercept does not match its standard points.")

    reference = data.get("reference_emission")
    if metric_mode == "unmixing":
        if not isinstance(reference, dict):
            raise ValueError("Unmixing presets must include a wavelength-based reference spectrum.")
        wavelengths = reference.get("wavelengths")
        emission = reference.get("emission")
        if not isinstance(wavelengths, list) or not isinstance(emission, list) or len(wavelengths) < 2:
            raise ValueError("Invalid reference spectrum in curve preset.")
        wavelengths = [_finite_number(v, "Reference wavelength") for v in wavelengths]
        emission = [_finite_number(v, "Reference emission") for v in emission]
        if len(emission) != len(wavelengths) or any(
            right <= left for left, right in zip(wavelengths, wavelengths[1:])
        ):
            raise ValueError("Reference wavelengths must be strictly increasing and match the spectrum.")
        window = reference.get("window_nm")
        if not isinstance(window, list) or len(window) != 2:
            raise ValueError("Invalid unmixing wavelength window.")
        window = [_finite_number(v, "Unmixing wavelength") for v in window]
        if window[0] >= window[1]:
            raise ValueError("Invalid unmixing wavelength window.")
        if sum(window[0] <= wavelength <= window[1] for wavelength in wavelengths) < 3:
            raise ValueError("Unmixing reference must cover at least three wavelengths.")
        auc = _finite_number(reference.get("auc"), "Reference emission AUC")
        if auc <= 0:
            raise ValueError("Reference emission AUC must be positive.")
        ref_conc = _finite_number(reference.get("ref_conc"), "Reference concentration")
        reference = {
            "wavelengths": wavelengths,
            "emission": emission,
            "auc": auc,
            "window_nm": window,
            "ref_conc": ref_conc,
        }
    elif reference is not None:
        raise ValueError("Legacy presets must not contain an unmixing reference spectrum.")

    return {
        "schema_version": 1,
        "analysis_mode": data.get("analysis_mode"),
        "fluorophore": data["fluorophore"],
        "metric_mode": metric_mode,
        "regression_method": regression_method,
        "concentrations": concentrations,
        "means": means,
        "stds": stds,
        "fit": fit,
        "reference_emission": reference,
    }


def save_curve_preset(path, cache, analysis_mode, fluorophore):
    if not cache or not all(key in cache for key in ("concentrations", "means", "stds", "slope", "intercept")):
        raise ValueError("Build a Step 4 curve before saving a preset.")
    unmix = cache.get("unmix")
    metric_mode = cache.get("metric_mode")
    reference = None
    if metric_mode == "unmixing":
        if not unmix:
            raise ValueError("The curve is missing its unmixing reference spectrum.")
        reference = {
            "wavelengths": [float(v) for v in unmix["wls"]],
            "emission": [float(v) for v in unmix["E"]],
            "auc": float(unmix["auc_E"]),
            "window_nm": [float(v) for v in unmix["window"]],
            "ref_conc": float(unmix["ref_conc"]),
        }
    fit = {key: cache[key] for key in _FIT_FIELDS if key in cache}
    fit.update({
        "slope": cache["slope"],
        "intercept": cache["intercept"],
        "r_value": cache.get("r_value", 0.0),
        "r_squared": cache.get("r_squared", 0.0),
    })
    data = validate_curve_preset({
        "schema_version": 1,
        "analysis_mode": analysis_mode,
        "fluorophore": fluorophore,
        "metric_mode": metric_mode,
        "regression_method": cache.get("regression_method"),
        "concentrations": cache["concentrations"],
        "means": cache["means"],
        "stds": cache["stds"],
        "fit": fit,
        "reference_emission": reference,
    }, expected_mode=analysis_mode)
    with Path(path).open("w", encoding="utf-8") as preset_file:
        json.dump(data, preset_file, ensure_ascii=False, indent=2, allow_nan=False)


def load_curve_preset(path, expected_mode):
    with Path(path).open("r", encoding="utf-8") as preset_file:
        data = json.load(preset_file)
    return validate_curve_preset(data, expected_mode=expected_mode)
