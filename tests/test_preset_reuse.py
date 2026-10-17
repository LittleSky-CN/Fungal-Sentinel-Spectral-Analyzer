"""Save -> load -> reuse coverage for the three preset types.

Preset taxonomy (authoritative source: Android ``data/preset/PresetTypes``):

===================  ==========================================================
WAVELENGTH           Step 1, pixel <-> wavelength mapping
SPD                  Step 2, per-channel spectral response
CURVE                Step 4, standard curve -> concentration
===================  ==========================================================

All three are now persisted on the desktop build:
``wavelength_calibration.json`` (Step 1), ``spd_calibration_params.json``
(Step 2) and ``concentration_curve.json`` (Step 4).

The Step 2 preset carries its wavelength mapping plus the raw lamp profiles and
reference-SPD grid, so a preset loaded under a drifted mapping is *recomputed*
(Mode B) and is equivalent to re-running Step 2.  Drift itself only ever warns.
"""
import json
import math

import numpy as np
import pytest

from core.calibration import LinearMapping, SpectralResponseCalibrator
from core.curve_preset import (
    load_curve_preset, save_curve_preset, validate_curve_preset,
)
from core.quantification import calculate_concentration
from core.unmixing import unmix_spectrum

FLUOROPHORE = "Ypet (Venus variant)"
REGRESSION_METHODS = ("ols", "weighted_1y", "weighted_1y2")
METRIC_MODES = ("unmixing", "legacy")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _synthetic_calibrator(mapping=None):
    """A calibrator built from a smooth synthetic lamp / SPD pair."""
    mapping = mapping or LinearMapping(-2.738911, 3198.4744)
    # 1000-pixel window around the R/G/B peaks (1490 / 1780 / 1930 px), i.e.
    # roughly 365-730 nm with the mapping above.
    pixels = np.arange(1200.0, 2200.0, dtype=np.float64)
    true_wl = np.linspace(380.0, 720.0, 341)
    true_val = np.exp(-0.5 * ((true_wl - 540.0) / 70.0) ** 2) + 0.05
    lamp = np.column_stack([
        40.0 * np.exp(-0.5 * ((pixels - 1930.0) / 90.0) ** 2),
        70.0 * np.exp(-0.5 * ((pixels - 1780.0) / 90.0) ** 2),
        95.0 * np.exp(-0.5 * ((pixels - 1490.0) / 90.0) ** 2),
    ])
    cal = SpectralResponseCalibrator(mapping)
    cal.calibrate(pixels, lamp, true_wl, true_val)
    return cal


def _sensor_grid_calibrator(mapping=None, n_px=3072):
    """A calibrator built exactly the way the GUI does it: the pixel grid is
    0-based over the whole sensor column, and the lamp profile is indexed by
    pixel (so it is unaffected by a mapping change -- which is what makes Mode B
    meaningful)."""
    mapping = mapping or LinearMapping(-2.738911, 3198.4744)
    pixels = np.arange(n_px, dtype=np.float64)
    true_wl = np.linspace(380.0, 720.0, 341)
    true_val = np.exp(-0.5 * ((true_wl - 540.0) / 70.0) ** 2) + 0.05
    lamp = np.column_stack([
        40.0 * np.exp(-0.5 * ((pixels - 1493.0) / 120.0) ** 2),
        70.0 * np.exp(-0.5 * ((pixels - 1767.0) / 120.0) ** 2),
        95.0 * np.exp(-0.5 * ((pixels - 1931.0) / 120.0) ** 2),
    ])
    cal = SpectralResponseCalibrator(mapping)
    cal.calibrate(pixels, lamp, true_wl, true_val)
    return cal


def _unmix_cache(metric_mode, regression_method, analysis_mode="lod_iupac"):
    """A ``cache_step4``-shaped dict exactly as gui/main_window.py builds it."""
    rng = np.random.default_rng(20241015)
    wls = np.linspace(460.0, 640.0, 181)
    window = (500.0, 610.0)
    window_mask = (wls >= window[0]) & (wls <= window[1])
    emission = np.exp(-0.5 * ((wls - 527.0) / 22.0) ** 2)
    blank = 0.05 + 0.02 * np.exp(-0.5 * ((wls - 470.0) / 25.0) ** 2)

    concentrations = [100.0, 60.0, 20.0]
    means, stds = [], []
    per_row = []
    for conc in concentrations:
        alphas = []
        for _ in range(3):
            noise = rng.normal(0.0, 2e-4, wls.size)
            spectrum = (conc / 1000.0) * emission + 0.9 * blank + 0.01 + noise
            alphas.append(unmix_spectrum(spectrum, emission, blank, window_mask)["alpha"])
        metrics = [a * 12.5 for a in alphas]
        means.append(float(np.mean(metrics)))
        stds.append(float(np.std(metrics, ddof=1)))
        per_row.append({"conc": conc, "alpha": float(np.mean(alphas)), "s": 0.9,
                        "k": 0.01, "r2": 0.99})

    sample_spectrum = 0.045 * emission + 0.9 * blank + 0.01
    alpha_u = unmix_spectrum(sample_spectrum, emission, blank, window_mask)["alpha"]
    sample_mean_area = alpha_u * 12.5

    calc = calculate_concentration(concentrations, means, sample_mean_area,
                                   mode=analysis_mode,
                                   regression_method=regression_method)
    if metric_mode == "unmixing":
        unmix = {"window": window, "ref_conc": 100.0, "auc_E": 12.5,
                 "per_row": per_row, "alpha": alpha_u, "u_s": 0.9, "u_k": 0.01,
                 "wls": wls, "E": emission, "B": blank,
                 "sample_mean_spec": sample_spectrum}
    else:
        unmix = None
    cache = dict(calc)
    cache.update({"concentrations": concentrations, "means": means, "stds": stds,
                  "metric_mode": metric_mode, "t_stat": 12.0, "unmix": unmix,
                  "blocks": [], "scatter_peak_wl": 470.0,
                  "fluorophore": FLUOROPHORE})
    return cache


def _sample_metric_from_preset(preset):
    """Mirror of gui/main_window.calculate_with_curve_preset: the concentration
    the app reports for the current sample once a preset is active."""
    fit = preset["fit"]
    sample_mean_area = _unmix_cache(preset["metric_mode"],
                                    preset["regression_method"])["sample_mean_area"]
    return (sample_mean_area - fit["intercept"]) / fit["slope"]


# ---------------------------------------------------------------------------
# SPD preset (Step 2)
# ---------------------------------------------------------------------------
def test_spd_params_round_trip_is_bit_exact(tmp_path):
    original = _synthetic_calibrator()
    path = tmp_path / "spd_calibration_params.json"
    original.save_params(str(path))

    loaded = SpectralResponseCalibrator.load_params(str(path), original.mapping)

    assert np.array_equal(original.wavelengths, loaded.wavelengths)
    assert np.array_equal(original.response_R, loaded.response_R)
    assert np.array_equal(original.response_G, loaded.response_G)
    assert np.array_equal(original.response_B, loaded.response_B)
    assert np.array_equal(original.final_mask, loaded.final_mask)
    assert loaded.valid_range == original.valid_range
    assert loaded.spd_threshold_ratio == original.spd_threshold_ratio


def test_spd_params_reuse_reproduces_the_same_correction(tmp_path):
    """A reloaded SPD preset must correct the very same spectrum identically."""
    original = _synthetic_calibrator()
    path = tmp_path / "spd.json"
    original.save_params(str(path))
    loaded = SpectralResponseCalibrator.load_params(str(path), original.mapping)

    signal = np.abs(np.sin(np.linspace(0.0, 6.0, original.wavelengths.size))) + 0.01
    from core.calibration import correct_by_response
    before = correct_by_response(signal, original.response_G)
    after = correct_by_response(signal, loaded.response_G)
    assert np.array_equal(before, after)


def test_spd_params_json_stores_the_wavelength_mapping_and_raw_lamp(tmp_path):
    """DEF-P02 (fixed): the Step 2 JSON now carries the slope/intercept it was
    computed with -- so the file can be checked against the active mapping -- and
    it also carries the raw lamp profiles plus the raw reference SPD grid, which
    lets ``load_params`` recompute the response on the current mapping (Mode B)."""
    cal = _synthetic_calibrator()
    path = tmp_path / "spd.json"
    cal.save_params(str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["version"] == "1.4.6"
    assert payload["schema_version"] == 2
    for legacy_key in ("valid_range", "smooth_window", "spd_threshold_ratio",
                       "wavelengths", "response_R", "response_G", "response_B"):
        assert legacy_key in payload, "legacy key must survive for old files"

    assert math.isclose(payload["slope"], cal.mapping.slope_k, rel_tol=1e-12)
    assert math.isclose(payload["intercept"], cal.mapping.intercept_b, rel_tol=1e-12)
    assert payload["sensor_height"] == len(cal.wavelengths)
    assert payload["rb_swapped"] is False

    # Mode B raw inputs -- 6 significant digits, same length as the response grid
    assert len(payload["lamp_R"]) == len(payload["wavelengths"])
    assert len(payload["lamp_G"]) == len(payload["wavelengths"])
    assert len(payload["lamp_B"]) == len(payload["wavelengths"])
    assert np.allclose(payload["lamp_G"], cal.lamp_rgb[:, 1], rtol=1e-5, atol=1e-15)
    reference = payload["spd_reference"]
    assert len(reference["wavelengths"]) == len(reference["values"]) >= 2
    assert np.allclose(reference["values"], cal.spd_reference[1], rtol=1e-5, atol=1e-15)


def test_load_params_accepts_a_missing_mapping_then_breaks_later(tmp_path):
    """DEF-P02 (characterisation): ``load_params`` defaults ``mapping`` to None
    and does not complain.  gui.load_spd_params() shows a success dialog and the
    failure only surfaces later as an AttributeError inside Step 3."""
    cal = _synthetic_calibrator()
    path = tmp_path / "spd.json"
    cal.save_params(str(path))

    blind = SpectralResponseCalibrator.load_params(str(path))
    assert blind.mapping is None
    with pytest.raises(AttributeError):
        blind.mapping.pixel_to_wavelength(np.arange(3.0))


def test_load_params_drops_the_aligned_source_spectrum(tmp_path):
    """DEF-P03 (characterisation): ``load_params`` rebuilds ``final_mask`` from
    the response curves but never restores ``spd_aligned``, so anything reading
    it after a load sees None."""
    cal = _synthetic_calibrator()
    assert cal.spd_aligned is not None
    path = tmp_path / "spd.json"
    cal.save_params(str(path))

    loaded = SpectralResponseCalibrator.load_params(str(path), cal.mapping)
    assert loaded.spd_aligned is None
    assert np.array_equal(loaded.final_mask, cal.final_mask)


def test_loaded_spd_response_is_resampled_onto_the_current_mapping(tmp_path):
    """DEF-P04: with a different mapping active the saved response is silently
    interpolated onto the new wavelength axis instead of being rejected, so a
    reused SPD preset does not reproduce the original calibration."""
    from core.calibration import correct_by_response
    cal = _synthetic_calibrator()
    path = tmp_path / "spd.json"
    cal.save_params(str(path))

    drifting = LinearMapping(cal.mapping.slope_k, cal.mapping.intercept_b + 5.0)
    loaded = SpectralResponseCalibrator.load_params(str(path), drifting)

    # The saved response is indexed by the ORIGINAL ascending wavelength grid.
    # gui._interp_response() resamples it onto whatever grid the CURRENT mapping
    # produces; a drifted mapping therefore shifts the response in wavelength
    # instead of raising an error.
    grid = cal.wavelengths
    shifted = grid + 5.0 / abs(cal.mapping.slope_k)     # 5 px of drift, ≈1.8 nm
    assert np.array_equal(loaded.wavelengths, grid)

    response_original = np.interp(grid, loaded.wavelengths, loaded.response_G)
    response_shifted = np.interp(shifted, loaded.wavelengths, loaded.response_G)
    assert np.array_equal(response_original, response_shifted) is False


def test_loaded_spd_response_is_recomputed_on_the_current_mapping(tmp_path):
    """Mode B: a preset loaded under a drifted mapping must be identical to
    re-running Step 2 with that mapping -- the baked response carries a
    ``SPD(λ-δ)/SPD(λ)`` error that no amount of interpolation can remove."""
    cal = _sensor_grid_calibrator()
    path = tmp_path / "spd.json"
    cal.save_params(str(path))

    drifted = LinearMapping(cal.mapping.slope_k, cal.mapping.intercept_b + 15.0)
    loaded = SpectralResponseCalibrator.load_params(str(path), drifted)
    fresh = _sensor_grid_calibrator(drifted)

    assert loaded.recomputed is True
    assert loaded.drift_level == "WARN"
    assert loaded.drift_nm == pytest.approx(15.0 / abs(cal.mapping.slope_k), abs=1e-9)
    assert np.allclose(loaded.wavelengths, fresh.wavelengths, rtol=0.0, atol=1e-9)
    assert np.array_equal(loaded.final_mask, fresh.final_mask)
    # 6 significant digits on the raw lamp inputs -> ~5e-6 relative worst case
    for channel in ("R", "G", "B"):
        assert np.allclose(getattr(loaded, f"response_{channel}"),
                           getattr(fresh, f"response_{channel}"), rtol=1e-5, atol=1e-12)


def test_mode_a_still_deviates_when_recompute_is_disabled(tmp_path):
    """The same comparison with Mode B off -- this is the deviation the fix removes."""
    cal = _sensor_grid_calibrator()
    path = tmp_path / "spd.json"
    cal.save_params(str(path))

    drifted = LinearMapping(cal.mapping.slope_k, cal.mapping.intercept_b + 15.0)
    fresh = _sensor_grid_calibrator(drifted)
    mode_a = SpectralResponseCalibrator.load_params(str(path), drifted, recompute=False)

    assert mode_a.recomputed is False
    mask = fresh.final_mask
    assert mask.sum() >= 20
    denominator = np.maximum(np.abs(fresh.response_G[mask]), 1e-12)
    deviation = np.abs(mode_a.response_G[mask] - fresh.response_G[mask]) / denominator
    assert float(np.max(deviation)) > 0.05


def test_load_params_falls_back_to_mode_a_on_legacy_files(tmp_path):
    """A preset without ``lamp_*`` must still load (backward compatibility)."""
    cal = _sensor_grid_calibrator()
    path = tmp_path / "spd_legacy.json"
    cal.save_params(str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("lamp_R", "lamp_G", "lamp_B", "spd_reference", "slope", "intercept"):
        payload.pop(key, None)
    path.write_text(json.dumps(payload), encoding="utf-8")

    drifted = LinearMapping(cal.mapping.slope_k, cal.mapping.intercept_b + 15.0)
    loaded = SpectralResponseCalibrator.load_params(str(path), drifted)

    assert loaded.recomputed is False
    assert loaded.drift_level == "UNKNOWN"      # no embedded mapping -> cannot measure
    assert np.array_equal(loaded.wavelengths, cal.wavelengths)


def test_drift_level_thresholds():
    """L3 is a soft hint only: 1.9 nm silent, 3 nm yellow, 12 nm orange."""
    from core.calibration import drift_level
    assert drift_level(None) == "UNKNOWN"
    assert drift_level(0.0) == "OK"
    assert drift_level(1.9) == "OK"
    assert drift_level(2.0) == "OK"
    assert drift_level(3.0) == "WARN"
    assert drift_level(10.0) == "WARN"
    assert drift_level(12.0) == "SEVERE"
    assert drift_level(float("nan")) == "UNKNOWN"


def test_load_params_rejects_a_length_mismatch(tmp_path):
    cal = _sensor_grid_calibrator()
    path = tmp_path / "spd.json"
    cal.save_params(str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["response_G"] = payload["response_G"][:-1]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match its wavelength grid"):
        SpectralResponseCalibrator.load_params(str(path), cal.mapping)


def test_load_params_rejects_a_non_monotonic_wavelength_grid(tmp_path):
    cal = _sensor_grid_calibrator()
    path = tmp_path / "spd.json"
    cal.save_params(str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    wavelengths = payload["wavelengths"]
    wavelengths[len(wavelengths) // 2] = wavelengths[len(wavelengths) // 2 - 1]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="strictly monotonic"):
        SpectralResponseCalibrator.load_params(str(path), cal.mapping)


def test_load_params_rejects_a_degenerate_response(tmp_path):
    cal = _sensor_grid_calibrator()
    path = tmp_path / "spd.json"
    cal.save_params(str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["response_R"] = [0.0] * len(payload["wavelengths"])
    payload["response_G"] = [0.0] * len(payload["wavelengths"])
    payload["response_B"] = [0.0] * len(payload["wavelengths"])
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="at least 20 are required"):
        SpectralResponseCalibrator.load_params(str(path), cal.mapping)


def test_load_params_rejects_a_preset_that_does_not_cover_the_current_axis(tmp_path):
    """L2: ``np.interp`` would silently clamp outside the stored grid."""
    cal = _sensor_grid_calibrator()
    path = tmp_path / "spd.json"
    cal.save_params(str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    keep = (np.asarray(payload["wavelengths"]) >= 500.0) & (
        np.asarray(payload["wavelengths"]) <= 600.0)
    for key in ("wavelengths", "response_R", "response_G", "response_B",
                "lamp_R", "lamp_G", "lamp_B"):
        payload[key] = [v for v, flag in zip(payload[key], keep) if flag]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="当前波长轴未被 preset 覆盖"):
        SpectralResponseCalibrator.load_params(str(path), cal.mapping)


def test_load_params_never_rejects_on_drift_alone(tmp_path):
    """L3 is soft: a 40 px drift is reportable but must still load."""
    cal = _sensor_grid_calibrator()
    path = tmp_path / "spd.json"
    cal.save_params(str(path))

    wildly_drifted = LinearMapping(cal.mapping.slope_k, cal.mapping.intercept_b + 40.0)
    loaded = SpectralResponseCalibrator.load_params(str(path), wildly_drifted)

    assert loaded.drift_level == "SEVERE"
    assert loaded.drift_nm > 10.0


# ---------------------------------------------------------------------------
# CURVE preset (Step 4)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("regression_method", REGRESSION_METHODS)
@pytest.mark.parametrize("metric_mode", METRIC_MODES)
def test_curve_preset_round_trip_preserves_the_fit(tmp_path, metric_mode, regression_method):
    cache = _unmix_cache(metric_mode, regression_method)
    path = tmp_path / "concentration_curve.json"
    save_curve_preset(str(path), cache, "lod_iupac", FLUOROPHORE)

    preset = load_curve_preset(str(path), "lod_iupac")

    assert preset["schema_version"] == 1
    assert preset["analysis_mode"] == "lod_iupac"
    assert preset["fluorophore"] == FLUOROPHORE
    assert preset["metric_mode"] == metric_mode
    assert preset["regression_method"] == regression_method
    assert preset["fit"]["slope"] == cache["slope"]
    assert preset["fit"]["intercept"] == cache["intercept"]
    assert preset["concentrations"] == cache["concentrations"]
    assert preset["means"] == cache["means"]
    assert preset["stds"] == cache["stds"]
    assert (preset["reference_emission"] is not None) == (metric_mode == "unmixing")


@pytest.mark.parametrize("regression_method", REGRESSION_METHODS)
@pytest.mark.parametrize("metric_mode", METRIC_MODES)
def test_curve_preset_reuse_reproduces_the_original_concentration(
        tmp_path, metric_mode, regression_method):
    """The whole point of the CURVE preset: reload it, feed it the same sample
    and get the same number back."""
    cache = _unmix_cache(metric_mode, regression_method)
    path = tmp_path / "concentration_curve.json"
    save_curve_preset(str(path), cache, "lod_iupac", FLUOROPHORE)

    preset = load_curve_preset(str(path), "lod_iupac")
    reused = _sample_metric_from_preset(preset)

    assert math.isclose(reused, cache["sample_conc"], rel_tol=1e-12, abs_tol=1e-12)


@pytest.mark.parametrize("analysis_mode", ["lod_iupac", "legacy", "turbidity"])
def test_validate_curve_preset_accepts_every_analysis_mode(tmp_path, analysis_mode):
    """The ``mode=`` argument handed to ``calculate_concentration`` inside
    ``validate_curve_preset`` only selects the extra LoD/LoQ fields -- it never
    changes slope/intercept -- so no analysis mode can trip the recomputation
    check."""
    cache = _unmix_cache("unmixing", "ols", analysis_mode=analysis_mode)
    path = tmp_path / "curve.json"
    save_curve_preset(str(path), cache, analysis_mode, FLUOROPHORE)

    preset = load_curve_preset(str(path), analysis_mode)
    assert preset["analysis_mode"] == analysis_mode
    assert math.isclose(preset["fit"]["slope"], cache["slope"], rel_tol=1e-12)


def test_validate_curve_preset_tolerates_a_missing_analysis_mode(tmp_path):
    """Direct probe of the ``data.get("analysis_mode", "")`` fallback: even with
    the key absent (empty mode string) the slope/intercept check still passes."""
    cache = _unmix_cache("unmixing", "ols")
    path = tmp_path / "curve.json"
    save_curve_preset(str(path), cache, "lod_iupac", FLUOROPHORE)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("analysis_mode")

    validated = validate_curve_preset(payload)
    assert validated["analysis_mode"] is None
    assert math.isclose(validated["fit"]["slope"], cache["slope"], rel_tol=1e-12)


def test_curve_preset_rejects_a_foreign_analysis_mode(tmp_path):
    cache = _unmix_cache("unmixing", "ols")
    path = tmp_path / "curve.json"
    save_curve_preset(str(path), cache, "lod_iupac", FLUOROPHORE)

    with pytest.raises(ValueError, match="different analysis mode"):
        load_curve_preset(str(path), "legacy")


def test_curve_preset_rejects_a_tampered_slope(tmp_path):
    cache = _unmix_cache("unmixing", "ols")
    path = tmp_path / "curve.json"
    save_curve_preset(str(path), cache, "lod_iupac", FLUOROPHORE)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["fit"]["slope"] *= 1.05

    with pytest.raises(ValueError, match="slope does not match"):
        validate_curve_preset(payload)


def test_curve_preset_requires_the_unmixing_reference(tmp_path):
    cache = _unmix_cache("unmixing", "ols")
    path = tmp_path / "curve.json"
    save_curve_preset(str(path), cache, "lod_iupac", FLUOROPHORE)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metric_mode"] = "unmixing"
    payload.pop("reference_emission")

    with pytest.raises(ValueError, match="[Uu]nmixing presets must include"):
        validate_curve_preset(payload)


def test_curve_preset_requires_a_built_curve():
    with pytest.raises(ValueError, match="Build a Step 4 curve"):
        save_curve_preset("unused.json", {}, "lod_iupac", FLUOROPHORE)


# ---------------------------------------------------------------------------
# WAVELENGTH preset (Step 1) -- missing on the desktop build
# ---------------------------------------------------------------------------
def test_wavelength_preset_has_a_persistence_api(tmp_path):
    """DEF-P05 (fixed): the desktop build can now save and load a Step 1
    WAVELENGTH preset, so the pixel<->nm mapping survives a restart instead of
    living in ``FSSA_GUI.mapping`` for the lifetime of the process."""
    import core.wavelength_preset as wavelength_preset

    step2_api = {"save_params", "load_params"} <= set(dir(SpectralResponseCalibrator))
    assert step2_api, "Step 2 persistence disappeared -- update this test"

    import core.curve_preset as curve_preset
    step4_api = {"save_curve_preset", "load_curve_preset"} <= set(dir(curve_preset))
    assert step4_api, "Step 4 persistence disappeared -- update this test"

    step1_api = {"validate_wavelength_preset", "save_wavelength_preset",
                 "load_wavelength_preset", "mapping_from_preset",
                 "mapping_drift_nm"}
    for required in step1_api:
        assert callable(getattr(wavelength_preset, required)), f"missing {required}"

    # and it really round-trips a mapping
    path = tmp_path / "wavelength_calibration.json"
    wavelength_preset.save_wavelength_preset(
        str(path), LinearMapping(-2.738911, 3198.4744), sensor_size=(4080, 3072))
    restored = wavelength_preset.mapping_from_preset(
        wavelength_preset.load_wavelength_preset(str(path)))
    assert restored.slope_k == -2.738911
    assert restored.intercept_b == 3198.4744


def test_linear_mapping_carries_everything_a_wavelength_preset_needs():
    """If/when a WAVELENGTH preset is added, this is the state it must persist
    (mirrors the columns of the Android ``PresetEntity`` for
    ``PresetTypes.WAVELENGTH``: slope, intercept, gErrorNm, roiStart/roiEnd,
    sensorWidth/Height, peakR/G/B, wlR/G/B, exposureSeconds)."""
    mapping = LinearMapping(-2.738911, 3198.4744)
    # 522.5 nm lands near the measured green centroid of the 10.15 data set.
    assert math.isclose(mapping.wavelength_to_pixel(522.5), 1767.39, abs_tol=0.05)
    assert math.isclose(mapping.pixel_to_wavelength(1767.39), 522.5, abs_tol=0.02)
    assert mapping.pixel_to_wavelength(mapping.wavelength_to_pixel(505.0)) == pytest.approx(505.0)
    assert mapping.pixel_to_wavelength(mapping.wavelength_to_pixel(462.5)) == pytest.approx(462.5)
