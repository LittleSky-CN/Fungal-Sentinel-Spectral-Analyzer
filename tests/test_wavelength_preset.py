"""Step 1 WAVELENGTH preset: round trip, validation and drift measurement.

Mirrors ``tests/test_curve_preset.py`` -- the WAVELENGTH preset is the third
preset type and follows the same validate / save / load contract.
"""
import json
import math

import pytest

from core.calibration import LinearMapping, TwoPointValidation
from core.image_processing import PeakResult
from core.wavelength_preset import (
    APP_VERSION, MIN_ABS_SLOPE, SCHEMA_VERSION,
    load_wavelength_preset, mapping_drift_nm, mapping_from_preset,
    save_wavelength_preset, validate_wavelength_preset,
)

MAPPING = LinearMapping(-2.738911, 3198.4744)
SENSOR_SIZE = (4080, 3072)          # (width, height)
KNOWN_WLS = {"R": 622.5, "G": 522.5, "B": 462.5}


def _peaks():
    """Three PeakResult objects shaped like the ones Step 1 actually produces."""
    return [
        PeakResult("R", 622.5, "red", 1493.50, 1493, 12.0, 1400, 1590, 4800.0,
                   estimator="gauss", n_maxima=1, quality="good", centroid_legacy=1493.9),
        PeakResult("G", 522.5, "green", 1767.39, 1767, 10.5, 1680, 1855, 5200.0,
                   estimator="gauss", n_maxima=1, quality="good", centroid_legacy=1766.8),
        PeakResult("B", 462.5, "blue", 1931.73, 1932, 9.5, 1850, 2010, 3900.0,
                   estimator="edges", n_maxima=2, quality="suspect", centroid_legacy=1930.1),
    ]


def _validation(error_nm=5.512):
    residual_px = error_nm * abs(MAPPING.slope_k)
    predicted = MAPPING.wavelength_to_pixel(522.5)
    return TwoPointValidation(("B", "R"), "G", MAPPING.slope_k, MAPPING.intercept_b,
                              predicted, predicted + residual_px, residual_px, error_nm)


def _write(tmp_path, **overrides):
    path = tmp_path / "wavelength_calibration.json"
    kwargs = dict(
        name="10.15 positioning",
        validation=_validation(),
        peaks=_peaks(),
        x_roi=(1204, 2871),
        sensor_size=SENSOR_SIZE,
        known_wls=KNOWN_WLS,
        exposure_seconds=0.05,
        frames=3,
        rb_swapped=False,
        device_model="",
        analysis_mode="lod_iupac",
    )
    kwargs.update(overrides)
    save_wavelength_preset(str(path), kwargs.pop("mapping", MAPPING), **kwargs)
    return path


# ---------------------------------------------------------------------------
# round trip
# ---------------------------------------------------------------------------
def test_wavelength_preset_round_trip_preserves_the_mapping(tmp_path):
    path = _write(tmp_path)
    preset = load_wavelength_preset(str(path))

    assert preset["preset_type"] == "WAVELENGTH"
    assert preset["schema_version"] == SCHEMA_VERSION
    assert preset["app_version"] == APP_VERSION
    assert math.isclose(preset["slope"], MAPPING.slope_k, rel_tol=1e-12)
    assert math.isclose(preset["intercept"], MAPPING.intercept_b, rel_tol=1e-12)
    assert math.isclose(preset["gErrorNm"], 5.512, abs_tol=1e-9)
    assert preset["roiStart"] == 1204 and preset["roiEnd"] == 2871
    assert preset["sensorWidth"] == 4080 and preset["sensorHeight"] == 3072
    assert preset["wlR"] == 622.5 and preset["wlG"] == 522.5 and preset["wlB"] == 462.5
    assert preset["exposureSeconds"] == 0.05 and preset["frames"] == 3
    assert preset["rbSwapped"] is False
    assert preset["analysisMode"] == "lod_iupac"

    # nested mirror agrees with the flat keys
    assert preset["mapping"] == {"slope_k": preset["slope"], "intercept_b": preset["intercept"]}
    assert preset["roi"] == {"start": 1204, "end": 2871}
    assert preset["device"]["image_width"] == 4080
    assert preset["device"]["image_height"] == 3072
    assert preset["capture"] == {"exposure_seconds": 0.05, "frames": 3, "rb_swapped": False}


def test_wavelength_preset_round_trip_restores_a_usable_mapping(tmp_path):
    path = _write(tmp_path)
    preset = load_wavelength_preset(str(path))
    mapping = mapping_from_preset(preset)

    assert math.isclose(mapping.wavelength_to_pixel(522.5), 1767.39, abs_tol=0.05)
    assert math.isclose(mapping.pixel_to_wavelength(1767.39), 522.5, abs_tol=0.02)


def test_wavelength_preset_keeps_subpixel_peak_positions(tmp_path):
    """Python stores the sub-pixel centroid; Android rounds to whole pixels."""
    path = _write(tmp_path)
    preset = load_wavelength_preset(str(path))

    assert preset["peakR"] == 1493.50
    assert preset["peakG"] == 1767.39
    assert preset["peakB"] == 1931.73
    assert [p["name"] for p in preset["peaks"]] == ["R", "G", "B"]
    assert preset["peaks"][2]["quality"] == "SUSPECT"
    assert preset["peaks"][2]["estimator"] == "edges"
    assert preset["peaks"][2]["n_maxima"] == 2
    assert preset["peaks"][2]["centroid_legacy_px"] == 1930.1


def test_wavelength_preset_round_trip_preserves_rb_swapped(tmp_path):
    path = _write(tmp_path, rb_swapped=True)
    preset = load_wavelength_preset(str(path))
    assert preset["rbSwapped"] is True
    assert preset["capture"]["rb_swapped"] is True


def test_wavelength_preset_requires_a_mapping(tmp_path):
    with pytest.raises(ValueError, match="Run Step 1"):
        save_wavelength_preset(str(tmp_path / "x.json"), None)


# ---------------------------------------------------------------------------
# rejection paths
# ---------------------------------------------------------------------------
def test_wavelength_preset_rejects_a_missing_schema_version(tmp_path):
    path = _write(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("schema_version")

    with pytest.raises(ValueError, match="Unsupported or invalid"):
        validate_wavelength_preset(payload)


@pytest.mark.parametrize("schema_version", [0, 2, "1", True])
def test_wavelength_preset_rejects_a_foreign_schema_version(tmp_path, schema_version):
    path = _write(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = schema_version

    with pytest.raises(ValueError, match="Unsupported or invalid"):
        validate_wavelength_preset(payload)


def test_wavelength_preset_rejects_a_foreign_preset_type(tmp_path):
    path = _write(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["preset_type"] = "SPD"

    with pytest.raises(ValueError, match="Not a WAVELENGTH preset"):
        validate_wavelength_preset(payload)


def test_wavelength_preset_rejects_a_tiny_slope(tmp_path):
    """A flat mapping makes ``pixel_to_wavelength`` divide by zero."""
    with pytest.raises(ValueError, match="slope is too small"):
        _write(tmp_path, mapping=LinearMapping(MIN_ABS_SLOPE * 0.5, 100.0))


def test_wavelength_preset_rejects_a_missing_intercept(tmp_path):
    path = _write(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("intercept")
    payload["mapping"].pop("intercept_b")

    with pytest.raises(ValueError, match="Missing intercept"):
        validate_wavelength_preset(payload)


def test_wavelength_preset_rejects_a_missing_slope(tmp_path):
    path = _write(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("slope")
    payload["mapping"].pop("slope_k")

    with pytest.raises(ValueError, match="Missing slope"):
        validate_wavelength_preset(payload)


@pytest.mark.parametrize("height", [0, -3072])
def test_wavelength_preset_rejects_a_non_positive_sensor_height(tmp_path, height):
    """sensorHeight is the profile length -- it defines the SPD array size."""
    with pytest.raises(ValueError, match="sensorHeight must be positive"):
        _write(tmp_path, sensor_size=(4080, height))


def test_wavelength_preset_rejects_a_non_positive_sensor_height_on_load(tmp_path):
    path = _write(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["sensorHeight"] = 0
    payload["device"]["image_height"] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="sensorHeight must be positive"):
        load_wavelength_preset(str(path))


def test_wavelength_preset_rejects_an_inverted_roi(tmp_path):
    with pytest.raises(ValueError, match="roiStart must not be greater"):
        _write(tmp_path, x_roi=(2871, 1204))


def test_wavelength_preset_rejects_a_foreign_analysis_mode(tmp_path):
    path = _write(tmp_path)

    with pytest.raises(ValueError, match="different analysis mode"):
        load_wavelength_preset(str(path), "legacy")


def test_wavelength_preset_rejects_a_tampered_mapping(tmp_path):
    path = _write(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["slope"] *= 1.05

    reloaded = validate_wavelength_preset(payload)
    assert not math.isclose(reloaded["slope"], MAPPING.slope_k, rel_tol=1e-9)
    assert reloaded["mapping"]["slope_k"] == reloaded["slope"]


def test_wavelength_preset_accepts_a_legacy_file_without_nested_groups(tmp_path):
    """Older / hand-written files only carry the flat keys -- still accepted."""
    payload = {
        "preset_type": "WAVELENGTH",
        "schema_version": 1,
        "slope": MAPPING.slope_k,
        "intercept": MAPPING.intercept_b,
        "gErrorNm": 5.512,
        "roiStart": 1204,
        "roiEnd": 2871,
        "sensorWidth": 4080,
        "sensorHeight": 3072,
        "peakR": 1493.5,
        "peakG": 1767.39,
        "peakB": 1931.73,
        "wlR": 622.5,
        "wlG": 522.5,
        "wlB": 462.5,
        "exposureSeconds": 0.05,
        "frames": 3,
        "rbSwapped": False,
    }
    validated = validate_wavelength_preset(payload)
    assert validated["peaks"] is None
    assert validated["roi"] == {"start": 1204, "end": 2871}
    assert validated["quality"] == "GOOD"


# ---------------------------------------------------------------------------
# drift
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shift_px, expected_nm", [(0.0, 0.0), (1.0, 0.36511), (5.0, 1.82555)])
def test_mapping_drift_nm_scales_with_the_pixel_shift(tmp_path, shift_px, expected_nm):
    preset = validate_wavelength_preset(json.loads(_write(tmp_path).read_text(encoding="utf-8")))
    drifted = LinearMapping(MAPPING.slope_k, MAPPING.intercept_b + shift_px)

    drift = mapping_drift_nm(preset, drifted)
    assert drift == pytest.approx(expected_nm, abs=1e-4)


def test_mapping_drift_nm_is_none_without_a_mapping(tmp_path):
    preset = validate_wavelength_preset(json.loads(_write(tmp_path).read_text(encoding="utf-8")))
    assert mapping_drift_nm(preset, None) is None


def test_mapping_drift_nm_is_none_without_a_sensor_height(tmp_path):
    payload = json.loads(_write(tmp_path).read_text(encoding="utf-8"))
    payload.pop("sensorHeight")
    payload["device"].pop("image_height")
    preset = validate_wavelength_preset(payload)

    assert mapping_drift_nm(preset, MAPPING) is None
