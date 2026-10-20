"""Python <-> Android (MycoSpec v2.2.6) image-processing parity.

Every rule in here is transcribed from
``app/src/main/java/org/fungalsentinel/app/analysis/FrameProfileExtractor.kt``
and ``WavelengthCalibrator.kt``.  The point of the module is that the row
profiles fed into Step 1 / Step 2 are produced by the *same* algorithm on both
ends, so a given DNG yields the same numbers on desktop and on phone.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import tifffile as tiff

from core.image_processing import (
    CROSSTALK_G_FROM_R, CROSSTALK_R_FROM_G, DEFAULT_CFA,
    _cfa_index_map, _cfa_pattern_from_page, _demosaic_channel, _normalise_samples,
    auto_detect_x_roi, calibrate_wavelength, column_energy, demosaic_bayer,
    find_peak_auto, make_channel_profiles, parse_known_wavelengths,
    read_dng_frame, resolve_x_roi,
)


RGGB = (0, 1, 1, 2)
BGGR = (2, 1, 1, 0)


def _raw_channels(sensor, black=0.0, exposure=1.0, cfa=RGGB):
    """Interpolated channels *before* the crosstalk correction."""
    values = _normalise_samples(sensor, black, exposure)
    cfa_map = _cfa_index_map(*values.shape, cfa)
    return tuple(_demosaic_channel(values, cfa_map, target) for target in (0, 1, 2))


def _bayer_sensor(cfa, height=8, width=8, values=None):
    """A synthetic raw grid whose every 2x2 block matches ``cfa``."""
    idx = _cfa_index_map(height, width, cfa)
    if values is None:
        return idx.astype(np.uint16) * 100 + 10
    return values


class FakeTag:
    def __init__(self, value):
        self.value = value


class FakePage:
    def __init__(self, tags):
        self.tags = tags


# ---------------------------------------------------------------------------
# 1. CFA tag 33422
# ---------------------------------------------------------------------------
def test_cfa_tag_bytes_are_read_verbatim():
    """Android reads the four tag bytes directly (:489)."""
    page = FakePage({33422: FakeTag(b"\x02\x01\x01\x00")})   # BGGR
    assert _cfa_pattern_from_page(page) == BGGR
    page = FakePage({33422: FakeTag(b"\x00\x01\x01\x02")})   # RGGB
    assert _cfa_pattern_from_page(page) == RGGB


def test_cfa_tag_one_based_values_are_normalised():
    """min==1 && max==3 -> subtract 1 (:494)."""
    page = FakePage({33422: FakeTag(b"\x01\x02\x02\x03")})
    assert _cfa_pattern_from_page(page) == RGGB


@pytest.mark.parametrize("raw", [b"\x00\x00\x00\x00", b"\x03\x03\x03\x03",
                                 b"\x00\x01\x02\x03", b"\x01\x01\x01\x02"])
def test_invalid_cfa_tag_falls_back_to_rggb(raw):
    assert _cfa_pattern_from_page(FakePage({33422: FakeTag(raw)})) == DEFAULT_CFA


def test_missing_cfa_tag_falls_back_to_rggb():
    assert _cfa_pattern_from_page(FakePage({})) == DEFAULT_CFA


def test_cfa_index_map_matches_the_android_lookup():
    idx = _cfa_index_map(4, 4, BGGR)
    for y in range(4):
        for x in range(4):
            assert idx[y, x] == BGGR[(y & 1) * 2 + (x & 1)]


# ---------------------------------------------------------------------------
# 2. demosaic + crosstalk
# ---------------------------------------------------------------------------
def test_direct_samples_are_used_verbatim():
    """A pixel already carrying the target colour is returned as-is (:322-325)."""
    red, green, blue = _raw_channels(_bayer_sensor(RGGB))
    raw = np.asarray(_bayer_sensor(RGGB), dtype=np.float64)
    assert red[0, 0] == pytest.approx(raw[0, 0])        # (0, 0) is R in RGGB
    assert blue[1, 1] == pytest.approx(raw[1, 1])       # (1, 1) is B


def test_black_level_and_exposure_are_applied_before_demosaic():
    red, green, blue = _raw_channels(_bayer_sensor(RGGB), black=10.0, exposure=2.0)
    assert red[0, 0] == pytest.approx(0.0)      # (10 - 10) / 2
    assert blue[1, 1] == pytest.approx(100.0)   # (210 - 10) / 2


def test_rgb_channels_follow_the_cfa_tag():
    """The same raw grid decoded as RGGB vs BGGR must swap R and B."""
    sensor = _bayer_sensor(BGGR, 16, 16)
    as_bggr = _raw_channels(sensor, cfa=BGGR)
    as_rggb = _raw_channels(sensor, cfa=RGGB)
    assert np.allclose(as_bggr[0], as_rggb[2], atol=1e-5)
    assert np.allclose(as_bggr[2], as_rggb[0], atol=1e-5)
    assert np.allclose(as_bggr[1], as_rggb[1], atol=1e-5)


def test_green_uses_only_orthogonal_neighbours():
    """Bayer: every orthogonal neighbour of a non-green pixel is green."""
    sensor = _bayer_sensor(RGGB, 8, 8)
    raw = np.asarray(sensor, dtype=np.float64)
    _, green, _ = _raw_channels(sensor)
    # (1, 1) is B; its orthogonal neighbours (0,1) (2,1) (1,0) (1,2) are all G
    expected = (raw[0, 1] + raw[2, 1] + raw[1, 0] + raw[1, 2]) / 4.0
    assert green[1, 1] == pytest.approx(expected)


def test_red_uses_diagonal_neighbours_before_orthogonal():
    """On a green pixel the diagonals are green too -> orthogonal fallback."""
    sensor = _bayer_sensor(RGGB, 8, 8)
    raw = np.asarray(sensor, dtype=np.float64)
    red, _, _ = _raw_channels(sensor)
    # (0, 1) is G; diagonals are all G, orthogonal holds (0,0) and (0,2) which are R
    assert red[0, 1] == pytest.approx((raw[0, 0] + raw[0, 2]) / 2.0)
    # (1, 1) is B; all four diagonals are R -> mean of four
    assert red[1, 1] == pytest.approx(
        (raw[0, 0] + raw[0, 2] + raw[2, 0] + raw[2, 2]) / 4.0)


def test_out_of_bounds_neighbours_are_skipped():
    """A corner pixel simply averages the neighbours that exist."""
    sensor = _bayer_sensor(RGGB, 8, 8)
    raw = np.asarray(sensor, dtype=np.float64)
    _, green, _ = _raw_channels(sensor)
    assert green[0, 0] == pytest.approx((raw[1, 0] + raw[0, 1]) / 2.0)


def test_crosstalk_correction_is_order_sensitive():
    """g is corrected first, and r uses the *corrected* g (:304-305)."""
    # R is deliberately much brighter than G so the corrected g clamps to zero
    # and the two orderings differ instead of both saturating at zero.
    idx = _cfa_index_map(8, 8, RGGB)
    sensor = np.where(idx == 0, 1000, np.where(idx == 1, 100, 50)).astype(np.uint16)
    rgb = demosaic_bayer(sensor, 0.0, 1.0, RGGB)
    raw = np.asarray(sensor, dtype=np.float64)
    r_raw, g_raw = raw[0, 0], (raw[0, 1] + raw[1, 0]) / 2.0
    g_fixed = max(0.0, g_raw - CROSSTALK_G_FROM_R * r_raw)
    r_fixed = max(0.0, r_raw - CROSSTALK_R_FROM_G * g_fixed)
    assert rgb[0, 0, 0] == pytest.approx(r_fixed)
    assert rgb[0, 0, 1] == pytest.approx(g_fixed)
    # the wrong order would use the *uncorrected* g -- and give a different r
    assert r_fixed != pytest.approx(max(0.0, r_raw - CROSSTALK_R_FROM_G * g_raw))


def test_crosstalk_never_goes_negative():
    sensor = np.zeros((8, 8), dtype=np.uint16)
    rgb = demosaic_bayer(sensor, 0.0, 1.0, RGGB)
    assert np.all(rgb >= 0.0)


# ---------------------------------------------------------------------------
# 3. column energy / ROI
# ---------------------------------------------------------------------------
def test_column_energy_is_the_top_k_quantile_of_the_raw_column():
    """Android: kTop = max(4, h*0.005 + 4); p995 takes index int(0.005*h)."""
    height, width = 1000, 4
    rng = np.random.default_rng(7)
    sensor = rng.integers(0, 1000, size=(height, width)).astype(np.uint16)
    col_energy = column_energy(sensor, 0.0, 1.0)

    values = sensor.astype(np.float64)
    k_top = max(4, int(height * 0.005) + 4)
    index = min(k_top - 1, int(0.005 * height))
    for x in range(width):
        top = np.sort(values[:, x])[::-1][:k_top]
        assert col_energy[x] == pytest.approx(float(top[index]), rel=1e-6)


def test_column_energy_ignores_black_and_exposure():
    sensor = np.full((500, 3), 300, dtype=np.uint16)
    assert column_energy(sensor, 100.0, 2.0) == pytest.approx([100.0] * 3)


def test_auto_detect_x_roi_adds_roughly_80px_of_margin():
    """Android: max(0, first - 80) .. min(width - 1, last - 1 + 80) (:138)."""
    width = 4000
    profile = np.zeros(width)
    profile[1500:1600] = 10.0        # a single bright band
    start, end = auto_detect_x_roi(profile, width)
    assert abs(start - (1500 - 80)) <= 20
    assert abs(end - (1600 + 80)) <= 20


def test_auto_detect_x_roi_clamps_the_upper_bound_to_the_last_column():
    """min(width - 1, ...) -- the ROI must never fall off the right edge."""
    width = 4000
    profile = np.zeros(width)
    profile[-200:] = 10.0
    start, end = auto_detect_x_roi(profile, width)
    assert end == width
    assert 0 <= start < width


def test_auto_detect_x_roi_falls_back_to_the_centre_band():
    start, end = auto_detect_x_roi(np.zeros(1000), 1000)
    assert start == 350 and end == 650


def test_resolve_x_roi_prefers_the_fixed_region():
    """resolveXRoi: an explicit ROI is used verbatim, never re-detected (:146)."""
    assert resolve_x_roi(np.zeros(2000), 2000, (500, 900)) == (500, 901)
    # clamped and never inverted
    assert resolve_x_roi(np.zeros(2000), 2000, (-5, 99999)) == (0, 2000)


def test_fixed_roi_is_stable_across_frames():
    """A preset ROI must produce identical profiles no matter the content."""
    rng = np.random.default_rng(3)
    rgb = rng.random((64, 40, 3)).astype(np.float32)
    first = make_channel_profiles(rgb, resolve_x_roi(None, 40, (10, 20)))
    second = make_channel_profiles(rgb, resolve_x_roi(None, 40, (10, 20)))
    assert np.array_equal(first["red"], second["red"])


# ---------------------------------------------------------------------------
# 4. R/B swap self-check
# ---------------------------------------------------------------------------
def _gaussian_profile(n, center, width, height):
    x = np.arange(n, dtype=np.float64)
    return height * np.exp(-0.5 * ((x - center) / width) ** 2)


def test_rb_swap_is_detected_when_r_and_b_sit_on_the_wrong_sides_of_g():
    """Android: pR < pG && pB > pG -> start from the swapped state (:47-49).

    G at 175 px makes the swapped fit (blue/green/red as R/G/B) self-consistent,
    so it is kept rather than being discarded by the fallback.
    """
    size = 400
    profiles = {
        "red": _gaussian_profile(size, 100, 8, 50.0),
        "green": _gaussian_profile(size, 175, 8, 60.0),
        "blue": _gaussian_profile(size, 300, 8, 40.0),
    }
    fit = calibrate_wavelength(profiles, parse_known_wavelengths(None))
    assert fit is not None
    assert fit["swapped"] is True
    assert abs(fit["error_nm"]) < 1.0


def test_no_swap_for_the_natural_pixel_order():
    """R above G and B below G -> no swap, same G position."""
    size = 400
    profiles = {
        "red": _gaussian_profile(size, 300, 8, 50.0),
        "green": _gaussian_profile(size, 175, 8, 60.0),
        "blue": _gaussian_profile(size, 100, 8, 40.0),
    }
    fit = calibrate_wavelength(profiles, parse_known_wavelengths(None))
    assert fit is not None
    assert fit["swapped"] is False
    assert abs(fit["error_nm"]) < 1.0


def test_swap_fallback_picks_the_smaller_g_error():
    """Android retries the other swap state when the first one FAILEDs (:53-61)."""
    size = 400
    # Unswapped the fit predicts G at 175 px (40 nm off); swapped it predicts
    # 225 px, exactly where G sits -- so the fallback must pick the swap.
    profiles = {
        "red": _gaussian_profile(size, 300, 8, 50.0),
        "green": _gaussian_profile(size, 225, 8, 60.0),
        "blue": _gaussian_profile(size, 100, 8, 40.0),
    }
    fit = calibrate_wavelength(profiles, parse_known_wavelengths(None))
    assert fit is not None
    assert fit["swapped"] is True
    assert abs(fit["error_nm"]) < 1.0


def test_calibrate_wavelength_returns_none_without_peaks():
    knowns = parse_known_wavelengths(None)
    flat = {"red": np.zeros(200), "green": np.zeros(200), "blue": np.zeros(200)}
    assert calibrate_wavelength(flat, knowns) is None


def test_peak_detection_is_unaffected_by_the_channel_label():
    """find_peak_auto only reads the nominal wavelength, never the channel."""
    knowns = parse_known_wavelengths(None)
    profile = _gaussian_profile(400, 250, 6, 30.0)
    as_red = find_peak_auto(profile, knowns["R"])
    as_blue = find_peak_auto(profile, knowns["B"])
    assert as_red.y_centroid == pytest.approx(as_blue.y_centroid)


# ---------------------------------------------------------------------------
# 5. Real 10.15 DNG -- pins the CFA verdict so the wrong conclusion cannot come
#    back.  Opt-in like tests/test_preset_reuse_realdata.py (FSSA_E2E=1).
#
#    Known LED lines of the 10.15 positioning frame and the pixel rows they
#    land on under the calibrated mapping (px = -2.7398 * lambda + 3198.91):
#        R 622.5 nm -> px 1493      G 522.5 nm -> px 1767      B 462.5 nm -> px 1931
#    The G centroid is measured ~15 px above its predicted row (+5.5 nm); that
#    residual is the documented Step 1 tolerance, not an error.
# ---------------------------------------------------------------------------
DATA_DIR = Path("D:/Documents/iGem/Python/FSSA_Documents/10.15")
POSITIONING = DATA_DIR / "positioning.dng"
RED_PEAK_PX, GREEN_PEAK_PX, BLUE_PEAK_PX = 1493.0, 1767.0, 1931.0
PEAK_TOL_PX = 40.0

_real_data = pytest.mark.skipif(
    os.environ.get("FSSA_E2E") != "1" or not POSITIONING.is_file(),
    reason="set FSSA_E2E=1 and mount the 10.15 DNG sample set to run this test",
)


def test_android_cfa_lookup_places_red_at_the_odd_odd_pixel():
    """cfa[(y & 1) * 2 + (x & 1)] -- for BGGR the red sample is at (odd, odd)."""
    assert BGGR[(1 & 1) * 2 + (1 & 1)] == 0
    assert BGGR[(0 & 1) * 2 + (0 & 1)] == 2
    assert BGGR[(0 & 1) * 2 + (1 & 1)] == 1
    assert BGGR[(1 & 1) * 2 + (0 & 1)] == 1


@_real_data
def test_cfa_tag_of_the_reference_frame_is_bggr():
    """tag 33422 of positioning.dng is b'\\x02\\x01\\x01\\x00' == BGGR.

    Any hard-coded RGGB assumption would swap R and B on this sensor.
    """
    with tiff.TiffFile(str(POSITIONING)) as tif:
        cfa = _cfa_pattern_from_page(tif.pages[0])
    assert cfa == BGGR
    assert cfa[0] == 2 and cfa[3] == 0            # B G / G R
    assert cfa[(1 & 1) * 2 + (1 & 1)] == 0        # red sits at (odd, odd)


@_real_data
def test_decoded_peaks_are_physically_consistent():
    """Each filter must peak on its own LED line.

    A previous QA conclusion claimed the red peak should move to ~1930 px once
    the CFA tag is honoured.  That is wrong: 1930 px is the 462.5 nm line, i.e.
    where the *blue* filter peaks -- an RGGB decode would put red there and is
    therefore physically impossible on this sensor.  The negative assertion
    below locks the correct verdict.
    """
    frame = read_dng_frame(POSITIONING)
    assert frame["cfa"] == BGGR
    profiles = make_channel_profiles(frame["rgb"], frame["x_roi"])
    knowns = parse_known_wavelengths(None)
    red = find_peak_auto(profiles["red"], knowns["R"]).y_centroid
    green = find_peak_auto(profiles["green"], knowns["G"]).y_centroid
    blue = find_peak_auto(profiles["blue"], knowns["B"]).y_centroid

    assert red == pytest.approx(RED_PEAK_PX, abs=PEAK_TOL_PX)
    assert green == pytest.approx(GREEN_PEAK_PX, abs=PEAK_TOL_PX)
    assert blue == pytest.approx(BLUE_PEAK_PX, abs=PEAK_TOL_PX)
    # the inverted decode would place red on the blue line -- must never happen
    assert abs(red - BLUE_PEAK_PX) > PEAK_TOL_PX
    assert abs(blue - RED_PEAK_PX) > PEAK_TOL_PX


@_real_data
def test_rb_swap_is_not_triggered_on_the_reference_frame():
    """``swapped`` must end up False on 10.15, and here is why.

    The pixel order is R(1493) < G(1782) < B(1931), which satisfies Android's
    ``pR < pG && pB > pG`` heuristic, so the FIRST attempt runs in the swapped
    state and yields gErr ~ +47 nm -> FAILED.  The fallback then tries the
    unswapped state, gets gErr ~ +5.57 nm (WARNING) and wins.  The fallback
    branch is what saves this data set; without it the calibration would be
    reported as swapped.
    """
    frame = read_dng_frame(POSITIONING)
    profiles = make_channel_profiles(frame["rgb"], frame["x_roi"])
    fit = calibrate_wavelength(profiles, parse_known_wavelengths(None))
    assert fit is not None
    assert fit["swapped"] is False
    assert fit["quality"] in ("PASS", "WARNING")
    assert abs(fit["error_nm"]) < 10.0
