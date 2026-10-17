"""End-to-end preset reuse on the real 10.15 DNG sample set.

Opt-in: decoding seven 25 MB DNGs takes a couple of minutes, so this module is
skipped unless ``FSSA_E2E=1`` is exported::

    FSSA_E2E=1 python -m pytest tests/test_preset_reuse_realdata.py -q

Layout of ``D:/Documents/iGem/Python/FSSA_Documents/10.15`` (per the user's
own note in ``新建 文本文档.txt``)::

    positioning = step 1      spd = step 2
    S   = Sample              B0  = Blank
    B1  = 100 mg/L            B2  = 60 mg/L            B3  = 20 mg/L

The harness mirrors ``gui/main_window.py`` call-for-call
(``import``-free: the GUI module needs tkinter, which headless CI lacks).
"""
import os
from pathlib import Path

import numpy as np
import pytest

from config import FLUOROPHORES
from core.calibration import (
    LinearMapping, SpectralResponseCalibrator, analyze_blank_subtracted_replicates,
    correct_by_response,
)
from core.curve_preset import load_curve_preset, save_curve_preset
from core.image_processing import (
    auto_detect_x_roi, find_peak_auto, get_average_profiles, get_exposure_time,
    load_true_spd_data, make_channel_profiles, parse_known_wavelengths,
    read_dng_linear_rgb, read_dng_frame, np_trapz,
)
from core.quantification import calculate_concentration
from core.unmixing import (
    unmix_spectrum, select_reference_emission, default_unmix_window,
    slope_t_statistic,
)

DATA = Path("D:/Documents/iGem/Python/FSSA_Documents/10.15")
TRUE_SPD = Path("D:/Documents/iGem/Python/FSSA_Documents/true_spd.csv")

pytestmark = pytest.mark.skipif(
    os.environ.get("FSSA_E2E") != "1" or not DATA.is_dir(),
    reason="set FSSA_E2E=1 and mount the 10.15 DNG sample set to run this module",
)

FLUOROPHORE = "Ypet (Venus variant)"
ANALYSIS_MODE = "lod_iupac"
SAMPLE = "S"
BLANK = "B0"
STANDARDS = [(100.0, "B1"), (60.0, "B2"), (20.0, "B3")]


# ---------------------------------------------------------------------------
# one shared decode of the seven DNGs
# ---------------------------------------------------------------------------
class Session:
    """Decoded profiles + the GUI's in-memory state, mirroring FSSA_GUI."""

    def __init__(self):
        # 【v1.4.6】一次解码同时拿到 RGB、曝光与列能量 ROI（Android profileFromDng 口径）
        frame = read_dng_frame(DATA / "positioning.dng")
        rgb_first, exp_first = frame["rgb"], frame["exposure"]
        self.exp_first = exp_first
        self.x_roi = frame["x_roi"]
        self.frames = {}
        for name in ("positioning", "spd", "S", "B0", "B1", "B2", "B3"):
            rgb, exp = read_dng_linear_rgb(DATA / f"{name}.dng")
            prof = make_channel_profiles(rgb, self.x_roi)
            self.frames[name] = {
                "red": prof["red"], "green": prof["green"], "blue": prof["blue"],
                "exp": exp,
            }
        self.mapping = None
        self.calibrator = None

    # --- Step 1 ----------------------------------------------------------
    def step1(self):
        knowns = parse_known_wavelengths(None)
        avg = self.frames["positioning"]
        peaks = [find_peak_auto(avg[knowns[k].channel_name], knowns[k]) for k in ("R", "G", "B")]
        p_b = next(p for p in peaks if p.name == "B")
        p_r = next(p for p in peaks if p.name == "R")
        p_g = next(p for p in peaks if p.name == "G")
        k = (p_r.y_centroid - p_b.y_centroid) / (p_r.wavelength_nm - p_b.wavelength_nm)
        b = p_b.y_centroid - k * p_b.wavelength_nm
        self.mapping = LinearMapping(k, b)
        self.peaks = peaks
        self.wl_err = (p_g.y_centroid - (k * p_g.wavelength_nm + b)) / abs(k)
        return self.mapping

    # --- Step 2 ----------------------------------------------------------
    def step2(self):
        p = self.frames["spd"]
        y_pixels = np.arange(len(p["red"]), dtype=np.float64)
        measured = np.column_stack((p["red"], p["green"], p["blue"]))
        true_wl, true_val = load_true_spd_data(TRUE_SPD)
        self.calibrator = SpectralResponseCalibrator(self.mapping)
        self.calibrator.calibrate(y_pixels, measured, true_wl, true_val)
        return self.calibrator

    # --- helpers (copied from gui/main_window.py) ------------------------
    def _interp_response(self, wavelengths, ch_name):
        resp = {"red": self.calibrator.response_R,
                "green": self.calibrator.response_G,
                "blue": self.calibrator.response_B}[ch_name]
        return np.interp(wavelengths, self.calibrator.wavelengths, resp)

    def raw_channel_spectra(self, names, fluor):
        ch_idx = {"red": 0, "green": 1, "blue": 2}[fluor["channel"]]
        specs, wls = [], None
        for n in names:
            p = self.frames[n]
            rgb = np.column_stack([p["red"], p["green"], p["blue"]])
            wavelengths = self.mapping.pixel_to_wavelength(np.arange(len(rgb)))
            if wavelengths[0] > wavelengths[-1]:
                wavelengths = wavelengths[::-1]
                rgb = rgb[::-1, :]
            if self.calibrator:
                response = self._interp_response(wavelengths, fluor["channel"])
                intensity = correct_by_response(rgb[:, ch_idx], response)
            else:
                intensity = rgb[:, ch_idx]
            specs.append(intensity)
            wls = wavelengths
        return wls, specs

    def net_areas(self, blank_names, sample_names, fluor):
        bp = self.frames[blank_names[0]]
        blank_rgb = np.column_stack([bp["red"], bp["green"], bp["blue"]])
        sample_rgb = np.stack([
            np.column_stack([self.frames[n]["red"], self.frames[n]["green"],
                             self.frames[n]["blue"]]) for n in sample_names])
        ci = {"red": 0, "green": 1, "blue": 2}[fluor["channel"]]
        wavelengths = self.mapping.pixel_to_wavelength(np.arange(len(blank_rgb)))
        if wavelengths[0] > wavelengths[-1]:
            wavelengths = wavelengths[::-1]
            blank_rgb = blank_rgb[::-1, :]
            sample_rgb = sample_rgb[:, ::-1, :]
        response = self._interp_response(wavelengths, fluor["channel"]) if self.calibrator else None
        tw, fw = fluor["peak_wl"], fluor["fwhm"]
        mask = (wavelengths >= tw - fw / 2) & (wavelengths <= tw + fw / 2)
        areas, _ = analyze_blank_subtracted_replicates(
            blank_rgb[:, ci], sample_rgb[:, :, ci], wavelengths, mask, response)
        return areas.tolist()


def build_curve(sess, metric_mode, regression_method):
    """Mirror of gui/main_window.calculate_concentration."""
    fluor = FLUOROPHORES[FLUOROPHORE]
    concentrations, means, stds = [], [], []

    if metric_mode == "unmixing":
        rows = []
        for conc, sname in STANDARDS:
            wls_s, s_specs = sess.raw_channel_spectra([sname], fluor)
            _, b_specs = sess.raw_channel_spectra([BLANK], fluor)
            rows.append({"conc": conc, "wls": wls_s, "sample": s_specs,
                         "blank": b_specs, "exp": sess.frames[sname]["exp"]})
        wls = rows[0]["wls"]
        emis_mask = (wls >= fluor["peak_wl"] - fluor["fwhm"] / 2) & (
            wls <= fluor["peak_wl"] + fluor["fwhm"] / 2)
        ref_index, E, auc_E = select_reference_emission(
            wls,
            [(f"#{i + 1} ({r['conc']:g})", np.mean(r["sample"], axis=0),
              np.mean(r["blank"], axis=0)) for i, r in enumerate(rows)],
            emis_mask)
        ref_row = rows[ref_index]
        b_ref = np.mean(ref_row["blank"], axis=0)
        sm = (wls > 460) & (wls < 520)
        if np.any(sm):
            scatter_wl = float(wls[sm][int(np.argmax(b_ref[sm]))])
            lo = min(max(scatter_wl + 2.0, fluor["peak_wl"] - 30.0), fluor["peak_wl"] - 5.0)
        else:
            lo, _ = default_unmix_window(fluor["peak_wl"])
        hi = fluor["peak_wl"] + 90.0
        win_mask = (wls >= lo) & (wls <= hi)

        per_row = []
        for r in rows:
            b_mean = np.mean(r["blank"], axis=0)
            alphas = [unmix_spectrum(sp, E, b_mean, win_mask)["alpha"] for sp in r["sample"]]
            metrics = [a * auc_E for a in alphas]
            concentrations.append(r["conc"])
            means.append(float(np.mean(metrics)))
            stds.append(float(np.std(metrics, ddof=1)) if len(metrics) > 1 else 0.0)
            per_row.append({"conc": r["conc"], "alpha": float(np.mean(alphas)),
                            "s": 0.0, "k": 0.0, "r2": 1.0})

        _, u_samples = sess.raw_channel_spectra([SAMPLE], fluor)
        _, u_blanks = sess.raw_channel_spectra([BLANK], fluor)
        b_u = np.mean(u_blanks, axis=0)
        alphas_u = [unmix_spectrum(sp, E, b_u, win_mask)["alpha"] for sp in u_samples]
        sample_mean_area = float(np.mean(alphas_u)) * auc_E
        unmix = {"window": (lo, hi), "ref_conc": ref_row["conc"], "auc_E": auc_E,
                 "per_row": per_row, "alpha": float(np.mean(alphas_u)), "u_s": 0.0,
                 "u_k": 0.0, "wls": wls, "E": E, "B": b_u,
                 "sample_mean_spec": np.mean(u_samples, axis=0)}
    else:
        unmix = None
        for conc, sname in STANDARDS:
            areas = sess.net_areas([BLANK], [sname], fluor)
            concentrations.append(conc)
            means.append(float(np.mean(areas)))
            stds.append(float(np.std(areas, ddof=1)) if len(areas) > 1 else 0.0)
        sample_mean_area = float(np.mean(sess.net_areas([BLANK], [SAMPLE], fluor)))

    calc = calculate_concentration(concentrations, means, sample_mean_area,
                                   mode=ANALYSIS_MODE,
                                   regression_method=regression_method)
    cache = dict(calc)
    cache.update({"concentrations": concentrations, "means": means, "stds": stds,
                  "metric_mode": metric_mode,
                  "t_stat": slope_t_statistic(calc["r_value"], len(concentrations)),
                  "unmix": unmix, "blocks": [], "scatter_peak_wl": float("nan"),
                  "fluorophore": FLUOROPHORE})
    return cache


def apply_preset(sess, preset):
    """Mirror of gui/main_window.calculate_with_curve_preset."""
    fluor = FLUOROPHORES[preset["fluorophore"]]
    if preset["metric_mode"] == "unmixing":
        wls_s, s_specs = sess.raw_channel_spectra([SAMPLE], fluor)
        _, b_specs = sess.raw_channel_spectra([BLANK], fluor)
        ref = preset["reference_emission"]
        emission = np.interp(wls_s, ref["wavelengths"], ref["emission"], left=0.0, right=0.0)
        lo, hi = ref["window_nm"]
        win = (wls_s >= lo) & (wls_s <= hi)
        b_mean = np.mean(b_specs, axis=0)
        metrics = [unmix_spectrum(sp, emission, b_mean, win)["alpha"] * ref["auc"]
                   for sp in s_specs]
        sample_mean_area = float(np.mean(metrics))
    else:
        sample_mean_area = float(np.mean(sess.net_areas([BLANK], [SAMPLE], fluor)))
    fit = preset["fit"]
    return (sample_mean_area - fit["intercept"]) / fit["slope"], sample_mean_area


@pytest.fixture(scope="module")
def session():
    sess = Session()
    sess.step1()
    sess.step2()
    return sess


# ---------------------------------------------------------------------------
def test_step1_produces_a_usable_mapping(session):
    assert session.mapping is not None
    assert session.mapping.slope_k != 0.0
    # The 10.15 positioning frame is a WARNING-tier calibration (|G error| > 2.5 nm)
    assert abs(session.wl_err) < 10.0


def test_step2_calibrates_three_channels(session):
    cal = session.calibrator
    assert cal.response_R is not None
    assert cal.response_G is not None
    assert cal.response_B is not None
    assert np.any(cal.response_G > 0)
    assert cal.final_mask.sum() > 100


def test_spd_preset_round_trip_is_bit_exact(session, tmp_path):
    cal = session.calibrator
    path = tmp_path / "spd_calibration_params.json"
    cal.save_params(str(path))

    loaded = SpectralResponseCalibrator.load_params(str(path), session.mapping)

    assert np.array_equal(cal.wavelengths, loaded.wavelengths)
    assert np.array_equal(cal.response_R, loaded.response_R)
    assert np.array_equal(cal.response_G, loaded.response_G)
    assert np.array_equal(cal.response_B, loaded.response_B)


def test_reloaded_spd_preset_gives_identical_spectra(session, tmp_path):
    """Reload the SPD preset, then redo the whole analysis: the spectra must be
    bit-identical to the ones from the live calibration."""
    fluor = FLUOROPHORES[FLUOROPHORE]
    cal = session.calibrator
    path = tmp_path / "spd.json"
    cal.save_params(str(path))

    before_wls, before_specs = session.raw_channel_spectra([SAMPLE], fluor)

    session.calibrator = SpectralResponseCalibrator.load_params(str(path), session.mapping)
    try:
        after_wls, after_specs = session.raw_channel_spectra([SAMPLE], fluor)
    finally:
        session.calibrator = cal

    assert np.array_equal(before_wls, after_wls)
    assert np.array_equal(before_specs[0], after_specs[0])


def test_spd_preset_recompute_matches_a_fresh_calibration_after_drift(session, tmp_path):
    """Mode B on real DNGs: load a preset under a 15 px drifted mapping and it
    must reproduce a fresh Step 2 run on that same mapping."""
    cal = session.calibrator
    path = tmp_path / "spd_drift.json"
    cal.save_params(str(path))

    drifted = LinearMapping(session.mapping.slope_k, session.mapping.intercept_b + 15.0)
    loaded = SpectralResponseCalibrator.load_params(str(path), drifted)
    assert loaded.recomputed is True
    assert loaded.drift_nm == pytest.approx(15.0 / abs(session.mapping.slope_k), abs=1e-9)

    fresh = SpectralResponseCalibrator(drifted)
    frame = session.frames["spd"]
    pixels = np.arange(len(frame["red"]), dtype=np.float64)
    measured = np.column_stack((frame["red"], frame["green"], frame["blue"]))
    true_wl, true_val = load_true_spd_data(TRUE_SPD)
    fresh.calibrate(pixels, measured, true_wl, true_val)

    mask = fresh.final_mask
    assert mask.sum() > 20
    for channel in ("R", "G", "B"):
        denominator = np.maximum(np.abs(getattr(fresh, f"response_{channel}")[mask]), 1e-12)
        deviation = np.abs(getattr(loaded, f"response_{channel}")[mask]
                           - getattr(fresh, f"response_{channel}")[mask]) / denominator
        assert float(np.max(deviation)) < 1e-4

    # Mode A on the same file is nowhere near that good -> the fix is what closes it
    mode_a = SpectralResponseCalibrator.load_params(str(path), drifted, recompute=False)
    denominator = np.maximum(np.abs(fresh.response_G[mask]), 1e-12)
    mode_a_deviation = np.abs(mode_a.response_G[mask] - fresh.response_G[mask]) / denominator
    assert float(np.max(mode_a_deviation)) > 0.1


@pytest.mark.parametrize("metric_mode", ["unmixing", "legacy"])
@pytest.mark.parametrize("regression_method", ["ols", "weighted_1y", "weighted_1y2"])
def test_curve_preset_reuse_reproduces_the_original_concentration(
        session, tmp_path, metric_mode, regression_method):
    cache = build_curve(session, metric_mode, regression_method)
    path = tmp_path / f"curve_{metric_mode}_{regression_method}.json"
    save_curve_preset(str(path), cache, ANALYSIS_MODE, FLUOROPHORE)

    preset = load_curve_preset(str(path), ANALYSIS_MODE)
    reused_conc, reused_metric = apply_preset(session, preset)

    assert preset["fit"]["slope"] == cache["slope"]
    assert preset["fit"]["intercept"] == cache["intercept"]
    assert reused_metric == pytest.approx(cache["sample_mean_area"], rel=1e-12, abs=1e-12)
    assert reused_conc == pytest.approx(cache["sample_conc"], rel=1e-12, abs=1e-12)
