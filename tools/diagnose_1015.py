#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""10.15 真实数据 Step 1->4 数值取证脚本（只读诊断，不改 core/ 与 gui/）。

用法:
    C:/Users/Lenovo/.workbuddy/binaries/python/versions/3.13.12/python.exe tools/diagnose_1015.py

目的：把 Step 1(波长标定) / Step 2(SPD 响应标定) / Step 3(扣空白+积分) /
Step 4(标准曲线) 的每一步中间数值原样打印出来，并落盘点表 JSON，
用于解释"同一批数据 Android R2=0.999 而 Python 只有 0.3"。

本脚本逐行复刻 gui/main_window.py 的现场计算路径（不含任何 tkinter）：
    step1_calibrate()          :1027-1076
    step2_calibrate()          :1105-1207
    get_net_areas_batch()      :971-1003
    get_raw_channel_spectra()  :1005-1024
    calculate_concentration()  :1462-1597
唯一差别是去掉了 tk 控件读写与绘图。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import linregress

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from config import FLUOROPHORES, DEFAULT_PARAMS                      # noqa: E402
from core.calibration import (                                        # noqa: E402
    LinearMapping, SpectralResponseCalibrator, correct_by_response,
    remove_isolated_spikes, analyze_blank_subtracted_replicates,
)
from core.image_processing import (                                   # noqa: E402
    read_dng_frame, read_dng_linear_rgb, make_channel_profiles, apply_rb_swap,
    resolve_x_roi, calibrate_wavelength, parse_known_wavelengths,
    load_true_spd_data, get_exposure_time, np_trapz,
)
from core.quantification import calculate_concentration               # noqa: E402
from core.unmixing import (                                           # noqa: E402
    select_reference_emission, unmix_spectrum, slope_t_statistic,
)

DATA_DIR = Path("D:/Documents/iGem/Python/FSSA_Documents/10.15")
OUT_DIR = REPO_ROOT / "analysis_output"

POS = DATA_DIR / "positioning.dng"
SPD = DATA_DIR / "spd.dng"
TRUE_SPD_CSV = DATA_DIR / "true_spd.csv"
F_BLANK = DATA_DIR / "B0.dng"          # <<< 空白 = B0（明确写死）
F_100 = DATA_DIR / "B1.dng"            # 100 mg/L
F_80 = DATA_DIR / "B2.dng"             # 80 mg/L
F_20 = DATA_DIR / "B3.dng"             # 20 mg/L
F_S = DATA_DIR / "S.dng"               # 未知样品

CHANNEL_INDEX = {"red": 0, "green": 1, "blue": 2}
CHANNELS = ("red", "green", "blue")

np_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

_LINES: list[str] = []


def say(text: str = "") -> None:
    print(text)
    _LINES.append(text)


def hdr(title: str) -> None:
    say("")
    say("=" * 78)
    say(title)
    say("=" * 78)


# ===========================================================================
# 0. 环境 / 残留 preset 体检
# ===========================================================================
def probe_stale_presets() -> None:
    hdr("0. 环境与残留 preset 体检")
    say(f"Python      : {sys.version.split()[0]}")
    say(f"numpy       : {np.__version__}")
    say(f"数据目录    : {DATA_DIR}")
    say(f"10.15 目录内 json/预设文件: "
        f"{[p.name for p in DATA_DIR.glob('*.json')] or '（无）'}")
    say("  -> Step 1/2 走现场标定，不加载任何 preset。")

    # 用户提示：旧 v2 SPD preset 会被 L0 版本闸门拒绝。扫描邻近目录做如实报告。
    neighbours = [
        DATA_DIR.parent / "spd_calibration_params.json",
        DATA_DIR.parent / "wavelength_calibration.json",
        DATA_DIR.parent / "concentration_curve.json",
        DATA_DIR / "spd_calibration_params.json",
        DATA_DIR / "wavelength_calibration.json",
        DATA_DIR / "concentration_curve.json",
    ]
    for path in neighbours:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text("utf-8"))
        except Exception as exc:                                   # pragma: no cover
            say(f"[残留文件] {path} 解析失败: {exc}")
            continue
        schema = data.get("schema_version")
        verdict = ("v3 兼容" if schema == 3
                   else "旧格式，当前 load_params() L0 闸门会直接拒绝加载")
        say(f"[残留文件] {path}")
        say(f"    schema_version={schema!r}  preset_type={data.get('preset_type')!r}"
            f"  app_version={data.get('app_version')!r}")
        say(f"    -> {verdict}")
        if schema != 3 and "lamp_R" in data:
            say("    -> 该文件仍带 v2 像素域载荷 (lamp_R/lamp_G/lamp_B/spd_reference)，"
                "v3 已删除这些键。")


# ===========================================================================
# 1. 数据加载（与 GUI 完全一致的解码路径）
# ===========================================================================
def load_frames() -> tuple[dict, float, tuple[int, int]]:
    hdr("1. DNG 解码（read_dng_linear_rgb，全帧 demosaic）")
    t0 = time.time()
    frames = {}
    for name, path in (("positioning", POS), ("spd", SPD), ("B0(blank)", F_BLANK),
                       ("B1(100)", F_100), ("B2(80)", F_80), ("B3(20)", F_20),
                       ("S(unknown)", F_S)):
        rgb, exposure = read_dng_linear_rgb(path)   # 与 GUI 的样品读取路径一致（未传 x_roi）
        frames[name] = {"path": path, "rgb": rgb, "exposure": float(exposure)}
        say(f"  {name:<12} shape={rgb.shape}  exposure={exposure:.4f}s  "
            f"dtype={rgb.dtype}")
    say(f"  解码耗时 {time.time() - t0:.1f}s")
    return frames, 0.0, (0, 0)


# ===========================================================================
# 2. Step 1 —— 波长标定
# ===========================================================================
def step1(frames) -> dict:
    hdr("2. Step 1 —— 波长标定（现场标定，knowns 用内置默认 R/G/B）")
    frame = read_dng_frame(POS)          # 与 step1_calibrate :1033 一致
    x_roi = frame["x_roi"]
    say(f"  ROI (half-open): x_roi={x_roi}  宽度={x_roi[1] - x_roi[0]}px"
        f"  / 全宽 {frame['width']}px")
    say(f"  传感器: {frame['width']}x{frame['height']}  CFA={frame['cfa']}"
        f"  black_level={frame['black_level']:.1f}")
    say(f"  positioning 曝光: {frame['exposure']:.4f}s")

    # get_average_profiles(pos_files, x_roi) —— 内部把 x_roi 当 fixed 传入
    # resolve_x_roi，返回 (start, end+1)，即比 x_roi 宽 1 px。照抄以保持口径一致。
    x0, x1 = resolve_x_roi(None, frame["width"], x_roi)
    say(f"  get_average_profiles 实际切片: [{x0}, {x1})  (比 x_roi 多 1px，"
        f"这是 resolve_x_roi 的既有行为)")

    rgb_pos = frames["positioning"]["rgb"]
    profiles = apply_rb_swap(
        make_channel_profiles(rgb_pos, (x0, x1)), False)
    knowns = parse_known_wavelengths(None)     # self.txt_file 默认 None
    say(f"  Known wavelengths: "
        + ", ".join(f"{k}={v.wavelength_nm} nm (±{v.tolerance_nm})"
                    for k, v in knowns.items()))

    fit = calibrate_wavelength(profiles, knowns)
    if fit is None:
        raise RuntimeError("Step 1: no usable R/G/B peaks in the positioning frame.")

    k, b = fit["k"], fit["b"]
    mapping = LinearMapping(k, b)
    rb_swapped = bool(fit["swapped"])
    peaks = list(fit["peaks"])
    profiles = apply_rb_swap(profiles, rb_swapped)

    say("")
    say(f"  Mapping      : p = {k:.6f} * lambda + {b:.4f}")
    say(f"  R/B swapped  : {rb_swapped}")
    say(f"  Quality      : {fit['quality']}   (PASS<=2.5nm, WARNING<=10nm, FAILED>10nm)")
    say(f"  G 点残差     : {fit['error_nm']:+.4f} nm  "
        f"(predicted y={fit['predicted']:.3f}px, residual={fit['residual']:+.3f}px)")
    for peak in peaks:
        say(f"    peak {peak.name}: y_centroid={peak.y_centroid:.3f}px  "
            f"estimator={peak.estimator}  n_maxima={peak.n_maxima}  "
            f"quality={peak.quality}  height={peak.peak_height:.4g}")
    wl_top = mapping.pixel_to_wavelength(0.0)
    wl_bot = mapping.pixel_to_wavelength(frame["height"] - 1)
    say(f"  像素->波长   : px0 -> {wl_top:.2f} nm, "
        f"px{frame['height'] - 1} -> {wl_bot:.2f} nm")
    say(f"  有效区间     : {DEFAULT_PARAMS['valid_wavelength_range']}")

    profiles = apply_rb_swap(profiles, False)   # 已 swap 过，保持最终态
    return {"mapping": mapping, "x_roi": x_roi, "rb_swapped": rb_swapped,
            "fit": fit, "peaks": peaks, "pos_profiles": profiles,
            "height": frame["height"], "width": frame["width"],
            "exposure": frame["exposure"], "cfa": frame["cfa"]}


# ===========================================================================
# 3. Step 2 —— SPD 响应标定
# ===========================================================================
def step2(frames, s1) -> SpectralResponseCalibrator:
    hdr("3. Step 2 —— SPD 响应标定（现场标定，spd.dng + true_spd.csv）")
    x_roi = s1["x_roi"]
    rgb_spd = frames["spd"]["rgb"]
    x0, x1 = resolve_x_roi(None, rgb_spd.shape[1], x_roi)
    spd_profiles = apply_rb_swap(make_channel_profiles(rgb_spd, (x0, x1)),
                                 s1["rb_swapped"])
    y_pixels = np.arange(len(spd_profiles["red"]), dtype=np.float64)
    measured_rgb = np.column_stack((spd_profiles["red"], spd_profiles["green"],
                                    spd_profiles["blue"]))
    true_wl, true_val = load_true_spd_data(TRUE_SPD_CSV)
    if true_wl is None:
        raise RuntimeError("Step 2: true_spd.csv 缺失。")
    say(f"  spd.dng 曝光 : {frames['spd']['exposure']:.4f}s")
    say(f"  true_spd.csv : {true_wl.size} 点, {true_wl.min():.1f}-{true_wl.max():.1f} nm, "
        f"峰值 {true_val.max():.4g} @ {true_wl[int(np.argmax(true_val))]:.1f} nm")

    calibrator = SpectralResponseCalibrator(s1["mapping"])
    calibrator.calibrate(y_pixels, measured_rgb, true_wl, true_val)
    calibrator.spd_source = TRUE_SPD_CSV.name
    calibrator.x_roi = tuple(int(v) for v in x_roi)
    calibrator.g_error_nm = s1["fit"]["error_nm"]
    calibrator.rb_swapped = s1["rb_swapped"]
    calibrator.exposure_seconds = frames["spd"]["exposure"]
    calibrator.frames = 1

    say("")
    say("  现场标定结果（SPD payload v3，规范波长网格 380-780nm @1nm）:")
    say(f"    grid_points   = {calibrator.grid_points}")
    say(f"    masked_count  = {calibrator.masked_count}  (响应非零的有效点数)")
    say(f"    effective_band= {calibrator.effective_band[0]:.1f} - "
        f"{calibrator.effective_band[1]:.1f} nm")
    say(f"    spd_threshold_ratio = {calibrator.spd_threshold_ratio}")
    say(f"    valid_range   = {calibrator.valid_range}")
    say(f"    sensor_height = {calibrator.sensor_height}")
    for cname, resp in (("R", calibrator.response_R), ("G", calibrator.response_G),
                        ("B", calibrator.response_B)):
        live = int(np.count_nonzero(resp != 0))
        if live:
            idx = int(np.argmax(resp))
            say(f"    response_{cname}: 有效点 {live:>3}/{calibrator.grid_points}  "
                f"峰值 {resp[idx]:.6g} a.u./SPD @ {calibrator.wavelengths[idx]:.1f} nm  "
                f"非零区间 {calibrator.wavelengths[np.flatnonzero(resp)[0]]:.0f}-"
                f"{calibrator.wavelengths[np.flatnonzero(resp)[-1]]:.0f} nm")
        else:
            say(f"    response_{cname}: 有效点 0  —— 该通道整条曲线为 0！")
    say("  resp_c(lambda) = Lamp_c(lambda) / SPD_true(lambda)  "
        "[per-channel, masked, NO SG, NO self-normalization]")
    return calibrator


# ===========================================================================
# 4. Step 3 —— 扣空白 + 积分
# ===========================================================================
def _interp_response(calibrator, wavelengths, ch_name):
    resp_map = {"red": calibrator.response_R, "green": calibrator.response_G,
                "blue": calibrator.response_B}
    return np.interp(wavelengths, calibrator.wavelengths, resp_map[ch_name])


def _sample_profiles(rgb, x_roi, rb_swapped):
    """GUI get_net_areas_batch :981 / get_raw_channel_spectra :1012 的取样口径。"""
    return apply_rb_swap(make_channel_profiles(rgb, x_roi), rb_swapped)


def _blank_profiles(rgb, x_roi, rb_swapped):
    """GUI get_average_profiles :974 的取样口径（resolve_x_roi 会 +1px）。"""
    x0, x1 = resolve_x_roi(None, rgb.shape[1], x_roi)
    return apply_rb_swap(make_channel_profiles(rgb, (x0, x1)), rb_swapped)


def step3(frames, s1, calibrator, fluor_id) -> dict:
    fluor = FLUOROPHORES[fluor_id]
    ch_name = fluor["channel"]
    ch_idx = CHANNEL_INDEX[ch_name]
    target_wl, fwhm = fluor["peak_wl"], fluor["fwhm"]
    lo_wl, hi_wl = target_wl - fwhm / 2, target_wl + fwhm / 2

    hdr(f"4. Step 3 —— 扣空白 + 积分   [fluorophore={fluor_id} "
        f"channel={ch_name} peak={target_wl}nm fwhm={fwhm}nm "
        f"window={lo_wl}-{hi_wl}nm]")
    say("  【配置】空白 = B0.dng（脚本内写死 BLANK_FILE = B0.dng）")

    x_roi, rb_swapped = s1["x_roi"], s1["rb_swapped"]
    blank = _blank_profiles(frames["B0(blank)"]["rgb"], x_roi, rb_swapped)
    blank_rgb = np.column_stack([blank[c] for c in CHANNELS])

    names = ["B0(blank)", "B1(100)", "B2(80)", "B3(20)", "S(unknown)"]
    sample_rgb = {}
    for name in names:
        sample_rgb[name] = np.column_stack(
            [_sample_profiles(frames[name]["rgb"], x_roi, rb_swapped)[c]
             for c in CHANNELS])

    n_px = blank_rgb.shape[0]
    wavelengths = s1["mapping"].pixel_to_wavelength(np.arange(n_px))
    descending = wavelengths[0] > wavelengths[-1]
    if descending:
        wavelengths = wavelengths[::-1]
        blank_rgb = blank_rgb[::-1, :]
        sample_rgb = {k: v[::-1, :] for k, v in sample_rgb.items()}
    say(f"  像素->波长{'降序，已翻转' if descending else '升序'}；"
        f"轴范围 {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")

    mask = (wavelengths >= lo_wl) & (wavelengths <= hi_wl)
    say(f"  积分窗内像素点数: {int(np.count_nonzero(mask))} "
        f"({wavelengths[mask].min():.2f} - {wavelengths[mask].max():.2f} nm)"
        if np.any(mask) else "  积分窗内无像素点！")

    response = _interp_response(calibrator, wavelengths, ch_name)
    max_resp = float(np.max(response))
    floor = max(1e-9, max_resp * 0.20)
    reliable = response > floor
    say(f"  SPD 响应(选中通道 {ch_name})：窗内 {int(np.count_nonzero(reliable[mask]))}"
        f"/{int(np.count_nonzero(mask))} 点 > floor(20%峰值={floor:.4g}) 会被真正相除")
    say(f"    resp 窗内 min/median/max = {response[mask].min():.4g} / "
        f"{np.median(response[mask]):.4g} / {response[mask].max():.4g}")
    say("  => SPD 校正标记: resp=APPLIED (per-channel absolute)")

    say("")
    say(f"  {'file':<12} {'raw均值':>12} {'净均值':>12} "
        f"{'净面积(R)':>13} {'净面积(G)':>13} {'净面积(B)':>13} {'选中通道净面积':>15}")
    results = {}
    for name in names:
        net_rgb = np.clip(sample_rgb[name] - blank_rgb, 0.0, None)
        per_ch = {}
        for c in CHANNELS:
            idx = CHANNEL_INDEX[c]
            resp_c = _interp_response(calibrator, wavelengths, c)
            corrected = correct_by_response(net_rgb[:, idx], resp_c)
            cleaned = remove_isolated_spikes(corrected)
            area = float(np_trapz(cleaned[mask], wavelengths[mask])) if np.count_nonzero(mask) > 1 else 0.0
            per_ch[c] = area
        raw_mean = float(np.mean(sample_rgb[name][mask][:, ch_idx]))
        net_mean = float(np.mean(net_rgb[mask][:, ch_idx]))
        results[name] = {"area": per_ch[ch_name], "per_channel": per_ch,
                         "raw_mean": raw_mean, "net_mean": net_mean}
        say(f"  {name:<12} {raw_mean:>12.6g} {net_mean:>12.6g} "
            f"{per_ch['red']:>13.6g} {per_ch['green']:>13.6g} {per_ch['blue']:>13.6g} "
            f"{per_ch[ch_name]:>15.6g}")
    say("  （B0 对自己扣空白，净值恒为 0；三通道净面积一并列出以便核对通道选择）")

    # 与 GUI 完全同口径的 batch 调用，验证上面的手算没有偏差
    areas, _mean_spec = analyze_blank_subtracted_replicates(
        blank_rgb[:, ch_idx],
        np.stack([sample_rgb[n][:, ch_idx] for n in names]),
        wavelengths, mask, response)
    say("")
    say("  [复算校验] analyze_blank_subtracted_replicates 同口径面积: "
        + ", ".join(f"{n}={a:.6g}" for n, a in zip(names, areas)))

    # 不做 SPD 校正时的面积（对照）
    areas_nospd, _ = analyze_blank_subtracted_replicates(
        blank_rgb[:, ch_idx],
        np.stack([sample_rgb[n][:, ch_idx] for n in names]),
        wavelengths, mask, None)
    say("  [对照] 未做 SPD 校正的面积:                                 "
        + ", ".join(f"{n}={a:.6g}" for n, a in zip(names, areas_nospd)))

    # ---- 实测净谱峰位：判断"真实发射带到底落在哪个窗口" ----
    sel = ["B1(100)", "B2(80)", "B3(20)", "S(unknown)"]
    specs = []
    for n in sel:
        net = np.clip(sample_rgb[n] - blank_rgb, 0.0, None)
        specs.append(remove_isolated_spikes(
            correct_by_response(net[:, ch_idx], response)))
    mean_spec = np.mean(np.stack(specs), axis=0)
    vis = (wavelengths >= 430.0) & (wavelengths <= 660.0)
    if np.any(vis):
        pk = int(np.argmax(mean_spec[vis]))
        say("")
        say(f"  实测净谱（{ch_name} 通道，B1/B2/B3/S 扣 B0 并做 SPD 校正后的平均谱）:")
        say(f"    峰位 = {wavelengths[vis][pk]:.1f} nm   峰值 = {mean_spec[vis][pk]:.6g}")
        wl_v, sp_v = wavelengths[vis], mean_spec[vis]
        prof = [f"{w:.0f}nm:{np.interp(w, wl_v, sp_v):.4g}"
                for w in range(440, 661, 20)]
        say("    粗剖面 " + "  ".join(prof))
        # 各候选窗口内的平均强度（同一条净谱）
        for fid2, cfg2 in FLUOROPHORES.items():
            if cfg2["channel"] != ch_name:
                continue
            lo2, hi2 = cfg2["peak_wl"] - cfg2["fwhm"] / 2, cfg2["peak_wl"] + cfg2["fwhm"] / 2
            m2 = (wavelengths >= lo2) & (wavelengths <= hi2)
            if np.any(m2):
                say(f"    候选窗 {fid2:<12} {lo2:.1f}-{hi2:.1f} nm  "
                    f"窗内均值={np.mean(mean_spec[m2]):.6g}  "
                    f"窗内积分={float(np_trapz(mean_spec[m2], wavelengths[m2])):.6g}")
        # ---- 逐文件剖面：形状是否随浓度变化（浓度致峰移 / 内滤效应）----
        grid = np.arange(470.0, 601.0, 10.0)
        say(f"    逐文件净谱剖面 ({ch_name} 通道，单位同前):")
        say("        " + "  ".join(f"{int(w):>7d}" for w in grid) + "   nm")
        for n, spec in zip(sel, specs):
            say(f"      {n:<11}"
                + "  ".join(f"{np.interp(w, wavelengths, spec):>7.4g}" for w in grid))

    return {"wavelengths": wavelengths, "mask": mask,
            "blank_rgb": blank_rgb, "sample_rgb": sample_rgb,
            "response": response, "areas": dict(zip(names, areas)),
            "areas_nospd": dict(zip(names, areas_nospd)),
            "per_file": results, "fluor": fluor, "fluor_id": fluor_id,
            "ch_name": ch_name, "ch_idx": ch_idx}


# ===========================================================================
# 5. 标准曲线点表构建
# ===========================================================================
STD_SETS = {
    # 只把 B1/B2/B3 当作标准点，B0 只做空白
    "3pts_blank_not_a_point": [(100.0, "B1(100)"), (80.0, "B2(80)"), (20.0, "B3(20)")],
    # 额外把 B0 当成 0 mg/L 的标准点（sample=B0, blank=B0 -> 面积恒 0）
    "4pts_blank_as_zero": [(0.0, "B0(blank)"), (100.0, "B1(100)"),
                           (80.0, "B2(80)"), (20.0, "B3(20)")],
}


def _net_area_batch(blank_rgb, sample_rgb, wavelengths, mask, calibrator,
                    fluor, s1):
    """复刻 get_net_areas_batch 的单行计算。"""
    ch_idx = CHANNEL_INDEX[fluor["channel"]]
    response = _interp_response(calibrator, wavelengths, fluor["channel"])
    areas, _ = analyze_blank_subtracted_replicates(
        blank_rgb[:, ch_idx], sample_rgb[:, ch_idx][None, :],
        wavelengths, mask, response)
    return float(areas[0])


def _raw_channel_spectra(frames, names, x_roi, rb_swapped, wavelengths_desc,
                         calibrator, fluor):
    """复刻 get_raw_channel_spectra：不做扣空白，只做 SPD 校正。"""
    ch_idx = CHANNEL_INDEX[fluor["channel"]]
    out = []
    for name in names:
        prof = _sample_profiles(frames[name]["rgb"], x_roi, rb_swapped)
        spec = np.column_stack([prof[c] for c in CHANNELS])
        wl = wavelengths_desc.copy()
        if wl[0] > wl[-1]:
            wl = wl[::-1]
            spec = spec[::-1, :]
        intensity = correct_by_response(spec[:, ch_idx],
                                        _interp_response(calibrator, wl, fluor["channel"]))
        out.append(intensity)
    return out


def build_points(frames, s1, calibrator, s3, fluor_id, metric_mode, std_set_name):
    """返回 (concentrations, means, stds, sample_metric, extra_info)"""
    fluor = FLUOROPHORES[fluor_id]
    x_roi, rb_swapped = s1["x_roi"], s1["rb_swapped"]
    wavelengths = s3["wavelengths"]
    mask = s3["mask"]
    blank_rgb = s3["blank_rgb"]
    sample_rgb = s3["sample_rgb"]
    std_set = STD_SETS[std_set_name]

    extra = {}
    concentrations, means, stds, rows = [], [], [], []

    if metric_mode == "unmixing":
        # ---- 复刻 calculate_concentration :1483-1547 ----
        wls_desc = s1["mapping"].pixel_to_wavelength(
            np.arange(frames["B0(blank)"]["rgb"].shape[0]))
        emis_mask = ((wls_desc >= fluor["peak_wl"] - fluor["fwhm"] / 2)
                     & (wls_desc <= fluor["peak_wl"] + fluor["fwhm"] / 2))
        row_spectra = []
        for conc, name in std_set:
            s_specs = _raw_channel_spectra(frames, [name], x_roi, rb_swapped,
                                           wls_desc, calibrator, fluor)
            b_specs = _raw_channel_spectra(frames, ["B0(blank)"], x_roi, rb_swapped,
                                           wls_desc, calibrator, fluor)
            row_spectra.append({"conc": conc, "name": name, "sample": s_specs,
                                "blank": b_specs})
        wls = wls_desc[::-1] if wls_desc[0] > wls_desc[-1] else wls_desc
        if wls_desc[0] > wls_desc[-1]:
            emis_mask = emis_mask[::-1]
        candidates = [(f"{r['name']} (C={r['conc']:g})",
                       np.mean(r["sample"], axis=0), np.mean(r["blank"], axis=0))
                      for r in row_spectra]
        ref_index, E, auc_E = select_reference_emission(wls, candidates, emis_mask)
        ref_row = row_spectra[ref_index]
        B_ref = np.mean(ref_row["blank"], axis=0)
        sm = (wls > 460) & (wls < 520)
        if np.any(sm):
            scatter_wl = float(wls[sm][int(np.argmax(B_ref[sm]))])
            lo = min(max(scatter_wl + 2.0, fluor["peak_wl"] - 30.0), fluor["peak_wl"] - 5.0)
        else:
            lo = fluor["peak_wl"] - 20.0
        hi = fluor["peak_wl"] + 90.0
        win_mask = (wls >= lo) & (wls <= hi)

        for r in row_spectra:
            B_mean = np.mean(r["blank"], axis=0)
            alphas = [unmix_spectrum(spec, E, B_mean, win_mask)["alpha"]
                      for spec in r["sample"]]
            metrics = [a * auc_E for a in alphas]
            concentrations.append(r["conc"])
            means.append(float(np.mean(metrics)))
            stds.append(float(np.std(metrics, ddof=1)) if len(metrics) > 1 else 0.0)
            rows.append({"conc": r["conc"], "name": r["name"],
                         "alpha": float(np.mean(alphas)), "metric": means[-1]})

        u_specs = _raw_channel_spectra(frames, ["S(unknown)"], x_roi, rb_swapped,
                                       wls_desc, calibrator, fluor)
        u_blanks = _raw_channel_spectra(frames, ["B0(blank)"], x_roi, rb_swapped,
                                        wls_desc, calibrator, fluor)
        B_u = np.mean(u_blanks, axis=0)
        alpha_u = float(np.mean([unmix_spectrum(s, E, B_u, win_mask)["alpha"]
                                 for s in u_specs]))
        sample_metric = alpha_u * auc_E
        extra = {"unmix_window": [float(lo), float(hi)],
                 "scatter_peak_wl": scatter_wl if np.any(sm) else float("nan"),
                 "ref_standard": f"{ref_row['name']} (C={ref_row['conc']:g})",
                 "auc_E": float(auc_E), "alpha_sample": alpha_u,
                 "per_row_alpha": [{"conc": r["conc"], "name": r["name"],
                                    "alpha": r["alpha"], "metric": r["metric"]}
                                   for r in rows]}
    else:
        # ---- 复刻 calculate_concentration :1586-1593 (legacy 空白色积分) ----
        for conc, name in std_set:
            area = _net_area_batch(blank_rgb, sample_rgb[name], wavelengths, mask,
                                   calibrator, fluor, s1)
            concentrations.append(conc)
            means.append(area)
            stds.append(0.0)
            rows.append({"conc": conc, "name": name, "metric": area})
        sample_metric = _net_area_batch(blank_rgb, sample_rgb["S(unknown)"],
                                        wavelengths, mask, calibrator, fluor, s1)
        extra = {"integration_window": [float(wavelengths[mask].min()),
                                        float(wavelengths[mask].max())]}

    extra["rows"] = rows
    return concentrations, means, stds, sample_metric, extra


# ===========================================================================
# 6. 三种 R² 对照
# ===========================================================================
def reference_fits(concentrations, means):
    """纯线性 OLS R²、log10(浓度) R²、log-log R²。"""
    c = np.asarray(concentrations, dtype=float)
    y = np.asarray(means, dtype=float)
    out = {}

    res = linregress(c, y)
    out["ols_linear"] = {"r_squared": float(res.rvalue ** 2), "slope": float(res.slope),
                         "intercept": float(res.intercept), "n": int(c.size),
                         "x": "concentration", "y": "area"}

    pos = c > 0
    n_pos = int(np.count_nonzero(pos))
    if n_pos >= 3:
        lx = np.log10(c[pos])
        r = linregress(lx, y[pos])
        out["log10x"] = {"r_squared": float(r.rvalue ** 2), "slope": float(r.slope),
                         "intercept": float(r.intercept), "n": n_pos,
                         "x": "log10(concentration)", "y": "area",
                         "note": "0 mg/L 点被剔除（log10(0) 无定义）"}
        ok = pos & (y > 0)
        if np.count_nonzero(ok) >= 3:
            r2 = linregress(np.log10(c[ok]), np.log10(y[ok]))
            out["loglog"] = {"r_squared": float(r2.rvalue ** 2),
                             "slope": float(r2.slope), "intercept": float(r2.intercept),
                             "n": int(np.count_nonzero(ok)),
                             "x": "log10(concentration)", "y": "log10(area)",
                             "note": "0 mg/L 与非正面积点均剔除"}
    if np.any(c <= 0):
        lx = np.log10(c + 1.0)
        r = linregress(lx, y)
        out["log10_conc_plus_1"] = {
            "r_squared": float(r.rvalue ** 2), "slope": float(r.slope),
            "intercept": float(r.intercept), "n": int(c.size),
            "x": "log10(concentration + 1)", "y": "area",
            "note": "保留 0 mg/L 点的另一种处理方式（+1 平移）"}
    return out


def r2_ols(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return float(linregress(x, y).rvalue ** 2)


# ===========================================================================
# 7. main
# ===========================================================================
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    probe_stale_presets()
    frames, _, _ = load_frames()
    s1 = step1(frames)
    calibrator = step2(frames, s1)

    report = {
        "dataset": "10.15",
        "generated_by": "tools/diagnose_1015.py",
        "data_dir": str(DATA_DIR),
        "blank_file": "B0.dng",
        "step1": {
            "k": float(s1["fit"]["k"]), "b": float(s1["fit"]["b"]),
            "quality": s1["fit"]["quality"],
            "g_error_nm": float(s1["fit"]["error_nm"]),
            "residual_px": float(s1["fit"]["residual"]),
            "predicted_y_px": float(s1["fit"]["predicted"]),
            "rb_swapped": bool(s1["rb_swapped"]),
            "x_roi": [int(v) for v in s1["x_roi"]],
            "peaks": [{"name": p.name, "y_centroid": float(p.y_centroid),
                       "estimator": p.estimator, "n_maxima": int(p.n_maxima),
                       "quality": p.quality} for p in s1["peaks"]],
            "knowns": {"R": 622.5, "G": 522.5, "B": 462.5},
        },
        "step2": {
            "on_site": True,
            "grid_points": calibrator.grid_points,
            "masked_count": calibrator.masked_count,
            "effective_band": [float(v) for v in calibrator.effective_band],
            "spd_threshold_ratio": float(calibrator.spd_threshold_ratio),
            "valid_range": [float(v) for v in calibrator.valid_range],
            "valid_points": {c: int(np.count_nonzero(getattr(calibrator, f"response_{c.upper()}") != 0))
                             for c in ("r", "g", "b")},
            "spd_correction": "APPLIED (per-channel absolute)",
        },
        "step4_variants": [],
    }

    # ---------------- Step 3：对每个候选荧光素都出一次净面积 ----------------
    step3_cache = {}
    for fluor_id in ("YPET", "FLUORESCEIN", "EGFP", "MCHERRY"):
        try:
            step3_cache[fluor_id] = step3(frames, s1, calibrator, fluor_id)
        except Exception as exc:      # 该配置跑不通本身也是结论，原样报告
            type_, _, _tb = sys.exc_info()
            say(f"\n  [Step 3 失败] fluorophore={fluor_id}: "
                f"{type_.__name__}: {exc}")
            say("  -> 该荧光素在后续 Step 4 组合中跳过。")

    # ---------------- Step 4：主要配置组合 ----------------
    variants = []
    combos = []
    for fluor_id in ("YPET", "FLUORESCEIN", "EGFP"):
        if fluor_id not in step3_cache:
            continue
        for metric_mode in ("unmixing", "legacy"):
            for std_set_name in STD_SETS:
                for reg_method in ("ols", "weighted_1y2"):
                    combos.append((fluor_id, metric_mode, std_set_name, reg_method))

    hdr("5. Step 4 —— 标准曲线（多配置全跑）")
    for fluor_id, metric_mode, std_set_name, reg_method in combos:
        s3 = step3_cache[fluor_id]
        try:
            concs, means, stds, sample_metric, extra = build_points(
                frames, s1, calibrator, s3, fluor_id, metric_mode, std_set_name)
        except Exception as exc:
            say("")
            say("-" * 78)
            say(f"{fluor_id} metric={metric_mode} stds={std_set_name} -> "
                f"失败: {type(exc).__name__}: {exc}")
            continue
        calc = calculate_concentration(concs, means, sample_metric,
                                       mode="lod_iupac",
                                       regression_method=reg_method)
        refs = reference_fits(concs, means)
        variants.append({
            "fluorophore": fluor_id,
            "metric_mode": metric_mode,
            "std_set": std_set_name,
            "regression_method": reg_method,
            "points": [{"concentration": float(c), "area": float(m)}
                       for c, m in zip(concs, means)],
            "point_rows": [{"concentration": r["conc"], "file": r["name"],
                            "blank": "B0.dng", "metric": r["metric"]}
                           for r in extra["rows"]],
            "sample": {"file": "S.dng", "blank": "B0.dng", "metric": float(sample_metric)},
            "python_fit": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                           for k, v in calc.items()},
            "reference_fits": refs,
            "extra": {k: v for k, v in extra.items() if k != "rows"},
        })

    # ---------------- 逐 variant 打印 ----------------
    for v in variants:
        tag = (f"{v['fluorophore']:<11} metric={v['metric_mode']:<8} "
               f"stds={v['std_set']:<24} reg={v['regression_method']}")
        say("")
        say("-" * 78)
        say(tag)
        say("-" * 78)
        pts = ", ".join(f"({p['concentration']:g}, {p['area']:.6g})"
                        for p in v["points"])
        say(f"  点表 (浓度 mg/L, metric): [{pts}]")
        for row in v["point_rows"]:
            extra_alpha = (f" alpha={row.get('alpha', float('nan')):.6g}"
                           if "alpha" in row else "")
            say(f"    C={row['concentration']:<6g} file={row['file']:<12} "
                f"blank={row['blank']:<8} metric={row['metric']:.6g}{extra_alpha}")
        say(f"  S 样品 metric = {v['sample']['metric']:.6g}")
        pf = v["python_fit"]
        say(f"  Python 报出: R2={pf['r_squared']:.6f}  slope={pf['slope']:.6g}  "
            f"intercept={pf['intercept']:.6g}  method={pf['regression_method']}")
        say(f"  Python 预测 S 浓度 = {pf['sample_conc']:.6g}")
        rf = v["reference_fits"]
        say(f"  对照 (a) 纯线性 OLS R2          = {rf['ols_linear']['r_squared']:.6f}")
        if "log10x" in rf:
            say(f"  对照 (b) log10(C) vs 面积 R2    = {rf['log10x']['r_squared']:.6f}  "
                f"(n={rf['log10x']['n']}, {rf['log10x']['note']})")
        if "loglog" in rf:
            say(f"  对照 (c) log-log R2             = {rf['loglog']['r_squared']:.6f}  "
                f"(n={rf['loglog']['n']})")
        if "log10_conc_plus_1" in rf:
            say(f"  对照 (d) log10(C+1) vs 面积 R2  = "
                f"{rf['log10_conc_plus_1']['r_squared']:.6f}  "
                f"(n={rf['log10_conc_plus_1']['n']}, {rf['log10_conc_plus_1']['note']})")

    report["step4_variants"] = variants

    # ---------------- 汇总表 ----------------
    hdr("6. 汇总：Python R2 vs 线性 R2 vs log10 R2")
    say(f"  {'fluor':<11} {'metric':<9} {'stds':<24} {'reg':<13} "
        f"{'Py R2':>9} {'linR2':>9} {'log10R2':>9} {'S conc':>12}")
    for v in variants:
        rf = v["reference_fits"]
        log_r2 = rf.get("log10x", {}).get("r_squared", float("nan"))
        say(f"  {v['fluorophore']:<11} {v['metric_mode']:<9} {v['std_set']:<24} "
            f"{v['regression_method']:<13} {v['python_fit']['r_squared']:>9.6f} "
            f"{rf['ols_linear']['r_squared']:>9.6f} {log_r2:>9.6f} "
            f"{v['python_fit']['sample_conc']:>12.6g}")

    # ---------------- 落盘 ----------------
    points_path = OUT_DIR / "step4_points_1015.json"
    points_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    full_path = OUT_DIR / "diagnose_1015_full.json"
    full_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    txt_path = OUT_DIR / "diagnose_1015_report.txt"
    txt_path.write_text("\n".join(_LINES) + "\n", encoding="utf-8")

    hdr("7. 落盘")
    say(f"  点表 + 全量 : {points_path}")
    say(f"  同名副本     : {full_path}")
    say(f"  文本报告     : {txt_path}")
    say(f"  总耗时 {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
