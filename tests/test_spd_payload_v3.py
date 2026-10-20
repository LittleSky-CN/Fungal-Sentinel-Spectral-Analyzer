"""SPD payload v3 contract tests -- the wavelength-indexed preset format.

These are the guarantees that make a Step 2 preset portable between the desktop
build and Android (MycoSpec):

  1. ``schema_version`` must be exactly 3; anything else is refused with an
     actionable message (no silent migration).
  2. The response curves are indexed by wavelength, so two mappings several nm
     apart must yield byte-identical curves.  This is the whole point of v3.
  3. Every array is exactly ``grid_points`` long -- including the optional ones
     Android writes and the desktop build does not.
  4. A descending wavelength axis is refused instead of silently mis-interpolated
     by ``np.interp`` in ``gui.main_window._interp_response``.
  5. Save -> load is numerically stable, and the residual 1-nm-grid resampling
     error stays far below the 1% budget it was given.

Every test here was mutation-checked: the behaviour under test was deliberately
broken and the test was confirmed to go red before the code was restored.
"""
import json

import numpy as np
import pytest

from core.calibration import (
    LinearMapping, SpectralResponseCalibrator, canonical_grid, correct_by_response,
)

# 1 nm 网格 + correct_by_response 的定量预算：实测 ~5e-5，留两个数量级余量
QUANT_BUDGET = 1e-2
# 有效波段内 response 的最大相对偏差预算：实测 ~9e-4
RESPONSE_BUDGET = 1e-2


def _sensor_grid_calibrator(mapping=None, n_px=3072):
    """全 sensor 列的 fixture（与 gui 的真实用法一致）：np.arange(3072)。"""
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
    return cal, pixels, lamp, true_wl, true_val


def _write(cal, tmp_path, name="spd.json"):
    path = tmp_path / name
    cal.save_params(str(path))
    return path


def _mutate(payload, path, **changes):
    payload.update(changes)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1) L0 版本闸门
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("schema_version", [None, 0, 1, 2, 4, "3", 3.0, True])
def test_non_v3_schema_is_refused(tmp_path, schema_version):
    cal, *_ = _sensor_grid_calibrator()
    path = _write(cal, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = schema_version
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        SpectralResponseCalibrator.load_params(str(path), cal.mapping)

    message = str(excinfo.value)
    assert "schema_version" in message, "错误信息必须点名 schema_version"
    assert "3" in message, "错误信息必须说清当前要求哪个版本"
    # 用户要能据此行动，而不是猜
    assert "重新导出" in message or "重做" in message


def test_missing_schema_version_is_refused(tmp_path):
    cal, *_ = _sensor_grid_calibrator()
    path = _write(cal, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("schema_version")
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        SpectralResponseCalibrator.load_params(str(path), cal.mapping)


def test_version_gate_fires_even_when_strict_is_off(tmp_path):
    """L0 不受 strict 影响：旧格式永远不会被静默接受。"""
    cal, *_ = _sensor_grid_calibrator()
    path = _write(cal, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        SpectralResponseCalibrator.load_params(str(path), cal.mapping, strict=False)


# ---------------------------------------------------------------------------
# 2) 与像素解耦：两个不同 mapping 下加载结果全等
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pixel_shift", [8.0, 15.0, 40.0])
def test_response_is_identical_under_two_different_mappings(tmp_path, pixel_shift):
    cal, *_ = _sensor_grid_calibrator()
    path = _write(cal, tmp_path)
    drifted = LinearMapping(cal.mapping.slope_k,
                            cal.mapping.intercept_b + pixel_shift)
    drift_nm = pixel_shift / abs(cal.mapping.slope_k)
    assert 2.0 < drift_nm < 20.0, "fixture 必须落在真实的漂移量级"

    a = SpectralResponseCalibrator.load_params(str(path), cal.mapping)
    b = SpectralResponseCalibrator.load_params(str(path), drifted)

    # 索引轴必须是规范波长网格，而不是任何一个 mapping 推出来的像素轴
    assert np.array_equal(a.wavelengths, canonical_grid())
    assert np.array_equal(a.wavelengths, b.wavelengths)
    assert np.array_equal(a.final_mask, b.final_mask)
    for channel in ("R", "G", "B"):
        assert np.array_equal(getattr(a, f"response_{channel}"),
                              getattr(b, f"response_{channel}")), \
            f"response_{channel} 随 mapping 变了 —— v3 解耦失效"

    # 换 mapping 后重跑一次 Step 2，落点仍是同一条网格（而不是跟着 mapping 走）
    fresh = _sensor_grid_calibrator(drifted)[0]
    assert np.array_equal(fresh.wavelengths, b.wavelengths), \
        "重跑 Step 2 后网格变了 —— 说明仍在用像素当索引轴"

    # 漂移仍被记录（诊断），但绝不导致拒绝
    assert b.drift_nm == pytest.approx(drift_nm, abs=1e-9)
    assert b.drift_level in ("WARN", "SEVERE")


def test_response_is_identical_with_no_mapping_at_all(tmp_path):
    """极端解耦：mapping=None 也要得到同一条曲线。"""
    cal, *_ = _sensor_grid_calibrator()
    path = _write(cal, tmp_path)

    blind = SpectralResponseCalibrator.load_params(str(path))
    assert blind.mapping is None
    for channel in ("R", "G", "B"):
        assert np.array_equal(getattr(blind, f"response_{channel}"),
                              getattr(cal, f"response_{channel}"))


# ---------------------------------------------------------------------------
# 3) 长度必须 == grid_points（含 Android 才会写的可选数组）
# ---------------------------------------------------------------------------
def test_every_array_is_exactly_grid_points_long(tmp_path):
    cal, *_ = _sensor_grid_calibrator()
    path = _write(cal, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    n = payload["grid_points"]
    assert n == canonical_grid().size
    for key in ("wavelengths", "response_r", "response_g", "response_b", "mask"):
        assert len(payload[key]) == n, key


@pytest.mark.parametrize("key", ["response_r", "response_g", "response_b", "mask"])
def test_short_required_array_is_refused(tmp_path, key):
    cal, *_ = _sensor_grid_calibrator()
    path = _write(cal, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    _mutate(payload, path, **{key: payload[key][:-1]})

    with pytest.raises(ValueError, match="does not match its wavelength grid"):
        SpectralResponseCalibrator.load_params(str(path), cal.mapping)


@pytest.mark.parametrize("key", ["pos_r", "pos_g", "pos_b", "corr_r", "corr_g",
                                 "corr_b", "weight_b", "weight_g", "weight_r",
                                 "fused"])
def test_short_optional_array_is_refused(tmp_path, key):
    """Android 会写这些可选键；长度不符必须拒绝，不能静默忽略。"""
    cal, *_ = _sensor_grid_calibrator()
    path = _write(cal, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[key] = [0.0] * (payload["grid_points"] - 1)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=key):
        SpectralResponseCalibrator.load_params(str(path), cal.mapping)


def test_correct_length_optional_arrays_are_tolerated(tmp_path):
    """长度对的可选键（含标量类）必须能加载 —— Python 端不写但要能读。"""
    cal, *_ = _sensor_grid_calibrator()
    path = _write(cal, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    n = payload["grid_points"]
    extras = {key: [float(i) for i in range(n)]
              for key in ("pos_r", "pos_g", "pos_b", "corr_r", "corr_g", "corr_b",
                          "weight_b", "weight_g", "weight_r", "fused")}
    extras.update({"crossing_bg": 495.0, "crossing_gr": 545.0,
                   "agreement_pct": 97.5,
                   "peak_r": [12.5, 622.0], "peak_g": [20.0, 522.0],
                   "peak_b": [30.0, 462.0], "fused_peak": [18.0, 527.0]})
    payload.update(extras)
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = SpectralResponseCalibrator.load_params(str(path), cal.mapping)
    assert np.array_equal(loaded.wavelengths, cal.wavelengths)
    assert np.array_equal(loaded.response_G, cal.response_G)


def test_grid_points_mismatch_is_refused(tmp_path):
    cal, *_ = _sensor_grid_calibrator()
    path = _write(cal, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["grid_points"] = payload["grid_points"] + 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="grid_points"):
        SpectralResponseCalibrator.load_params(str(path), cal.mapping)


def test_axis_block_must_describe_the_grid(tmp_path):
    cal, *_ = _sensor_grid_calibrator()
    path = _write(cal, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["axis"]["step_nm"] = 2.0
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="do not match the axis block"):
        SpectralResponseCalibrator.load_params(str(path), cal.mapping)


def test_missing_axis_block_is_refused(tmp_path):
    cal, *_ = _sensor_grid_calibrator()
    path = _write(cal, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("axis")
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing its 'axis' block"):
        SpectralResponseCalibrator.load_params(str(path), cal.mapping)


# ---------------------------------------------------------------------------
# 4) 降序波长轴必须被拒绝
# ---------------------------------------------------------------------------
def test_descending_wavelength_axis_is_refused(tmp_path):
    """旧实现接受降序，而 gui._interp_response 的 np.interp 要求 xp 升序，
    降序 preset 走 Mode A 会静默算错。v3 直接拒绝。"""
    cal, *_ = _sensor_grid_calibrator()
    path = _write(cal, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("wavelengths", "response_r", "response_g", "response_b", "mask"):
        payload[key] = payload[key][::-1]
    payload["axis"]["start_nm"], payload["axis"]["stop_nm"] = (
        payload["axis"]["stop_nm"], payload["axis"]["start_nm"])
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert payload["wavelengths"][0] > payload["wavelengths"][-1]
    with pytest.raises(ValueError, match="strictly monotonic"):
        SpectralResponseCalibrator.load_params(str(path), cal.mapping)


def test_flat_wavelength_axis_is_still_refused(tmp_path):
    cal, *_ = _sensor_grid_calibrator()
    path = _write(cal, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    wavelengths = payload["wavelengths"]
    wavelengths[len(wavelengths) // 2] = wavelengths[len(wavelengths) // 2 - 1]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="strictly monotonic"):
        SpectralResponseCalibrator.load_params(str(path), cal.mapping)


# ---------------------------------------------------------------------------
# 5) 数值稳定性：1 nm 网格重采样到底引入多少误差
# ---------------------------------------------------------------------------
def _pixel_domain_truth(mapping, pixels, lamp, true_wl, true_val):
    """标定当时的逐像素响应（升序域），作为"真值"参照。"""
    from scipy.interpolate import interp1d
    wl = np.asarray(mapping.pixel_to_wavelength(pixels), dtype=np.float64)
    if wl[0] > wl[-1]:
        wl, lamp = wl[::-1], lamp[::-1, :]
    spd = interp1d(true_wl, true_val, bounds_error=False, fill_value=0.0)(wl)
    mask = (wl >= 430.0) & (wl <= 660.0) & (spd > float(np.max(spd)) * 0.1)
    response = lamp / (spd[:, None] + 1e-6)
    response[~mask, :] = 0.0
    return wl, mask, response


def test_grid_resampling_stays_inside_the_error_budget(tmp_path):
    """把 response 从 ~0.36 nm/px 重采样到 1 nm 网格引入的误差，
    以及它一路除进 Step 3 后的定量误差，都必须远小于 1%。"""
    cal, pixels, lamp, true_wl, true_val = _sensor_grid_calibrator()
    path = _write(cal, tmp_path)
    loaded = SpectralResponseCalibrator.load_params(str(path), cal.mapping)

    wl, mask_px, truth = _pixel_domain_truth(
        cal.mapping, pixels, lamp, true_wl, true_val)
    grid = loaded.wavelengths
    assert grid.size == canonical_grid().size

    # 只比较 v3 有效波段内部：波段两端各约 1 nm 是掩码硬零造成的过渡带，
    # 那里响应被强制置零，逐点相对偏差本来就无意义（见报告）。
    live = loaded.final_mask
    lo, hi = float(grid[live][0]), float(grid[live][-1])
    inside = mask_px & (wl >= lo) & (wl <= hi)

    worst_response = 0.0
    for index, channel in enumerate(("R", "G", "B")):
        round_tripped = np.interp(wl, grid, getattr(loaded, f"response_{channel}"))
        reference = truth[:, index][inside]
        deviation = np.abs(round_tripped[inside] - reference) / np.maximum(
            np.abs(reference), 1e-12)
        worst_response = max(worst_response, float(np.max(deviation)))
    assert worst_response < RESPONSE_BUDGET, f"response 相对偏差 {worst_response:.3e}"

    # 最终定量：一个 GFP 形状的发射谱，响应校正后在 527±25 nm 上积分
    emission = np.exp(-0.5 * ((wl - 527.0) / 25.0) ** 2) + 0.02
    band = (wl >= 502.0) & (wl <= 552.0)
    worst_quant = 0.0
    for index, channel in enumerate(("R", "G", "B")):
        reference_area = np.trapezoid(
            correct_by_response(emission, truth[:, index])[band], wl[band])
        if reference_area <= 0.0:
            continue
        round_tripped = np.interp(wl, grid, getattr(loaded, f"response_{channel}"))
        v3_area = np.trapezoid(
            correct_by_response(emission, round_tripped)[band], wl[band])
        worst_quant = max(worst_quant,
                          abs(v3_area - reference_area) / abs(reference_area))
    assert worst_quant < QUANT_BUDGET, f"定量相对偏差 {worst_quant:.3e}"


def test_save_load_round_trip_is_stable(tmp_path):
    """同一份 preset 反复保存/加载不会漂移。"""
    cal, *_ = _sensor_grid_calibrator()
    path = _write(cal, tmp_path)
    first = SpectralResponseCalibrator.load_params(str(path), cal.mapping)
    second_path = _write(first, tmp_path, "spd2.json")
    second = SpectralResponseCalibrator.load_params(str(second_path), cal.mapping)

    for channel in ("R", "G", "B"):
        np.testing.assert_allclose(getattr(second, f"response_{channel}"),
                                   getattr(first, f"response_{channel}"),
                                   rtol=1e-15, atol=0.0)
    assert np.array_equal(second.final_mask, first.final_mask)
    assert second.grid_points == first.grid_points == canonical_grid().size
