"""Step 1 的标定质量必须体现在状态行上。

用户是拿这个软件出实验数据的：Step 1 实际是 WARNING/FAILED 却显示绿色 "Done!"，
会让人误以为数据可信。这里守住四条：

  * WARNING -> 状态文本不含 "Done!"，颜色不是 green
  * FAILED  -> 红色，且必须弹窗告知（但仍设置 mapping，不硬阻断）
  * PASS    -> 仍是绿色（别把正常路径也改坏）
  * 导入 SUSPECT / FAILED 的 WAVELENGTH preset -> 状态不是绿色，弹窗不许叫 "Success"

两端的词汇不一致（实时标定 PASS/WARNING/FAILED，preset 侧 GOOD/SUSPECT/FAILED/PRESET），
统一由 ``core.wavelength_preset.step1_quality_tier`` 归一化，这里一并验证映射本身。
"""
import json

import pytest

# 复用同目录 sibling 的 Tk stub（没有 tkinter 的 CI 解释器才能 import gui.main_window）
from test_gui_preset_flow import Recorder, StatusLog, Stub, mw  # noqa: E402

from core.wavelength_preset import step1_quality_tier  # noqa: E402
from gui.main_window import _step1_live_status, _step1_preset_status  # noqa: E402

PASS_LIKE = ("PASS", "GOOD")
WARNING_LIKE = ("WARNING", "SUSPECT", "PRESET")


def _gui():
    """最小 FSSA_GUI：只塞 Step 1 两条路径真正碰到的属性。"""
    from matplotlib.figure import Figure

    gui = mw.FSSA_GUI.__new__(mw.FSSA_GUI)
    gui.root = Stub()
    gui.analysis_mode = "lod_iupac"
    gui.mapping = None
    gui.x_roi = None
    gui.rb_swapped = False
    gui.calibrator = None
    gui.spd_params_path = None
    gui.txt_file = None
    gui.csv_file = None
    gui.cache_step1 = {}
    gui.cache_step2 = {}
    gui.txt_result = Stub()
    gui.fig = Figure(figsize=(4, 3), dpi=72)
    gui.ax = gui.fig.add_subplot(111)
    gui.canvas_plot = Stub()
    for step in (1, 2, 3, 4):
        setattr(gui, f"lbl_status{step}", Stub())
    gui.messages = Recorder()
    mw.messagebox = gui.messages
    gui.log = StatusLog()
    gui.log_status = gui.log
    return gui


def _status(gui, step=1):
    """最后一次写入指定步骤的状态行 ``(step, text, colour)``。"""
    entries = [entry for entry in gui.log.entries if entry[0] == step]
    assert entries, "没有写状态行"
    return entries[-1]


def _run_step1(gui, monkeypatch, quality, error_nm):
    """走真实的 ``step1_calibrate()``，只把 DNG 读取与拟合换成桩。"""
    import types

    import numpy as np

    ns = types.SimpleNamespace
    gui.pos_selector = Stub()
    gui.pos_selector.get_files = lambda: ["pos.dng"]
    monkeypatch.setattr(mw, "parse_known_wavelengths",
                        lambda path: {"R": ns(wavelength_nm=622.5, tolerance_nm=15.0),
                                      "G": ns(wavelength_nm=522.5, tolerance_nm=15.0),
                                      "B": ns(wavelength_nm=462.5, tolerance_nm=15.0)})
    monkeypatch.setattr(mw, "read_dng_frame",
                        lambda path: {"rgb": np.zeros((64, 64, 3)),
                                      "exposure": 0.25, "x_roi": (0, 64),
                                      "cfa": (2, 1, 1, 0)})
    monkeypatch.setattr(mw, "get_average_profiles",
                        lambda paths, x_roi: {"red": np.arange(16.0),
                                              "green": np.arange(16.0) + 1.0,
                                              "blue": np.arange(16.0) + 2.0})
    monkeypatch.setattr(mw, "apply_rb_swap", lambda profiles, swapped: profiles)
    peaks = [ns(name=name, y_centroid=y, centroid_px=y, lo=y - 1.0, hi=y + 1.0,
                y_max=1.0, baseline=0.0, height=1.0, estimator="centroid",
                n_maxima=1, quality="GOOD", centroid_legacy=y)
             for name, y in (("R", 10.0), ("G", 20.0), ("B", 30.0))]
    monkeypatch.setattr(mw, "calibrate_wavelength", lambda profiles, knowns: {
        "peaks": peaks, "k": -2.738911, "b": 3198.4744, "predicted": 20.0,
        "residual": 1.0, "error_nm": error_nm, "swapped": False,
        "quality": quality})

    gui.step1_calibrate()


def _write_preset(tmp_path, quality, g_error, name="wavelength.json"):
    """直接写 payload JSON，绕开 save_wavelength_preset 的 quality 推导。"""
    path = tmp_path / name
    payload = {
        "preset_type": "WAVELENGTH", "schema_version": 1, "app_version": "1.5.0",
        "created_at": "2026-01-01T00:00:00Z", "name": "field-wl",
        "slope": -2.738911, "intercept": 3198.4744,
        "gErrorNm": g_error, "quality": quality,
        "analysisMode": "lod_iupac",
    }
    if quality is None:
        payload.pop("quality")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 词汇归一化（两端映射）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("quality,tier", [
    ("PASS", "PASS"), ("GOOD", "PASS"),
    ("WARNING", "WARNING"), ("SUSPECT", "WARNING"), ("PRESET", "WARNING"),
    ("FAILED", "FAILED"),
])
def test_quality_tier_normalises_both_vocabularies(quality, tier):
    assert step1_quality_tier(quality) == tier
    assert step1_quality_tier(quality.lower()) == tier, "大小写不敏感"


@pytest.mark.parametrize("quality", [None, "", "UNKNOWN", 3])
def test_unknown_quality_never_claims_success(quality):
    """未知/缺失标签按最保守的 WARNING 处理 —— 不许显示成成功。"""
    assert step1_quality_tier(quality) == "WARNING"


# ---------------------------------------------------------------------------
# 实时标定：状态行的文案与颜色
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("quality", PASS_LIKE)
def test_pass_status_stays_green_and_says_done(quality):
    text, colour = _step1_live_status(quality, 0.25, 0.80)
    assert colour == "green"
    assert "Done!" in text and "(PASS)" in text


@pytest.mark.parametrize("quality", WARNING_LIKE)
def test_warning_status_is_orange_and_never_says_done(quality):
    text, colour = _step1_live_status(quality, 0.25, 5.50)
    assert colour != "green", "WARNING 不能显示成绿色"
    assert colour == "orange"
    assert "Done!" not in text, "WARNING 不许出现光秃秃的 Done!"
    assert "WARNING" in text and "5.50" in text


@pytest.mark.parametrize("quality", ["FAILED"])
def test_failed_status_is_red(quality):
    text, colour = _step1_live_status(quality, 0.25, 15.00)
    assert colour == "red"
    assert "Done!" not in text
    assert "FAILED" in text and "mapping unreliable" in text


# ---------------------------------------------------------------------------
# 实时标定：走真实 step1_calibrate()
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("quality,error_nm,colour", [
    ("PASS", 0.8, "green"),
    ("WARNING", 5.5, "orange"),
    ("FAILED", 15.0, "red"),
])
def test_step1_calibrate_colours_the_status_line(
        monkeypatch, quality, error_nm, colour):
    gui = _gui()
    _run_step1(gui, monkeypatch, quality, error_nm)

    step, text, actual = _status(gui)
    assert actual == colour
    if quality == "PASS":
        assert "Done!" in text
    else:
        assert "Done!" not in text, f"{quality} 不该显示成成功"
    assert f"{error_nm:+.2f}" in text, "残差必须出现在状态行里"


def test_step1_calibrate_warns_the_user_when_failed(monkeypatch):
    """FAILED 仍设置 mapping（不硬阻断），但必须弹窗说明下游定量不可信。"""
    gui = _gui()
    _run_step1(gui, monkeypatch, "FAILED", 15.0)

    assert gui.mapping is not None, "保持尽力而为的结果，不硬阻断"
    assert gui.messages.warnings, "FAILED 必须弹窗，不能静默成功"
    title, message = gui.messages.warnings[-1]
    assert "FAILED" in title
    assert "15.00" in message and "建议重新" in message
    step, text, colour = _status(gui)
    assert colour == "red"


@pytest.mark.parametrize("quality", WARNING_LIKE)
def test_step1_calibrate_stays_silent_for_warning(monkeypatch, quality):
    """WARNING 只染橙，不弹窗 —— 没到 FAILED 那个量级，别骚扰用户。"""
    gui = _gui()
    _run_step1(gui, monkeypatch, quality, 5.5)

    assert not gui.messages.warnings
    step, text, colour = _status(gui)
    assert colour == "orange"


def test_step1_calibrate_stores_the_quality_for_the_report(monkeypatch):
    """details_window / 报告靠 cache_step1["quality"] 显示质量，不能漏。"""
    gui = _gui()
    _run_step1(gui, monkeypatch, "WARNING", 5.5)
    assert gui.cache_step1["quality"] == "WARNING"
    assert step1_quality_tier(gui.cache_step1["quality"]) == "WARNING"


# ---------------------------------------------------------------------------
# 导入 WAVELENGTH preset
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("quality,g_error,colour", [
    ("GOOD", 0.8, "green"),
    ("SUSPECT", 5.5, "orange"),
    ("FAILED", 15.0, "red"),
    ("PRESET", None, "orange"),
    (None, 1.0, "orange"),          # 老文件没有 quality 字段
])
def test_preset_status_colour_follows_the_quality_tag(quality, g_error, colour):
    text, actual = _step1_preset_status("field.json", quality, g_error)
    assert actual == colour
    if colour == "green":
        assert "quality: GOOD" in text
    else:
        assert "Done!" not in text
        assert "WARNING" in text or "FAILED" in text


@pytest.mark.parametrize("quality,expect_success", [
    ("GOOD", True), ("SUSPECT", False), ("FAILED", False),
])
def test_loading_a_preset_never_calls_a_bad_result_success(
        tmp_path, monkeypatch, quality, expect_success):
    from core.calibration import LinearMapping

    gui = _gui()
    path = _write_preset(tmp_path, quality, 5.5 if quality != "GOOD" else 0.8)
    monkeypatch.setattr(mw.filedialog, "askopenfilename", lambda **kw: str(path))

    gui.load_wavelength_preset()

    step, text, colour = _status(gui)
    if expect_success:
        assert colour == "green"
        assert gui.messages.infos and gui.messages.infos[-1][0] == "Success"
        assert not gui.messages.warnings
    else:
        assert colour != "green", f"quality={quality} 的 preset 不该显示成绿色"
        assert not gui.messages.infos, "非 GOOD 的 preset 不许弹 'Success'"
        assert gui.messages.warnings, "必须弹窗说明标定质量"
        title, message = gui.messages.warnings[-1]
        assert "Success" not in title
        assert quality in message
    assert gui.mapping is not None, "仍然要恢复 mapping"
    assert gui.cache_step1["quality"] == quality
