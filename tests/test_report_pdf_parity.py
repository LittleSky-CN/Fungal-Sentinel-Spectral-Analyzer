"""报告窗口 vs 导出 PDF 的内容一致性（FSSA v1.5.0）。

``gui/details_window.py`` 里每个 Tab 的参数是双重承载的：

* ``_add_param_panel`` -> ``tk.Text``：只在报告窗口里可见，不进 PDF；
* ``figN_params`` / ``fig5`` -> matplotlib Figure：``export_pdf_report`` 逐页存进 PDF。

所以「窗口里看得到的东西 PDF 里都有」等价于「每个 tab 的参数都有一个
figure 版，且该 figure 的版式真的装得下这段文本」。本模块守这条不变量：

1. 每个 tab 的参数文本都有对应的 ``figN_params``（不为 ``None``），且内容与
   ``tk.Text`` 里的一致（折行只插入 ``\\n``，去掉全部空白后必须逐字相同）；
2. 四步数据齐全时 ``figs_to_export`` 恰好是 9 个 figure；
3. 任意长的参数文本（200 行 + 500 字符长行）生成的 figure 不得裁掉底部或右侧；
4. Tab5 的预设溯源文本真的落到了 PDF 里。

tkinter 在无头环境不可用，这里复用 ``tests/test_report_provenance.py`` 的桩化
手法，然后真的构造 ``DetailsWindow``。
"""
import importlib.util
import sys
import types


# ---------------------------------------------------------------------------
# Tk 桩（同 tests/test_report_provenance.py:18-74）
# ---------------------------------------------------------------------------
class Stub:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return Stub()

    def __call__(self, *args, **kwargs):
        return Stub()

    def __getitem__(self, name):
        return Stub()

    def __setitem__(self, name, value):
        return None

    def __bool__(self):
        return True


class StubError(Exception):
    """Stand-in for ``tk.TclError``."""


def _stub_module(name):
    module = types.ModuleType(name)

    def _lookup(attr):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        return StubError if attr.endswith("Error") else Stub

    module.__getattr__ = _lookup
    return module


def _install_tk_stubs():
    if importlib.util.find_spec("tkinter") is not None:
        return False
    tkinter = _stub_module("tkinter")
    for sub in ("ttk", "filedialog", "messagebox", "scrolledtext", "font",
                "colorchooser", "commondialog", "constants"):
        module = _stub_module(f"tkinter.{sub}")
        sys.modules[f"tkinter.{sub}"] = module
        setattr(tkinter, sub, module)
    sys.modules["tkinter"] = tkinter
    sys.modules["matplotlib.backends.backend_tkagg"] = _stub_module(
        "matplotlib.backends.backend_tkagg")
    return True


_install_tk_stubs()

import matplotlib  # noqa: E402

matplotlib.use = lambda *args, **kwargs: None

import numpy as np  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from pypdf import PdfReader  # noqa: E402

from core.calibration import TwoPointValidation  # noqa: E402
from core.image_processing import PeakResult  # noqa: E402
from gui import details_window as dw  # noqa: E402

dw._THEME_ON = False


# ---------------------------------------------------------------------------
# 合成的四步缓存：刚好够 _build_tab1.._build_tab5 跑完
# ---------------------------------------------------------------------------
def _peak(name, wl, y, ch):
    return PeakResult(name=name, wavelength_nm=wl, channel_name=ch, y_centroid=float(y),
                      y_max=int(y), baseline=0.0, lo=int(y) - 3, hi=int(y) + 3,
                      peak_height=1.0)


def _cache1():
    n = 120
    rgb = np.zeros((n, 8, 3))
    rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2] = 0.4, 0.5, 0.6
    return {
        "rgb": rgb, "x_roi": (2, 6),
        "profiles": {c: np.linspace(0.2, 1.0, n) for c in ("red", "green", "blue")},
        "peaks": [_peak("R", 622.5, 30, "red"),
                  _peak("G", 522.5, 60, "green"),
                  _peak("B", 462.5, 90, "blue")],
        "validation": TwoPointValidation(("B", "R"), "G", -0.5, 380.0,
                                         55.0, 60.0, 5.0, -10.0),
        "exp_time": 0.05, "n_pos": 1,
    }


def _cache2():
    wls = np.linspace(400.0, 700.0, 61)
    return {
        "wavelengths": wls,
        "R": np.linspace(0.5, 1.0, 61), "G": np.linspace(1.0, 0.5, 61),
        "B": np.full(61, 0.7),
        "pos_R": np.linspace(0.6, 1.0, 61), "pos_G": np.linspace(0.7, 0.9, 61),
        "pos_B": np.linspace(0.8, 0.6, 61),
        "spd_threshold_ratio": 0.05, "lamp_R_n": 1,
    }


def _cache3():
    wls = np.linspace(450.0, 650.0, 51)
    return {
        "wls": wls, "intensity": np.exp(-((wls - 527.0) / 20.0) ** 2),
        "mask": (wls >= 517.0) & (wls <= 537.0),
        "fluor": {"color": "green", "peak_wl": 527.0, "fwhm": 20.0, "channel": "green"},
        "mean_area": 1.23, "peak": 0.98,
        "features": {"peak_wl": 527.1, "peak_intensity": 0.97, "fwhm": 21.0,
                     "auc": 20.5, "centroid": 528.0},
        "spd_applied": True, "fluor_name": "Ypet (Venus variant)",
    }


def _cache4():
    return {
        "concentrations": [20.0, 60.0, 100.0], "means": [1.0, 3.0, 5.0],
        "stds": [0.1, 0.2, 0.3], "slope": 0.05, "intercept": 0.0,
        "sample_mean_area": 2.5, "sample_conc": 50.0, "r_squared": 0.999,
        "mode": "lod_iupac", "lod_conc": 1.5, "loq_conc": 4.5,
        "lod_signal": 0.075, "loq_signal": 0.225,
        "lod_source": "full-range curve", "metric_mode": "legacy",
        "t_stat": 12.0, "regression_method": "ols", "unmix": None,
        "blocks": [{"exp": 0.05, "n": 3, "slope": 0.05, "r2": 0.999,
                    "syx": 0.01, "t": 12.0, "lod": 1.5}],
    }


PROVENANCE = {
    "step1": {"kind": "live", "preset_name": None, "file_name": None, "file": None,
              "imported_at": "2026-10-17 10:30:00",
              "params": {"mapping_slope": "-2.739846", "g_error_nm": "+5.567"}},
    "step2": {"kind": "preset", "preset_type": "SPD", "preset_name": "spd.json",
              "file_name": "spd.json", "file": "D:/presets/spd.json",
              "imported_at": "2026-10-17 10:31:00",
              "params": {"drift_level": "OK", "drift_nm": "0.00"}},
    "step3": {"kind": "live", "preset_name": None, "file_name": None, "file": None,
              "imported_at": "2026-10-17 10:32:00",
              "params": {"fluor_name": "Ypet (Venus variant)", "peak_wl": "527.0"}},
    "step4": {"kind": "preset", "preset_type": "CURVE", "preset_name": "curve.json",
              "file_name": "curve.json", "file": "D:/presets/curve.json",
              "imported_at": "2026-10-17 10:33:00",
              "params": {"slope": "0.05", "r_squared": "0.999"}},
}


def _window(monkeypatch=None):
    """构造一个四步齐全的 DetailsWindow，并顺带抓下每个 tab 的参数文本。"""
    captured = []
    original = dw.DetailsWindow._add_param_panel

    def spy(self, tab, content, title=None, height=8):
        captured.append(content)
        return original(self, tab, content, title=title, height=height)

    dw.DetailsWindow._add_param_panel = spy
    try:
        win = dw.DetailsWindow(Stub(), _cache1(), _cache2(), _cache3(), _cache4(),
                               None, preset_provenance=PROVENANCE)
    finally:
        dw.DetailsWindow._add_param_panel = original
    win._captured_params = captured
    return win


def _strip(s):
    """去掉全部空白：折行只插入 ``\\n``，所以折行前后应当逐字相同。"""
    return "".join(s.split())


def _text_bbox(fig):
    """用真实 renderer 量出 fig.text 的包围盒（像素）。"""
    fig.set_canvas(FigureCanvasAgg(fig))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    return fig.texts[0].get_window_extent(renderer=renderer)


# ---------------------------------------------------------------------------
# 1. 每个 tab 的参数都有 figure 版，且内容与窗口里的一致
# ---------------------------------------------------------------------------
def test_every_tab_has_a_params_figure():
    win = _window()
    for name in ("fig1_params", "fig2_params", "fig3_params", "fig4_params", "fig5"):
        fig = getattr(win, name)
        assert isinstance(fig, Figure), f"{name} 缺失：该 tab 的参数不会进 PDF"


def test_param_figures_carry_exactly_the_panel_text():
    win = _window()
    panels = win._captured_params
    assert len(panels) == 5, "5 个 tab 各有一个参数面板"

    pairs = [(win.fig1_params, panels[0]), (win.fig2_params, panels[1]),
             (win.fig3_params, panels[2]), (win.fig4_params, panels[3]),
             (win.fig5, panels[4])]
    for fig, panel in pairs:
        drawn = fig.texts[0].get_text()
        assert _strip(drawn) == _strip(panel), (
            "figure 里的参数文本与窗口 tk.Text 的不一致\n"
            f"--- figure ---\n{drawn[:400]}\n--- panel ---\n{panel[:400]}")


# ---------------------------------------------------------------------------
# 2. 四步齐全时导出列表恰好 9 个 figure
# ---------------------------------------------------------------------------
def test_figs_to_export_holds_all_nine():
    win = _window()
    assert len(win.figs_to_export) == 9, (
        f"应导出 9 个 figure（4 张曲线图 + 5 页参数），实际 {len(win.figs_to_export)}")
    for fig in win.figs_to_export:
        assert isinstance(fig, Figure)
    expected = [win.fig1, win.fig1_params, win.fig2, win.fig2_params,
                win.fig3, win.fig3_params, win.fig4, win.fig4_params, win.fig5]
    assert win.figs_to_export == expected


def test_figures_of_unfinished_steps_are_filtered_out():
    """只做了 Step 1 时，figs_to_export 只能有 fig1/fig1_params/fig5 三个。"""
    win = dw.DetailsWindow(Stub(), _cache1(), {}, {}, {}, None)
    assert win.figs_to_export == [win.fig1, win.fig1_params, win.fig5]
    assert win.fig2_params is None and win.fig4_params is None


# ---------------------------------------------------------------------------
# 3. 版式：任意长的参数文本都必须装得下（不裁底、不裁右）
# ---------------------------------------------------------------------------
LONG_LINE = "X" * 500
STRESS_TEXT = "\n".join(f"row {i:03d} " + "value " * 8 for i in range(200))
STRESS_TEXT += "\n" + LONG_LINE


def test_long_params_text_is_not_clipped_at_the_bottom():
    win = _window()
    fig = win._create_params_figure("Stress Test", STRESS_TEXT)
    bb = _text_bbox(fig)
    assert bb.y0 >= -1.0, (
        f"参数文本被裁掉底部：包围盒下沿 y0={bb.y0:.0f}px，"
        f"画布高 {fig.bbox.height:.0f}px（figsize 高 "
        f"{fig.get_size_inches()[1]:.2f}in），溢出 {-bb.y0:.0f}px")


def test_long_params_text_is_not_clipped_on_the_right():
    win = _window()
    fig = win._create_params_figure("Stress Test", STRESS_TEXT)
    bb = _text_bbox(fig)
    assert bb.x1 <= fig.bbox.width + 1.0, (
        f"参数文本被裁掉右侧：包围盒右沿 x1={bb.x1:.0f}px，"
        f"画布宽 {fig.bbox.width:.0f}px，溢出 {bb.x1 - fig.bbox.width:.0f}px")


def test_long_line_is_wrapped_so_nothing_is_lost():
    win = _window()
    fig = win._create_params_figure("Stress Test", STRESS_TEXT)
    drawn = fig.texts[0].get_text()
    assert _strip(drawn) == _strip(STRESS_TEXT), "折行不得丢字符"
    assert max(len(line) for line in drawn.split("\n")) < len(LONG_LINE), (
        "500 字符的长行没有被折行")
    assert fig.get_size_inches()[1] <= 60.0, "自适应高度必须有上限"


def test_short_params_keep_the_compact_default_height():
    """短参数不该被撑成一张巨长的纸。"""
    win = _window()
    fig = win._create_params_figure("Tiny", "[A]\n  b: 1\n")
    assert fig.get_size_inches()[1] == 6.0


def test_provenance_figure_still_fits_its_text():
    win = _window()
    fig = win.fig5
    bb = _text_bbox(fig)
    assert bb.y0 >= -1.0 and bb.x1 <= fig.bbox.width + 1.0
    assert 6.0 <= fig.get_size_inches()[1] <= 60.0


# ---------------------------------------------------------------------------
# 4. Tab5 的溯源文本真的进了 PDF
# ---------------------------------------------------------------------------
def test_provenance_text_lands_in_the_pdf(tmp_path):
    win = _window()
    target = tmp_path / "report.pdf"
    seen = {}
    orig = (dw.filedialog, dw.messagebox)
    dw.filedialog = types.SimpleNamespace(asksaveasfilename=lambda **kw: str(target))
    dw.messagebox = types.SimpleNamespace(
        showinfo=lambda *a, **k: seen.setdefault("info", a),
        showerror=lambda *a, **k: seen.setdefault("error", a))
    try:
        win.export_pdf_report(win.figs_to_export)
    finally:
        dw.filedialog, dw.messagebox = orig

    assert "error" not in seen, seen
    assert target.is_file()

    reader = PdfReader(str(target))
    assert len(reader.pages) == 9, f"9 页报告，实际 {len(reader.pages)} 页"
    blob = "\n".join(page.extract_text() or "" for page in reader.pages)

    for frag in ("Preset Provenance", "Imported preset (SPD)", "Imported preset (CURVE)",
                 "Live calibration / fit", "spd.json", "curve.json",
                 "mapping_slope: -2.739846", "drift_level: OK",
                 "fluor_name: Ypet (Venus variant)", "r_squared: 0.999"):
        assert frag in blob, f"溯源片段 {frag!r} 没进 PDF"

    # 四个 tab 的参数标题也必须各自出现在 PDF 里
    for frag in ("[Wavelength Mapping Equation]", "[Method Notes]",
                 "[Measurement Status]", "[Regression Analysis]"):
        assert frag in blob, f"参数片段 {frag!r} 没进 PDF"


def test_export_pdf_report_survives_an_empty_or_none_figure_list(tmp_path):
    win = _window()
    target = tmp_path / "x.pdf"
    seen = {}
    orig = (dw.filedialog, dw.messagebox)
    dw.filedialog = types.SimpleNamespace(asksaveasfilename=lambda **kw: str(target))
    dw.messagebox = types.SimpleNamespace(
        showinfo=lambda *a, **k: seen.setdefault("info", a),
        showerror=lambda *a, **k: seen.setdefault("error", a))
    try:
        win.export_pdf_report([])          # 空列表：静默返回，不得建文件、不得报错
        win.export_pdf_report([None])      # None：必须被当作空处理，不能崩
        win.export_pdf_report(None)
    finally:
        dw.filedialog, dw.messagebox = orig

    assert not target.exists()
    assert "error" not in seen, f"空/None 不该弹错误框：{seen}"
