"""End-to-end flow completeness: preset provenance + auto-continue + auto report.

The GUI module needs tkinter, which headless CI lacks, so this file reuses the
permissive Tk stubbing from ``tests/test_gui_preset_flow.py`` and then drives the
real ``FSSA_GUI`` methods on a bare instance (``__new__`` so no widget tree is
built).  Every test executes the real function body -- the only things faked are
the file-selection dialogs and the raw-DNG image readers.

Covered behaviours (the three user-reported flow gaps):
  * Step 4 preset import must continue into the calculation instead of stopping
    at the "imported" dialog.
  * A successful calculation must schedule the report window.
  * Every step must record whether its parameters came from an imported preset
    or from a live calibration in this session.
"""
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from config import FLUOROPHORES, canonical_fluorophore, fluorophore_display_name
from core.calibration import LinearMapping, SpectralResponseCalibrator
from core.quantification import calculate_concentration as fit_curve

# 荧光素一律用 canonical ID（= Android enum name）。需要给用户看的文字（下拉框、
# 日志、报告）再从 FLUOROPHORES 里取 display_name。这里不写死具体 key，避免
# 下次改 key 又全量破一次（之前 assert payload["version"] == "1.4.7" 就吃过亏）。
YPET = canonical_fluorophore("Ypet (Venus variant)")
MCHERRY = canonical_fluorophore("mCherry")
FLUORESCEIN = canonical_fluorophore("Fluorescein (sodium salt)")


# ---------------------------------------------------------------------------
# Tk stubs -- identical technique to tests/test_gui_preset_flow.py
# ---------------------------------------------------------------------------
class Stub:
    """A permissive stand-in for any Tk widget / constant."""

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
    """Stand-in for ``tk.TclError`` so ``except`` clauses stay valid."""


def _stub_module(name):
    module = types.ModuleType(name)
    # ``importlib.util.find_spec`` refuses modules without a spec, and sibling
    # test modules probe it to decide whether they need their own stubs -- so
    # give the stub a spec instead of leaving ``__spec__`` as None.
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)

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

import matplotlib
matplotlib.use = lambda *args, **kwargs: None

from matplotlib.figure import Figure          # noqa: E402  (after the stub hook)

from gui import main_window as mw            # noqa: E402

mw._THEME_ON = False


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------
class RecordingRoot(Stub):
    """A Tk root that records ``after()`` scheduling instead of running it."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.after_calls = []

    def after(self, delay, callback):
        self.after_calls.append((delay, callback))

    def scheduled(self, callback):
        return [delay for delay, cb in self.after_calls if cb == callback]


class Recorder:
    """Captures every messagebox call so the test can assert on user feedback."""

    def __init__(self):
        self.warnings, self.infos, self.errors = [], [], []
        self.confirm_answer = True

    def showwarning(self, title, message, **kwargs):
        self.warnings.append((title, message))

    def showinfo(self, title, message, **kwargs):
        self.infos.append((title, message))

    def showerror(self, title, message, **kwargs):
        self.errors.append((title, message))

    def askyesno(self, title, message, **kwargs):
        self.warnings.append((title, message))
        return self.confirm_answer


class StatusLog:
    def __init__(self):
        self.entries = []

    def __call__(self, step, message, color="black"):
        self.entries.append((step, message, color))


class FakeWin:
    """Stand-in for a live ``DetailsWindow.win``."""

    def __init__(self, exists=True):
        self._exists = exists
        self.lifted = False
        self.destroyed = False

    def winfo_exists(self):
        return self._exists

    def lift(self):
        self.lifted = True

    def focus_force(self):
        return None

    def destroy(self):
        self.destroyed = True
        self._exists = False


def _make_gui(analysis_mode="legacy"):
    """A bare FSSA_GUI with only the attributes the driven handlers touch."""
    gui = mw.FSSA_GUI.__new__(mw.FSSA_GUI)
    gui.root = RecordingRoot()
    gui.analysis_mode = analysis_mode
    gui.is_turbidity_mode = (analysis_mode == "turbidity")
    gui.mapping = LinearMapping(-2.738911, 3198.4744)
    gui.x_roi = None
    gui.calibrator = None
    gui.rb_swapped = False
    gui.spd_params_path = None
    gui.txt_file = None
    gui.csv_file = None
    gui.current_fluorophore = YPET
    gui.cache_step1 = {}
    gui.cache_step2 = {}
    gui.cache_step3 = {}
    gui.cache_step4 = {}
    gui.curve_preset = None
    gui.preset_provenance = {"step1": None, "step2": None, "step3": None, "step4": None}
    gui.details_window = None
    gui.turbidity_window = None
    gui._lod_headline = None
    gui.txt_result = Stub()
    gui.fig = Figure(figsize=(4, 3), dpi=72)
    gui.ax = gui.fig.add_subplot(111)
    gui.canvas_plot = Stub()
    for step in (1, 2, 3, 4):
        setattr(gui, f"lbl_status{step}", Stub())
    gui.messages = Recorder()
    gui.log = StatusLog()
    gui.log_status = gui.log
    # Step 3 / Step 4 widget handles the preset handlers poke at
    gui.sample_selector = types.SimpleNamespace(get_files=lambda: [])
    gui.blank_selector = types.SimpleNamespace(get_files=lambda: [])
    gui.fluor_combo = Stub()
    gui.btn_calc_conc = Stub()
    # 【v1.4.8】turbidity 模式为 None；非 turbidity 由 _setup_page3 建立
    gui.lbl_fluor_channel = None
    # 【v1.4.7】非 turbidity 模式下 Step 4 预设控件真实存在；turbidity 下这三个句柄为 None
    gui.curve_preset_frame = Stub()
    gui.btn_clear_curve_preset = Stub()
    gui.lbl_curve_preset = Stub()
    return gui


@pytest.fixture
def gui(monkeypatch):
    instance = _make_gui()
    monkeypatch.setattr(mw, "messagebox", instance.messages)
    return instance


def _ns(**kwargs):
    return types.SimpleNamespace(**kwargs)


def _areas_patch(gui, value=3.0):
    """Replace the DNG reader with a deterministic net-area source."""
    def _fake(blank_paths, sample_paths, fluor_config):
        wls = np.linspace(400.0, 700.0, 301)
        intensity = np.exp(-0.5 * ((wls - 530.0) / 40.0) ** 2)
        mask = (wls >= 510.0) & (wls <= 550.0)
        return [float(value)] * max(len(sample_paths), 1), wls, intensity, mask

    gui.get_net_areas_batch = _fake
    return _fake


# ---------------------------------------------------------------------------
# Fixture data: a real Step 4 curve preset on disk
# ---------------------------------------------------------------------------
def _write_curve_preset(tmp_path, name="curve.json", fluorophore=None):
    """``fluorophore`` 默认写 canonical；传旧显示名可模拟 Python 存量 preset。"""
    concentrations = [0.0, 1.0, 2.0]
    means = [1.0, 3.0, 5.0]
    fit = fit_curve(concentrations, means, 0.0, mode="legacy")
    from core.curve_preset import save_curve_preset
    fluor = YPET if fluorophore is None else fluorophore
    cache = {
        **fit,
        "concentrations": concentrations,
        "means": means,
        "stds": [0.1, 0.2, 0.3],
        "metric_mode": "legacy",
        "fluorophore": fluor,
        "regression_method": "ols",
    }
    path = tmp_path / name
    save_curve_preset(str(path), cache, "legacy", fluor)
    return path


def _write_spd_preset(tmp_path, name="spd.json"):
    pixels = np.arange(3072, dtype=np.float64)
    true_wl = np.linspace(380.0, 720.0, 341)
    true_val = np.exp(-0.5 * ((true_wl - 540.0) / 70.0) ** 2) + 0.05
    lamp = np.column_stack([
        40.0 * np.exp(-0.5 * ((pixels - 1493.0) / 120.0) ** 2),
        70.0 * np.exp(-0.5 * ((pixels - 1767.0) / 120.0) ** 2),
        95.0 * np.exp(-0.5 * ((pixels - 1931.0) / 120.0) ** 2),
    ])
    mapping = LinearMapping(-2.738911, 3198.4744)
    calibrator = SpectralResponseCalibrator(mapping)
    calibrator.calibrate(pixels, lamp, true_wl, true_val)
    path = tmp_path / name
    calibrator.save_params(str(path))
    return path


def _make_calibrator():
    """A Step-2-ready calibrator, so Step 2 counts as completed."""
    pixels = np.arange(3072, dtype=np.float64)
    true_wl = np.linspace(380.0, 720.0, 341)
    true_val = np.exp(-0.5 * ((true_wl - 540.0) / 70.0) ** 2) + 0.05
    lamp = np.column_stack([
        40.0 * np.exp(-0.5 * ((pixels - 1493.0) / 120.0) ** 2),
        70.0 * np.exp(-0.5 * ((pixels - 1767.0) / 120.0) ** 2),
        95.0 * np.exp(-0.5 * ((pixels - 1931.0) / 120.0) ** 2),
    ])
    calibrator = SpectralResponseCalibrator(LinearMapping(-2.738911, 3198.4744))
    calibrator.calibrate(pixels, lamp, true_wl, true_val)
    return calibrator


def _write_wavelength_preset(tmp_path, name="wavelength.json"):
    from core.wavelength_preset import save_wavelength_preset
    mapping = LinearMapping(-2.738911, 3198.4744)
    peaks = [
        _ns(name="R", y_centroid=900.0, centroid_px=900.0, lo=880.0, hi=920.0,
            y_max=1.0, baseline=0.0, height=1.0, estimator="centroid",
            n_maxima=1, quality="GOOD", centroid_legacy=900.0),
        _ns(name="G", y_centroid=1300.0, centroid_px=1300.0, lo=1280.0, hi=1320.0,
            y_max=1.0, baseline=0.0, height=1.0, estimator="centroid",
            n_maxima=1, quality="GOOD", centroid_legacy=1300.0),
        _ns(name="B", y_centroid=1600.0, centroid_px=1600.0, lo=1580.0, hi=1620.0,
            y_max=1.0, baseline=0.0, height=1.0, estimator="centroid",
            n_maxima=1, quality="GOOD", centroid_legacy=1600.0),
    ]
    path = tmp_path / name
    save_wavelength_preset(
        str(path), mapping, name="bench-wl", peaks=peaks,
        x_roi=(100, 3000), sensor_size=(4096, 3072),
        known_wls={"R": 622.5, "G": 522.5, "B": 462.5},
        exposure_seconds=0.25, frames=3, rb_swapped=False,
        analysis_mode="legacy")
    return path


# ---------------------------------------------------------------------------
# _record_provenance -- the contract details_window.py relies on
# ---------------------------------------------------------------------------
def test_record_provenance_stringifies_every_param(gui):
    gui._record_provenance(
        2, "preset", preset_type="SPD", file_path="D:/tmp/spd.json",
        preset_name="bench", params={"f": 1.234567891, "i": 3, "b": True,
                                     "s": "text", "n": None, "t": (1, 2)})

    entry = gui.preset_provenance["step2"]
    assert entry["kind"] == "preset"
    assert entry["preset_type"] == "SPD"
    assert entry["file_name"] == "spd.json"
    assert entry["preset_name"] == "bench"
    assert entry["imported_at"]
    params = entry["params"]
    assert set(params) == {"f", "i", "b", "s", "n", "t"}
    assert all(isinstance(value, str) for value in params.values())
    assert params["f"] == "1.23457"          # float -> %.6g
    assert params["i"] == "3"
    assert params["s"] == "text"


def test_record_provenance_resolves_the_file_path(gui, tmp_path):
    path = tmp_path / "sub" / "preset.json"
    path.parent.mkdir()
    path.write_text("{}", encoding="utf-8")

    gui._record_provenance(1, "preset", preset_type="WAVELENGTH", file_path=str(path))

    entry = gui.preset_provenance["step1"]
    assert entry["file"] == str(path.resolve())
    assert entry["file_name"] == "preset.json"


def test_record_provenance_tolerates_a_missing_container(gui):
    del gui.preset_provenance
    gui._record_provenance(4, "live", params={"slope": 2.0})
    assert gui.preset_provenance["step4"]["kind"] == "live"
    assert gui.preset_provenance["step4"]["file_name"] is None


# ---------------------------------------------------------------------------
# Step 1 -- live calibration and WAVELENGTH preset
# ---------------------------------------------------------------------------
def test_step1_live_records_a_live_provenance(gui, monkeypatch):
    gui.pos_selector = _ns(get_files=lambda: ["pos.dng"])
    monkeypatch.setattr(mw, "parse_known_wavelengths",
                        lambda path: {"R": _ns(wavelength_nm=622.5),
                                      "G": _ns(wavelength_nm=522.5),
                                      "B": _ns(wavelength_nm=462.5)})
    monkeypatch.setattr(mw, "read_dng_frame",
                        lambda path: {"rgb": np.zeros((3072, 4096, 3)),
                                      "exposure": 0.25, "x_roi": (100, 3000),
                                      "cfa": (2, 1, 1, 0)})
    monkeypatch.setattr(mw, "get_average_profiles",
                        lambda paths, x_roi: {"red": np.arange(10.0),
                                              "green": np.arange(10.0) + 1.0,
                                              "blue": np.arange(10.0) + 2.0})
    monkeypatch.setattr(mw, "apply_rb_swap", lambda profiles, swapped: profiles)
    monkeypatch.setattr(
        mw, "calibrate_wavelength",
        lambda profiles, knowns: {
            "peaks": [_ns(name="R", y_centroid=900.0, quality="GOOD"),
                      _ns(name="G", y_centroid=1300.0, quality="GOOD"),
                      _ns(name="B", y_centroid=1600.0, quality="GOOD")],
            "k": -2.738911, "b": 3198.4744, "predicted": 1300.0,
            "residual": 0.5, "error_nm": 0.8, "swapped": False,
            "quality": "GOOD"})

    gui.step1_calibrate()

    entry = gui.preset_provenance["step1"]
    assert entry["kind"] == "live"
    assert entry["preset_type"] is None
    assert entry["file"] is None
    assert entry["params"]["cfa"] == "(2, 1, 1, 0)"
    assert entry["params"]["frames"] == "1"
    assert entry["params"]["rb_swapped"] == "False"
    assert "R@" in entry["params"]["peaks"]


def test_step1_preset_records_the_wavelength_preset(gui, monkeypatch, tmp_path):
    path = _write_wavelength_preset(tmp_path)
    monkeypatch.setattr(mw.filedialog, "askopenfilename", lambda **kw: str(path))

    gui.load_wavelength_preset()

    entry = gui.preset_provenance["step1"]
    assert entry["kind"] == "preset"
    assert entry["preset_type"] == "WAVELENGTH"
    assert entry["file_name"] == "wavelength.json"
    assert entry["preset_name"] == "bench-wl"
    assert entry["params"]["frames"] == "3"
    assert entry["params"]["sensor_size"] == "(4096, 3072)"
    assert gui.messages.infos, "a successful load must confirm to the user"


# ---------------------------------------------------------------------------
# Step 2 -- live calibration and SPD preset
# ---------------------------------------------------------------------------
def test_step2_live_records_a_live_provenance(gui, monkeypatch):
    pixels = np.arange(3072, dtype=np.float64)
    lamp = np.column_stack([
        40.0 * np.exp(-0.5 * ((pixels - 1493.0) / 120.0) ** 2),
        70.0 * np.exp(-0.5 * ((pixels - 1767.0) / 120.0) ** 2),
        95.0 * np.exp(-0.5 * ((pixels - 1931.0) / 120.0) ** 2),
    ])
    gui.spd_selector = _ns(get_files=lambda: ["spd1.dng", "spd2.dng"])
    gui.csv_file = "true_spd.csv"
    monkeypatch.setattr(mw, "get_average_profiles",
                        lambda paths, x_roi: {"red": lamp[:, 0],
                                              "green": lamp[:, 1],
                                              "blue": lamp[:, 2]})
    monkeypatch.setattr(mw, "apply_rb_swap", lambda profiles, swapped: profiles)
    monkeypatch.setattr(mw, "load_true_spd_data",
                        lambda path: (np.linspace(380.0, 720.0, 341),
                                      np.exp(-0.5 * ((np.linspace(380.0, 720.0, 341)
                                                      - 540.0) / 70.0) ** 2) + 0.05))
    monkeypatch.setattr(mw, "get_exposure_time", lambda path: 0.5)

    gui.step2_calibrate()

    entry = gui.preset_provenance["step2"]
    assert entry["kind"] == "live"
    assert entry["params"]["frames"] == "2"
    assert entry["params"]["exposure"] == "0.5"
    assert entry["params"]["spd_csv"] == "true_spd.csv"
    assert "x_roi" in entry["params"]


def test_step2_cache_is_entirely_on_the_wavelength_grid(gui, monkeypatch):
    """【SPD payload v3】cache_step2 里所有曲线都必须落在同一条规范波长网格上。

    定位光剖面（pos_*）原本是逐像素的；不重采样的话 details_window 的
    ``len(pos_R) == len(wls)`` 会不成立，Figure 4a/4b 会静默退化成占位文本。
    """
    from core.calibration import canonical_grid
    pixels = np.arange(3072, dtype=np.float64)
    lamp = np.column_stack([
        40.0 * np.exp(-0.5 * ((pixels - 1493.0) / 120.0) ** 2),
        70.0 * np.exp(-0.5 * ((pixels - 1767.0) / 120.0) ** 2),
        95.0 * np.exp(-0.5 * ((pixels - 1931.0) / 120.0) ** 2),
    ])
    gui.spd_selector = _ns(get_files=lambda: ["spd1.dng"])
    gui.csv_file = "true_spd.csv"
    gui.cache_step1["profiles"] = {"red": lamp[:, 0], "green": lamp[:, 1],
                                   "blue": lamp[:, 2]}
    monkeypatch.setattr(mw, "get_average_profiles",
                        lambda paths, x_roi: {"red": lamp[:, 0],
                                              "green": lamp[:, 1],
                                              "blue": lamp[:, 2]})
    monkeypatch.setattr(mw, "apply_rb_swap", lambda profiles, swapped: profiles)
    monkeypatch.setattr(mw, "load_true_spd_data",
                        lambda path: (np.linspace(380.0, 720.0, 341),
                                      np.exp(-0.5 * ((np.linspace(380.0, 720.0, 341)
                                                      - 540.0) / 70.0) ** 2) + 0.05))
    monkeypatch.setattr(mw, "get_exposure_time", lambda path: 0.5)

    gui.step2_calibrate()

    grid_size = canonical_grid().size
    cache = gui.cache_step2
    np.testing.assert_allclose(cache["wavelengths"], canonical_grid(),
                               rtol=0.0, atol=1e-9)
    for key in ("R", "G", "B", "pos_R", "pos_G", "pos_B", "measured_total",
                "corrected_total", "final_mask"):
        assert len(cache[key]) == grid_size, f"{key} 不在波长网格上"
    # 随 Mode B 一起消失的键
    for dead in ("lamp_R", "lamp_G", "lamp_B", "spd_aligned", "spd_reference",
                 "recomputed"):
        assert dead not in cache, f"{dead} 不该再出现在 cache_step2"


def test_step2_preset_records_the_spd_preset(gui, tmp_path):
    path = _write_spd_preset(tmp_path)

    assert gui._load_spd_params_from(str(path), silent=True) is True

    entry = gui.preset_provenance["step2"]
    assert entry["kind"] == "preset"
    assert entry["preset_type"] == "SPD"
    assert entry["file_name"] == "spd.json"
    assert "drift_nm" in entry["params"]
    assert "drift_level" in entry["params"]


# ---------------------------------------------------------------------------
# Step 3
# ---------------------------------------------------------------------------
def test_step3_live_records_the_blank_and_sample_files(gui):
    gui.blank_selector = _ns(get_files=lambda: ["blank.dng"])
    gui.sample_selector = _ns(get_files=lambda: ["sample_a.dng", "sample_b.dng"])
    _areas_patch(gui)

    gui.step3_analyze()

    entry = gui.preset_provenance["step3"]
    assert entry["kind"] == "live"
    assert entry["params"]["blank_files"] == "blank.dng"
    assert entry["params"]["sample_files"] == "sample_a.dng, sample_b.dng"
    assert entry["params"]["n_sample"] == "2"


# ---------------------------------------------------------------------------
# Step 4 -- the auto-continue gap (Fact 2)
# ---------------------------------------------------------------------------
def test_load_curve_preset_schedules_the_calculation_when_step3_is_ready(
        gui, monkeypatch, tmp_path):
    path = _write_curve_preset(tmp_path)
    gui.calibrator = _make_calibrator()          # Step 2 done
    gui.blank_selector = _ns(get_files=lambda: ["blank.dng"])
    gui.sample_selector = _ns(get_files=lambda: ["sample.dng"])
    monkeypatch.setattr(mw.filedialog, "askopenfilename", lambda **kw: str(path))

    gui.load_curve_preset()

    assert gui.root.scheduled(gui.calculate_concentration) == [150]
    entry = gui.preset_provenance["step4"]
    assert entry["kind"] == "preset"
    assert entry["preset_type"] == "CURVE"
    assert entry["file_name"] == "curve.json"
    assert entry["params"]["n_standards"] == "3"
    assert entry["params"]["metric_mode"] == "legacy"
    # 溯源/报告是给人看的：必须是本地化显示名，不能是 "YPET" 这种大写串
    assert entry["params"]["fluorophore"] == "Ypet (Venus variant)"
    # the "stopped at the import dialog" dialog must be gone
    assert not gui.messages.infos


def test_load_curve_preset_names_the_missing_blank(gui, monkeypatch, tmp_path):
    path = _write_curve_preset(tmp_path)
    gui.calibrator = _make_calibrator()          # Step 1 & 2 done
    gui.blank_selector = _ns(get_files=lambda: [])
    gui.sample_selector = _ns(get_files=lambda: ["sample.dng"])
    monkeypatch.setattr(mw.filedialog, "askopenfilename", lambda **kw: str(path))

    gui.load_curve_preset()

    assert not gui.root.after_calls, "nothing may run without a blank"
    assert gui.messages.infos
    text = gui.messages.infos[-1][1].lower()
    assert "blank" in text and "sample" not in text


def test_load_curve_preset_names_both_missing_inputs(gui, monkeypatch, tmp_path):
    path = _write_curve_preset(tmp_path)
    gui.calibrator = _make_calibrator()          # Step 1 & 2 done
    gui.blank_selector = _ns(get_files=lambda: [])
    gui.sample_selector = _ns(get_files=lambda: [])
    monkeypatch.setattr(mw.filedialog, "askopenfilename", lambda **kw: str(path))

    gui.load_curve_preset()

    assert not gui.root.after_calls
    text = gui.messages.infos[-1][1].lower()
    assert "blank" in text and "sample" in text


# ---------------------------------------------------------------------------
# Step 4 -- calculation keeps the preset provenance and schedules the report
# ---------------------------------------------------------------------------
def test_calculate_with_curve_preset_keeps_the_preset_provenance(
        gui, monkeypatch, tmp_path):
    path = _write_curve_preset(tmp_path)
    gui.blank_selector = _ns(get_files=lambda: ["blank.dng"])
    gui.sample_selector = _ns(get_files=lambda: ["sample.dng"])
    monkeypatch.setattr(mw.filedialog, "askopenfilename", lambda **kw: str(path))
    _areas_patch(gui, value=3.0)
    gui.load_curve_preset()

    gui.calculate_with_curve_preset()

    entry = gui.preset_provenance["step4"]
    assert entry["kind"] == "preset"
    assert entry["file_name"] == "curve.json"
    assert entry["params"]["predicted_conc"] == "1"
    assert gui.root.scheduled(gui._auto_show_report) == [200]


def test_calculate_concentration_live_records_provenance_and_shows_the_report(
        gui):
    _areas_patch(gui, value=4.0)
    gui.metric_mode = _ns(get=lambda: "legacy")
    gui.regression_method = _ns(get=lambda: "ols")
    gui.blank_selector = _ns(get_files=lambda: ["blank.dng"])
    gui.sample_selector = _ns(get_files=lambda: ["sample.dng"])
    gui.std_rows = [{"conc_var": _ns(get=lambda: "0.0"),
                     "blank": _ns(get_files=lambda: ["b.dng"]),
                     "sample": _ns(get_files=lambda: ["s.dng"])},
                    {"conc_var": _ns(get=lambda: "2.0"),
                     "blank": _ns(get_files=lambda: ["b.dng"]),
                     "sample": _ns(get_files=lambda: ["s.dng"])}]

    gui.calculate_concentration()

    entry = gui.preset_provenance["step4"]
    assert entry["kind"] == "live"
    assert entry["params"]["regression_method"] == "ols"
    assert entry["params"]["n_standards"] == "2"
    assert gui.root.scheduled(gui._auto_show_report) == [200]


def test_calculate_concentration_delegates_to_the_preset_path_without_a_second_report(
        gui, monkeypatch, tmp_path):
    path = _write_curve_preset(tmp_path)
    monkeypatch.setattr(mw.filedialog, "askopenfilename", lambda **kw: str(path))
    gui.load_curve_preset()
    gui.root.after_calls.clear()
    calls = []
    monkeypatch.setattr(gui, "calculate_with_curve_preset", lambda: calls.append(1))

    gui.calculate_concentration()

    assert calls == [1], "the preset branch must delegate exactly once"
    assert gui.root.after_calls == []


# ---------------------------------------------------------------------------
# D3 -- an SPD preset the user rejects must not stay in the report provenance
# ---------------------------------------------------------------------------
def test_spd_preset_rollback_clears_the_step2_provenance(gui, monkeypatch, tmp_path):
    """SEVERE drift + "no" must leave no Step 2 source in the report.

    Before the fix the rollback restored ``calibrator`` / ``spd_params_path`` /
    ``cache_step2`` but kept the provenance of the previous (discarded) import,
    so the report still claimed an SPD preset the user had just refused.
    """
    path = _write_spd_preset(tmp_path)
    assert gui._load_spd_params_from(str(path), silent=True) is True
    assert gui.preset_provenance["step2"]["kind"] == "preset"

    monkeypatch.setattr(gui, "_report_spd_drift", lambda calibrator: False)
    assert gui._load_spd_params_from(str(path), silent=True) is False

    assert gui.preset_provenance["step2"] is None
    assert gui.cache_step2 == {}
    assert gui.spd_params_path is None


# ---------------------------------------------------------------------------
# Report window -- idempotency and the provenance hand-off (Facts 1 and 3)
# ---------------------------------------------------------------------------
def test_show_details_window_reuses_the_existing_window(gui, monkeypatch):
    created = []

    class FakeDetails:
        def __init__(self, *args, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(mw, "DetailsWindow", FakeDetails)
    existing = _ns(win=FakeWin(exists=True))
    gui.details_window = existing

    gui.show_details_window()

    assert created == []
    assert existing.win.lifted is True


def test_show_details_window_passes_the_provenance(gui, monkeypatch):
    created = []

    class FakeDetails:
        def __init__(self, *args, **kwargs):
            created.append(kwargs)
            self.win = FakeWin(exists=False)

    monkeypatch.setattr(mw, "DetailsWindow", FakeDetails)
    gui.preset_provenance["step1"] = {"kind": "preset"}

    gui.show_details_window()

    assert len(created) == 1
    assert created[0]["preset_provenance"] is gui.preset_provenance


def test_auto_show_report_opens_the_details_window(gui, monkeypatch):
    opened = []
    monkeypatch.setattr(
        gui, "show_details_window",
        lambda force_refresh=False: opened.append(force_refresh))

    gui._auto_show_report()

    assert opened == [True], "the auto-popped report must force a rebuild"


def test_show_details_window_without_a_mapping_does_not_open(gui, monkeypatch):
    created = []

    class FakeDetails:
        def __init__(self, *args, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(mw, "DetailsWindow", FakeDetails)
    gui.mapping = None

    gui.show_details_window()

    assert created == []
    assert gui.messages.infos


# ---------------------------------------------------------------------------
# D1 -- the auto-popped report must show the newest result, not a stale window
# ---------------------------------------------------------------------------
def test_auto_show_report_rebuilds_a_live_window(gui, monkeypatch):
    """``DetailsWindow`` renders once in ``__init__`` and has no refresh API.

    Before the fix ``_auto_show_report()`` only lifted the surviving window, so a
    second calculation left the old ``cache_step4`` and the old provenance on
    screen.
    """
    created = []

    class FakeDetails:
        def __init__(self, *args, **kwargs):
            created.append(kwargs)
            self.win = FakeWin(exists=True)

    monkeypatch.setattr(mw, "DetailsWindow", FakeDetails)
    stale = _ns(win=FakeWin(exists=True))
    gui.details_window = stale
    gui.preset_provenance["step4"] = {"kind": "preset",
                                      "preset_type": "CURVE",
                                      "file_name": "curve.json"}

    gui._auto_show_report()

    assert stale.win.destroyed is True, "the stale report must be destroyed, not lifted"
    assert stale.win.lifted is False
    assert len(created) == 1, "a fresh report window must be built"
    assert gui.details_window is not stale
    # the rebuilt window must carry the newest provenance
    assert created[0]["preset_provenance"] is gui.preset_provenance
    assert created[0]["preset_provenance"]["step4"]["file_name"] == "curve.json"


def test_manual_details_call_keeps_reusing_the_window(gui, monkeypatch):
    """The toolbar「📋 Details」path must still lift, never rebuild."""
    created = []

    class FakeDetails:
        def __init__(self, *args, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(mw, "DetailsWindow", FakeDetails)
    live = _ns(win=FakeWin(exists=True))
    gui.details_window = live

    gui.show_details_window()

    assert created == []
    assert live.win.destroyed is False
    assert live.win.lifted is True


# ---------------------------------------------------------------------------
# D2 -- clearing the curve preset must clear its provenance
# ---------------------------------------------------------------------------
def test_clear_curve_preset_clears_the_step4_provenance(gui, monkeypatch, tmp_path):
    path = _write_curve_preset(tmp_path)
    monkeypatch.setattr(mw.filedialog, "askopenfilename", lambda **kw: str(path))

    gui.load_curve_preset()
    assert gui.preset_provenance["step4"]["kind"] == "preset"

    gui.clear_curve_preset()

    assert gui.curve_preset is None
    assert gui.preset_provenance["step4"] is None, (
        "the report must not keep claiming an imported CURVE preset after it is cleared")


# ---------------------------------------------------------------------------
# D4 -- auto-continue must not fire before Step 1 / Step 2 are ready
# ---------------------------------------------------------------------------
def test_load_curve_preset_without_step1_does_not_schedule(gui, monkeypatch, tmp_path):
    path = _write_curve_preset(tmp_path)
    gui.mapping = None
    gui.calibrator = _make_calibrator()
    gui.blank_selector = _ns(get_files=lambda: ["blank.dng"])
    gui.sample_selector = _ns(get_files=lambda: ["sample.dng"])
    monkeypatch.setattr(mw.filedialog, "askopenfilename", lambda **kw: str(path))

    gui.load_curve_preset()

    assert gui.root.scheduled(gui.calculate_concentration) == []
    text = gui.messages.infos[-1][1].lower()
    assert "step 1" in text and "step 2" not in text, "the hint must name Step 1"
    assert gui.curve_preset is not None, "the preset itself stays loaded"


def test_load_curve_preset_without_step2_does_not_schedule(gui, monkeypatch, tmp_path):
    path = _write_curve_preset(tmp_path)
    gui.calibrator = None
    gui.blank_selector = _ns(get_files=lambda: ["blank.dng"])
    gui.sample_selector = _ns(get_files=lambda: ["sample.dng"])
    monkeypatch.setattr(mw.filedialog, "askopenfilename", lambda **kw: str(path))

    gui.load_curve_preset()

    assert gui.root.scheduled(gui.calculate_concentration) == []
    text = gui.messages.infos[-1][1].lower()
    assert "step 2" in text and "step 1" not in text, "the hint must name Step 2"


def test_load_curve_preset_keeps_the_built_curve_when_it_cannot_rerun(
        gui, monkeypatch, tmp_path):
    """No auto-rerun -> the previously computed curve must survive."""
    path = _write_curve_preset(tmp_path)
    gui.mapping = None
    gui.cache_step4 = {"slope": 2.0, "intercept": 0.0}
    monkeypatch.setattr(mw.filedialog, "askopenfilename", lambda **kw: str(path))

    gui.load_curve_preset()

    assert gui.cache_step4 == {"slope": 2.0, "intercept": 0.0}


# ---------------------------------------------------------------------------
# v1.4.7 -- turbidity 模式没有 Step 4，曲线预设入口必须整体失效
# ---------------------------------------------------------------------------
class ConfigProbe:
    """Records every .config() write so a hidden widget can be proven untouched."""

    def __init__(self):
        self.config_calls = []

    def config(self, *args, **kwargs):
        self.config_calls.append((args, kwargs))


def test_turbidity_mode_curve_preset_handlers_touch_no_step4_widget(
        monkeypatch, tmp_path):
    """Turbidity never builds page4; the preset handlers must no-op, not crash."""
    gui = _make_gui("turbidity")
    gui.curve_preset_frame = None
    gui.btn_clear_curve_preset = None
    gui.lbl_curve_preset = None
    probe = ConfigProbe()
    gui.btn_calc_conc = probe          # 真实 turbidity 实例里该控件根本不存在
    monkeypatch.setattr(mw, "messagebox", gui.messages)
    monkeypatch.setattr(mw.filedialog, "askopenfilename",
                        lambda **kw: str(tmp_path / "curve.json"))

    gui.load_curve_preset()
    gui.clear_curve_preset()

    assert probe.config_calls == [], "Step 4 控件不该被 turbidity 模式的预设处理函数触碰"
    assert gui.curve_preset is None
    assert gui.preset_provenance["step4"] is None
    assert gui.messages.errors == []
    assert gui.messages.infos == []
    assert gui.root.scheduled(gui.calculate_concentration) == []


def test_turbidity_mode_show_page_does_not_touch_the_preset_frame():
    """show_page must not pack/pack_forget a frame the mode never created."""
    gui = _make_gui("turbidity")
    gui.curve_preset_frame = None
    gui.pages_packed, gui.pages_forgotten = [], []
    for step in (1, 2, 3, 4):
        setattr(gui, f"page{step}", _ns(pack=lambda *a, **k: gui.pages_packed.append(step),
                                        pack_forget=lambda: gui.pages_forgotten.append(step)))
    gui.btn_prev = ConfigProbe()
    gui.btn_next = ConfigProbe()
    gui.lbl_page_num = ConfigProbe()
    gui.max_steps = 3

    for step in (1, 2, 3):
        mw.FSSA_GUI.show_page(gui, step)

    assert gui.pages_packed == [1, 2, 3]
    assert gui.btn_prev.config_calls and gui.btn_next.config_calls


class _ResultText:
    """Records what step3_analyze writes into the results pane."""

    def __init__(self):
        self.logged = ""

    def insert(self, *args):
        if args and isinstance(args[-1], str):
            self.logged += args[-1]

    def see(self, *args):
        return None


def test_fluor_channel_label_reports_the_configured_channel():
    """Step 3 must show which R/G/B channel each fluorophore is read from.

    The report's Tab 3 already carried a "Channel" line; the Step 3 pane and the
    run log only ever showed the fluorophore name, so the user picked a dye
    without being told which channel it would be measured on.
    """
    from config import FLUOROPHORES

    for name, cfg in FLUOROPHORES.items():
        gui = _make_gui()
        gui.lbl_fluor_channel = ConfigProbe()
        gui.current_fluorophore = name
        mw.FSSA_GUI._update_fluor_channel_label(gui)
        calls = gui.lbl_fluor_channel.config_calls
        assert calls, f"no label update for {name}"
        text = calls[-1][1].get("text", "")
        assert cfg["channel"].upper() in text, f"{name}: {text!r} lacks channel"
        assert str(cfg["peak_wl"]) in text, f"{name}: {text!r} lacks peak"
        assert str(cfg["fwhm"]) in text, f"{name}: {text!r} lacks FWHM"


def test_fluor_channel_label_is_inert_without_the_widget():
    """Turbidity mode never builds the fluorophore combo, so the label is None."""
    gui = _make_gui(analysis_mode="turbidity")
    gui.lbl_fluor_channel = None
    mw.FSSA_GUI._update_fluor_channel_label(gui)
    mw.FSSA_GUI.on_fluorophore_change(gui, None)


def test_fluor_channel_label_follows_the_combo_selection():
    gui = _make_gui()
    gui.lbl_fluor_channel = ConfigProbe()
    # 下拉框里是 display_name，内部必须存 canonical
    gui.fluor_combo = types.SimpleNamespace(get=lambda: "mCherry")
    mw.FSSA_GUI.on_fluorophore_change(gui, None)
    assert gui.current_fluorophore == MCHERRY
    text = gui.lbl_fluor_channel.config_calls[-1][1].get("text", "")
    assert "RED" in text, text


def test_step3_log_reports_the_analysis_channel():
    """The run log must name the channel, matching what the report Tab 3 shows."""
    gui = _make_gui()
    gui.current_fluorophore = MCHERRY
    gui.blank_selector = types.SimpleNamespace(get_files=lambda: ["blank.dng"])
    gui.sample_selector = types.SimpleNamespace(get_files=lambda: ["sample.dng"])
    gui.txt_result = _ResultText()

    wls = np.linspace(400.0, 700.0, 256)
    intensity = np.exp(-((wls - 610.0) ** 2) / (2 * 20.0 ** 2))
    mask = np.zeros_like(wls, dtype=bool)
    mask[::2] = True
    gui.get_net_areas_batch = lambda *a, **k: (np.array([1.0, 1.5]), wls, intensity, mask)

    mw.FSSA_GUI.step3_analyze(gui)

    assert "Analysis channel: RED" in gui.txt_result.logged
    assert "Fluorophore: mCherry" in gui.txt_result.logged
    # turbidity 走固定的 green 通道，同样要在日志里报出来
    turb = _make_gui(analysis_mode="turbidity")
    turb.blank_selector = types.SimpleNamespace(get_files=lambda: ["blank.dng"])
    turb.sample_selector = types.SimpleNamespace(get_files=lambda: ["sample.dng"])
    turb.txt_result = _ResultText()
    turb.get_net_areas_batch = lambda *a, **k: (np.array([1.0]), wls, intensity, mask)
    mw.FSSA_GUI.step3_analyze(turb)
    assert "Analysis channel: GREEN" in turb.txt_result.logged


class _ComboProbe:
    """Records config() writes on the fluorophore dropdown."""

    def __init__(self):
        self.config_calls = []
        self.value = None

    def set(self, v):
        self.value = v

    def get(self):
        return self.value

    def config(self, *args, **kwargs):
        self.config_calls.append((args, kwargs))

    def bind(self, *args, **kwargs):
        return None


def test_curve_preset_does_not_lock_the_fluorophore_dropdown(
        gui, monkeypatch, tmp_path):
    """Loading a curve preset must not disable the Step 3 dye dropdown.

    ``calculate_with_curve_preset`` recomputes the sample area with the
    curve's own ``preset["fluorophore"]`` -- what Step 3 has selected is
    irrelevant to it, and the log already reports ``Preset fluorophore``.
    Locking the combobox only stopped users from analysing other dyes:
    the popdown still opened but selections were ignored (disabled state),
    which looked exactly like "the data was filled in but the UI didn't
    change".
    """
    path = _write_curve_preset(tmp_path)
    gui.calibrator = _make_calibrator()
    gui.blank_selector = _ns(get_files=lambda: ["blank.dng"])
    gui.sample_selector = _ns(get_files=lambda: ["sample.dng"])
    probe = _ComboProbe()
    gui.fluor_combo = probe
    monkeypatch.setattr(mw.filedialog, "askopenfilename", lambda **kw: str(path))

    gui.load_curve_preset()

    assert probe.value == "Ypet (Venus variant)"
    disabled = [k for _, k in probe.config_calls if k.get("state") == "disabled"]
    assert not disabled, probe.config_calls


def test_wheel_target_check_survives_combobox_popdown_paths():
    """``bind_all`` also receives wheel events from Combobox popdowns, which
    are standalone Tcl windows: there ``event.widget`` is a plain widget-path
    string, and walking ``.master`` used to raise
    ``AttributeError: 'str' object has no attribute 'master'`` on every wheel
    tick over the dye dropdown."""
    root_w, canvas = object(), object()
    assert mw.FSSA_GUI._wheel_target_in_region(
        ".!combopopdown.f.l", root_w, canvas) is False
    assert mw.FSSA_GUI._wheel_target_in_region(None, root_w, canvas) is False

    class W:
        def __init__(self, master=None):
            self.master = master

    leaf, mid, top = W(), W(), W()
    leaf.master, mid.master = mid, top
    assert mw.FSSA_GUI._wheel_target_in_region(leaf, top, mid) is True
    assert mw.FSSA_GUI._wheel_target_in_region(leaf, top, object()) is False
    assert mw.FSSA_GUI._wheel_target_in_region(top, top, mid) is False


def test_fluorescein_sodium_option_matches_the_requested_parameters():
    """User-requested dye: emission 520 nm on the green channel, excitation
    ~494 nm recorded as metadata. The dropdown, the channel label, the
    calculation and the plot all read ``FLUOROPHORES`` — adding the entry is
    the whole change, so this pins the entry itself."""
    from config import FLUOROPHORES

    cfg = FLUOROPHORES.get(FLUORESCEIN)
    assert cfg is not None, "dye missing from the dropdown source dict"
    assert cfg["display_name"] == "Fluorescein (sodium salt)"
    assert cfg["peak_wl"] == 520
    assert cfg["channel"] == "green"
    assert cfg["fwhm"] > 0
    assert cfg["color"] == "green"
    assert cfg["excitation_nm"] == 494


def test_step3_targets_fluorescein_emission_at_520nm():
    """With Fluorescein selected the drawn 'Target Center' vertical line must
    sit at 520 nm, so the recorded peak lands on the right position (the
    sample in the field photo is fluorescein sodium, not EGFP)."""
    gui = _make_gui()
    gui.current_fluorophore = FLUORESCEIN
    gui.blank_selector = types.SimpleNamespace(get_files=lambda: ["blank.dng"])
    gui.sample_selector = types.SimpleNamespace(get_files=lambda: ["sample.dng"])

    wls = np.linspace(400.0, 700.0, 256)
    intensity = np.exp(-((wls - 521.0) ** 2) / (2 * 18.0 ** 2))
    mask = np.zeros_like(wls, dtype=bool)
    mask[::2] = True
    gui.get_net_areas_batch = lambda *a, **k: (np.array([1.0, 1.5]), wls, intensity, mask)

    mw.FSSA_GUI.step3_analyze(gui)

    vline_xs = [line.get_xdata()[0] for line in gui.ax.get_lines()
                if len(set(np.asarray(line.get_xdata()).tolist())) == 1]
    assert 520.0 in vline_xs, f"target line not at 520 nm: {vline_xs}"


# ---------------------------------------------------------------------------
# 荧光素 canonical ID —— 跨端 preset 互通（Android MycoSpec enum name）
# ---------------------------------------------------------------------------
class _Combo:
    """记录 set/get 的假下拉框（无头环境没有真 ttk.Combobox）。"""

    def __init__(self, parent=None, values=None, **kwargs):
        self.values = list(values or [])
        self.current = None

    def set(self, value):
        self.current = value

    def get(self):
        return self.current

    def pack(self, **kwargs):
        return None

    def bind(self, *args, **kwargs):
        return None


@pytest.mark.parametrize("raw,expected", [
    # canonical 本身，大小写不敏感
    ("YPET", "YPET"), ("ypet", "YPET"), ("Ypet", "YPET"),
    ("EGFP", "EGFP"), ("egfp", "EGFP"),
    ("MCHERRY", "MCHERRY"), ("mcherry", "MCHERRY"),
    ("CFP", "CFP"), ("cfp", "CFP"),
    ("MTURQUOISE2", "MTURQUOISE2"), ("mturquoise2", "MTURQUOISE2"),
    ("FLUORESCEIN", "FLUORESCEIN"), ("fluorescein", "FLUORESCEIN"),
    # Python 旧显示名：存量 preset 里写的就是这些
    ("Ypet (Venus variant)", "YPET"),
    ("mCherry", "MCHERRY"),
    ("mTurquoise2", "MTURQUOISE2"),
    ("Fluorescein (sodium salt)", "FLUORESCEIN"),
    # 常见变体：多余空格 / 连字符
    ("Fluorescein", "FLUORESCEIN"),
    ("m-turquoise2", "MTURQUOISE2"),
    ("mTurquoise 2", "MTURQUOISE2"),
    ("  ypet  ", "YPET"),
])
def test_canonical_fluorophore_resolves_every_known_spelling(raw, expected):
    assert canonical_fluorophore(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   ", "DsRed", "mCherry2", "not a dye"])
def test_canonical_fluorophore_returns_none_for_unknown(raw):
    assert canonical_fluorophore(raw) is None


@pytest.mark.parametrize("android_id", ["MCHERRY", "mcherry", "MCherry"])
def test_curve_preset_written_by_android_loads(gui, monkeypatch, tmp_path, android_id):
    """Android 导出的 preset 写 canonical ID（enum name），Python 必须能加载。

    前两个 case 是 enum name 本身；第三个是大小写/分隔符被改过的写法，
    用来保证解析不是"只有完全等于 key 才认"。
    """
    path = _write_curve_preset(tmp_path, fluorophore=android_id)
    monkeypatch.setattr(mw.filedialog, "askopenfilename", lambda **kw: str(path))
    gui.fluor_combo = _Combo()

    gui.load_curve_preset()

    assert gui.messages.errors == []
    assert gui.current_fluorophore == "MCHERRY"
    assert gui.fluor_combo.get() == "mCherry"      # 下拉框显示 display_name


def test_curve_preset_written_by_older_python_still_loads(gui, monkeypatch, tmp_path):
    """存量 Python preset 写的是旧显示名 mCherry；改 key 不能让它加载失败。"""
    path = _write_curve_preset(tmp_path, fluorophore="mCherry")
    monkeypatch.setattr(mw.filedialog, "askopenfilename", lambda **kw: str(path))
    gui.fluor_combo = _Combo()

    gui.load_curve_preset()

    assert gui.messages.errors == []
    assert gui.current_fluorophore == "MCHERRY"
    assert gui.fluor_combo.get() == "mCherry"      # 用户看到的文字没变


def test_fluorophore_dropdown_lists_display_names(gui, monkeypatch):
    """下拉框 values 必须是 display_name，不能把 'YPET' 这种大写串显示给用户。"""
    built = []

    class _SpyCombo(_Combo):
        def __init__(self, parent=None, values=None, **kwargs):
            built.append(list(values or []))
            super().__init__(parent, values, **kwargs)

    monkeypatch.setattr(mw.ttk, "Combobox", _SpyCombo)
    gui.page3 = Stub()

    mw.FSSA_GUI._setup_page3(gui)

    assert built, "Step 3 did not build the fluorophore dropdown"
    assert built[0] == [cfg["display_name"] for cfg in FLUOROPHORES.values()]
    assert "Ypet (Venus variant)" in built[0] and "mCherry" in built[0]
    assert "YPET" not in built[0] and "MCHERRY" not in built[0]


def test_exported_curve_preset_carries_the_canonical_id(gui, monkeypatch, tmp_path):
    """导出的 preset 必须写 canonical ID，否则 Android 端对不上。"""
    from core.curve_preset import load_curve_preset
    gui.cache_step4 = {
        "concentrations": [0.0, 1.0, 2.0], "means": [1.0, 3.0, 5.0], "stds": [0.1, 0.2, 0.3],
        "slope": 2.0, "intercept": 1.0, "r_value": 0.99, "r_squared": 0.98,
        "metric_mode": "legacy", "regression_method": "ols",
        "fluorophore": "mCherry",          # 存量写法，导出时要规范化
    }
    out = tmp_path / "exported.json"
    monkeypatch.setattr(mw.filedialog, "asksaveasfilename", lambda **kw: str(out))

    gui.export_curve_preset()

    assert gui.messages.errors == []
    assert load_curve_preset(str(out), "legacy")["fluorophore"] == "MCHERRY"
