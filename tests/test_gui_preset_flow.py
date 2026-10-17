"""Headless logic walkthrough of the two GUI preset handlers.

``gui/main_window.py`` needs tkinter, which headless CI lacks, so this module
installs a permissive stub for the Tk layer and then drives
``FSSA_GUI.step1_calibrate`` / ``FSSA_GUI.load_spd_params`` on a bare instance
(``__new__`` instead of ``__init__``, so no widget tree is built).

It covers exactly what a real click does -- validation order, commit order and
the drift warning tiers -- without being able to render anything.
"""
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from core.calibration import LinearMapping, SpectralResponseCalibrator

DATA_DIR = Path("D:/Documents/iGem/Python/FSSA_Documents/10.15")
POSITIONING = DATA_DIR / "positioning.dng"
_real_data = pytest.mark.skipif(
    os.environ.get("FSSA_E2E") != "1" or not POSITIONING.is_file(),
    reason="set FSSA_E2E=1 and mount the 10.15 DNG sample set to run this test",
)


# ---------------------------------------------------------------------------
# Tk stubs
# ---------------------------------------------------------------------------
class Stub:
    """A permissive stand-in for any Tk widget / constant.

    It is a real class so it can also be used as a base class (``gui.widgets``
    subclasses ``ttk.Frame``), and it swallows every attribute and call.
    """

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


_STUBBED = _install_tk_stubs()

# matplotlib.use('TkAgg') would try to load the real backend; the figure
# machinery we drive here is plain Agg-free and never renders.
import matplotlib
matplotlib.use = lambda *args, **kwargs: None

from matplotlib.figure import Figure          # noqa: E402  (after the stub hook)

from gui import main_window as mw            # noqa: E402

# The theme module may import successfully against the stubs; force the classic
# look so the walkthrough only exercises the code paths we care about.
mw._THEME_ON = False


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


def _make_gui():
    """A bare FSSA_GUI with only the attributes the two handlers touch."""
    gui = mw.FSSA_GUI.__new__(mw.FSSA_GUI)
    gui.root = Stub()
    gui.analysis_mode = "lod_iupac"
    gui.is_turbidity_mode = False
    gui.mapping = None
    gui.x_roi = None
    gui.calibrator = None
    gui.rb_swapped = False
    gui.spd_params_path = None
    gui.txt_file = None
    gui.csv_file = None
    gui.current_fluorophore = "Ypet (Venus variant)"
    gui.cache_step1 = {}
    gui.cache_step2 = {}
    gui.cache_step3 = {}
    gui.cache_step4 = {}
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


def _sensor_grid_calibrator(mapping=None, n_px=3072):
    mapping = mapping or LinearMapping(-2.738911, 3198.4744)
    pixels = np.arange(n_px, dtype=np.float64)
    true_wl = np.linspace(380.0, 720.0, 341)
    true_val = np.exp(-0.5 * ((true_wl - 540.0) / 70.0) ** 2) + 0.05
    lamp = np.column_stack([
        40.0 * np.exp(-0.5 * ((pixels - 1493.0) / 120.0) ** 2),
        70.0 * np.exp(-0.5 * ((pixels - 1767.0) / 120.0) ** 2),
        95.0 * np.exp(-0.5 * ((pixels - 1931.0) / 120.0) ** 2),
    ])
    calibrator = SpectralResponseCalibrator(mapping)
    calibrator.calibrate(pixels, lamp, true_wl, true_val)
    return calibrator


# ---------------------------------------------------------------------------
# load_spd_params -- commit order and the soft drift tiers
# ---------------------------------------------------------------------------
def test_load_spd_params_refuses_before_step_1(tmp_path):
    """The mapping must be checked *before* anything else (DEF-P02 follow-up)."""
    gui = _make_gui()
    mw.filedialog.askopenfilename = lambda **kwargs: str(tmp_path / "spd.json")

    gui.load_spd_params()

    assert gui.messages.warnings, "the user must be told why nothing happened"
    assert gui.calibrator is None


def test_load_spd_params_keeps_the_old_calibrator_when_the_file_is_broken(tmp_path):
    """Regression: the old code assigned first and validated afterwards."""
    good = _sensor_grid_calibrator()
    path = tmp_path / "spd.json"
    good.save_params(str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["response_G"] = payload["response_G"][:-1]       # L1 length mismatch
    path.write_text(json.dumps(payload), encoding="utf-8")

    gui = _make_gui()
    gui.mapping = good.mapping
    gui.calibrator = good
    previous_cache = dict(gui.cache_step2)
    mw.filedialog.askopenfilename = lambda **kwargs: str(path)

    gui.load_spd_params()

    assert gui.calibrator is good, "a failed load must not clobber the calibrator"
    assert gui.cache_step2 == previous_cache
    assert gui.messages.errors, "the failure must be surfaced to the user"


def test_load_spd_params_commits_only_after_a_clean_load(tmp_path):
    path = tmp_path / "spd.json"
    _sensor_grid_calibrator().save_params(str(path))

    gui = _make_gui()
    gui.mapping = LinearMapping(-2.738911, 3198.4744)
    mw.filedialog.askopenfilename = lambda **kwargs: str(path)

    gui.load_spd_params()

    assert gui.calibrator is not None
    assert gui.spd_params_path == str(path)
    assert gui.cache_step2["loaded_from_file"] is True
    assert gui.messages.infos, "a successful load must confirm to the user"


def test_load_spd_params_warns_in_yellow_for_a_moderate_drift(tmp_path):
    """L3, 2-10 nm: a warning, never a rejection."""
    path = tmp_path / "spd.json"
    _sensor_grid_calibrator().save_params(str(path))

    gui = _make_gui()
    gui.mapping = LinearMapping(-2.738911, 3198.4744 + 15.0)    # ~5.5 nm
    mw.filedialog.askopenfilename = lambda **kwargs: str(path)

    gui.load_spd_params()

    assert gui.calibrator is not None, "drift must never reject a preset"
    assert gui.calibrator.drift_level == "WARN"
    assert any("漂移" in title or "drift" in title.lower()
               for title, _ in gui.messages.warnings), gui.messages.warnings


def test_load_spd_params_rolls_back_when_the_user_rejects_a_severe_drift(tmp_path):
    """L3, >10 nm: ask for confirmation and restore the previous calibrator."""
    path = tmp_path / "spd.json"
    good = _sensor_grid_calibrator()
    good.save_params(str(path))

    gui = _make_gui()
    gui.mapping = LinearMapping(-2.738911, 3198.4744 + 45.0)    # ~16 nm
    gui.calibrator = good
    gui.messages.confirm_answer = False
    mw.filedialog.askopenfilename = lambda **kwargs: str(path)

    gui.load_spd_params()

    assert any("严重" in title or "severe" in title.lower()
               for title, _ in gui.messages.warnings), gui.messages.warnings
    assert gui.calibrator is good, "a cancelled load must roll back"


# ---------------------------------------------------------------------------
# step1_calibrate -- everything the WAVELENGTH preset needs
# ---------------------------------------------------------------------------
@_real_data
def test_step1_calibrate_populates_the_preset_fields():
    gui = _make_gui()
    gui.pos_selector = types.SimpleNamespace(get_files=lambda: [str(POSITIONING)])

    gui.step1_calibrate()

    assert gui.mapping is not None
    cache = gui.cache_step1
    assert cache["sensor_size"] == (4096, 3072)      # (width, height)
    assert cache["known_wls"] == {"R": 622.5, "G": 522.5, "B": 462.5}
    assert cache["rb_swapped"] is False
    assert cache["cfa"] == (2, 1, 1, 0)
    assert cache["validation"].wavelength_error_nm == pytest.approx(
        cache["wl_err"], abs=1e-12)
    assert abs(cache["wl_err"]) < 10.0

    # and the preset can actually be written from that cache
    from core.wavelength_preset import load_wavelength_preset, save_wavelength_preset
    target = Path(os.environ.get("TMPDIR", "/tmp")) / "wl_from_gui.json"
    save_wavelength_preset(
        str(target), gui.mapping, validation=cache["validation"],
        peaks=cache["peaks"], x_roi=gui.x_roi, sensor_size=cache["sensor_size"],
        known_wls=cache["known_wls"], rb_swapped=cache["rb_swapped"],
        analysis_mode=gui.analysis_mode)
    preset = load_wavelength_preset(str(target))
    assert preset["rbSwapped"] is False
    assert preset["sensorHeight"] == 3072
    assert abs(preset["gErrorNm"] - cache["wl_err"]) < 1e-9
