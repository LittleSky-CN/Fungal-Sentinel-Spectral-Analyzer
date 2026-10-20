"""Report provenance: every step must keep its preset source visible.

``gui/details_window.py`` needs tkinter, which headless CI lacks, so this
module reuses the stub technique from ``tests/test_gui_preset_flow.py``
(module-level ``__getattr__`` stubs for tkinter and
``matplotlib.backends.backend_tkagg``) and then really constructs
``DetailsWindow`` -- the widget tree is built against the stubs, so
``_build_tab5`` executes for real.
"""
import importlib.util
import sys
import types


# ---------------------------------------------------------------------------
# Tk stubs (same technique as tests/test_gui_preset_flow.py)
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

import matplotlib
matplotlib.use = lambda *args, **kwargs: None

from matplotlib.figure import Figure                      # noqa: E402

from gui import details_window as dw                      # noqa: E402

dw._THEME_ON = False


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
def _preset(step, preset_type, name="p.json", params=None, path=None):
    return {
        "step": step,
        "kind": "preset",
        "preset_type": preset_type,
        "file": path or f"D:/presets/{name}",
        "file_name": name,
        "preset_name": name,
        "imported_at": "2026-10-17 10:30:00",
        "params": dict(params or {}),
    }


def _live(step, params=None):
    return {
        "step": step,
        "kind": "live",
        "preset_type": None,
        "file": None,
        "file_name": None,
        "preset_name": None,
        "imported_at": "2026-10-17 10:31:00",
        "params": dict(params or {}),
    }


FULL = {
    "step1": _preset(1, "WAVELENGTH", "wl_preset.json",
                     {"mapping_slope": "-2.73891", "g_error_nm": "0.42"}),
    "step2": _preset(2, "SPD", "spd_params.json",
                     {"drift_level": "OK", "drift_nm": "1.25"},
                     path="D:/Documents/iGem/spd_params.json"),
    "step3": _live(3, {"fluor_name": "Ypet", "peak_wl": "527.0"}),
    "step4": _preset(4, "CURVE", "curve_preset.json",
                     {"slope": "12.5", "r_squared": "0.9981"}),
}


def _window(provenance):
    return dw.DetailsWindow(Stub(), {}, {}, {}, {}, None, preset_provenance=provenance)


# ---------------------------------------------------------------------------
# backwards compatibility
# ---------------------------------------------------------------------------
def test_six_positional_args_still_constructs():
    """Old call sites must not break -- preset_provenance defaults to None."""
    win = dw.DetailsWindow(Stub(), {}, {}, {}, {}, None)
    assert win.preset_provenance == {}
    assert win.fig5 is not None


# ---------------------------------------------------------------------------
# provenance is stored and rendered
# ---------------------------------------------------------------------------
def test_full_provenance_is_stored_verbatim():
    win = _window(FULL)
    assert win.preset_provenance is FULL
    assert win.preset_provenance["step2"]["file_name"] == "spd_params.json"


def test_full_provenance_renders_every_step():
    win = _window(FULL)
    text = win._provenance_text()

    assert "Imported preset (WAVELENGTH)" in text
    assert "Imported preset (SPD)" in text
    assert "Imported preset (CURVE)" in text
    assert "Live calibration / fit" in text
    for frag in ("wl_preset.json", "spd_params.json", "curve_preset.json",
                 "D:/Documents/iGem/spd_params.json", "2026-10-17 10:30:00"):
        assert frag in text
    for frag in ("mapping_slope: -2.73891", "g_error_nm: 0.42", "drift_level: OK",
                 "fluor_name: Ypet", "slope: 12.5", "r_squared: 0.9981"):
        assert frag in text
    assert "Not completed" not in text


def test_build_tab5_is_idempotent_and_produces_a_figure():
    """_build_tab5 must run without raising and leave a real Figure behind."""
    win = _window(FULL)
    win._build_tab5()
    assert isinstance(win.fig5, Figure)
    assert win.fig5.get_size_inches()[1] >= 6.0


# ---------------------------------------------------------------------------
# None safety
# ---------------------------------------------------------------------------
def test_missing_step_is_reported_not_skipped():
    provenance = {"step1": _preset(1, "WAVELENGTH", "wl.json", {"a": "1"}),
                  "step2": None,
                  "step3": _live(3, {"b": "2"}),
                  "step4": None}
    win = _window(provenance)
    text = win._provenance_text()

    assert "Not completed / not imported" in text
    assert text.count("Not completed / not imported") == 2
    assert "Step 2" in text and "Step 4" in text
    assert "wl.json" in text
    win._build_tab5()          # must not raise


def test_empty_provenance_still_builds_the_tab():
    win = _window({})
    text = win._provenance_text()
    assert text.count("Not completed / not imported") == 4
    for step in (1, 2, 3, 4):
        assert f"[Step {step}:" in text
    win._build_tab5()
    assert win.fig5 is not None


def test_malformed_entry_does_not_crash():
    win = _window({"step1": "garbage", "step2": {"kind": "unknown-kind"}})
    text = win._provenance_text()
    assert "Not completed / not imported" in text
    assert "unknown-kind" in text


# ---------------------------------------------------------------------------
# the provenance page reaches the PDF export
# ---------------------------------------------------------------------------
def test_provenance_figure_is_exported_to_pdf(tmp_path):
    win = _window(FULL)
    target = tmp_path / "report.pdf"
    seen = {}

    dw.filedialog = types.SimpleNamespace(
        asksaveasfilename=lambda **kwargs: str(target))
    dw.messagebox = types.SimpleNamespace(
        showinfo=lambda *a, **k: seen.setdefault("info", a),
        showerror=lambda *a, **k: seen.setdefault("error", a))

    win.export_pdf_report([win.fig5])

    assert target.is_file(), seen
    assert "error" not in seen
    assert "info" in seen, "a successful write must confirm to the user"
    assert target.stat().st_size > 0

    with open(target, "rb") as fh:
        head = fh.read(5)
    assert head == b"%PDF-", "the export must be a real PDF document"
