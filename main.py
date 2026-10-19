# main.py
"""FSSA v1.5.0 Main Entry Point.

[v1.5.0] Added the Fluorescein (sodium salt) fluorophore option (emission 520 nm / excitation 494 nm / green
channel / FWHM 25); fluorophore identifiers are unified as canonical IDs (canonical_fluorophore() was added for
normalised parsing) so curve presets interoperate with the Android app.
- analysis mode selection (LoD/IUPAC, Legacy, Turbidity)
- optional neon theme (gui/theme.py + config.py, falls back to the classic look when missing)
- unified team icon (gui/app_icon.py: multi-size transparent .ico; Toplevels inherit it automatically)
- Windows high-DPI awareness
"""
import tkinter as tk
from tkinter import ttk

from gui.main_window import FSSA_GUI

# ---- safe import of the icon module (degrades silently when missing, startup is unaffected) ----
try:
    from gui.app_icon import apply_app_icon
except Exception:
    def apply_app_icon(root):
        return False

# ---- safe read of the theme configuration ----
try:
    from config import THEME_ENABLED
except Exception:
    THEME_ENABLED = False

try:
    from config import PIXEL_FONT
except Exception:
    PIXEL_FONT = {"path": None, "family": None}


def _apply_theme_if_enabled(root):
    """Must be called after tk.Tk() has been created."""
    if not THEME_ENABLED:
        return
    try:
        from gui.theme import apply_theme
        from gui import theme as _t
        apply_theme(root, PIXEL_FONT.get("path"), PIXEL_FONT.get("family"))
        # radio style dedicated to the mode selection page: yellow text on black, green when selected (original interaction preserved)
        style = ttk.Style(root)
        style.configure("Sel.TRadiobutton",
                        background=_t.BLACK, foreground=_t.YELLOW,
                        font=_t.H(10), indicatormargin=4)
        style.map("Sel.TRadiobutton",
                  foreground=[("selected", _t.GREEN), ("active", _t.GREEN)],
                  background=[("active", _t.BLACK), ("pressed", _t.BLACK)],
                  indicatorcolor=[("selected", _t.GREEN),
                                  ("!selected", _t.MAGENTA_DARK)])
    except Exception as e:
        print(f"[theme] skipped due to error: {e}")


# ---- Windows high-DPI awareness ----
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass


class ModeSelector:
    def __init__(self, root):
        self.root = root
        apply_app_icon(root)            # team icon: initial selection window
        _apply_theme_if_enabled(root)

        self.root.title("FSSA v1.5.0 - Analysis Mode Selection")
        self.root.geometry("600x650")
        self.root.resizable(True, True)
        self.root.eval('tk::PlaceWindow . center')

        ttk.Label(root, text="Fungal Sentinel Spectral Analyzer",
                  style="Title.TLabel" if THEME_ENABLED else None,
                  **({} if THEME_ENABLED else {"font": ("Arial", 16, "bold")})).pack(pady=(30, 10))
        ttk.Label(root, text="Please select the analysis mode for this session:",
                  font=("Arial", 11)).pack(pady=(0, 20))

        mode_frame = ttk.Frame(root)
        mode_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=15)

        self.mode_var = tk.StringVar(value="lod_iupac")
        rb = {"style": "Sel.TRadiobutton"} if THEME_ENABLED else {}
        ttk.Radiobutton(
            mode_frame,
            text="Fluorescence Quantification (LoD/LoQ)\n- IUPAC Standard Limit Analysis\n- Weighted Regression Support",
            variable=self.mode_var, value="lod_iupac", **rb
        ).pack(anchor=tk.W, pady=10)
        ttk.Radiobutton(
            mode_frame,
            text="Standard Fluorescence (Legacy)\n- Fast OLS linear regression\n- Routine sample testing",
            variable=self.mode_var, value="legacy", **rb
        ).pack(anchor=tk.W, pady=10)
        ttk.Radiobutton(
            mode_frame,
            text="Turbidity Measurement Mode\n- Dedicated UI for scattering analysis\n- Microplate-style spectral reporting",
            variable=self.mode_var, value="turbidity", **rb
        ).pack(anchor=tk.W, pady=10)

        btn_frame = ttk.Frame(root)
        btn_frame.pack(fill=tk.X, padx=30, pady=25)
        if THEME_ENABLED:
            ttk.Button(btn_frame, text="Exit", command=root.destroy, width=12,
                       style="Pale.TButton").pack(side=tk.RIGHT, padx=(15, 0))
            ttk.Button(btn_frame, text="Start Analysis", command=self.launch_main, width=15,
                       style="Magenta.TButton").pack(side=tk.RIGHT)
        else:
            ttk.Button(btn_frame, text="Exit", command=root.destroy, width=12).pack(side=tk.RIGHT, padx=(15, 0))
            ttk.Button(btn_frame, text="Start Analysis", command=self.launch_main, width=15).pack(side=tk.RIGHT)

    def launch_main(self):
        selected_mode = self.mode_var.get()
        self.root.destroy()

        main_root = tk.Tk()
        apply_app_icon(main_root)       # team icon: main window (Details / turbidity Toplevels inherit it via default)
        _apply_theme_if_enabled(main_root)

        app = FSSA_GUI(main_root, analysis_mode=selected_mode)
        main_root.mainloop()


if __name__ == "__main__":
    selector_root = tk.Tk()
    selector_root.withdraw()
    app = ModeSelector(selector_root)
    selector_root.deiconify()
    selector_root.mainloop()