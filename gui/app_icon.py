# gui/app_icon.py
"""Unified application icon helper (Windows .ico + fallback PNG)."""
import sys
import tkinter as tk
from pathlib import Path

def _assets_dir() -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / "assets"

def apply_app_icon(root: tk.Tk | tk.Toplevel) -> bool:
    ico = _assets_dir() / "fssa.ico"
    png = _assets_dir() / "logo_256.png"

    # Windows: make the taskbar icon follow this program (otherwise the default Tk feather is shown / grouping breaks)
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "iGEM.CAU-China.FungalSentinel.1.6.0")
    except Exception:
        pass

    applied = False
    if ico.exists():
        try:
            root.iconbitmap(str(ico))            # the current window
            root.iconbitmap(default=str(ico))    # [critical] every later Toplevel inherits it by default
            applied = True
        except tk.TclError:
            pass
    if not applied and png.exists():
        try:  # Linux/macOS fallback: iconphoto likewise propagates to child windows with True
            img = tk.PhotoImage(file=str(png))
            root._fssa_icon_ref = img            # guard against garbage collection
            root.iconphoto(True, img)
            applied = True
        except tk.TclError:
            pass
    return applied