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

    # Windows：让任务栏图标跟随本程序（否则显示默认 Tk 羽毛/分组错乱）
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "iGEM.CAU-China.FungalSentinel.1.4.7")
    except Exception:
        pass

    applied = False
    if ico.exists():
        try:
            root.iconbitmap(str(ico))            # 当前窗口
            root.iconbitmap(default=str(ico))    # 【关键】之后所有 Toplevel 默认继承
            applied = True
        except tk.TclError:
            pass
    if not applied and png.exists():
        try:  # Linux/mac 回退：iconphoto 同样用 True 传播到子窗口
            img = tk.PhotoImage(file=str(png))
            root._fssa_icon_ref = img            # 防垃圾回收
            root.iconphoto(True, img)
            applied = True
        except tk.TclError:
            pass
    return applied