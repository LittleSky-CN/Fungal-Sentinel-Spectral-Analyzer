# gui/widgets.py
"""Reusable GUI components for FSSA (theme-aware, fallback-safe)."""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import List

# ---- 可选主题依赖（缺失时自动回退） ----
try:
    from gui import theme as _theme
except Exception:
    _theme = None
try:
    from config import THEME_ENABLED as _THEME_FLAG
except Exception:
    _THEME_FLAG = False

_THEME_ON = bool(_theme is not None and _THEME_FLAG)

_BTN_STYLES = {
    "magenta": "Magenta.TButton",
    "green":   "Green.TButton",
    "yellow":  "Yellow.TButton",
    "pale":    "Pale.TButton",
}

def themed_button(master, text, kind="pale", **kw):
    """创建按钮；主题开启时使用霓虹风格，否则保持原生 ttk 外观。"""
    if _THEME_ON:
        kw.setdefault("style", _BTN_STYLES.get(kind, "Pale.TButton"))
    return ttk.Button(master, text=text, **kw)

def themed_label(master, text, surface="dark", **kw):
    """创建标签；surface='card' 用于奶油色卡片内部（黑字），'dark' 用于黑底（亮字）。"""
    if _THEME_ON:
        kw.setdefault("style", "Card.TLabel" if surface == "card" else "TLabel")
        return ttk.Label(master, text=text, **kw)
    if surface == "card":
        kw.setdefault("foreground", "#000000")
    kw.setdefault("font", ("Arial", 9, "bold"))
    return ttk.Label(master, text=text, **kw)

class MultiFileSelector(ttk.Frame):
    """多 DNG 文件选择组件（1-5 张），支持 dark / card 两种表面配色。"""
    def __init__(self, master, title, max_files=5, surface="dark", **kwargs):
        if _THEME_ON and surface == "card":
            kwargs.setdefault("style", "Card.TFrame")      # 卡片内容器用奶油底
        super().__init__(master, **kwargs)
        self.max_files = max_files
        self.files = []

        themed_label(self, title, surface=surface).pack(anchor=tk.W)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, pady=2)
        themed_button(btn_frame, f"Add (Max {max_files})", kind="yellow",
                      width=14, command=self.add_files).pack(side=tk.LEFT)
        themed_button(btn_frame, "Clear", kind="pale",
                      width=6, command=self.clear_files).pack(side=tk.LEFT, padx=5)

        if _THEME_ON:
            self.listbox = tk.Listbox(
                self, height=3, font=("Consolas", 8), selectmode=tk.EXTENDED,
                bg=_theme.CREAM, fg=_theme.INK,
                selectbackground=_theme.MAGENTA, selectforeground=_theme.INK,
                highlightthickness=1, highlightbackground=_theme.CYAN, borderwidth=0)
        else:
            self.listbox = tk.Listbox(self, height=3, font=("Consolas", 8),
                                      selectmode=tk.EXTENDED)
        self.listbox.pack(fill=tk.X, expand=True)

        scrollbar = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.listbox.xview)
        self.listbox.config(xscrollcommand=scrollbar.set)
        scrollbar.pack(fill=tk.X)

    def add_files(self):
        paths = filedialog.askopenfilenames(title="Select DNG Files",
                                            filetypes=[("DNG", "*.dng")])
        if not paths:
            return
        remaining = self.max_files - len(self.files)
        if remaining <= 0:
            messagebox.showinfo("Limit Reached", f"Maximum {self.max_files} files allowed.")
            return
        for p in paths[:remaining]:
            self.files.append(p)
            self.listbox.insert(tk.END, Path(p).name)

    def clear_files(self):
        self.files = []
        self.listbox.delete(0, tk.END)

    def get_files(self) -> List[str]:
        return self.files