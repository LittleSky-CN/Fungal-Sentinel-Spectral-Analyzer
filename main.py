# main.py
"""FSSA v1.4.7 Main Entry Point.
- 分析模式选择 (LoD/IUPAC、Legacy、Turbidity)
- 可选霓虹主题 (gui/theme.py + config.py, 缺失时自动回退经典外观)
- 统一团队图标 (gui/app_icon.py: 多尺寸透明底 .ico; Toplevel 自动继承)
- Windows 高 DPI 感知
"""
import tkinter as tk
from tkinter import ttk

from gui.main_window import FSSA_GUI

# ---- 图标模块安全导入 (缺失时静默降级, 不影响启动) ----
try:
    from gui.app_icon import apply_app_icon
except Exception:
    def apply_app_icon(root):
        return False

# ---- 安全读取主题配置 ----
try:
    from config import THEME_ENABLED
except Exception:
    THEME_ENABLED = False

try:
    from config import PIXEL_FONT
except Exception:
    PIXEL_FONT = {"path": None, "family": None}


def _apply_theme_if_enabled(root):
    """必须在 tk.Tk() 创建之后调用。"""
    if not THEME_ENABLED:
        return
    try:
        from gui.theme import apply_theme
        from gui import theme as _t
        apply_theme(root, PIXEL_FONT.get("path"), PIXEL_FONT.get("family"))
        # 模式选择页专用单选样式：黑底黄字，选中后绿字（保留原交互）
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


# ---- Windows 高 DPI 感知 ----
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass


class ModeSelector:
    def __init__(self, root):
        self.root = root
        apply_app_icon(root)            # 团队图标：初始选择窗口
        _apply_theme_if_enabled(root)

        self.root.title("FSSA v1.4.7 - Analysis Mode Selection")
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
        apply_app_icon(main_root)       # 团队图标：主窗口（Details/浊度等 Toplevel 经 default 自动继承）
        _apply_theme_if_enabled(main_root)

        app = FSSA_GUI(main_root, analysis_mode=selected_mode)
        main_root.mainloop()


if __name__ == "__main__":
    selector_root = tk.Tk()
    selector_root.withdraw()
    app = ModeSelector(selector_root)
    selector_root.deiconify()
    selector_root.mainloop()