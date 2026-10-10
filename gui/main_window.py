# gui/main_window.py
"""Main GUI Window for FSSA v1.4.4 (Spectral Unmixing LoD + Weighted Fusion + Turbidity).
Theme-aware (gui/theme.py optional): neon-pop look when enabled, classic look otherwise.
"""
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import linregress

from config import FLUOROPHORES
from core.image_processing import (
    get_exposure_time, read_dng_linear_rgb, auto_detect_x_roi, get_average_profiles,
    parse_known_wavelengths, find_peak_auto, make_channel_profiles,
    load_true_spd_data, np_trapz
)
from core.calibration import LinearMapping, TwoPointValidation, SpectralResponseCalibrator
from core.quantification import calculate_concentration
from core.unmixing import (unmix_spectrum, build_reference_emission, emission_auc,
                           default_unmix_window, slope_t_statistic)
from gui.widgets import MultiFileSelector
from gui.details_window import DetailsWindow

# ---- 主题安全配置 ----
try:
    from config import THEME_ENABLED
except Exception:
    THEME_ENABLED = False
try:
    from config import PIXEL_FONT
except Exception:
    PIXEL_FONT = {"path": None, "family": None}
try:
    from gui import theme as _theme
except Exception:
    _theme = None
_THEME_ON = bool(THEME_ENABLED and _theme is not None)


class FSSA_GUI:
    def __init__(self, root, analysis_mode="lod_iupac"):
        self.root = root
        self.analysis_mode = analysis_mode
        self.is_turbidity_mode = (analysis_mode == "turbidity")

        title_suffix = "TURBIDITY" if self.is_turbidity_mode else analysis_mode.upper()
        self.root.title(f"Fungal Sentinel Spectral Analyzer (FSSA) v1.4.4 [{title_suffix}] - iGEM 2026")
        self.root.geometry("1600x950")
        self.root.minsize(1400, 900)

        self.txt_file = None
        self.csv_file = None
        self.mapping = None
        self.x_roi = None
        self.calibrator = None
        self.current_fluorophore = "Ypet (Venus variant)"
        self.current_page = 1
        self.current_chart_filename = "FSSA_Chart"

        # 【修复】这里只初始化空字典，不要在这里计算 SPD 数据！
        self.cache_step1 = {}
        self.cache_step2 = {}
        self.cache_step3 = {}
        self.cache_step4 = {}
        
        self.turbidity_window = None
        self._lod_headline = None

        # 【主题】仅在此处应用全局样式（必须在 setup_ui 之前、root 存在之后）
        if _THEME_ON:
            _theme.apply_theme(self.root, PIXEL_FONT.get("path"), PIXEL_FONT.get("family"))
            self._extend_theme_styles()

        self.setup_ui()

    # ================= 主题辅助 =================
    def _extend_theme_styles(self):
        st = ttk.Style(self.root)
        st.configure("Head.TLabel", background=_theme.BLACK, foreground=_theme.CYAN, font=_theme.H(12))
        st.configure("Sub.TLabel", background=_theme.BLACK, foreground=_theme.CREAM, font=_theme.H(9))
        st.configure("CardSub.TLabel", background=_theme.CREAM, foreground=_theme.INK, font=_theme.H(9))

    def _btn(self, master, text, kind="pale", **kw):
        if _THEME_ON:
            kw.setdefault("style", {"magenta": "Magenta.TButton", "green": "Green.TButton",
                                    "yellow": "Yellow.TButton", "pale": "Pale.TButton"}[kind])
        return ttk.Button(master, text=text, **kw)

    def _head_label(self, master, text, fallback_fg):
        if _THEME_ON:
            return ttk.Label(master, text=text, style="Head.TLabel")
        return ttk.Label(master, text=text, font=("Arial", 12, "bold"), foreground=fallback_fg)

    def _sub_label(self, master, text, **kw):
        if _THEME_ON:
            kw.setdefault("style", "Sub.TLabel")
            return ttk.Label(master, text=text, **kw)
        kw.setdefault("font", ("Arial", 9, "bold"))
        return ttk.Label(master, text=text, **kw)

    def _card_label(self, master, text, **kw):
        if _THEME_ON:
            kw.setdefault("style", "CardSub.TLabel")
        return ttk.Label(master, text=text, **kw)

    def _status_label(self, master):
        if _THEME_ON:
            return ttk.Label(master, text="Status: Ready", style="Status.TLabel", wraplength=300)
        return ttk.Label(master, text="Status: Ready", foreground="gray", wraplength=300)

    def _radio(self, master, **kw):
        if _THEME_ON:
            kw.setdefault("style", "Card.TRadiobutton")
        return ttk.Radiobutton(master, **kw)

    def _dyn_fg(self, name):
        if not _THEME_ON:
            return {"black": "black", "gray": "gray"}.get(name, name)
        return {"black": _theme.GREEN, "gray": "#8f9a8f"}.get(name, name)

    def _style_fig(self, fig):
        if _THEME_ON:
            _theme.style_figure(fig)
        return fig

    def _c4_colors(self):
        if _THEME_ON:
            m = _theme.MPL
            return {"std": m["std"], "std_err": "#7FF7FF", "fit": m["fit"], "sample": m["sample"],
                    "sample_edge": _theme.BLACK, "lod": m["lod"], "loq": m["loq"]}
        return {"std": "blue", "std_err": "lightblue", "fit": "red", "sample": "red",
                "sample_edge": "darkred", "lod": "orange", "loq": "purple"}

    def _spec_line_color(self):
        return _theme.CREAM if _THEME_ON else "k"

    def _centroid_color(self):
        return _theme.CREAM if _THEME_ON else "black"

    # ================= UI Setup =================
    def setup_ui(self):
        left_frame = ttk.Frame(self.root, width=480)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        left_frame.pack_propagate(False)

        # 【主题】横幅 + 版本号：必须在 left_frame 创建之后
        if _THEME_ON:
            self.banner = _theme.StepBanner(left_frame)
            self.banner.pack(fill=tk.X, pady=(10, 0))
            ttk.Label(left_frame, text=f"FSSA v1.4.4  [{self.analysis_mode.upper()}]",
                      style="Title.TLabel").pack(pady=(6, 0))
        else:
            ttk.Label(left_frame, text=f"FSSA v1.4.4\n({self.analysis_mode.upper()})",
                      font=("Arial", 14, "bold")).pack(pady=15)

        self.content_frame = ttk.Frame(left_frame)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.page1 = ttk.Frame(self.content_frame)
        self.page2 = ttk.Frame(self.content_frame)
        self.page3 = ttk.Frame(self.content_frame)
        self.page4 = ttk.Frame(self.content_frame)

        self._setup_page1()
        self._setup_page2()
        self._setup_page3()
        if not self.is_turbidity_mode:
            self._setup_page4()

        self.max_steps = 3 if self.is_turbidity_mode else 4

        nav_frame = ttk.Frame(left_frame)
        nav_frame.pack(fill=tk.X, padx=10, pady=15)
        self.btn_prev = self._btn(nav_frame, "< Prev", kind="pale", command=self.prev_page)
        self.btn_prev.pack(side=tk.LEFT)
        if _THEME_ON:
            self.lbl_page_num = ttk.Label(nav_frame, text=f"Step 1 / {self.max_steps}", style="Sub.TLabel")
        else:
            self.lbl_page_num = ttk.Label(nav_frame, text=f"Step 1 / {self.max_steps}", font=("Arial", 11, "bold"))
        self.lbl_page_num.pack(side=tk.LEFT, expand=True)
        self.btn_next = self._btn(nav_frame, "Next >", kind="magenta", command=self.next_page)
        self.btn_next.pack(side=tk.RIGHT)

        self.show_page(1)

        right_frame = ttk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        toolbar = ttk.Frame(right_frame)
        toolbar.pack(fill=tk.X, padx=10, pady=5)
        if _THEME_ON:
            ttk.Label(toolbar, text="Real-time Visualization", style="Head.TLabel").pack(side=tk.LEFT)
        else:
            ttk.Label(toolbar, text="Real-time Visualization", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        if not self.is_turbidity_mode:
            self._btn(toolbar, "💧 Turbidity", kind="green", command=self.open_turbidity_window).pack(side=tk.RIGHT, padx=2)
        self._btn(toolbar, "📋 Details", kind="green", command=self.show_details_window).pack(side=tk.RIGHT, padx=2)
        self._btn(toolbar, " Export Chart", kind="green", command=self.export_chart).pack(side=tk.RIGHT)

        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Welcome to FSSA v1.4.4\nPlease proceed with Step 1: Positioning.")
        self.ax.axis('off')
        self._style_fig(self.fig)
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10)

        res_frame = ttk.LabelFrame(right_frame, text="Analysis Results & Logs", padding=5)
        res_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.txt_result = scrolledtext.ScrolledText(res_frame, height=8, font=("Consolas", 10))
        self.txt_result.pack(fill=tk.X)
        # 【主题】日志终端化：必须在 txt_result 创建之后
        if _THEME_ON:
            self.txt_result.configure(bg=_theme.BLACK, fg=_theme.CONSOLE, insertbackground=_theme.CONSOLE,
                                      selectbackground=_theme.MAGENTA, selectforeground=_theme.INK,
                                      relief="flat", highlightthickness=1,
                                      highlightbackground=_theme.CYAN)

    def _setup_page1(self):
        f = self.page1
        self._head_label(f, "Step 1: Wavelength Calibration", "blue").pack(anchor=tk.W, pady=(0, 10))
        self.pos_selector = MultiFileSelector(f, "Positioning DNG (1-5 images)", max_files=5)
        self.pos_selector.pack(fill=tk.X, pady=5)
        self._sub_label(f, "Wavelength Notes TXT (Optional):").pack(anchor=tk.W, pady=(10, 0))
        txt_frame = ttk.Frame(f); txt_frame.pack(fill=tk.X)
        self._btn(txt_frame, "Select TXT", kind="yellow", command=self.select_txt).pack(side=tk.LEFT)
        self.lbl_txt = self._sub_label(txt_frame, "Default"); self.lbl_txt.pack(side=tk.LEFT, padx=5)
        self.lbl_txt.config(foreground=self._dyn_fg("gray"))
        self._btn(f, "Run Wavelength Calibration", kind="magenta", command=self.step1_calibrate).pack(fill=tk.X, pady=15)
        self.lbl_status1 = self._status_label(f); self.lbl_status1.pack(anchor=tk.W)

    def _setup_page2(self):
        f = self.page2
        self._head_label(f, "Step 2: Spectral Response Cal. (SPD)", "green").pack(anchor=tk.W, pady=(0, 10))
        self.spd_selector = MultiFileSelector(f, "SPD Calibration DNG (1-5 images)", max_files=5)
        self.spd_selector.pack(fill=tk.X, pady=5)
        self._sub_label(f, "True SPD CSV (Optional):").pack(anchor=tk.W, pady=(10, 0))
        csv_frame = ttk.Frame(f); csv_frame.pack(fill=tk.X)
        self._btn(csv_frame, "Select CSV", kind="yellow", command=self.select_csv).pack(side=tk.LEFT)
        self.lbl_csv = self._sub_label(csv_frame, "None (Skip SPD Cal.)"); self.lbl_csv.pack(side=tk.LEFT, padx=5)
        self.lbl_csv.config(foreground=self._dyn_fg("gray"))

        btn_frame_step2 = ttk.Frame(f)
        btn_frame_step2.pack(fill=tk.X, pady=15)
        self._btn(btn_frame_step2, "Run Spectral Response Cal.", kind="magenta",
                  command=self.step2_calibrate).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        io_frame = ttk.Frame(f)
        io_frame.pack(fill=tk.X, pady=5)
        self._btn(io_frame, "💾 Export SPD Params", kind="yellow", command=self.export_spd_params).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self._btn(io_frame, "📂 Load SPD Params", kind="yellow", command=self.load_spd_params).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        self.lbl_status2 = self._status_label(f); self.lbl_status2.pack(anchor=tk.W)

    def _setup_page3(self):
        f = self.page3
        if self.is_turbidity_mode:
            self._head_label(f, "Step 3: Turbidity & Spectral Analysis", "teal").pack(anchor=tk.W, pady=(0, 10))
            ttk.Separator(f, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        else:
            self._head_label(f, "Step 3: Sample Analysis", "purple").pack(anchor=tk.W, pady=(0, 10))
            self._sub_label(f, "Select Fluorophore:").pack(anchor=tk.W)
            self.fluor_combo = ttk.Combobox(f, values=list(FLUOROPHORES.keys()), state="readonly", width=25)
            self.fluor_combo.set(self.current_fluorophore); self.fluor_combo.pack(pady=2, anchor=tk.W)
            self.fluor_combo.bind("<<ComboboxSelected>>", self.on_fluorophore_change)
            ttk.Separator(f, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        self.blank_selector = MultiFileSelector(f, "Blank / Background (1-5 images)", max_files=5)
        self.blank_selector.pack(fill=tk.X, pady=5)
        self.sample_selector = MultiFileSelector(f, "Sample DNG (1-5 images)", max_files=5)
        self.sample_selector.pack(fill=tk.X, pady=5)

        btn_text = "Analyze Turbidity & Spectrum" if self.is_turbidity_mode else "Extract Sample Spectrum"
        self._btn(f, btn_text, kind="magenta", command=self.step3_analyze).pack(fill=tk.X, pady=15)
        self.lbl_status3 = self._status_label(f); self.lbl_status3.pack(anchor=tk.W)

    def _setup_page4(self):
        f = self.page4
        self._head_label(f, "Step 4: Concentration Quantification", "red").pack(anchor=tk.W, pady=(0, 10))

        canvas_frame = ttk.Frame(f); canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.std_canvas = tk.Canvas(canvas_frame, highlightthickness=0)
        if _THEME_ON:
            self.std_canvas.configure(bg=_theme.BLACK)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.std_canvas.yview)
        self.std_rows_frame = ttk.Frame(self.std_canvas)
        self.std_rows_frame.bind("<Configure>", lambda e: self.std_canvas.configure(scrollregion=self.std_canvas.bbox("all")))
        self.std_canvas.create_window((0, 0), window=self.std_rows_frame, anchor="nw")
        self.std_canvas.configure(yscrollcommand=scrollbar.set)
        self.std_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_mousewheel(event):
            # 仅当鼠标位于主窗口内才滚动标准品列表（避免与子窗口冲突）
            if event.widget.winfo_toplevel() is not self.root:
                return
            if event.num == 4: self.std_canvas.yview_scroll(-1, "units")
            elif event.num == 5: self.std_canvas.yview_scroll(1, "units")
            else: self.std_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.std_rows = []

        metric_frame = ttk.LabelFrame(f, text="Quantification Metric", padding=5)
        metric_frame.pack(fill=tk.X, pady=5)
        self.metric_mode = tk.StringVar(value="unmixing")
        self._radio(metric_frame, text="Spectral Unmixing: sample = α·E + s·Blank + k  [Recommended, ~10x lower LoD]",
                    variable=self.metric_mode, value="unmixing").pack(anchor=tk.W)
        self._radio(metric_frame, text="Plain blank-subtracted AUC (Legacy)",
                    variable=self.metric_mode, value="legacy").pack(anchor=tk.W)

        reg_frame = ttk.LabelFrame(f, text="Regression Method", padding=5)
        reg_frame.pack(fill=tk.X, pady=5)
        self.regression_method = tk.StringVar(value="ols")
        self._radio(reg_frame, text="Ordinary Least Squares (OLS)", variable=self.regression_method, value="ols").pack(anchor=tk.W)
        self._radio(reg_frame, text="Weighted (1/y) - For count data", variable=self.regression_method, value="weighted_1y").pack(anchor=tk.W)
        self._radio(reg_frame, text="Weighted (1/y²) - For fluorescence (Recommended)", variable=self.regression_method, value="weighted_1y2").pack(anchor=tk.W)

        btn_frame = ttk.Frame(f); btn_frame.pack(fill=tk.X, pady=5)
        self.btn_add_std = self._btn(btn_frame, "+ Add Std", kind="yellow", command=self.add_std_row)
        self.btn_add_std.pack(side=tk.LEFT, padx=2)
        self.btn_calc_conc = self._btn(btn_frame, "Build Curve & Calculate", kind="magenta", command=self.calculate_concentration)
        self.btn_calc_conc.pack(side=tk.LEFT, padx=2)
        self.add_std_row(); self.add_std_row()
        self.lbl_status4 = self._status_label(f); self.lbl_status4.pack(anchor=tk.W, pady=5)

    # ================= Navigation & Helpers =================
    def show_page(self, page_num):
        self.current_page = page_num
        for p in [self.page1, self.page2, self.page3, self.page4]: p.pack_forget()
        getattr(self, f"page{page_num}").pack(fill=tk.BOTH, expand=True)
        self.lbl_page_num.config(text=f"Step {page_num} / {self.max_steps}")
        self.btn_prev.config(state="normal" if page_num > 1 else "disabled")
        self.btn_next.config(state="normal" if page_num < self.max_steps else "disabled")
        # 【主题】横幅步骤同步：必须在 banner 存在后调用
        if _THEME_ON and hasattr(self, "banner"):
            self.banner.set_text(f"STEP {page_num}/{self.max_steps}")

    def prev_page(self): self.show_page(self.current_page - 1)
    def next_page(self): self.show_page(self.current_page + 1)

    def select_txt(self):
        path = filedialog.askopenfilename(filetypes=[("Text", "*.txt")])
        if path:
            self.txt_file = path
            self.lbl_txt.config(text=Path(path).name, foreground=self._dyn_fg("black"))

    def select_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if path:
            self.csv_file = path
            self.lbl_csv.config(text=Path(path).name, foreground=self._dyn_fg("black"))

    def on_fluorophore_change(self, event): self.current_fluorophore = self.fluor_combo.get()

    def log_status(self, step, msg, color="black"):
        if _THEME_ON:
            color = {"black": _theme.CREAM, "gray": "#8f9a8f", "orange": _theme.YELLOW,
                     "green": _theme.GREEN, "red": "#FF5252"}.get(color, color)
        getattr(self, f"lbl_status{step}").config(text=f"Status: {msg}", foreground=color)
        self.root.update()

    def set_chart_title(self, title):
        self.ax.set_title(title, fontsize=13, fontweight='bold')
        self.current_chart_filename = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')

    def export_chart(self):
        file_path = filedialog.asksaveasfilename(title="Export Chart", initialfile=f"{self.current_chart_filename}.png",
                                                   defaultextension=".png", filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if file_path:
            try:
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Saved to:\n{file_path}")
            except Exception as e: messagebox.showerror("Error", str(e))

    # ================= SPD Params Persistence =================
    def export_spd_params(self):
        if not self.calibrator or self.calibrator.response_R is None:
            messagebox.showwarning("Warning", "Please run Spectral Response Calibration first!")
            return
        file_path = filedialog.asksaveasfilename(title="Export SPD Parameters", initialfile="spd_calibration_params.json",
                                                   defaultextension=".json", filetypes=[("JSON Params", "*.json")])
        if file_path:
            try:
                self.calibrator.save_params(file_path)
                messagebox.showinfo("Success", f"SPD parameters exported to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def load_spd_params(self):
        file_path = filedialog.askopenfilename(title="Load SPD Parameters", filetypes=[("JSON Params", "*.json")])
        if not file_path: return
        try:
            self.calibrator = SpectralResponseCalibrator.load_params(file_path, self.mapping)
            self.cache_step2 = {
                "wavelengths": self.calibrator.wavelengths,
                "R": self.calibrator.response_R, "G": self.calibrator.response_G, "B": self.calibrator.response_B,
                "spd_threshold_ratio": self.calibrator.spd_threshold_ratio,
                "smooth_window": self.calibrator.smooth_window,
                "loaded_from_file": True,
                # 【补充】占位键：让 details 窗口不报错，但 _build_tab2 的 else 分支会显示提示
                "measured_total": None,
                "corrected_total": None,
            }
            self.log_status(2, f"Loaded from {Path(file_path).name}", "green")
            messagebox.showinfo("Success", "SPD parameters loaded successfully!\nNote: fusion/comparison charts need a fresh calibration.")

            wls_all = self.calibrator.wavelengths
            r_all, g_all, b_all = self.calibrator.response_R, self.calibrator.response_G, self.calibrator.response_B
            vmask = (r_all > 0) | (g_all > 0) | (b_all > 0)
            if np.any(vmask):
                wp, rp, gp, bp = wls_all[vmask], r_all[vmask], g_all[vmask], b_all[vmask]
            else:
                wp, rp, gp, bp = wls_all, r_all, g_all, b_all
            self.ax.clear()
            self.ax.plot(wp, rp, 'r-', label='R (Absolute, Loaded)')
            self.ax.plot(wp, gp, 'g-', label='G (Absolute, Loaded)')
            self.ax.plot(wp, bp, 'b-', label='B (Absolute, Loaded)')
            self.set_chart_title("Step 2: Per-Channel Absolute Response (Loaded)")
            self.ax.set_xlabel("Wavelength (nm)"); self.ax.set_ylabel("Sensitivity (a.u. / SPD unit)")
            margin = (wp[-1] - wp[0]) * 0.05
            self.ax.set_xlim(wp[0] - margin, wp[-1] + margin)
            self.ax.legend(); self._style_fig(self.fig); self.fig.tight_layout(); self.canvas_plot.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load SPD params: {str(e)}")

    # ================= Core Signal Extraction =================
    def _resp_interp_guarded(self, wavelengths, ch_name):
        resp_map = {"red": self.calibrator.response_R, "green": self.calibrator.response_G, "blue": self.calibrator.response_B}
        resp_ch = resp_map[ch_name]
        ch_max = float(np.max(resp_ch)) if resp_ch is not None and len(resp_ch) else 0.0
        floor = 0.05 * ch_max if ch_max > 0 else 1e-9
        resp_interp = np.interp(wavelengths, self.calibrator.wavelengths, resp_ch)
        return np.where(resp_interp > floor, resp_interp, floor)

    def calculate_net_signal_single(self, avg_blank_rgb, sample_path, fluor_config):
        ch_name = fluor_config['channel']
        ch_idx = {"red": 0, "green": 1, "blue": 2}[ch_name]
        rgb_s, _ = read_dng_linear_rgb(sample_path)
        prof_s = make_channel_profiles(rgb_s, self.x_roi)
        sample_rgb = np.column_stack([prof_s[c] for c in ["red", "green", "blue"]])
        net_rgb = np.clip(sample_rgb - avg_blank_rgb, 0, None)
        y_pixels = np.arange(len(net_rgb))
        wavelengths = self.mapping.pixel_to_wavelength(y_pixels)
        if wavelengths[0] > wavelengths[-1]:
            wavelengths = wavelengths[::-1]
            net_rgb = net_rgb[::-1, :]
        if self.calibrator:
            true_intensity = net_rgb[:, ch_idx] / self._resp_interp_guarded(wavelengths, ch_name)
        else:
            true_intensity = net_rgb[:, ch_idx]
        target_wl, fwhm = fluor_config['peak_wl'], fluor_config['fwhm']
        mask = (wavelengths >= target_wl - fwhm / 2) & (wavelengths <= target_wl + fwhm / 2)
        area = np_trapz(true_intensity[mask], wavelengths[mask]) if np.any(mask) else 0
        return area, wavelengths, true_intensity, mask

    def get_net_areas_batch(self, blank_paths, sample_paths, fluor_config):
        avg_blank_rgb = np.column_stack([get_average_profiles(blank_paths, self.x_roi)[c] for c in ["red", "green", "blue"]])
        areas = []
        last_wls, last_int, last_mask = None, None, None
        for s_path in sample_paths:
            area, wls, intensity, mask = self.calculate_net_signal_single(avg_blank_rgb, s_path, fluor_config)
            areas.append(area)
            last_wls, last_int, last_mask = wls, intensity, mask
        return areas, last_wls, last_int, last_mask

    def get_raw_channel_spectra(self, paths, fluor_config):
        """Per-shot SPD-corrected channel spectra WITHOUT blank subtraction (for unmixing)."""
        ch_name = fluor_config['channel']
        ch_idx = {"red": 0, "green": 1, "blue": 2}[ch_name]
        specs, wls_out = [], None
        for p in paths:
            rgb_s, _ = read_dng_linear_rgb(p)
            prof = make_channel_profiles(rgb_s, self.x_roi)
            spec_rgb = np.column_stack([prof[c] for c in ["red", "green", "blue"]])
            y_pixels = np.arange(len(spec_rgb))
            wavelengths = self.mapping.pixel_to_wavelength(y_pixels)
            if wavelengths[0] > wavelengths[-1]:
                wavelengths = wavelengths[::-1]; spec_rgb = spec_rgb[::-1, :]
            if self.calibrator:
                intensity = spec_rgb[:, ch_idx] / self._resp_interp_guarded(wavelengths, ch_name)
            else:
                intensity = spec_rgb[:, ch_idx]
            specs.append(intensity); wls_out = wavelengths
        return wls_out, specs

    # ================= Steps =================
    def step1_calibrate(self):
        pos_files = self.pos_selector.get_files()
        if not pos_files: messagebox.showwarning("Warning", "Please add Positioning DNG(s)!"); return
        self.log_status(1, "Calibrating...", "orange")
        try:
            knowns = parse_known_wavelengths(self.txt_file)
            rgb_first, exp_first = read_dng_linear_rgb(pos_files[0])
            self.x_roi = auto_detect_x_roi(rgb_first)
            avg_profiles = get_average_profiles(pos_files, self.x_roi)
            peaks = [find_peak_auto(avg_profiles[knowns[k].channel_name], knowns[k]) for k in ["R", "G", "B"]]
            p1 = next(p for p in peaks if p.name == "B")
            p2 = next(p for p in peaks if p.name == "R")
            hold = next(p for p in peaks if p.name == "G")
            k = (p2.y_centroid - p1.y_centroid) / (p2.wavelength_nm - p1.wavelength_nm)
            b = p1.y_centroid - k * p1.wavelength_nm
            self.mapping = LinearMapping(k, b)
            pred_y = k * hold.wavelength_nm + b
            res_px = hold.y_centroid - pred_y
            wl_err = res_px / abs(k)
            validation = TwoPointValidation(("B", "R"), "G", k, b, pred_y, hold.y_centroid, res_px, wl_err)
            self.log_status(1, f"Done! Exp: {exp_first:.4f}s | G-err: {wl_err:+.2f} nm", "green")
            log_text = f"\n=== Step 1: Wavelength Calibration ===\nExposure Time Normalized: {exp_first:.4f}s\nMapping: p = {k:.4f} * λ + {b:.2f}\nG-point Error: {wl_err:+.2f} nm\n"
            self.txt_result.insert(tk.END, log_text); self.txt_result.see(tk.END)
            self.cache_step1 = {"rgb": rgb_first, "x_roi": self.x_roi, "profiles": avg_profiles, "peaks": peaks,
                                "mapping": self.mapping, "validation": validation, "exp_time": exp_first,
                                "wl_err": wl_err, "n_pos": len(pos_files)}
            self.ax.clear()
            colors = {"red": "red", "green": "green", "blue": "blue"}
            cc = self._centroid_color()
            for name, prof in avg_profiles.items(): self.ax.plot(prof, color=colors[name], label=f"{name.capitalize()} (Avg)", linewidth=1.5)
            for p in peaks: self.ax.axvline(p.y_centroid, linestyle="--", color=cc, label=f"{p.name}")
            self.set_chart_title("Step 1: Wavelength Calibration Profiles")
            self.ax.set_xlabel("Y Pixel"); self.ax.set_ylabel("Intensity"); self.ax.legend(fontsize=8)
            self._style_fig(self.fig); self.fig.tight_layout(); self.canvas_plot.draw()
        except Exception as e: self.log_status(1, f"Error: {e}", "red"); messagebox.showerror("Error", str(e))

    def step2_calibrate(self):
        if not self.mapping: messagebox.showwarning("Warning", "Please complete Step 1 first!"); return
        spd_files = self.spd_selector.get_files()
        if not spd_files: messagebox.showwarning("Warning", "Please add SPD DNG(s)!"); return
        self.log_status(2, "Calibrating Response...", "orange")
        try:
            avg_spd_rgb_dict = get_average_profiles(spd_files, self.x_roi)
            y_pixels = np.arange(len(avg_spd_rgb_dict["red"]), dtype=np.float64)
            measured_rgb = np.column_stack((avg_spd_rgb_dict["red"], avg_spd_rgb_dict["green"], avg_spd_rgb_dict["blue"]))
            true_wl, true_val = None, None
            if self.csv_file: true_wl, true_val = load_true_spd_data(self.csv_file)
            if true_wl is None: messagebox.showwarning("Warning", "True SPD CSV is required for Step 2!"); return

            self.calibrator = SpectralResponseCalibrator(self.mapping)
            self.calibrator.calibrate(y_pixels, measured_rgb, true_wl, true_val)

            wl_order_check = self.mapping.pixel_to_wavelength(y_pixels)
            lamp_aligned = measured_rgb[::-1, :] if wl_order_check[0] > wl_order_check[-1] else measured_rgb

            # 【关键修复】计算 measured_total 和 corrected_total，供 Details Tab 2 第二张子图使用
            measured_total = np.sum(measured_rgb, axis=1)
            resp_sum = self.calibrator.response_R + self.calibrator.response_G + self.calibrator.response_B
            resp_sum = np.where(resp_sum > 1e-6, resp_sum, 1e-6)
            corrected_total = measured_total / resp_sum
            corrected_total = np.nan_to_num(corrected_total, nan=0.0, posinf=0.0, neginf=0.0)

            # 【关键修复】缓存字典必须包含这两个键
            self.cache_step2 = {
                "wavelengths": self.calibrator.wavelengths,
                "R": self.calibrator.response_R, "G": self.calibrator.response_G, "B": self.calibrator.response_B,
                "lamp_R": lamp_aligned[:, 0], "lamp_G": lamp_aligned[:, 1], "lamp_B": lamp_aligned[:, 2],
                "lamp_R_n": len(spd_files),
                "spd_aligned": self.calibrator.spd_aligned,
                "final_mask": self.calibrator.final_mask,
                "spd_threshold_ratio": self.calibrator.spd_threshold_ratio,
                "smooth_window": self.calibrator.smooth_window,
                "loaded_from_file": False,
                "measured_total": measured_total,
                "corrected_total": corrected_total,
                "n_spd_shots": len(spd_files),  # 【新增】用于参数面板显示
            }

            self.log_status(2, "Absolute per-channel response generated.", "green")
            log_text = ("\n=== Step 2: Spectral Response Calibration ===\n"
                        f"Averaged {len(spd_files)} SPD image(s).\n"
                        "resp_c(λ) = Lamp_c(λ) / SPD_true(λ)  [per channel, masked, NO SG, NO self-normalization]\n")
            for cname, resp in (("R", self.calibrator.response_R), ("G", self.calibrator.response_G), ("B", self.calibrator.response_B)):
                if np.any(resp > 0):
                    pk = int(np.argmax(resp))
                    log_text += f"  Response {cname}: peak {resp[pk]:.4g} a.u./SPD @ {self.calibrator.wavelengths[pk]:.1f} nm\n"
            self.txt_result.insert(tk.END, log_text); self.txt_result.see(tk.END)

            wls_all = self.calibrator.wavelengths
            r_all, g_all, b_all = self.calibrator.response_R, self.calibrator.response_G, self.calibrator.response_B
            vmask = (r_all > 0) | (g_all > 0) | (b_all > 0)
            if np.any(vmask):
                wp, rp, gp, bp = wls_all[vmask], r_all[vmask], g_all[vmask], b_all[vmask]
            else:
                wp, rp, gp, bp = wls_all, r_all, g_all, b_all
            self.ax.clear()
            self.ax.plot(wp, rp, 'r-', label='R (Absolute)')
            self.ax.plot(wp, gp, 'g-', label='G (Absolute)')
            self.ax.plot(wp, bp, 'b-', label='B (Absolute)')
            self.set_chart_title("Step 2: Per-Channel Absolute Response Curves")
            self.ax.set_xlabel("Wavelength (nm)"); self.ax.set_ylabel("Sensitivity (a.u. / SPD unit)")
            margin = (wp[-1] - wp[0]) * 0.05
            self.ax.set_xlim(wp[0] - margin, wp[-1] + margin)
            self.ax.legend(); self._style_fig(self.fig); self.fig.tight_layout(); self.canvas_plot.draw()
        except Exception as e: self.log_status(2, f"Error: {e}", "red"); messagebox.showerror("Error", str(e))

    def step3_analyze(self):
        if not self.mapping: messagebox.showwarning("Warning", "Please complete Step 1 first!"); return
        blank_files = self.blank_selector.get_files()
        sample_files = self.sample_selector.get_files()
        if not sample_files or not blank_files: messagebox.showwarning("Warning", "Please add Blank & Sample DNG(s)!"); return
        self.log_status(3, "Extracting...", "orange")
        try:
            spd_status = "APPLIED (Loaded)" if self.cache_step2.get("loaded_from_file") else ("APPLIED (per-channel absolute)" if self.calibrator else "SKIPPED (Raw RGB)")
            if self.is_turbidity_mode:
                fluor_config = {"channel": "green", "peak_wl": 550, "fwhm": 150, "color": "teal"}
            else:
                fluor_config = FLUOROPHORES[self.current_fluorophore]

            areas, wls, intensity, mask = self.get_net_areas_batch(blank_files, sample_files, fluor_config)
            vis = (wls >= 400.0) & (wls <= 700.0)
            if np.any(vis):
                wls, intensity, mask = wls[vis], intensity[vis], mask[vis]
            mean_area = np.mean(areas)
            peak = np.max(intensity[mask]) if np.any(mask) else 0

            from core.spectral_features import calculate_spectral_features
            features = calculate_spectral_features(wls, intensity, mask)

            log_text = f"\n=== Step 3: {'Turbidity' if self.is_turbidity_mode else 'Sample'} Analysis ===\n"
            log_text += f"SPD Correction Status: {spd_status}\n"
            if not self.is_turbidity_mode: log_text += f"Fluorophore: {self.current_fluorophore}\n"
            log_text += f"Net Area (Mean): {mean_area:.2f} | Peak: {peak:.2f}\n"
            log_text += f"--- Microplate Reader Style Parameters ---\n"
            log_text += f"Peak Wavelength: {features['peak_wl']:.1f} nm\n"
            log_text += f"Peak Intensity: {features['peak_intensity']:.2f} a.u.\n"
            log_text += f"FWHM: {features['fwhm']:.1f} nm\n"
            log_text += f"AUC (Area Under Curve): {features['auc']:.2f}\n"
            log_text += f"Centroid (Center of Mass): {features['centroid']:.1f} nm\n"

            if self.is_turbidity_mode:
                try:
                    from core.turbidity import TurbidityAnalyzer
                    analyzer = TurbidityAnalyzer(self.mapping, self.calibrator)
                    turb_result = analyzer.calculate_turbidity_from_spectrum(intensity, wls, blank_spectrum=None)
                    log_text += f"\n--- Turbidity Analysis ---\n"
                    log_text += f"Turbidity (Est.): {turb_result.turbidity_ntu:.2f} NTU\n"
                    log_text += f"Scattering Ratio (B/G): {turb_result.scattering_ratio:.3f}\n"
                    log_text += f"Quality Flag: {turb_result.quality_flag}\n"
                except Exception as turb_err: log_text += f"\n[Turbidity Calc Warning]: {str(turb_err)}\n"

            self.txt_result.insert(tk.END, log_text); self.txt_result.see(tk.END)
            self.log_status(3, "Analysis complete.", "green")

            self.cache_step3 = {"wls": wls, "intensity": intensity, "mask": mask, "fluor": fluor_config,
                                "mean_area": mean_area, "peak": peak, "features": features,
                                "spd_applied": bool(self.calibrator),
                                "fluor_name": None if self.is_turbidity_mode else self.current_fluorophore}

            self.ax.clear()
            self.ax.plot(wls, intensity, color=self._spec_line_color(), label='Net Spectrum')
            self.ax.fill_between(wls, intensity, where=mask, color=fluor_config['color'], alpha=0.4, label='Integration')
            self.ax.axvline(fluor_config['peak_wl'], color='red', linestyle='--', label='Target Center')
            title = "Step 3: Turbidity Scattering Spectrum" if self.is_turbidity_mode else "Step 3: Net Fluorescence Spectrum"
            self.set_chart_title(title)
            self.ax.set_xlabel("Wavelength (nm)"); self.ax.set_ylabel("True Intensity"); self.ax.legend()
            self._style_fig(self.fig); self.fig.tight_layout(); self.canvas_plot.draw()
        except Exception as e: self.log_status(3, f"Error: {e}", "red"); messagebox.showerror("Error", str(e))

    # ================= Step 4 =================
    def add_std_row(self):
        if len(self.std_rows) >= 10: messagebox.showinfo("Info", "Max 10 standards."); return
        row_frame = ttk.LabelFrame(self.std_rows_frame, text=f"Standard {len(self.std_rows) + 1}", padding=5)
        row_frame.pack(fill=tk.X, pady=5)
        top_frame = ttk.Frame(row_frame, style="Card.TFrame") if _THEME_ON else ttk.Frame(row_frame)
        self._card_label(top_frame, "Conc:").pack(side=tk.LEFT)
        conc_var = tk.StringVar(value="0.0")
        ttk.Entry(top_frame, textvariable=conc_var, width=6).pack(side=tk.LEFT, padx=5)
        self._btn(top_frame, "X", kind="pale", width=3, command=lambda r=row_frame: self.remove_std_row(r)).pack(side=tk.RIGHT)
        mini_blank = MultiFileSelector(row_frame, "Blank (1-5)", max_files=5, surface="card"); mini_blank.pack(fill=tk.X, pady=2)
        mini_sample = MultiFileSelector(row_frame, "Sample (1-5)", max_files=5, surface="card"); mini_sample.pack(fill=tk.X, pady=2)
        self.std_rows.append({"frame": row_frame, "conc_var": conc_var, "blank": mini_blank, "sample": mini_sample})
        self.update_std_buttons()

    def remove_std_row(self, row_frame):
        if len(self.std_rows) <= 2: messagebox.showinfo("Info", "Min 2 required."); return
        for i, row in enumerate(self.std_rows):
            if row["frame"] == row_frame: row_frame.destroy(); self.std_rows.pop(i); break
        self.update_std_buttons()

    def update_std_buttons(self):
        self.btn_add_std.config(state="normal" if len(self.std_rows) < 10 else "disabled")

    def calculate_concentration(self):
        if not self.mapping: messagebox.showwarning("Warning", "Please complete Step 1 & 2 first!"); return
        std_data = []
        for row in self.std_rows:
            try: conc = float(row["conc_var"].get())
            except ValueError: messagebox.showerror("Error", "Invalid conc."); return
            b_files = row["blank"].get_files(); s_files = row["sample"].get_files()
            if not b_files or not s_files: messagebox.showerror("Error", "Missing Blank/Sample in standards."); return
            std_data.append({"conc": conc, "blank": b_files, "sample": s_files})
        self.log_status(4, "Calculating...", "orange")
        try:
            fluor = FLUOROPHORES[self.current_fluorophore]
            metric_mode = self.metric_mode.get()
            concentrations, means, stds = [], [], []
            unmix_info = None
            block_stats = []
            scatter_wl = float('nan')

            if metric_mode == "unmixing":
                if not self.sample_selector.get_files() or not self.blank_selector.get_files():
                    messagebox.showerror("Error", "Unmixing requires Step 3 Blank & Sample DNG(s)."); return
                row_spectra = []
                for data in std_data:
                    wls_s, s_specs = self.get_raw_channel_spectra(data["sample"], fluor)
                    wls_b, b_specs = self.get_raw_channel_spectra(data["blank"], fluor)
                    exp_s = float(np.mean([get_exposure_time(f) for f in data["sample"]]))
                    row_spectra.append({"conc": data["conc"], "wls": wls_s, "sample": s_specs,
                                        "blank": b_specs, "exp": exp_s})
                wls = row_spectra[0]["wls"]
                ref_row = max(row_spectra, key=lambda r: r["conc"])
                E = build_reference_emission(np.mean(ref_row["sample"], axis=0), np.mean(ref_row["blank"], axis=0))

                B_ref = np.mean(ref_row["blank"], axis=0)
                sm = (wls > 460) & (wls < 520)
                if np.any(sm):
                    scatter_wl = float(wls[sm][int(np.argmax(B_ref[sm]))])
                    lo = min(max(scatter_wl + 2.0, fluor["peak_wl"] - 30.0), fluor["peak_wl"] - 5.0)
                else:
                    lo, _ = default_unmix_window(fluor["peak_wl"])
                hi = fluor["peak_wl"] + 90.0
                win_mask = (wls >= lo) & (wls <= hi)
                emis_mask = (wls >= fluor["peak_wl"] - fluor["fwhm"] / 2) & (wls <= fluor["peak_wl"] + fluor["fwhm"] / 2)
                auc_E = emission_auc(wls, E, emis_mask)
                if auc_E <= 0: raise RuntimeError("Reference emission AUC <= 0; check fluorophore/window.")

                per_row = []
                for r in row_spectra:
                    B_mean = np.mean(r["blank"], axis=0)
                    alphas, fs, fk, fr = [], [], [], []
                    for spec in r["sample"]:
                        res = unmix_spectrum(spec, E, B_mean, win_mask)
                        alphas.append(res["alpha"]); fs.append(res["scatter_scale"]); fk.append(res["offset"]); fr.append(res["r2"])
                    metrics = [a * auc_E for a in alphas]
                    r["mean_metric"] = float(np.mean(metrics))
                    concentrations.append(r["conc"])
                    means.append(r["mean_metric"])
                    stds.append(float(np.std(metrics, ddof=1)) if len(metrics) > 1 else 0.0)
                    per_row.append({"conc": r["conc"], "alpha": float(np.mean(alphas)), "s": float(np.mean(fs)),
                                    "k": float(np.mean(fk)), "r2": float(np.mean(fr))})

                wls_u, u_samples = self.get_raw_channel_spectra(self.sample_selector.get_files(), fluor)
                _, u_blanks = self.get_raw_channel_spectra(self.blank_selector.get_files(), fluor)
                B_u = np.mean(u_blanks, axis=0)
                u_a, u_s, u_k = [], [], []
                for spec in u_samples:
                    res = unmix_spectrum(spec, E, B_u, win_mask)
                    u_a.append(res["alpha"]); u_s.append(res["scatter_scale"]); u_k.append(res["offset"])
                alpha_u = float(np.mean(u_a))
                sample_mean_area = alpha_u * auc_E
                unmix_info = {"window": (lo, hi), "ref_conc": ref_row["conc"], "auc_E": auc_E, "per_row": per_row,
                              "alpha": alpha_u, "u_s": float(np.mean(u_s)), "u_k": float(np.mean(u_k)),
                              "wls": wls, "E": E, "B": B_u, "sample_mean_spec": np.mean(u_samples, axis=0)}

                blocks = {}
                for r in row_spectra:
                    blocks.setdefault(round(r["exp"], 3), []).append(r)
                for e, rows in sorted(blocks.items()):
                    if len(rows) < 3:
                        block_stats.append({"exp": e, "n": len(rows), "slope": float('nan'), "r2": float('nan'),
                                            "syx": float('nan'), "t": float('nan'), "lod": float('nan'),
                                            "max_conc": max(x["conc"] for x in rows)})
                        continue
                    xs = np.array([x["conc"] for x in rows]); ys = np.array([x["mean_metric"] for x in rows])
                    rr = linregress(xs, ys)
                    n = len(xs)
                    syx = float(np.sqrt(np.sum((ys - (rr.slope * xs + rr.intercept)) ** 2) / (n - 2)))
                    t = slope_t_statistic(rr.rvalue, n)
                    lod = 3.3 * syx / rr.slope if rr.slope > 0 else float('nan')
                    block_stats.append({"exp": e, "n": n, "slope": rr.slope, "intercept": rr.intercept,
                                        "r2": rr.rvalue ** 2, "syx": syx, "t": t, "lod": lod,
                                        "max_conc": float(xs.max())})
                eligible = [b for b in block_stats if b["n"] >= 3 and np.isfinite(b["t"]) and b["t"] >= 3
                            and np.isfinite(b["lod"]) and b["lod"] > 0]
                headline = min(eligible, key=lambda b: b["max_conc"]) if eligible else None

                lods = []
                for shift in (-10.0, -5.0, 0.0, 5.0, 10.0):
                    wm = (wls >= lo + shift) & (wls <= hi)
                    if np.sum(wm) < 10: continue
                    ms = []
                    for r in row_spectra:
                        Bm = np.mean(r["blank"], axis=0)
                        aa = [unmix_spectrum(sp, E, Bm, wm)["alpha"] for sp in r["sample"]]
                        ms.append(float(np.mean(aa)))
                    rr = linregress(concentrations, ms)
                    if rr.slope > 0 and len(ms) > 2:
                        syx = np.sqrt(np.sum((np.array(ms) - (rr.slope * np.array(concentrations) + rr.intercept)) ** 2) / (len(ms) - 2))
                        lods.append(3.3 * syx / rr.slope)
                unmix_info["lod_range"] = (float(min(lods)), float(max(lods))) if lods else None
                self._lod_headline = headline
            else:
                self._lod_headline = None
                for data in std_data:
                    areas, _, _, _ = self.get_net_areas_batch(data["blank"], data["sample"], fluor)
                    concentrations.append(data["conc"]); means.append(float(np.mean(areas)))
                    stds.append(float(np.std(areas, ddof=1)) if len(areas) > 1 else 0.0)
                sample_areas, _, _, _ = self.get_net_areas_batch(self.blank_selector.get_files(), self.sample_selector.get_files(), fluor)
                sample_mean_area = float(np.mean(sample_areas))

            reg_method = self.regression_method.get()
            calc_result = calculate_concentration(concentrations, means, sample_mean_area,
                                                  mode=self.analysis_mode, regression_method=reg_method)
            t_stat = slope_t_statistic(calc_result["r_value"], len(concentrations))

            headline = getattr(self, "_lod_headline", None)
            if headline is not None and self.analysis_mode == "lod_iupac" and "lod_conc" in calc_result:
                calc_result["lod_fullrange"] = calc_result["lod_conc"]
                calc_result["loq_fullrange"] = calc_result["loq_conc"]
                calc_result["lod_conc"] = 3.3 * headline["syx"] / headline["slope"]
                calc_result["loq_conc"] = 10.0 * headline["syx"] / headline["slope"]
                calc_result["lod_source"] = f"single-exposure block @ {headline['exp']:.3f}s (n={headline['n']}, t={headline['t']:.1f})"
                calc_result["lod_source_exp"] = headline["exp"]
            elif self.analysis_mode == "lod_iupac" and "lod_conc" in calc_result:
                calc_result["lod_source"] = "full-range curve (no eligible single-exposure block)"

            log_text = f"\n=== Step 4: Concentration Quantification ({self.analysis_mode.upper()}) ===\n"
            log_text += f"Quantification Metric: {metric_mode.upper()}\n"
            log_text += f"SPD Correction: {'APPLIED' if self.calibrator else 'SKIPPED'}\n"
            log_text += f"Regression Method: {reg_method.upper()}\n"
            log_text += f"Curve: I = {calc_result['slope']:.2f}*C + {calc_result['intercept']:.2f} (R²={calc_result['r_squared']:.4f})\n"
            log_text += f"Slope t-statistic: {t_stat:.2f} (require >= 3)\n"
            if t_stat < 3: log_text += "  [WARNING] Calibration slope NOT significant; LoD/LoQ unreliable!\n"
            log_text += f"Sample Metric: {calc_result['sample_mean_area']:.2f}\n"
            log_text += f"PREDICTED CONCENTRATION: {calc_result['sample_conc']:.3f} (a.u.)\n"
            if unmix_info is not None:
                lo, hi = unmix_info["window"]
                log_text += f"--- Spectral Unmixing: sample = α·E + s·Blank + k ---\n"
                log_text += f"Scatter peak (map coords): {scatter_wl:.1f} nm | Unmixing window: [{lo:.1f}, {hi:.1f}] nm (start beyond scatter)\n"
                log_text += f"Reference E from std conc = {unmix_info['ref_conc']} | AUC(E): {unmix_info['auc_E']:.2f}\n"
                for pr in unmix_info["per_row"]:
                    log_text += f"  C={pr['conc']:<10g} α={pr['alpha']:.4f}  s={pr['s']:.3f}  k={pr['k']:.3f}  R²={pr['r2']:.4f}\n"
                log_text += f"  Sample:   α={unmix_info['alpha']:.4f}  s={unmix_info['u_s']:.3f}  k={unmix_info['u_k']:.3f}\n"
                if block_stats:
                    log_text += "--- Exposure-block LoD selection (lowest block, t >= 3) ---\n"
                    for b in block_stats:
                        tag = "  <-- HEADLINE" if (headline and abs(b["exp"] - headline["exp"]) < 1e-9) else ""
                        if b["n"] < 3:
                            log_text += f"  @{b['exp']:.3f}s: n={b['n']} (skipped, <3 levels){tag}\n"
                        else:
                            log_text += (f"  @{b['exp']:.3f}s: n={b['n']} slope={b['slope']:.4g} R²={b['r2']:.4f} "
                                         f"Sy/x={b['syx']:.4g} t={b['t']:.1f} LoD={b['lod']:.4g}{tag}\n")
                if unmix_info["lod_range"]:
                    log_text += f"LoD window-sensitivity range: [{unmix_info['lod_range'][0]:.3g}, {unmix_info['lod_range'][1]:.3g}] (a.u.)\n"
            if self.analysis_mode == "lod_iupac" and "lod_conc" in calc_result:
                log_text += f"--- IUPAC Limit Analysis (headline) ---\n"
                log_text += f"Source: {calc_result.get('lod_source', 'full-range curve')}\n"
                log_text += f"LIMIT OF DETECTION (LoD, 3.3σ): {calc_result['lod_conc']:.4g} (a.u.)\n"
                log_text += f"LIMIT OF QUANTIFICATION (LoQ, 10σ): {calc_result['loq_conc']:.4g} (a.u.)\n"
                if "lod_fullrange" in calc_result:
                    log_text += f"[reference] full-range LoD/LoQ: {calc_result['lod_fullrange']:.4g} / {calc_result['loq_fullrange']:.4g}\n"
            self.txt_result.insert(tk.END, log_text); self.txt_result.see(tk.END)
            self.log_status(4, "Done.", "green")

            self.cache_step4 = calc_result
            self.cache_step4.update({"concentrations": concentrations, "means": means, "stds": stds,
                                     "metric_mode": metric_mode, "t_stat": t_stat, "unmix": unmix_info,
                                     "blocks": block_stats, "scatter_peak_wl": scatter_wl})

            pc = self._c4_colors()
            self.ax.clear()
            self.ax.errorbar(concentrations, means, yerr=stds, fmt='o', color=pc["std"], capsize=5,
                             label='Standards (Mean±SD)', zorder=5, ecolor=pc["std_err"])
            x_line = np.linspace(min(concentrations) * 0.8, max(concentrations) * 1.2, 100)
            self.ax.plot(x_line, calc_result['slope'] * x_line + calc_result['intercept'], '--', color=pc["fit"],
                         linewidth=2, label=f'Fit ({reg_method.upper()}, R²={calc_result["r_squared"]:.3f})')
            self.ax.scatter([calc_result['sample_conc']], [calc_result['sample_mean_area']], color=pc["sample"],
                            marker='*', s=150, label=f'Sample (Conc: {calc_result["sample_conc"]:.3f})',
                            zorder=6, edgecolors=pc["sample_edge"])
            has_inset = False
            if self.analysis_mode == "lod_iupac" and "lod_conc" in calc_result:
                if "lod_fullrange" not in calc_result:
                    self.ax.axhline(y=calc_result["lod_signal"], color=pc["lod"], linestyle='-.', linewidth=1.5, label='LoD Signal')
                    self.ax.axhline(y=calc_result["loq_signal"], color=pc["loq"], linestyle=':', linewidth=1.5, label='LoQ Signal')
                self.ax.axvline(x=calc_result["lod_conc"], color=pc["lod"], linestyle='-.', alpha=0.6, label='LoD (headline)')
                self.ax.axvline(x=calc_result["loq_conc"], color=pc["loq"], linestyle=':', alpha=0.6, label='LoQ (headline)')
            self.ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

            max_conc = max(concentrations) if concentrations else 1.0
            x_thresh = max(calc_result.get("loq_conc", max_conc * 0.3) * 1.5, max_conc * 0.3)
            points_in_region = sum(1 for c in concentrations if c <= x_thresh)
            if calc_result['sample_conc'] <= x_thresh: points_in_region += 1
            if points_in_region >= 2:
                has_inset = True
                axins = inset_axes(self.ax, width="45%", height="45%", loc='upper left', borderpad=1.5)
                axins.errorbar(concentrations, means, yerr=stds, fmt='o', color=pc["std"], capsize=3, markersize=4, ecolor=pc["std_err"])
                axins.plot(x_line, calc_result['slope'] * x_line + calc_result['intercept'], '--', color=pc["fit"], linewidth=1.5)
                axins.scatter([calc_result['sample_conc']], [calc_result['sample_mean_area']], color=pc["sample"], marker='*', s=50)
                if self.analysis_mode == "lod_iupac" and "lod_conc" in calc_result:
                    axins.axvline(x=calc_result["lod_conc"], color=pc["lod"], linestyle='-.', alpha=0.6)
                    axins.axvline(x=calc_result["loq_conc"], color=pc["loq"], linestyle=':', alpha=0.6)
                axins.set_xlim(0, x_thresh * 1.2)
                y_min_inset, y_max_inset = float('inf'), float('-inf')
                for c, m, s in zip(concentrations, means, stds):
                    if c <= x_thresh: y_min_inset = min(y_min_inset, m - s); y_max_inset = max(y_max_inset, m + s)
                if calc_result['sample_conc'] <= x_thresh:
                    y_min_inset = min(y_min_inset, calc_result['sample_mean_area']); y_max_inset = max(y_max_inset, calc_result['sample_mean_area'])
                if y_min_inset == float('inf'): y_min_inset, y_max_inset = 0, 1
                y_range = y_max_inset - y_min_inset
                axins.set_ylim(y_min_inset - 0.15 * y_range, y_max_inset + 0.15 * y_range)
                axins.set_title("Zoom-in: LoD/LoQ Region", fontsize=8); axins.tick_params(labelsize=7); axins.grid(True, linestyle=':', alpha=0.5)

            self.set_chart_title("Step 4: Standard Curve & Concentration")
            self.ax.set_xlabel("Concentration (a.u.)", fontsize=11)
            self.ax.set_ylabel("Unmixed Net Area" if metric_mode == "unmixing" else "Net Fluorescence Area", fontsize=11)
            self.ax.grid(True, linestyle=':', alpha=0.7)
            self._style_fig(self.fig)
            if has_inset:
                self.fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.10)
            else:
                self.fig.tight_layout()
            self.canvas_plot.draw()
        except Exception as e: self.log_status(4, f"Error: {e}", "red"); messagebox.showerror("Error", str(e))

    # ================= Windows =================
    def open_turbidity_window(self):
        if self.turbidity_window is None or not self.turbidity_window.win.winfo_exists():
            if self.mapping is None: messagebox.showwarning("Warning", "Please complete Step 1 first!"); return
            from gui.turbidity_window import TurbidityWindow
            self.turbidity_window = TurbidityWindow(self.root, self.mapping, self.calibrator)
            if self.cache_step3: self.turbidity_window.update_with_spectrum(self.cache_step3["wls"], self.cache_step3["intensity"])
        else:
            self.turbidity_window.win.lift(); self.turbidity_window.win.focus_force()

    def show_details_window(self):
        if not self.mapping:
            messagebox.showinfo("Info", "Please complete at least Step 1 to view details.")
            return
        DetailsWindow(self.root, self.cache_step1, self.cache_step2, self.cache_step3, self.cache_step4, self.mapping)