# gui/main_window.py
"""Main GUI Window for FSSA v1.4.2 (Integrated SPD Param Export/Import)."""
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

from config import FLUOROPHORES
from core.image_processing import (
    read_dng_linear_rgb, auto_detect_x_roi, get_average_profiles, 
    parse_known_wavelengths, find_peak_auto, make_channel_profiles, 
    load_true_spd_data, np_trapz
)
from core.calibration import LinearMapping, TwoPointValidation, SpectralResponseCalibrator
from core.quantification import calculate_concentration
from gui.widgets import MultiFileSelector
from gui.details_window import DetailsWindow

class FSSA_GUI:
    def __init__(self, root, analysis_mode="lod_iupac"):
        self.root = root
        self.analysis_mode = analysis_mode
        self.is_turbidity_mode = (analysis_mode == "turbidity")
        
        title_suffix = "TURBIDITY" if self.is_turbidity_mode else analysis_mode.upper()
        self.root.title(f"Fungal Sentinel Spectral Analyzer (FSSA) v1.4.2 [{title_suffix}] - iGEM 2026")
        self.root.geometry("1250x850")
        
        self.txt_file = None
        self.csv_file = None
        self.mapping = None
        self.x_roi = None
        self.calibrator = None
        self.current_fluorophore = "Ypet (Venus variant)"
        self.current_page = 1
        self.current_chart_filename = "FSSA_Chart"
        
        self.cache_step1 = {}
        self.cache_step2 = {}
        self.cache_step3 = {}
        self.cache_step4 = {}
        self.turbidity_window = None
        
        self.setup_ui()

    def setup_ui(self):
        left_frame = ttk.Frame(self.root, width=360)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        left_frame.pack_propagate(False)
        
        ttk.Label(left_frame, text=f"FSSA v1.4.2\n({self.analysis_mode.upper()})", font=("Arial", 14, "bold")).pack(pady=15)
        
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
        self.btn_prev = ttk.Button(nav_frame, text="< Prev", command=self.prev_page)
        self.btn_prev.pack(side=tk.LEFT)
        self.lbl_page_num = ttk.Label(nav_frame, text=f"Step 1 / {self.max_steps}", font=("Arial", 11, "bold"))
        self.lbl_page_num.pack(side=tk.LEFT, expand=True)
        self.btn_next = ttk.Button(nav_frame, text="Next >", command=self.next_page)
        self.btn_next.pack(side=tk.RIGHT)
        
        self.show_page(1)
        
        right_frame = ttk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        toolbar = ttk.Frame(right_frame)
        toolbar.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(toolbar, text="Real-time Visualization", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        
        if not self.is_turbidity_mode:
            ttk.Button(toolbar, text="💧 Turbidity", command=self.open_turbidity_window).pack(side=tk.RIGHT, padx=2)
        ttk.Button(toolbar, text=" Details", command=self.show_details_window).pack(side=tk.RIGHT, padx=2)
        ttk.Button(toolbar, text=" Export Chart", command=self.export_chart).pack(side=tk.RIGHT)
        
        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Welcome to FSSA v1.4.2\nPlease proceed with Step 1: Positioning.")
        self.ax.axis('off')
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10)
        
        res_frame = ttk.LabelFrame(right_frame, text="Analysis Results & Logs", padding=5)
        res_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.txt_result = scrolledtext.ScrolledText(res_frame, height=8, font=("Consolas", 10))
        self.txt_result.pack(fill=tk.X)

    def _setup_page1(self):
        f = self.page1
        ttk.Label(f, text="Step 1: Wavelength Calibration", font=("Arial", 12, "bold"), foreground="blue").pack(anchor=tk.W, pady=(0,10))
        self.pos_selector = MultiFileSelector(f, "Positioning DNG (1-5 images)", max_files=5)
        self.pos_selector.pack(fill=tk.X, pady=5)
        ttk.Label(f, text="Wavelength Notes TXT (Optional):", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(10,0))
        txt_frame = ttk.Frame(f); txt_frame.pack(fill=tk.X)
        ttk.Button(txt_frame, text="Select TXT", command=self.select_txt).pack(side=tk.LEFT)
        self.lbl_txt = ttk.Label(txt_frame, text="Default", foreground="gray"); self.lbl_txt.pack(side=tk.LEFT, padx=5)
        ttk.Button(f, text="Run Wavelength Calibration", command=self.step1_calibrate).pack(fill=tk.X, pady=15)
        self.lbl_status1 = ttk.Label(f, text="Status: Ready", foreground="gray", wraplength=300); self.lbl_status1.pack(anchor=tk.W)

    def _setup_page2(self):
        f = self.page2
        ttk.Label(f, text="Step 2: Spectral Response Cal. (SPD)", font=("Arial", 12, "bold"), foreground="green").pack(anchor=tk.W, pady=(0,10))
        self.spd_selector = MultiFileSelector(f, "SPD Calibration DNG (1-5 images)", max_files=5)
        self.spd_selector.pack(fill=tk.X, pady=5)
        ttk.Label(f, text="True SPD CSV (Optional):", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(10,0))
        csv_frame = ttk.Frame(f); csv_frame.pack(fill=tk.X)
        ttk.Button(csv_frame, text="Select CSV", command=self.select_csv).pack(side=tk.LEFT)
        self.lbl_csv = ttk.Label(csv_frame, text="None (Skip SPD Cal.)", foreground="gray"); self.lbl_csv.pack(side=tk.LEFT, padx=5)
        
        btn_frame_step2 = ttk.Frame(f)
        btn_frame_step2.pack(fill=tk.X, pady=15)
        ttk.Button(btn_frame_step2, text="Run Spectral Response Cal.", command=self.step2_calibrate).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        io_frame = ttk.Frame(f)
        io_frame.pack(fill=tk.X, pady=5)
        ttk.Button(io_frame, text="💾 Export SPD Params", command=self.export_spd_params).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(io_frame, text="📂 Load SPD Params", command=self.load_spd_params).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        self.lbl_status2 = ttk.Label(f, text="Status: Ready", foreground="gray", wraplength=300); self.lbl_status2.pack(anchor=tk.W)

    def _setup_page3(self):
        f = self.page3
        if self.is_turbidity_mode:
            ttk.Label(f, text="Step 3: Turbidity & Spectral Analysis", font=("Arial", 12, "bold"), foreground="teal").pack(anchor=tk.W, pady=(0,10))
            ttk.Separator(f, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        else:
            ttk.Label(f, text="Step 3: Sample Analysis", font=("Arial", 12, "bold"), foreground="purple").pack(anchor=tk.W, pady=(0,10))
            ttk.Label(f, text="Select Fluorophore:", font=("Arial", 9, "bold")).pack(anchor=tk.W)
            self.fluor_combo = ttk.Combobox(f, values=list(FLUOROPHORES.keys()), state="readonly", width=25)
            self.fluor_combo.set(self.current_fluorophore); self.fluor_combo.pack(pady=2, anchor=tk.W)
            self.fluor_combo.bind("<<ComboboxSelected>>", self.on_fluorophore_change)
            ttk.Separator(f, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
            
        self.blank_selector = MultiFileSelector(f, "Blank / Background (1-5 images)", max_files=5)
        self.blank_selector.pack(fill=tk.X, pady=5)
        self.sample_selector = MultiFileSelector(f, "Sample DNG (1-5 images)", max_files=5)
        self.sample_selector.pack(fill=tk.X, pady=5)
        
        btn_text = "Analyze Turbidity & Spectrum" if self.is_turbidity_mode else "Extract Sample Spectrum"
        ttk.Button(f, text=btn_text, command=self.step3_analyze).pack(fill=tk.X, pady=15)
        self.lbl_status3 = ttk.Label(f, text="Status: Ready", foreground="gray", wraplength=300); self.lbl_status3.pack(anchor=tk.W)

    def _setup_page4(self):
        f = self.page4
        ttk.Label(f, text="Step 4: Concentration Quantification", font=("Arial", 12, "bold"), foreground="red").pack(anchor=tk.W, pady=(0,10))
        
        canvas_frame = ttk.Frame(f); canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.std_canvas = tk.Canvas(canvas_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.std_canvas.yview)
        self.std_rows_frame = ttk.Frame(self.std_canvas)
        self.std_rows_frame.bind("<Configure>", lambda e: self.std_canvas.configure(scrollregion=self.std_canvas.bbox("all")))
        self.std_canvas.create_window((0, 0), window=self.std_rows_frame, anchor="nw")
        self.std_canvas.configure(yscrollcommand=scrollbar.set)
        self.std_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        def _on_mousewheel(event):
            if event.num == 4: self.std_canvas.yview_scroll(-1, "units")
            elif event.num == 5: self.std_canvas.yview_scroll(1, "units")
            else: self.std_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.std_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.std_canvas.bind_all("<Button-4>", _on_mousewheel)
        self.std_canvas.bind_all("<Button-5>", _on_mousewheel)
        
        self.std_rows = []
        reg_frame = ttk.LabelFrame(f, text="Regression Method", padding=5)
        reg_frame.pack(fill=tk.X, pady=5)
        self.regression_method = tk.StringVar(value="ols")
        ttk.Radiobutton(reg_frame, text="Ordinary Least Squares (OLS)", variable=self.regression_method, value="ols").pack(anchor=tk.W)
        ttk.Radiobutton(reg_frame, text="Weighted (1/y) - For count data", variable=self.regression_method, value="weighted_1y").pack(anchor=tk.W)
        ttk.Radiobutton(reg_frame, text="Weighted (1/y²) - For fluorescence (Recommended)", variable=self.regression_method, value="weighted_1y2").pack(anchor=tk.W)
        
        btn_frame = ttk.Frame(f); btn_frame.pack(fill=tk.X, pady=5)
        self.btn_add_std = ttk.Button(btn_frame, text="+ Add Std", command=self.add_std_row); self.btn_add_std.pack(side=tk.LEFT, padx=2)
        self.btn_calc_conc = ttk.Button(btn_frame, text="Build Curve & Calculate", command=self.calculate_concentration); self.btn_calc_conc.pack(side=tk.LEFT, padx=2)
        self.add_std_row(); self.add_std_row()
        self.lbl_status4 = ttk.Label(f, text="Status: Ready", foreground="gray", wraplength=300); self.lbl_status4.pack(anchor=tk.W, pady=5)

    def show_page(self, page_num):
        self.current_page = page_num
        for p in [self.page1, self.page2, self.page3, self.page4]: p.pack_forget()
        getattr(self, f"page{page_num}").pack(fill=tk.BOTH, expand=True)
        self.lbl_page_num.config(text=f"Step {page_num} / {self.max_steps}")
        self.btn_prev.config(state="normal" if page_num > 1 else "disabled")
        self.btn_next.config(state="normal" if page_num < self.max_steps else "disabled")

    def prev_page(self): self.show_page(self.current_page - 1)
    def next_page(self): self.show_page(self.current_page + 1)
    def select_txt(self):
        path = filedialog.askopenfilename(filetypes=[("Text", "*.txt")])
        if path: self.txt_file = path; self.lbl_txt.config(text=Path(path).name, foreground="black")
    def select_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if path: self.csv_file = path; self.lbl_csv.config(text=Path(path).name, foreground="black")
    def on_fluorophore_change(self, event): self.current_fluorophore = self.fluor_combo.get()
    def log_status(self, step, msg, color="black"):
        getattr(self, f"lbl_status{step}").config(text=f"Status: {msg}", foreground=color); self.root.update()
    def set_chart_title(self, title):
        self.ax.set_title(title, fontsize=13, fontweight='bold')
        self.current_chart_filename = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
    def export_chart(self):
        file_path = filedialog.asksaveasfilename(title="Export Chart", initialfile=f"{self.current_chart_filename}.png", defaultextension=".png", filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if file_path:
            try: 
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Saved to:\n{file_path}")
            except Exception as e: messagebox.showerror("Error", str(e))

    # ================= SPD 参数导出与导入 =================
    def export_spd_params(self):
        if not self.calibrator or self.calibrator.response_R is None:
            messagebox.showwarning("Warning", "Please run Spectral Response Calibration first!")
            return
        file_path = filedialog.asksaveasfilename(
            title="Export SPD Parameters", 
            initialfile="spd_calibration_params.json",
            defaultextension=".json", 
            filetypes=[("JSON Params", "*.json")]
        )
        if file_path:
            try:
                self.calibrator.save_params(file_path)
                messagebox.showinfo("Success", f"SPD parameters exported to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def load_spd_params(self):
        file_path = filedialog.askopenfilename(
            title="Load SPD Parameters", 
            filetypes=[("JSON Params", "*.json")]
        )
        if not file_path: return
        
        try:
            self.calibrator = SpectralResponseCalibrator.load_params(file_path, self.mapping)
            
            measured_total = np.zeros_like(self.calibrator.wavelengths)
            corrected_total = np.zeros_like(self.calibrator.wavelengths)
            
            self.cache_step2 = {
                "wavelengths": self.calibrator.wavelengths, 
                "R": self.calibrator.response_R, "G": self.calibrator.response_G, "B": self.calibrator.response_B,
                "measured_total": measured_total, 
                "corrected_total": corrected_total,
                "spd_threshold_ratio": self.calibrator.spd_threshold_ratio,
                "smooth_window": self.calibrator.smooth_window,
                "loaded_from_file": True
            }
            
            self.log_status(2, f"Loaded from {Path(file_path).name}", "green")
            messagebox.showinfo("Success", "SPD parameters loaded successfully!\nNote: Spectral comparison chart will be unavailable until re-calibration.")
            
            # 【v1.4.2 优化】截断至有效波段后绘图
            wls_all = self.calibrator.wavelengths
            r_all, g_all, b_all = self.calibrator.response_R, self.calibrator.response_G, self.calibrator.response_B
            valid_mask = (r_all > 0) | (g_all > 0) | (b_all > 0)
            if np.any(valid_mask):
                wls_plot, r_plot, g_plot, b_plot = wls_all[valid_mask], r_all[valid_mask], g_all[valid_mask], b_all[valid_mask]
            else:
                wls_plot, r_plot, g_plot, b_plot = wls_all, r_all, g_all, b_all
            
            self.ax.clear()
            self.ax.plot(wls_plot, r_plot, 'r-', label='R (Loaded)')
            self.ax.plot(wls_plot, g_plot, 'g-', label='G (Loaded)')
            self.ax.plot(wls_plot, b_plot, 'b-', label='B (Loaded)')
            self.set_chart_title("Step 2: Spectral Response Curves (Loaded)")
            self.ax.set_xlabel("Wavelength (nm)"); self.ax.set_ylabel("Normalized Sensitivity")
            margin = (wls_plot[-1] - wls_plot[0]) * 0.05
            self.ax.set_xlim(wls_plot[0] - margin, wls_plot[-1] + margin)
            self.ax.legend(); self.fig.tight_layout(); self.canvas_plot.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load SPD params: {str(e)}")

    def calculate_net_signal_single(self, avg_blank_rgb, sample_path, fluor_config):
        ch_name = fluor_config['channel']
        ch_idx = {"red":0, "green":1, "blue":2}[ch_name]
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
            resp_map = {"red": self.calibrator.response_R, "green": self.calibrator.response_G, "blue": self.calibrator.response_B}
            resp_ch = resp_map[ch_name]
            # 【v1.4.3】绝对响应度下的相对地板：该通道峰值的 5%，防止掩码边缘除零爆炸
            ch_max = float(np.max(resp_ch)) if resp_ch is not None and len(resp_ch) else 0.0
            floor = 0.05 * ch_max if ch_max > 0 else 1e-9
            resp_interp = np.interp(wavelengths, self.calibrator.wavelengths, resp_ch)
            resp_interp = np.where(resp_interp > floor, resp_interp, floor)
            true_intensity = net_rgb[:, ch_idx] / resp_interp
        else:
            true_intensity = net_rgb[:, ch_idx]
        target_wl, fwhm = fluor_config['peak_wl'], fluor_config['fwhm']
        mask = (wavelengths >= target_wl - fwhm/2) & (wavelengths <= target_wl + fwhm/2)
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
            p1 = next(p for p in peaks if p.name=="B")
            p2 = next(p for p in peaks if p.name=="R")
            hold = next(p for p in peaks if p.name=="G")
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
            self.cache_step1 = {"rgb": rgb_first, "x_roi": self.x_roi, "profiles": avg_profiles, "peaks": peaks, "mapping": self.mapping, "validation": validation, "exp_time": exp_first, "wl_err": wl_err}
            self.ax.clear()
            colors = {"red": "red", "green": "green", "blue": "blue"}
            for name, prof in avg_profiles.items(): self.ax.plot(prof, color=colors[name], label=f"{name.capitalize()} (Avg)", linewidth=1.5)
            for p in peaks: self.ax.axvline(p.y_centroid, linestyle="--", color="black", label=f"{p.name}")
            self.set_chart_title("Step 1: Wavelength Calibration Profiles")
            self.ax.set_xlabel("Y Pixel"); self.ax.set_ylabel("Intensity"); self.ax.legend(fontsize=8); self.fig.tight_layout(); self.canvas_plot.draw()
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

            # 【v1.4.3】逐通道缓存（升序对齐），绝不合并通道
            wl_order_check = self.mapping.pixel_to_wavelength(y_pixels)
            lamp_aligned = measured_rgb[::-1, :] if wl_order_check[0] > wl_order_check[-1] else measured_rgb

            self.cache_step2 = {
                "wavelengths": self.calibrator.wavelengths,
                "R": self.calibrator.response_R, "G": self.calibrator.response_G, "B": self.calibrator.response_B,
                "lamp_R": lamp_aligned[:, 0], "lamp_R_n": len(spd_files),
                "lamp_G": lamp_aligned[:, 1], "lamp_B": lamp_aligned[:, 2],
                "spd_aligned": self.calibrator.spd_aligned,
                "final_mask": self.calibrator.final_mask,
                "spd_threshold_ratio": self.calibrator.spd_threshold_ratio,
                "smooth_window": self.calibrator.smooth_window,
                "loaded_from_file": False
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

            # 主图：逐通道绝对响应度，截断至有效波段
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
            self.ax.legend(); self.fig.tight_layout(); self.canvas_plot.draw()
        except Exception as e: self.log_status(2, f"Error: {e}", "red"); messagebox.showerror("Error", str(e))

    def step3_analyze(self):
        if not self.mapping: messagebox.showwarning("Warning", "Please complete Step 1 first!"); return
        blank_files = self.blank_selector.get_files()
        sample_files = self.sample_selector.get_files()
        if not sample_files or not blank_files: messagebox.showwarning("Warning", "Please add Blank & Sample DNG(s)!"); return
        self.log_status(3, "Extracting...", "orange")
        try:
            spd_status = "APPLIED (Loaded)" if self.cache_step2.get("loaded_from_file") else ("APPLIED (Normalized)" if self.calibrator else "SKIPPED (Raw RGB)")
            if self.is_turbidity_mode:
                fluor_config = {"channel": "green", "peak_wl": 550, "fwhm": 150, "color": "teal"} 
            else:
                fluor_config = FLUOROPHORES[self.current_fluorophore]
                
            areas, wls, intensity, mask = self.get_net_areas_batch(blank_files, sample_files, fluor_config)
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
            
            self.cache_step3 = {"wls": wls, "intensity": intensity, "mask": mask, "fluor": fluor_config, "mean_area": mean_area, "peak": peak, "features": features, "spd_applied": bool(self.calibrator)}
            
            self.ax.clear()
            self.ax.plot(wls, intensity, 'k-', label='Net Spectrum')
            self.ax.fill_between(wls, intensity, where=mask, color=fluor_config['color'], alpha=0.4, label='Integration')
            self.ax.axvline(fluor_config['peak_wl'], color='red', linestyle='--', label='Target Center')
            title = "Step 3: Turbidity Scattering Spectrum" if self.is_turbidity_mode else "Step 3: Net Fluorescence Spectrum"
            self.set_chart_title(title)
            self.ax.set_xlabel("Wavelength (nm)"); self.ax.set_ylabel("True Intensity"); self.ax.legend(); self.fig.tight_layout(); self.canvas_plot.draw()
        except Exception as e: self.log_status(3, f"Error: {e}", "red"); messagebox.showerror("Error", str(e))

    def add_std_row(self):
        if len(self.std_rows) >= 10: messagebox.showinfo("Info", "Max 10 standards."); return
        row_frame = ttk.LabelFrame(self.std_rows_frame, text=f"Standard {len(self.std_rows)+1}", padding=5)
        row_frame.pack(fill=tk.X, pady=5)
        top_frame = ttk.Frame(row_frame); top_frame.pack(fill=tk.X)
        ttk.Label(top_frame, text="Conc:").pack(side=tk.LEFT)
        conc_var = tk.StringVar(value="0.0")
        ttk.Entry(top_frame, textvariable=conc_var, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="X", width=3, command=lambda r=row_frame: self.remove_std_row(r)).pack(side=tk.RIGHT)
        mini_blank = MultiFileSelector(row_frame, "Blank (1-5)", max_files=5); mini_blank.pack(fill=tk.X, pady=2)
        mini_sample = MultiFileSelector(row_frame, "Sample (1-5)", max_files=5); mini_sample.pack(fill=tk.X, pady=2)
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
            concentrations, means, stds = [], [], []
            for data in std_data:
                areas, _, _, _ = self.get_net_areas_batch(data["blank"], data["sample"], fluor)
                concentrations.append(data["conc"])
                means.append(np.mean(areas))
                stds.append(np.std(areas, ddof=1) if len(areas) > 1 else 0)
            sample_areas, _, _, _ = self.get_net_areas_batch(self.blank_selector.get_files(), self.sample_selector.get_files(), fluor)
            sample_mean_area = np.mean(sample_areas)
            
            reg_method = self.regression_method.get()
            calc_result = calculate_concentration(concentrations, means, sample_mean_area, mode=self.analysis_mode, regression_method=reg_method)
            
            log_text = f"\n=== Step 4: Concentration Quantification ({self.analysis_mode.upper()}) ===\n"
            log_text += f"SPD Correction: {'APPLIED' if self.calibrator else 'SKIPPED'}\n"
            log_text += f"Regression Method: {reg_method.upper()}\n"
            log_text += f"Curve: I = {calc_result['slope']:.2f}*C + {calc_result['intercept']:.2f} (R²={calc_result['r_squared']:.4f})\n"
            log_text += f"Sample Net Area: {calc_result['sample_mean_area']:.2f}\n"
            log_text += f"PREDICTED CONCENTRATION: {calc_result['sample_conc']:.3f} (a.u.)\n"
            
            if self.analysis_mode == "lod_iupac" and "lod_conc" in calc_result:
                log_text += f"--- IUPAC Limit Analysis ---\n"
                log_text += f"Residual Std Dev (Sy/x): {calc_result['syx']:.2f}\n"
                log_text += f"LIMIT OF DETECTION (LoD, 3.3σ): {calc_result['lod_conc']:.3f} (a.u.)\n"
                log_text += f"LIMIT OF QUANTIFICATION (LoQ, 10σ): {calc_result['loq_conc']:.3f} (a.u.)\n"
                
            self.txt_result.insert(tk.END, log_text); self.txt_result.see(tk.END)
            self.log_status(4, "Done.", "green")
            
            self.cache_step4 = calc_result
            self.cache_step4["concentrations"] = concentrations
            self.cache_step4["means"] = means
            self.cache_step4["stds"] = stds
            
            self.ax.clear()
            self.ax.errorbar(concentrations, means, yerr=stds, fmt='o', color='blue', capsize=5, label='Standards (Mean±SD)', zorder=5, ecolor='lightblue')
            x_line = np.linspace(min(concentrations)*0.8, max(concentrations)*1.2, 100)
            self.ax.plot(x_line, calc_result['slope'] * x_line + calc_result['intercept'], 'r--', linewidth=2, label=f'Fit ({reg_method.upper()}, R²={calc_result["r_squared"]:.3f})')
            self.ax.scatter([calc_result['sample_conc']], [calc_result['sample_mean_area']], color='red', marker='*', s=150, label=f'Sample (Conc: {calc_result["sample_conc"]:.3f})', zorder=6, edgecolors='darkred')
            
            if self.analysis_mode == "lod_iupac" and "lod_conc" in calc_result:
                self.ax.axhline(y=calc_result["lod_signal"], color='orange', linestyle='-.', linewidth=1.5, label=f'LoD Signal')
                self.ax.axhline(y=calc_result["loq_signal"], color='purple', linestyle=':', linewidth=1.5, label=f'LoQ Signal')
                self.ax.axvline(x=calc_result["lod_conc"], color='orange', linestyle='-.', alpha=0.5)
                self.ax.axvline(x=calc_result["loq_conc"], color='purple', linestyle=':', alpha=0.5)
                
            self.ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
            
            max_conc = max(concentrations) if concentrations else 1.0
            x_thresh = max(calc_result.get("loq_conc", max_conc * 0.3) * 1.5, max_conc * 0.3)
            points_in_region = sum(1 for c in concentrations if c <= x_thresh)
            if calc_result['sample_conc'] <= x_thresh: points_in_region += 1
            
            if points_in_region >= 2:
                axins = inset_axes(self.ax, width="45%", height="45%", loc='upper left', borderpad=1.5)
                axins.errorbar(concentrations, means, yerr=stds, fmt='o', color='blue', capsize=3, markersize=4, ecolor='lightblue')
                axins.plot(x_line, calc_result['slope'] * x_line + calc_result['intercept'], 'r--', linewidth=1.5)
                axins.scatter([calc_result['sample_conc']], [calc_result['sample_mean_area']], color='red', marker='*', s=50)
                if self.analysis_mode == "lod_iupac" and "lod_conc" in calc_result:
                    axins.axhline(y=calc_result["lod_signal"], color='orange', linestyle='-.', linewidth=1)
                    axins.axhline(y=calc_result["loq_signal"], color='purple', linestyle=':', linewidth=1)
                    axins.axvline(x=calc_result["lod_conc"], color='orange', linestyle='-.', alpha=0.5)
                    axins.axvline(x=calc_result["loq_conc"], color='purple', linestyle=':', alpha=0.5)
                axins.set_xlim(0, x_thresh * 1.2)
                y_min_inset, y_max_inset = float('inf'), float('-inf')
                for c, m, s in zip(concentrations, means, stds):
                    if c <= x_thresh: y_min_inset = min(y_min_inset, m - s); y_max_inset = max(y_max_inset, m + s)
                if calc_result['sample_conc'] <= x_thresh: y_min_inset = min(y_min_inset, calc_result['sample_mean_area']); y_max_inset = max(y_max_inset, calc_result['sample_mean_area'])
                if y_min_inset == float('inf'): y_min_inset, y_max_inset = 0, 1
                y_range = y_max_inset - y_min_inset
                axins.set_ylim(y_min_inset - 0.15 * y_range, y_max_inset + 0.15 * y_range)
                axins.set_title("Zoom-in: LoD/LoQ Region", fontsize=8)
                axins.tick_params(labelsize=7)
                axins.grid(True, linestyle=':', alpha=0.5)
                
            self.set_chart_title("Step 4: Standard Curve & Concentration")
            self.ax.set_xlabel("Concentration (a.u.)", fontsize=11)
            self.ax.set_ylabel("Net Fluorescence Area", fontsize=11)
            self.ax.grid(True, linestyle=':', alpha=0.7)
            self.fig.tight_layout()
            self.canvas_plot.draw()
        except Exception as e: self.log_status(4, f"Error: {e}", "red"); messagebox.showerror("Error", str(e))

    def open_turbidity_window(self):
        if self.turbidity_window is None or not self.turbidity_window.win.winfo_exists():
            if self.mapping is None: messagebox.showwarning("Warning", "Please complete Step 1 first!"); return
            from gui.turbidity_window import TurbidityWindow
            self.turbidity_window = TurbidityWindow(self.root, self.mapping, self.calibrator)
            if self.cache_step3: self.turbidity_window.update_with_spectrum(self.cache_step3["wls"], self.cache_step3["intensity"])
        else: self.turbidity_window.win.lift(); self.turbidity_window.win.focus_force()

    def show_details_window(self):
        if not self.mapping: messagebox.showinfo("Info", "Please complete at least Step 1 to view details."); return
        DetailsWindow(self.root, self.cache_step1, self.cache_step2, self.cache_step3, self.cache_step4, self.mapping)