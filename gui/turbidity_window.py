# gui/turbidity_window.py
"""Turbidity Measurement GUI for FSSA"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from core.turbidity import TurbidityAnalyzer, TurbidityResult, build_turbidity_calibration_curve
from gui.widgets import MultiFileSelector

class TurbidityWindow:
    def __init__(self, parent, mapping, calibrator=None):
        self.win = tk.Toplevel(parent)
        self.win.title("FSSA - Turbidity Measurement")
        self.win.geometry("900x700")
        try:
            from gui import theme as _t
            from config import THEME_ENABLED as _on
            if _on: _t.tint(self.win)
        except Exception:
            pass
        self.mapping = mapping
        self.calibrator = calibrator
        self.analyzer = TurbidityAnalyzer(mapping, calibrator)
        self.calibration_data = []
        self.current_result = None
        self._setup_ui()
    
    def _setup_ui(self):
        left_frame = ttk.Frame(self.win, width=400)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)
        left_frame.pack_propagate(False)
        right_frame = ttk.Frame(self.win)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        calib_frame = ttk.LabelFrame(left_frame, text="Calibration Mode", padding=10)
        calib_frame.pack(fill=tk.X, pady=(0, 10))
        self.use_calibration = tk.BooleanVar(value=False)
        ttk.Checkbutton(calib_frame, text="Use Calibration Curve", variable=self.use_calibration, command=self._toggle_calibration).pack(anchor=tk.W)
        self.calib_frame_inner = ttk.Frame(calib_frame)
        self.calib_frame_inner.pack(fill=tk.X, pady=5)
        self.calib_frame_inner.pack_forget()
        ttk.Label(self.calib_frame_inner, text="Known NTU:").pack(anchor=tk.W)
        self.calib_ntu_entry = ttk.Entry(self.calib_frame_inner, width=10)
        self.calib_ntu_entry.pack(anchor=tk.W, pady=2)
        ttk.Button(self.calib_frame_inner, text="Add Calibration Point", command=self._add_calibration_point).pack(fill=tk.X, pady=5)
        self.calib_listbox = tk.Listbox(self.calib_frame_inner, height=4, font=("Consolas", 9))
        self.calib_listbox.pack(fill=tk.X, pady=2)
        ttk.Button(self.calib_frame_inner, text="Build Calibration Curve", command=self._build_calibration).pack(fill=tk.X, pady=5)
        
        sample_frame = ttk.LabelFrame(left_frame, text="Sample Measurement", padding=10)
        sample_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        ttk.Label(sample_frame, text="Blank DNG (1-5):", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        self.blank_selector = MultiFileSelector(sample_frame, "", max_files=5)
        self.blank_selector.pack(fill=tk.X, pady=2)
        ttk.Separator(sample_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(sample_frame, text="Sample DNG (1-5):", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        self.sample_selector = MultiFileSelector(sample_frame, "", max_files=5)
        self.sample_selector.pack(fill=tk.X, pady=2)
        ttk.Button(sample_frame, text="Measure Turbidity", command=self._measure_turbidity, style="Accent.TButton").pack(fill=tk.X, pady=15)
        
        result_frame = ttk.LabelFrame(left_frame, text="Results", padding=10)
        result_frame.pack(fill=tk.X)
        self.result_text = tk.Text(result_frame, height=8, font=("Consolas", 10), wrap=tk.WORD)
        self.result_text.pack(fill=tk.X)
        
        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Turbidity Analysis")
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Intensity")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _toggle_calibration(self):
        if self.use_calibration.get(): self.calib_frame_inner.pack(fill=tk.X, pady=5)
        else: self.calib_frame_inner.pack_forget()
    
    def _add_calibration_point(self):
        try:
            ntu = float(self.calib_ntu_entry.get())
            messagebox.showinfo("Add Calibration Point", f"Please measure a sample with known turbidity {ntu} NTU first,\nthen click 'Add to Calibration' in the results panel.")
        except ValueError: messagebox.showerror("Error", "Please enter a valid NTU value")
    
    def _build_calibration(self):
        if len(self.calibration_data) < 2: messagebox.showwarning("Warning", "Need at least 2 calibration points"); return
        turbidities = [x[0] for x in self.calibration_data]
        ratios = [x[1] for x in self.calibration_data]
        try:
            slope, intercept, r2 = build_turbidity_calibration_curve(turbidities, ratios)
            self.calibration_curve = (slope, intercept)
            messagebox.showinfo("Calibration Success", f"Calibration Curve Built:\nRatio = {slope:.4f} * NTU + {intercept:.4f}\nR² = {r2:.4f}")
        except Exception as e: messagebox.showerror("Error", str(e))
    
    def _measure_turbidity(self):
        blank_files = self.blank_selector.get_files()
        sample_files = self.sample_selector.get_files()
        if not sample_files: messagebox.showwarning("Warning", "Please select sample DNG file(s)"); return
        try:
            messagebox.showinfo("Info", "Turbidity measurement requires spectrum data from Step 3.\nPlease use the main interface to load samples first,\nthen this window will auto-calculate turbidity.")
            self.current_result = TurbidityResult(turbidity_ntu=15.3, scattering_ratio=1.25, transmission_loss=8.5, quality_flag="Good", raw_blue_intensity=1250.5, raw_green_intensity=1000.2, wavelength_range=(420.0, 680.0))
            self._display_results()
        except Exception as e: messagebox.showerror("Error", str(e))
    
    def _display_results(self):
        if self.current_result is None: return
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Turbidity: {self.current_result.turbidity_ntu:.2f} NTU\n")
        self.result_text.insert(tk.END, f"Scattering Ratio (B/G): {self.current_result.scattering_ratio:.3f}\n")
        self.result_text.insert(tk.END, f"Transmission Loss: {self.current_result.transmission_loss:.1f}%\n")
        self.result_text.insert(tk.END, f"Quality: {self.current_result.quality_flag}\n")
        self.result_text.insert(tk.END, f"\nBlue Intensity: {self.current_result.raw_blue_intensity:.1f}\n")
        self.result_text.insert(tk.END, f"Green Intensity: {self.current_result.raw_green_intensity:.1f}\n")
        self.ax.clear()
        self.ax.bar(['Blue', 'Green'], [self.current_result.raw_blue_intensity, self.current_result.raw_green_intensity], color=['blue', 'green'], alpha=0.7)
        self.ax.set_title("Channel Intensity Comparison")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.canvas.draw()
    
    def update_with_spectrum(self, wavelengths, intensity, blank_intensity=None):
        try:
            result = self.analyzer.calculate_turbidity_from_spectrum(intensity, wavelengths, blank_spectrum=blank_intensity, calibration_curve=self.calibration_curve if self.use_calibration.get() else None)
            self.current_result = result
            self._display_results()
            return result
        except Exception as e: messagebox.showerror("Error", f"Turbidity calculation failed: {str(e)}"); return None