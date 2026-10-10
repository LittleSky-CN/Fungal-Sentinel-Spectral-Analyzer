# gui/details_window.py
"""FSSA v1.4.5 Detailed Analysis Report.
- Fixed Figure 4b algorithm: Use Corrected = Raw * SPD / Lamp to avoid division by zero/masked regions.
- Added Figure 4d: Channel weights visualization.
- Clipped all Figure 4 plots to Visible Range (400-700 nm).
- Restored Tab 3 and Tab 4.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages

from core.image_processing import normalize_for_display

# ---- Theme dependencies ----
try:
    from gui import theme as _theme
except Exception:
    _theme = None
try:
    from config import THEME_ENABLED as _THEME_FLAG
except Exception:
    _THEME_FLAG = False
_THEME_ON = bool(_theme is not None and _THEME_FLAG)

VISIBLE_RANGE = (400.0, 700.0)

def _style_fig(fig):
    if _THEME_ON:
        _theme.style_figure(fig)
    return fig

class DetailsWindow:
    def __init__(self, parent, cache1, cache2, cache3, cache4, mapping):
        self.win = tk.Toplevel(parent)
        self.win.title("FSSA Detailed Analysis Report & Parameters (v1.4.5)")
        self.win.geometry("2000x2000")
        if _THEME_ON:
            _theme.tint(self.win)
            
        self.cache1 = cache1 or {}
        self.cache2 = cache2 or {}
        self.cache3 = cache3 or {}
        self.cache4 = cache4 or {}
        self.mapping = mapping

        self._active_canvas = None
        self.notebook = ttk.Notebook(self.win)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig1 = self.fig2 = self.fig3 = self.fig4 = None
        self.fig1_params = self.fig2_params = self.fig3_params = self.fig4_params = None

        self._build_tab1()
        self._build_tab2() # Core fix: SPD Response & Positioning Correction
        self._build_tab3() # Restored
        self._build_tab4() # Restored

        btn_frame = ttk.Frame(self.win)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        figs_to_export = [f for f in (self.fig1, self.fig1_params, self.fig2, self.fig2_params, self.fig3, self.fig3_params, self.fig4, self.fig4_params) if f is not None]
        
        if _THEME_ON:
            ttk.Button(btn_frame, text="📥 Export Full Report (PDF)", style="Magenta.TButton", command=lambda: self.export_pdf_report(figs_to_export)).pack(side=tk.LEFT)
            ttk.Button(btn_frame, text="Close", style="Pale.TButton", command=self.win.destroy).pack(side=tk.RIGHT)
        else:
            ttk.Button(btn_frame, text="📥 Export Full Report (PDF)", command=lambda: self.export_pdf_report(figs_to_export)).pack(side=tk.LEFT)
            ttk.Button(btn_frame, text="Close", command=self.win.destroy).pack(side=tk.RIGHT)

        self.win.bind("<MouseWheel>", self._on_mousewheel)
        self.win.bind("<Button-4>", self._on_mousewheel)
        self.win.bind("<Button-5>", self._on_mousewheel)

    # ================= Helpers =================
    def _ink_line(self): return _theme.CREAM if _THEME_ON else "black"
    def _ch_colors(self):
        if _THEME_ON: return {"red": "#FF5C5C", "green": "#5CFF5C", "blue": "#5CA8FF"}
        return {"red": "red", "green": "green", "blue": "blue"}
    def _c4_palette(self):
        if _THEME_ON:
            m = _theme.MPL
            return {"std": m["std"], "std_err": "#7FF7FF", "fit": m["fit"], "sample": m["sample"], "sample_edge": _theme.BLACK, "lod": m["lod"], "loq": m["loq"]}
        return {"std": "blue", "std_err": "lightblue", "fit": "red", "sample": "red", "sample_edge": "darkred", "lod": "orange", "loq": "purple"}

    def _on_mousewheel(self, event):
        c = self._active_canvas
        if c is None: return
        if getattr(event, "num", None) == 4: c.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5: c.yview_scroll(1, "units")
        else:
            delta = getattr(event, "delta", 0)
            if delta: c.yview_scroll(int(-1 * (delta / 120)), "units")

    def _create_scrollable_frame(self, parent):
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True, pady=5)
        canvas = tk.Canvas(container, highlightthickness=0)
        if _THEME_ON: canvas.configure(bg=_theme.BLACK)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        def _on_canvas_configure(e):
            canvas.itemconfig(win_id, width=e.width)
            canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.bind("<Configure>", _on_canvas_configure)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.bind("<Enter>", lambda e: self._set_active_canvas(canvas))
        canvas.bind("<Leave>", lambda e: self._set_active_canvas(None))
        return canvas, inner

    def _set_active_canvas(self, canvas): self._active_canvas = canvas
    def _embed_figure(self, fig, scroll_canvas, inner):
        _style_fig(fig)
        widget = FigureCanvasTkAgg(fig, master=inner).get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)
        widget.bind("<Enter>", lambda e: self._set_active_canvas(scroll_canvas))
        widget.bind("<Leave>", lambda e: self._set_active_canvas(None))
        return widget

    def _add_param_panel(self, tab, content):
        frame = ttk.LabelFrame(tab, text="Calculated Parameters & Debug Info", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=(0, 8))
        if _THEME_ON:
            txt = tk.Text(frame, height=8, font=_theme.MONO, wrap=tk.WORD, bg=_theme.CREAM, fg=_theme.INK, relief="flat", highlightthickness=1, highlightbackground=_theme.CYAN)
        else:
            txt = tk.Text(frame, height=8, font=("Consolas", 10), wrap=tk.WORD, bg="#f8f9fa")
        txt.pack(fill=tk.X)
        txt.insert("1.0", content)
        txt.config(state=tk.DISABLED)
        return txt

    def _create_params_figure(self, title, params_text):
        fig = Figure(figsize=(15, 6), dpi=100)
        tcol = "black"
        if _THEME_ON:
            fig.patch.set_facecolor(_theme.CREAM)
            tcol = _theme.INK
        else:
            fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20, color=tcol)
        fig.text(0.05, 0.85, params_text, fontsize=11, family="monospace", verticalalignment="top", color=tcol)
        return fig

    def _clip_visible(self, wls, *arrays):
        mask = (wls >= VISIBLE_RANGE[0]) & (wls <= VISIBLE_RANGE[1])
        res = [wls[mask]]
        for arr in arrays:
            res.append(arr[mask])
        return res

    # ================= Tab 1: Wavelength Calibration =================
    def _build_tab1(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Step 1: Wavelength Cal.")
        if not self.cache1:
            ttk.Label(tab, text="Please complete Step 1.", font=("Arial", 12)).pack(pady=50)
            return
        sc, inner = self._create_scrollable_frame(tab)
        ch = self._ch_colors()
        ink = self._ink_line()
        self.fig1 = Figure(figsize=(15, 25), dpi=100)
        rgb = self.cache1["rgb"]
        x0, x1 = self.cache1["x_roi"]
        ax01 = self.fig1.add_subplot(3, 1, 1)
        ax01.imshow(normalize_for_display(rgb))
        ax01.add_patch(Rectangle((x0, 0), x1 - x0, rgb.shape[0], fill=False, linewidth=2, edgecolor='yellow', label='Auto X ROI'))
        for p in self.cache1["peaks"]:
            ax01.axhline(p.y_centroid, linestyle="--", linewidth=1.5, label=f"{p.name} ({p.wavelength_nm}nm)")
        ax01.set_title("Figure 1: Positioning DNG Preview with Auto ROI", fontsize=14)
        ax01.legend(fontsize=8, loc='upper right')
        ax02 = self.fig1.add_subplot(3, 1, 2)
        ys = np.arange(len(self.cache1["profiles"]["red"]))
        for name, prof in self.cache1["profiles"].items():
            ax02.plot(ys, prof, color=ch[name], linewidth=1.5, label=f"{name.capitalize()} Channel")
        for p in self.cache1["peaks"]:
            ax02.axvline(p.y_centroid, linestyle="--", color=ink, label=f"{p.name} Centroid")
            ax02.axvspan(p.lo, p.hi, alpha=0.1, color=ch[p.channel_name])
        ax02.set_title("Figure 2: Automatic 1D Profiles and Dynamic Peak Windows", fontsize=14)
        ax02.set_xlabel("Y Pixel", fontsize=12); ax02.set_ylabel("Intensity", fontsize=12)
        ax02.legend(fontsize=8, ncol=3); ax02.grid(True, linestyle=':', alpha=0.5)
        ax03 = self.fig1.add_subplot(3, 1, 3)
        v = self.cache1["validation"]
        lam_grid = np.linspace(440, 640, 200)
        pred_grid = v.slope_k * lam_grid + v.intercept_b
        ax03.plot(lam_grid, pred_grid, color=ink, linewidth=2, label=f"Fit: p = {v.slope_k:.4f}λ + {v.intercept_b:.2f}")
        ax03.scatter([462.5, 622.5], [self.cache1["peaks"][2].y_centroid, self.cache1["peaks"][0].y_centroid], color=ch["blue"], s=100, label="Train (B, R)", zorder=5)
        ax03.scatter([522.5], [self.cache1["peaks"][1].y_centroid], marker='x', s=150, color=ch["red"], label="Validate (G)", zorder=5)
        ax03.plot([522.5, 522.5], [v.predicted_y, self.cache1["peaks"][1].y_centroid], '--', color=ch["red"], linewidth=2, label=f"Error: {v.wavelength_error_nm:+.2f} nm")
        ax03.set_title("Figure 3: Two-point Fit (B+R) with Third-point Validation (G)", fontsize=14)
        ax03.set_xlabel("Wavelength (nm)", fontsize=12); ax03.set_ylabel("Y Pixel", fontsize=12)
        ax03.invert_yaxis(); ax03.legend(fontsize=9); ax03.grid(True, linestyle=':', alpha=0.5)
        self.fig1.tight_layout()
        self._embed_figure(self.fig1, sc, inner)
        params = (f"[Wavelength Mapping Equation]\n  Pixel (p) = {v.slope_k:.6f} * Wavelength(λ) + {v.intercept_b:.4f}\n\n"
                  f"[Calibration Metadata]\n  Exposure Time: {self.cache1.get('exp_time', float('nan')):.4f} s\n  Auto-detected X ROI: [{x0}, {x1}] (Width: {x1 - x0} px)\n  Positioning Shots: {self.cache1.get('n_pos', 'n/a')}\n\n"
                  f"[Validation Results]\n  Training Points: B (462.5nm), R (622.5nm)\n  Validation Point: G (522.5nm)\n  G-point Residual: {v.residual_px:+.2f} pixels\n  G-point Wavelength Error: {v.wavelength_error_nm:+.3f} nm")
        self._add_param_panel(tab, params)
        self.fig1_params = self._create_params_figure("Step 1: Wavelength Calibration Parameters", params)

    # ================= Tab 2: SPD Response & Correction (CORE FIX) =================
    def _build_tab2(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Step 2: SPD Response")
        if not self.cache2:
            ttk.Label(tab, text="Please complete Step 2.", font=("Arial", 12)).pack(pady=50)
            return
        sc, inner = self._create_scrollable_frame(tab)
        ch = self._ch_colors()
        
        # 1. Response Data (from Step 2)
        wls_resp = self.cache2["wavelengths"]
        r_resp = self.cache2["R"]
        g_resp = self.cache2["G"]
        b_resp = self.cache2["B"]
        if wls_resp[0] > wls_resp[-1]:
            wls_resp, r_resp, g_resp, b_resp = wls_resp[::-1], r_resp[::-1], g_resp[::-1], b_resp[::-1]
        
        # 2. Positioning (LED) Data (from Step 1) - Independent validation data
        pos_data_available = (self.cache1 and "profiles" in self.cache1 and "mapping" in self.cache1)
        if pos_data_available:
            profs = self.cache1["profiles"]
            mapping = self.cache1["mapping"]
            y_pix = np.arange(len(profs["red"]))
            wls_pos = mapping.pixel_to_wavelength(y_pix)
            if wls_pos[0] > wls_pos[-1]:
                wls_pos = wls_pos[::-1]
                r_raw = profs["red"][::-1]
                g_raw = profs["green"][::-1]
                b_raw = profs["blue"][::-1]
            else:
                r_raw = profs["red"]
                g_raw = profs["green"]
                b_raw = profs["blue"]
        else:
            wls_pos, r_raw, g_raw, b_raw = None, None, None, None

        self.fig2 = Figure(figsize=(15, 24), dpi=100)

        # --- Figure 4a: RAW Positioning (Visible Range) ---
        ax4a = self.fig2.add_subplot(4, 1, 1)
        if pos_data_available:
            wls_v, r_v, g_v, b_v = self._clip_visible(wls_pos, r_raw, g_raw, b_raw)
            ax4a.plot(wls_v, r_v, '-', color=ch["red"], label='R channel', linewidth=1.5)
            ax4a.plot(wls_v, g_v, '-', color=ch["green"], label='G channel', linewidth=1.5)
            ax4a.plot(wls_v, b_v, '-', color=ch["blue"], label='B channel', linewidth=1.5)
            ax4a.set_title("Figure 4a: Positioning Light Source - RAW Measured (per channel, absolute units)", fontsize=12, fontweight='bold')
            ax4a.set_ylabel("Linear Intensity (a.u./s)", fontsize=10)
            ax4a.legend(fontsize=8, loc='upper right')
            ax4a.grid(True, linestyle=':', alpha=0.5)
        else:
            ax4a.text(0.5, 0.5, "Step 1 data missing.", ha='center', va='center', transform=ax4a.transAxes)

        # --- Figure 4b: CORRECTED Positioning (CORE ALGORITHM FIX) ---
        # Formula: Corrected = Raw * SPD_true / Lamp
        # This avoids division by zero/masked Response regions.
        ax4b = self.fig2.add_subplot(4, 1, 2)
        if pos_data_available:
            # Get SPD_true and Lamp data from cache_step2
            spd_aligned = self.cache2.get("spd_aligned")
            lamp_r = self.cache2.get("lamp_R")
            lamp_g = self.cache2.get("lamp_G")
            lamp_b = self.cache2.get("lamp_B")
            
            if spd_aligned is not None and lamp_r is not None:
                # Interpolate SPD and Lamp to Positioning wavelengths
                spd_interp = np.interp(wls_pos, wls_resp, spd_aligned)
                lamp_r_interp = np.interp(wls_pos, wls_resp, lamp_r)
                lamp_g_interp = np.interp(wls_pos, wls_resp, lamp_g)
                lamp_b_interp = np.interp(wls_pos, wls_resp, lamp_b)
                
                # Calculate Corrected: Raw * SPD / Lamp
                # Add small epsilon to denominator to avoid division by zero where Lamp is 0
                eps = 1e-6
                r_corr = r_raw * spd_interp / (lamp_r_interp + eps)
                g_corr = g_raw * spd_interp / (lamp_g_interp + eps)
                b_corr = b_raw * spd_interp / (lamp_b_interp + eps)
                
                # Clip to visible range
                wls_v, r_v, g_v, b_v = self._clip_visible(wls_pos, r_corr, g_corr, b_corr)
                ax4b.plot(wls_v, r_v, '-', color=ch["red"], label='R channel', linewidth=1.5)
                ax4b.plot(wls_v, g_v, '-', color=ch["green"], label='G channel', linewidth=1.5)
                ax4b.plot(wls_v, b_v, '-', color=ch["blue"], label='B channel', linewidth=1.5)
                ax4b.set_title("Figure 4b: Positioning Light Source - SPD-CORRECTED (per channel, absolute units)", fontsize=12, fontweight='bold')
                ax4b.set_ylabel("Corrected Intensity (a.u./s)", fontsize=10)
                ax4b.set_xlabel("Wavelength (nm)", fontsize=10)
                ax4b.legend(fontsize=8, loc='upper right')
                ax4b.grid(True, linestyle=':', alpha=0.5)
            else:
                ax4b.text(0.5, 0.5, "SPD/Lamp data missing in cache.", ha='center', va='center', transform=ax4b.transAxes)
        else:
            ax4b.text(0.5, 0.5, "Step 1 data missing.", ha='center', va='center', transform=ax4b.transAxes)

        # --- Figure 4c: Camera Response (Visible Range) ---
        ax4c = self.fig2.add_subplot(4, 1, 3)
        wls_v, r_v, g_v, b_v = self._clip_visible(wls_resp, r_resp, g_resp, b_resp)
        ax4c.plot(wls_v, r_v, '-', color=ch["red"], label='R channel', linewidth=2)
        ax4c.plot(wls_v, g_v, '-', color=ch["green"], label='G channel', linewidth=2)
        ax4c.plot(wls_v, b_v, '-', color=ch["blue"], label='B channel', linewidth=2)
        ax4c.set_title("Figure 4c: Camera Absolute Response = Lamp_c / SPD_true (masked, NO SG, NO normalization)", fontsize=12, fontweight='bold')
        ax4c.set_ylabel("Sensitivity (a.u./SPD unit)", fontsize=10)
        ax4c.set_xlabel("Wavelength (nm)", fontsize=10)
        ax4c.legend(fontsize=8, loc='upper right')
        ax4c.grid(True, linestyle=':', alpha=0.5)

        # --- Figure 4d: Channel Weights (Visible Range) ---
        ax4d = self.fig2.add_subplot(4, 1, 4)
        wls_plot = np.linspace(420, 660, 500)
        h1, h2 = 483.0, 577.0 # Handoff points
        k = 15.0 # Sigmoid steepness
        
        w_B_raw = 1.0 / (1.0 + np.exp((wls_plot - h1) / k))
        w_R_raw = 1.0 / (1.0 + np.exp((h2 - wls_plot) / k))
        w_G_raw = (1.0 / (1.0 + np.exp((h1 - wls_plot) / k))) * (1.0 / (1.0 + np.exp((wls_plot - h2) / k))) * 1.5
        
        w_sum = w_B_raw + w_G_raw + w_R_raw
        w_B = w_B_raw / w_sum
        w_G = w_G_raw / w_sum
        w_R = w_R_raw / w_sum
        
        mask_vis = (wls_plot >= VISIBLE_RANGE[0]) & (wls_plot <= VISIBLE_RANGE[1])
        wls_v = wls_plot[mask_vis]
        
        ax4d.plot(wls_v, w_B[mask_vis], '-', color=ch["blue"], label='w_B(λ)', linewidth=2.5)
        ax4d.plot(wls_v, w_G[mask_vis], '-', color=ch["green"], label='w_G(λ)', linewidth=2.5)
        ax4d.plot(wls_v, w_R[mask_vis], '-', color=ch["red"], label='w_R(λ)', linewidth=2.5)
        ax4d.axvline(h1, color='gray', linestyle=':', label=f'B→G handoff {h1} nm')
        ax4d.axvline(h2, color='gray', linestyle='--', label=f'G→R handoff {h2} nm')
        ax4d.set_title("Figure 4d: Per-wavelength channel weights (a+b+c=1, continuous handoff)", fontsize=12, fontweight='bold')
        ax4d.set_ylabel("Weight", fontsize=10)
        ax4d.set_xlabel("Wavelength (nm)", fontsize=10)
        ax4d.legend(fontsize=9, loc='upper right')
        ax4d.grid(True, linestyle=':', alpha=0.5)
        ax4d.set_ylim(0, 1.05)

        self.fig2.tight_layout()
        self._embed_figure(self.fig2, sc, inner)

        # Parameters Text
        n_spd = self.cache2.get("n_spd_shots", "N/A")
        spd_ratio = self.cache2.get("spd_threshold_ratio", 0.05)
        total_resp = r_resp + g_resp + b_resp
        valid_mask = total_resp > 0
        eff_band = f"{wls_resp[valid_mask][0]:.1f} - {wls_resp[valid_mask][-1]:.1f} nm" if np.any(valid_mask) else "N/A"
        
        # Calculate peaks for text
        r_resp_peak_wl = wls_resp[np.argmax(r_resp)] if len(wls_resp) else 0
        
        params = (f"Step 2: SPD Response Parameters\n\n"
                  f"[Method Notes]\nCorrection is demonstrated on INDEPENDENT data (positioning LEDs), NOT on the SPD lamp itself -> avoids circular reasoning.\nChannels are processed SEPARATELY (never summed).\nNO Savitzky-Golay smoothing; NO self-normalization (absolute units).\n\n"
                  f"[Formulas]\nresp_c(λ) = Lamp_c(λ) / SPD_true(λ)  (dual-masked)\nCorrected_c(λ) = Pos_c(λ) * SPD_true(λ) / Lamp_c(λ)\n\n"
                  f"[Effective Band] {eff_band} | masked pixels: {len(wls_resp) - np.sum(valid_mask)}/{len(wls_resp)}\n"
                  f"[SPD Threshold Ratio] {spd_ratio}\n[Shots Averaged] SPD lamp: {n_spd} | Positioning: see Step 1\n\n"
                  f"[Per-channel Response Peaks (absolute)]\n"
                  f"R: {np.max(r_resp):.0f} a.u./SPD @ {r_resp_peak_wl:.1f} nm\n"
                  f"G: {np.max(g_resp):.0f} a.u./SPD @ {wls_resp[np.argmax(g_resp)]:.1f} nm\n"
                  f"B: {np.max(b_resp):.0f} a.u./SPD @ {wls_resp[np.argmax(b_resp)]:.1f} nm\n\n"
                  f"[Channel Weights Handoff]\nB→G: {h1} nm | G→R: {h2} nm\nWeights sum to 1.0 at every wavelength.")
        self._add_param_panel(tab, params)
        self.fig2_params = self._create_params_figure("Step 2: SPD Response Parameters", params)

    # ================= Tab 3: Sample Spectrum (Restored) =================
    def _build_tab3(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Step 3: Sample Spectrum")
        if not self.cache3:
            ttk.Label(tab, text="Please complete Step 3.", font=("Arial", 12)).pack(pady=50)
            return
        sc, inner = self._create_scrollable_frame(tab)
        ink = self._ink_line()
        ch = self._ch_colors()
        self.fig3 = Figure(figsize=(15, 8), dpi=100)
        ax3 = self.fig3.add_subplot(111)
        fluor = self.cache3["fluor"]
        wls, intensity, mask = self.cache3["wls"], self.cache3["intensity"], self.cache3["mask"]
        if wls[0] > wls[-1]:
            wls, intensity, mask = wls[::-1], intensity[::-1], mask[::-1]
        wls, intensity, mask = self._clip_visible(wls, intensity, mask)
        
        ax3.plot(wls, intensity, '-', color=ink, linewidth=2, label='Net Spectrum')
        ax3.fill_between(wls, intensity, where=mask, color=fluor['color'], alpha=0.4, label='Integration')
        ax3.axvline(fluor['peak_wl'], '--', color=ch["red"], label='Target Peak')
        ax3.set_title("Figure: Background-Subtracted & SPD-Corrected Spectrum (Visible Band Only)", fontsize=14)
        ax3.set_xlabel("Wavelength (nm)", fontsize=12); ax3.set_ylabel("True Intensity", fontsize=12)
        ax3.legend(fontsize=10); ax3.grid(True, linestyle=':', alpha=0.5)
        self.fig3.tight_layout()
        self._embed_figure(self.fig3, sc, inner)
        
        features = self.cache3.get("features", {})
        spd_status = "APPLIED (Absolute Sensitivity)" if self.cache3.get("spd_applied") else "SKIPPED (Raw RGB)"
        fluor_name = self.cache3.get("fluor_name") or ("Turbidity Mode (Green Channel)" if fluor.get("peak_wl") == 550 else "Custom/Unknown")
        params = (f"[Measurement Status]\n  Target: {fluor_name}\n  Channel: {fluor.get('channel', 'n/a')} | Target WL: {fluor.get('peak_wl', 0)} nm | FWHM Window: {fluor.get('fwhm', 0)} nm\n  SPD Correction: {spd_status}\n  Display Band: [{VISIBLE_RANGE[0]:.0f}, {VISIBLE_RANGE[1]:.0f}] nm\n\n"
                  f"[Microplate Reader Style Parameters]\n  Peak Wavelength: {features.get('peak_wl', 0):.2f} nm\n  Peak Intensity: {features.get('peak_intensity', 0):.2f} a.u.\n  FWHM (Full Width Half Max): {features.get('fwhm', 0):.2f} nm\n  AUC (Area Under Curve): {features.get('auc', 0):.2f}\n  Centroid (Center of Mass): {features.get('centroid', 0):.2f} nm\n\n"
                  f"[Integration]\n  Net Area (Mean): {self.cache3.get('mean_area', 0):.2f} | Peak in Window: {self.cache3.get('peak', 0):.2f}")
        self._add_param_panel(tab, params)
        self.fig3_params = self._create_params_figure("Step 3: Sample Spectrum Parameters", params)

    # ================= Tab 4: Concentration (Restored) =================
    def _build_tab4(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Step 4: Concentration")
        if not self.cache4:
            ttk.Label(tab, text="Please complete Step 4.", font=("Arial", 12)).pack(pady=50)
            return
        sc, inner = self._create_scrollable_frame(tab)
        pc = self._c4_palette()
        self.fig4 = Figure(figsize=(15, 8), dpi=100)
        ax4 = self.fig4.add_subplot(111)
        c4 = self.cache4
        r2 = c4.get("r_squared", c4.get("r_value", 0) ** 2)
        ax4.errorbar(c4["concentrations"], c4["means"], yerr=c4["stds"], fmt='o', color=pc["std"], capsize=5, label='Standards', ecolor=pc["std_err"])
        x_line = np.linspace(min(c4["concentrations"]) * 0.8, max(c4["concentrations"]) * 1.2, 100)
        ax4.plot(x_line, c4["slope"] * x_line + c4["intercept"], '--', color=pc["fit"], linewidth=2, label=f'Fit (R²={r2:.4f})')
        ax4.scatter([c4["sample_conc"]], [c4["sample_mean_area"]], color=pc["sample"], marker='*', s=150, label=f'Sample (Conc: {c4["sample_conc"]:.3f})', edgecolors=pc["sample_edge"])
        if c4.get("mode") == "lod_iupac" and "lod_conc" in c4:
            if "lod_fullrange" not in c4:
                ax4.axhline(y=c4["lod_signal"], color=pc["lod"], linestyle='-.', label='LoD Signal')
                ax4.axhline(y=c4["loq_signal"], color=pc["loq"], linestyle=':', label='LoQ Signal')
            ax4.axvline(x=c4["lod_conc"], color=pc["lod"], linestyle='-.', alpha=0.6, label='LoD (headline)')
            ax4.axvline(x=c4["loq_conc"], color=pc["loq"], linestyle=':', alpha=0.6, label='LoQ (headline)')
        ylab = "Unmixed Net Area" if c4.get("metric_mode") == "unmixing" else "Net Area"
        ax4.set_title("Figure: Standard Curve & Concentration Quantification", fontsize=14)
        ax4.set_xlabel("Concentration (a.u.)", fontsize=12); ax4.set_ylabel(ylab, fontsize=12)
        ax4.legend(loc='upper right', fontsize=10); ax4.grid(True, linestyle=':', alpha=0.7)
        self.fig4.tight_layout()
        self._embed_figure(self.fig4, sc, inner)
        
        reg_method = str(c4.get("regression_method", "OLS")).upper()
        params = (f"[Regression Analysis]\n  Method: {reg_method}\n  Equation: I = {c4['slope']:.6f} * C + ({c4['intercept']:.6f})\n  R-squared (R²): {r2:.6f}\n")
        if c4.get("metric_mode"): params += f"  Quantification Metric: {c4['metric_mode'].upper()}\n"
        if c4.get("t_stat") is not None: params += f"  Slope t-statistic: {c4['t_stat']:.2f} (require >= 3)\n"
        params += (f"\n[Sample Result]\n  Sample Net Area: {c4['sample_mean_area']:.2f}\n  PREDICTED CONCENTRATION: {c4['sample_conc']:.4f} (a.u.)\n")
        u = c4.get("unmix")
        if u:
            lo, hi = u["window"]
            params += (f"\n[Spectral Unmixing: sample = α·E + s·Blank + k]\n  Window: [{lo:.1f}, {hi:.1f}] nm | Scatter peak: {c4.get('scatter_peak_wl', float('nan')):.1f} nm\n  Reference E: std conc = {u['ref_conc']} | AUC(E): {u['auc_E']:.2f}\n")
            for pr in u.get("per_row", []):
                params += f"    C={pr['conc']:<10g} α={pr['alpha']:.4f}  s={pr['s']:.3f}  k={pr['k']:.3f}  R²={pr['r2']:.4f}\n"
            params += f"    SAMPLE     α={u['alpha']:.4f}  s={u['u_s']:.3f}  k={u['u_k']:.3f}\n"
            if u.get("lod_range"): params += f"  LoD window-sensitivity range: [{u['lod_range'][0]:.3g}, {u['lod_range'][1]:.3g}] (a.u.)\n"
        blocks = c4.get("blocks")
        if blocks:
            params += "\n[Exposure-block LoD selection (ICH Q2: n>=3, t>=3)]\n"
            for b in blocks:
                if b["n"] < 3: params += f"  @{b['exp']:.3f}s: n={b['n']} (skipped, <3 levels)\n"
                else: params += f"  @{b['exp']:.3f}s: n={b['n']} slope={b['slope']:.4g} R²={b['r2']:.4f} Sy/x={b['syx']:.4g} t={b['t']:.1f} LoD={b['lod']:.4g}\n"
        if c4.get("mode") == "lod_iupac" and "lod_conc" in c4:
            params += (f"\n[IUPAC Limit Analysis (headline)]\n  Source: {c4.get('lod_source', 'full-range curve')}\n  LIMIT OF DETECTION (LoD, 3.3σ): {c4['lod_conc']:.4f} (a.u.)\n  LIMIT OF QUANTIFICATION (LoQ, 10σ): {c4['loq_conc']:.4f} (a.u.)\n")
            if "lod_fullrange" in c4: params += f"  [reference] full-range LoD/LoQ: {c4['lod_fullrange']:.4g} / {c4['loq_fullrange']:.4g}\n"
        self._add_param_panel(tab, params)
        self.fig4_params = self._create_params_figure("Step 4: Concentration Parameters", params)

    # ================= PDF Export =================
    def export_pdf_report(self, figures):
        if not figures: return
        file_path = filedialog.asksaveasfilename(title="Export PDF Report", defaultextension=".pdf", filetypes=[("PDF Document", "*.pdf")])
        if file_path:
            try:
                with PdfPages(file_path) as pdf:
                    for fig in figures: pdf.savefig(fig, bbox_inches='tight')
                messagebox.showinfo("Success", f"Full Report (Charts + Parameters) saved to:\n{file_path}")
            except Exception as e: messagebox.showerror("Error", str(e))