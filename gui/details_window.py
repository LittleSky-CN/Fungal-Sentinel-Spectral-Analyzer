# gui/details_window.py
"""Detailed analysis report window with Parameter Panels and Scrollbars (v1.4.3)."""
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
from core.fusion import (compute_channel_weights, fuse_spectra,
                         find_handoff_wavelength, robustness_stats, weight_at)

class DetailsWindow:
    def __init__(self, parent, cache1, cache2, cache3, cache4, mapping):
        self.win = tk.Toplevel(parent)
        self.win.title("FSSA Detailed Analysis Report & Parameters (v1.4.3)")
        self.win.geometry("2000x2000")
        self.cache1 = cache1
        self.cache2 = cache2
        self.cache3 = cache3
        self.cache4 = cache4
        self.mapping = mapping
        self.export_figs = []
        self.export_param_figs = []
        self.notebook = ttk.Notebook(self.win)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self._build_tab1()
        self._build_tab2()
        self._build_tab3()
        self._build_tab4()
        btn_frame = ttk.Frame(self.win)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_frame, text="📥 Export Full Report (PDF)", command=self.export_pdf_report).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Close", command=self.win.destroy).pack(side=tk.RIGHT)

    def _create_scrollable_chart_area(self, parent_frame):
        canvas = tk.Canvas(parent_frame)
        scrollbar_y = ttk.Scrollbar(parent_frame, orient="vertical", command=canvas.yview)
        scrollbar_x = ttk.Scrollbar(parent_frame, orient="horizontal", command=canvas.xview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar_y.pack(side="right", fill="y")
        scrollbar_x.pack(side="bottom", fill="x")
        def _on_mousewheel(event):
            if event.num == 4: canvas.yview_scroll(-1, "units")
            elif event.num == 5: canvas.yview_scroll(1, "units")
            else: canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", _on_mousewheel)
        canvas.bind_all("<Button-5>", _on_mousewheel)
        return scrollable_frame

    def _create_tab_layout(self, title):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=title)
        chart_container = ttk.Frame(tab)
        chart_container.pack(fill=tk.BOTH, expand=True, pady=5)
        scrollable_chart_frame = self._create_scrollable_chart_area(chart_container)
        param_frame = ttk.LabelFrame(tab, text="Calculated Parameters & Debug Info", padding=5)
        param_frame.pack(fill=tk.X, pady=5)
        param_text = tk.Text(param_frame, height=14, font=("Consolas", 10), wrap=tk.WORD, bg="#f8f9fa")
        param_text.pack(fill=tk.X)
        param_text.config(state=tk.DISABLED)
        return tab, scrollable_chart_frame, param_text

    def _update_params(self, text_widget, content):
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)

    def _create_text_figure(self, title, text_content):
        fig = Figure(figsize=(8.5, 11), dpi=100)
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=30)
        fig.text(0.1, 0.9, text_content, fontsize=10, family='monospace', verticalalignment='top', wrap=True)
        return fig

    @staticmethod
    def _peak_in_band(wls, arr, band):
        w, a = wls[band], np.nan_to_num(arr[band], nan=-np.inf)
        if len(a) == 0 or not np.any(np.isfinite(a)): return 0.0, 0.0
        i = int(np.argmax(a))
        return float(w[i]), float(a[i])

    def _build_tab1(self):
        tab, chart_frame, param_text = self._create_tab_layout("Step 1: Wavelength Cal.")
        if not self.cache1:
            ttk.Label(tab, text="Please complete Step 1.", font=("Arial", 12)).pack(pady=50)
            return
        self.fig1 = Figure(figsize=(12, 18), dpi=100)
        rgb = self.cache1["rgb"]; x0, x1 = self.cache1["x_roi"]; v = self.cache1["validation"]
        ax01 = self.fig1.add_subplot(3, 1, 1)
        ax01.imshow(normalize_for_display(rgb))
        ax01.add_patch(Rectangle((x0, 0), x1 - x0, rgb.shape[0], fill=False, linewidth=2, edgecolor='yellow', label='Auto X ROI'))
        for p in self.cache1["peaks"]: ax01.axhline(p.y_centroid, linestyle="--", linewidth=1.5, label=f"{p.name} ({p.wavelength_nm}nm)")
        ax01.set_title("Figure 1: Positioning DNG Preview with Auto ROI", fontsize=14); ax01.legend(fontsize=9, loc='upper right')
        ax02 = self.fig1.add_subplot(3, 1, 2)
        ys = np.arange(len(self.cache1["profiles"]["red"]))
        colors = {"red": "red", "green": "green", "blue": "blue"}
        for name, prof in self.cache1["profiles"].items(): ax02.plot(ys, prof, color=colors[name], label=f"{name.capitalize()} Channel", linewidth=1.5)
        for p in self.cache1["peaks"]:
            ax02.axvline(p.y_centroid, linestyle="--", color="black", label=f"{p.name} Centroid")
            ax02.axvspan(p.lo, p.hi, alpha=0.15, color=colors[p.channel_name])
        ax02.set_title("Figure 2: Automatic 1D Profiles and Dynamic Peak Windows", fontsize=14)
        ax02.set_xlabel("Y Pixel", fontsize=12); ax02.legend(fontsize=9, ncol=3, loc='upper right'); ax02.grid(True, linestyle=':', alpha=0.5)
        ax03 = self.fig1.add_subplot(3, 1, 3)
        lam_grid = np.linspace(440, 640, 200)
        ax03.plot(lam_grid, v.slope_k * lam_grid + v.intercept_b, 'k-', linewidth=2, label=f"Fit: p = {v.slope_k:.4f}λ + {v.intercept_b:.2f}")
        ax03.scatter([462.5, 622.5], [self.cache1["peaks"][2].y_centroid, self.cache1["peaks"][0].y_centroid], color='blue', s=100, label="Train (B, R)", zorder=5)
        ax03.scatter([522.5], [self.cache1["peaks"][1].y_centroid], marker='x', s=150, color='red', label="Validate (G)", zorder=5)
        ax03.plot([522.5, 522.5], [v.predicted_y, self.cache1["peaks"][1].y_centroid], 'r--', linewidth=2, label=f"Error: {v.wavelength_error_nm:+.2f} nm")
        ax03.set_title("Figure 3: Two-point Fit (B+R) with Third-point Validation (G)", fontsize=14)
        ax03.set_xlabel("Wavelength (nm)", fontsize=12); ax03.set_ylabel("Y Pixel", fontsize=12); ax03.invert_yaxis(); ax03.legend(fontsize=10); ax03.grid(True, linestyle=':', alpha=0.5)
        self.fig1.tight_layout(pad=3.0)
        FigureCanvasTkAgg(self.fig1, master=chart_frame).get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.export_figs.append(self.fig1)
        params = (
            f"[Wavelength Mapping Equation]\n  Pixel (p) = {v.slope_k:.6f} * Wavelength(λ) + {v.intercept_b:.4f}\n\n"
            f"[Calibration Metadata]\n  Exposure Time: {self.cache1['exp_time']:.4f} s\n  Auto-detected X ROI: [{x0}, {x1}] (Width: {x1-x0} px)\n\n"
            f"[Validation Results]\n  Training Points: B (462.5nm), R (622.5nm)\n  Validation Point: G (522.5nm)\n"
            f"  G-point Residual: {v.residual_px:+.2f} pixels\n  G-point Wavelength Error: {v.wavelength_error_nm:+.3f} nm"
        )
        self._update_params(param_text, params)
        self.export_param_figs.append(self._create_text_figure("Step 1: Wavelength Calibration Parameters", params))

    def _build_tab2(self):
        tab, chart_frame, param_text = self._create_tab_layout("Step 2: SPD Response")
        if not self.cache2:
            ttk.Label(tab, text="Please complete Step 2.", font=("Arial", 12)).pack(pady=50)
            return
        wls = self.cache2["wavelengths"]
        r_resp, g_resp, b_resp = self.cache2["R"], self.cache2["G"], self.cache2["B"]
        if wls[0] > wls[-1]:
            wls, r_resp, g_resp, b_resp = wls[::-1], r_resp[::-1], g_resp[::-1], b_resp[::-1]
        valid_mask = (r_resp > 0) | (g_resp > 0) | (b_resp > 0)
        lo, hi = (float(wls[valid_mask][0]), float(wls[valid_mask][-1])) if np.any(valid_mask) else (400.0, 700.0)
        margin = (hi - lo) * 0.05
        band = (wls >= lo) & (wls <= hi)

        # 逐通道权重（a+b+c=1，连续接力，纯逐点运算）
        w_R, w_G, w_B = compute_channel_weights(r_resp, g_resp, b_resp)

        # 定位光源逐通道谱（独立数据，避免循环论证）
        has_pos = bool(self.cache1) and ("profiles" in self.cache1) and self.mapping is not None
        lamp_avail = ("lamp_R" in self.cache2) and ("spd_aligned" in self.cache2) and not self.cache2.get("loaded_from_file", False)
        pos, corr = {}, {}
        if has_pos:
            prof = self.cache1["profiles"]
            for c, k in (("R", "red"), ("G", "green"), ("B", "blue")):
                a = np.asarray(prof[k], dtype=np.float64).copy()
                if self.mapping.slope_k < 0: a = a[::-1]
                pos[c] = a if len(a) == len(wls) else None
            if any(v is None for v in pos.values()): has_pos = False
        full = has_pos and lamp_avail
        if full:
            for c, resp in (("R", r_resp), ("G", g_resp), ("B", b_resp)):
                idx = resp > 0
                cur = np.full(len(wls), np.nan)
                cur[idx] = pos[c][idx] / resp[idx]
                corr[c] = cur
            fused = fuse_spectra(corr["R"], corr["G"], corr["B"], w_R, w_G, w_B, valid=valid_mask)
            hand_BG = find_handoff_wavelength(wls, w_B, w_G)
            hand_GR = find_handoff_wavelength(wls, w_G, w_R)
            stats = robustness_stats(wls, fused, corr, {"R": w_R, "G": w_G, "B": w_B})
        else:
            fused = None; hand_BG = hand_GR = None; stats = {}

        n_sub = 4 if full else 1
        self.fig2 = Figure(figsize=(12, 5.0 * n_sub), dpi=100)
        ccolors = {"R": "red", "G": "green", "B": "blue"}

        if full:
            axA = self.fig2.add_subplot(n_sub, 1, 1)
            for c in ("R", "G", "B"):
                axA.plot(wls[band], pos[c][band], color=ccolors[c], linewidth=1.8, label=f'{c} channel')
            axA.set_title("Figure 4a: Positioning Light Source - RAW Measured (per channel, absolute units)", fontsize=13)
            axA.set_ylabel("Linear Intensity (a.u./s)", fontsize=11); axA.set_xlim(lo - margin, hi + margin)
            axA.legend(fontsize=9); axA.grid(True, linestyle=':', alpha=0.5)

            axB = self.fig2.add_subplot(n_sub, 1, 2)
            for c in ("R", "G", "B"):
                axB.plot(wls[band], corr[c][band], color=ccolors[c], linewidth=1.2, alpha=0.8, label=f'{c} channel')
            axB.plot(wls[band], fused[band], color='black', linewidth=2.6, label='FUSED  F(λ)=Σ w_c(λ)·C_c(λ)')
            axB.set_title("Figure 4b: SPD-CORRECTED per channel + FUSED single curve (absolute units)", fontsize=13)
            axB.set_ylabel("Corrected Intensity (a.u./s)", fontsize=11); axB.set_xlim(lo - margin, hi + margin)
            axB.legend(fontsize=9); axB.grid(True, linestyle=':', alpha=0.5)

            axC = self.fig2.add_subplot(n_sub, 1, 3)
            axC.plot(wls[band], w_B[band], color='blue', linewidth=4, alpha=0.75, label='w_B(λ)')
            axC.plot(wls[band], w_G[band], color='green', linewidth=4, alpha=0.75, label='w_G(λ)')
            axC.plot(wls[band], w_R[band], color='red', linewidth=4, alpha=0.75, label='w_R(λ)')
            if hand_BG: axC.axvline(hand_BG, color='gray', linestyle=':', linewidth=1.5, label=f'B→G handoff {hand_BG:.0f} nm')
            if hand_GR: axC.axvline(hand_GR, color='gray', linestyle='--', linewidth=1.5, label=f'G→R handoff {hand_GR:.0f} nm')
            axC.set_title("Figure 4c: Per-wavelength channel weights (a+b+c=1, continuous handoff)", fontsize=13)
            axC.set_ylabel("Weight", fontsize=11); axC.set_ylim(-0.05, 1.05); axC.set_xlim(lo - margin, hi + margin)
            axC.legend(fontsize=8, ncol=2); axC.grid(True, linestyle=':', alpha=0.5)
            axD = self.fig2.add_subplot(n_sub, 1, 4)
        else:
            axD = self.fig2.add_subplot(1, 1, 1)

        for c, resp in (("R", r_resp), ("G", g_resp), ("B", b_resp)):
            axD.plot(wls[band], resp[band], color=ccolors[c], linewidth=1.8, label=f'{c} channel')
        axD.set_title("Figure 4d: Camera Absolute Response = Lamp_c / SPD_true (masked, NO SG, NO normalization)", fontsize=13)
        axD.set_xlabel("Wavelength (nm)", fontsize=11); axD.set_ylabel("Sensitivity (a.u. / SPD unit)", fontsize=11)
        axD.set_xlim(lo - margin, hi + margin); axD.legend(fontsize=9); axD.grid(True, linestyle=':', alpha=0.5)

        self.fig2.tight_layout(pad=3.0)
        FigureCanvasTkAgg(self.fig2, master=chart_frame).get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.export_figs.append(self.fig2)

        # ---------------- 参数面板 ----------------
        lines = []
        lines.append("[Method Notes]")
        lines.append("  Correction demonstrated on INDEPENDENT data (positioning LEDs): no circular reasoning.")
        lines.append("  Channels processed SEPARATELY, then fused with per-wavelength weights (a+b+c=1).")
        lines.append("  NO Savitzky-Golay / moving-average: scalar-per-wavelength ops only.")
        lines.append("  ROI column averaging = repeats at IDENTICAL λ (keeps spectral resolution).")
        lines.append("")
        lines.append("[Formulas]")
        lines.append("  resp_c(λ)      = Lamp_c(λ) / SPD_true(λ)                      (masked)")
        lines.append("  C_c(λ)         = Pos_c(λ) / resp_c(λ)                          (corrected)")
        lines.append("  w_c(λ)         = resp_c(λ) / Σ_c resp_c(λ)                     (weights, Σw=1)")
        lines.append("  F(λ)           = w_R·C_R + w_G·C_G + w_B·C_B                   (fused, continuous)")
        lines.append("")
        lines.append(f"[Effective Band] {lo:.1f} - {hi:.1f} nm | masked pixels: {int(np.sum(valid_mask))} / {len(wls)}")
        lines.append(f"[SPD Threshold Ratio] {self.cache2.get('spd_threshold_ratio', 0.1)}")
        lines.append("")
        lines.append("[Channel Handoff (weight crossover)]")
        lines.append(f"  B→G: {('%.1f nm' % hand_BG) if hand_BG else 'n/a'}   G→R: {('%.1f nm' % hand_GR) if hand_GR else 'n/a'}")
        lines.append("[Weights at reference wavelengths]  (w_R, w_G, w_B)")
        for t in (450.0, 520.0, 600.0):
            lines.append(f"  @{t:.0f} nm: ({weight_at(wls, w_R, t):.3f}, {weight_at(wls, w_G, t):.3f}, {weight_at(wls, w_B, t):.3f})")
        lines.append("")
        lines.append("[Per-channel Response Peaks (absolute)]")
        for c, resp in (("R", r_resp), ("G", g_resp), ("B", b_resp)):
            wl_p, val = self._peak_in_band(wls, resp, band)
            lines.append(f"  {c}: {val:.4g} a.u./SPD @ {wl_p:.1f} nm")
        if full:
            lines.append("")
            lines.append("[Robustness: mean |C_c − Fused| / max|Fused| where w_c > 0.5]")
            for c in ("R", "G", "B"):
                n, rel = stats[c]
                lines.append(f"  {c}: {rel:.2f}%  over {n} pts" if np.isfinite(rel) else f"  {c}: n/a")
            wl_p, val = self._peak_in_band(wls, np.nan_to_num(fused), band)
            lines.append("")
            lines.append(f"[Fused Curve Peak] {val:.4g} a.u./s @ {wl_p:.1f} nm")
        else:
            lines.append("")
            lines.append("[Note] Fused curve unavailable (loaded JSON mode or Step 1 missing).")
        params = "\n".join(lines)
        self._update_params(param_text, params)
        self.export_param_figs.append(self._create_text_figure("Step 2: SPD Response & Fusion Parameters", params))

    def _build_tab3(self):
        tab, chart_frame, param_text = self._create_tab_layout("Step 3: Sample Spectrum")
        if not self.cache3:
            ttk.Label(tab, text="Please complete Step 3.", font=("Arial", 12)).pack(pady=50)
            return
        self.fig3 = Figure(figsize=(12, 8), dpi=100)
        ax3 = self.fig3.add_subplot(111)
        fluor = self.cache3["fluor"]
        wls, intensity, mask = self.cache3["wls"], self.cache3["intensity"], self.cache3["mask"]
        if wls[0] > wls[-1]: wls, intensity, mask = wls[::-1], intensity[::-1], mask[::-1]
        ax3.plot(wls, intensity, 'k-', label='Net Spectrum')
        ax3.fill_between(wls, intensity, where=mask, color=fluor['color'], alpha=0.4, label='Integration')
        ax3.axvline(fluor['peak_wl'], color='red', linestyle='--', label='Target Peak')
        ax3.set_title("Figure 5: Background-Subtracted & SPD-Corrected Spectrum", fontsize=14)
        ax3.set_xlabel("Wavelength (nm)", fontsize=12); ax3.set_ylabel("True Intensity", fontsize=12)
        ax3.legend(); ax3.grid(True, linestyle=':', alpha=0.5)
        self.fig3.tight_layout()
        FigureCanvasTkAgg(self.fig3, master=chart_frame).get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.export_figs.append(self.fig3)
        features = self.cache3.get("features", {})
        spd_status = "APPLIED (Loaded)" if self.cache2.get("loaded_from_file") else ("APPLIED (per-channel absolute)" if self.cache3.get("spd_applied") else "SKIPPED (Raw RGB)")
        params = (
            f"[Measurement Status]\n  Target: {fluor.get('name', 'Turbidity Mode') if isinstance(fluor, dict) else 'Turbidity Mode'}\n  SPD Correction: {spd_status}\n\n"
            f"[Microplate Reader Style Parameters]\n  Peak Wavelength: {features.get('peak_wl', 0):.2f} nm\n  Peak Intensity: {features.get('peak_intensity', 0):.2f} a.u.\n"
            f"  FWHM: {features.get('fwhm', 0):.2f} nm\n  AUC: {features.get('auc', 0):.2f}\n  Centroid: {features.get('centroid', 0):.2f} nm"
        )
        self._update_params(param_text, params)
        self.export_param_figs.append(self._create_text_figure("Step 3: Sample Spectrum Parameters", params))

    def _build_tab4(self):
        tab, chart_frame, param_text = self._create_tab_layout("Step 4: Concentration")
        if not self.cache4:
            ttk.Label(tab, text="Please complete Step 4.", font=("Arial", 12)).pack(pady=50)
            return
        self.fig4 = Figure(figsize=(12, 8), dpi=100)
        ax4 = self.fig4.add_subplot(111)
        c4 = self.cache4
        ax4.errorbar(c4["concentrations"], c4["means"], yerr=c4["stds"], fmt='o', color='blue', capsize=5, label='Standards', ecolor='lightblue')
        x_line = np.linspace(min(c4["concentrations"])*0.8, max(c4["concentrations"])*1.2, 100)
        ax4.plot(x_line, c4["slope"] * x_line + c4["intercept"], 'r--', label=f'Fit (R²={c4["r_squared"]:.3f})')
        ax4.scatter([c4["sample_conc"]], [c4["sample_mean_area"]], color='red', marker='*', s=150, label=f'Sample (Conc: {c4["sample_conc"]:.3f})')
        if c4.get("mode") == "lod_iupac" and "lod_conc" in c4:
            ax4.axhline(y=c4["lod_signal"], color='orange', linestyle='-.', label='LoD Signal')
            ax4.axhline(y=c4["loq_signal"], color='purple', linestyle=':', label='LoQ Signal')
            ax4.axvline(x=c4["lod_conc"], color='orange', linestyle='-.', alpha=0.5)
            ax4.axvline(x=c4["loq_conc"], color='purple', linestyle=':', alpha=0.5)
        ax4.set_title("Figure 6: Standard Curve & Concentration Quantification", fontsize=14)
        ax4.set_xlabel("Concentration (a.u.)", fontsize=12); ax4.set_ylabel("Net Area", fontsize=12)
        ax4.legend(loc='upper right'); ax4.grid(True, linestyle=':', alpha=0.7)
        self.fig4.tight_layout()
        FigureCanvasTkAgg(self.fig4, master=chart_frame).get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.export_figs.append(self.fig4)
        reg_method = c4.get("regression_method", "OLS").upper()
        params = (
            f"[Regression Analysis]\n  Method: {reg_method}\n  Equation: I = {c4['slope']:.4f} * C + ({c4['intercept']:.4f})\n  R²: {c4['r_squared']:.6f}\n\n"
            f"[Sample Result]\n  Sample Net Area: {c4['sample_mean_area']:.2f}\n  PREDICTED CONCENTRATION: {c4['sample_conc']:.4f} (a.u.)\n"
        )
        if c4.get("mode") == "lod_iupac" and "lod_conc" in c4:
            params += (
                f"\n[IUPAC Limit Analysis]\n  Sy/x: {c4['syx']:.4f}\n  LoD (3.3σ): {c4['lod_conc']:.4f} (a.u.)\n  LoQ (10σ): {c4['loq_conc']:.4f} (a.u.)"
            )
        self._update_params(param_text, params)
        self.export_param_figs.append(self._create_text_figure("Step 4: Concentration Parameters", params))

    def export_pdf_report(self):
        file_path = filedialog.asksaveasfilename(title="Export Full PDF Report", defaultextension=".pdf", filetypes=[("PDF Document", "*.pdf")])
        if file_path:
            try:
                with PdfPages(file_path) as pdf:
                    for fig in self.export_figs: pdf.savefig(fig, bbox_inches='tight')
                    for fig in self.export_param_figs: pdf.savefig(fig, bbox_inches='tight')
                messagebox.showinfo("Success", f"Full Report (Charts + Parameters) saved to:\n{file_path}")
            except Exception as e: messagebox.showerror("Error", str(e))