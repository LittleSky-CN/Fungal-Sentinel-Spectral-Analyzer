# gui/details_window.py
"""FSSA v1.5.0 Detailed Analysis Report.

【v1.5.0】新增 Fluorescein (sodium salt) 荧光素选项（发射 520 nm / 激发 494 nm / green
通道 / FWHM 25）；荧光素标识统一为 canonical ID（新增 canonical_fluorophore() 归一化解
析），以便与 Android 端曲线预设互通。
Tab 2 fixed: Figure 4a-4d with ONE continuous pointwise-fused curve
(weights response-proportional, sum = 1, no cross-wavelength smoothing).
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

from core.image_processing import (
    normalize_for_display, QUALITY_PASS_NM, QUALITY_WARNING_NM,
)
from core.wavelength_preset import step1_quality_tier

try:
    from core.fusion import channel_weights, auto_handoffs, fuse_pointwise
except Exception:
    def auto_handoffs(w, rb, rg, rr, default_bg=483.0, default_gr=577.0):
        h_bg, h_gr = default_bg, default_gr
        m = (rb > 0) & (rg > 0)
        if np.any(m):
            h_bg = float(w[int(np.argmin(np.where(m, np.abs(rb - rg), np.inf)))])
        m2 = (rg > 0) & (rr > 0) & (w > h_bg + 20.0)
        if np.any(m2):
            h_gr = float(w[int(np.argmin(np.where(m2, np.abs(rg - rr), np.inf)))])
        return h_bg, h_gr
    def channel_weights(rb, rg, rr, eps=1e-6):
        total = rb + rg + rr + eps
        return rb / total, rg / total, rr / total
    def fuse_pointwise(weights, curves):
        (wB, wG, wR), (cB, cG, cR) = weights, curves
        return wB * cB + wG * cG + wR * cR

# ---- 可选主题依赖 ----
try:
    from gui import theme as _theme
except Exception:
    _theme = None
try:
    from config import THEME_ENABLED as _THEME_FLAG
except Exception:
    _THEME_FLAG = False
_THEME_ON = bool(_theme is not None and _THEME_FLAG)

VISIBLE_NM = (400.0, 700.0)


def _style_fig(fig):
    if _THEME_ON:
        _theme.style_figure(fig)
    return fig


def _safe_div(a, b):
    """Pointwise guarded division (per-wavelength scalar op only)."""
    ok = b > 1e-6
    return np.where(ok, a / np.where(ok, b, 1.0), 0.0)


class DetailsWindow:
    def __init__(self, parent, cache1, cache2, cache3, cache4, mapping, preset_provenance=None):
        self.win = tk.Toplevel(parent)
        self.win.title("FSSA v1.5.0 — Detailed Analysis Report & Parameters")
        self.win.geometry("2000x2000")
        if _THEME_ON:
            _theme.tint(self.win)
        self.cache1 = cache1 or {}
        self.cache2 = cache2 or {}
        self.cache3 = cache3 or {}
        self.cache4 = cache4 or {}
        self.mapping = mapping
        self.preset_provenance = preset_provenance or {}
        self._active_canvas = None

        self.notebook = ttk.Notebook(self.win)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig1 = self.fig2 = self.fig3 = self.fig4 = self.fig5 = None
        self.fig1_params = self.fig2_params = self.fig3_params = self.fig4_params = None

        self._build_tab1()
        self._build_tab2()
        self._build_tab3()
        self._build_tab4()
        self._build_tab5()

        btn_frame = ttk.Frame(self.win)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        self.figs_to_export = [f for f in (
            self.fig1, self.fig1_params, self.fig2, self.fig2_params,
            self.fig3, self.fig3_params, self.fig4, self.fig4_params,
            self.fig5) if f is not None]
        if _THEME_ON:
            ttk.Button(btn_frame, text="📥 Export Full Report (PDF)", style="Magenta.TButton",
                       command=lambda: self.export_pdf_report(self.figs_to_export)).pack(side=tk.LEFT)
            ttk.Button(btn_frame, text="Close", style="Pale.TButton",
                       command=self.win.destroy).pack(side=tk.RIGHT)
        else:
            ttk.Button(btn_frame, text=" Export Full Report (PDF)",
                       command=lambda: self.export_pdf_report(self.figs_to_export)).pack(side=tk.LEFT)
            ttk.Button(btn_frame, text="Close", command=self.win.destroy).pack(side=tk.RIGHT)

        self.win.bind("<MouseWheel>", self._on_mousewheel)
        self.win.bind("<Button-4>", self._on_mousewheel)
        self.win.bind("<Button-5>", self._on_mousewheel)

    # ---------------- 主题配色辅助 ----------------
    def _ink_line(self):
        return _theme.CREAM if _THEME_ON else "black"

    def _ch_colors(self):
        if _THEME_ON:
            return {"red": "#FF5C5C", "green": "#5CFF5C", "blue": "#5CA8FF"}
        return {"red": "red", "green": "green", "blue": "blue"}

    def _c4_palette(self):
        if _THEME_ON:
            m = _theme.MPL
            return {"std": m["std"], "std_err": "#7FF7FF", "fit": m["fit"],
                    "sample": m["sample"], "sample_edge": _theme.BLACK,
                    "lod": m["lod"], "loq": m["loq"]}
        return {"std": "blue", "std_err": "lightblue", "fit": "red",
                "sample": "red", "sample_edge": "darkred",
                "lod": "orange", "loq": "purple"}

    # ---------------- 滚轮 & 滚动容器 ----------------
    def _set_active_canvas(self, canvas):
        self._active_canvas = canvas

    def _on_mousewheel(self, event):
        c = self._active_canvas
        if c is None:
            return
        if getattr(event, "num", None) == 4:
            c.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            c.yview_scroll(1, "units")
        else:
            d = getattr(event, "delta", 0)
            if d:
                c.yview_scroll(int(-1 * (d / 120)), "units")

    def _create_scrollable_frame(self, parent):
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True, pady=5)
        canvas = tk.Canvas(container, highlightthickness=0)
        if _THEME_ON:
            canvas.configure(bg=_theme.BLACK)
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

    def _embed_figure(self, fig, scroll_canvas, inner):
        _style_fig(fig)
        widget = FigureCanvasTkAgg(fig, master=inner).get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)
        widget.bind("<Enter>", lambda e: self._set_active_canvas(scroll_canvas))
        widget.bind("<Leave>", lambda e: self._set_active_canvas(None))
        return widget

    def _add_param_panel(self, tab, content, title=None, height=8):
        frame = ttk.LabelFrame(tab, text=title or "Calculated Parameters & Debug Info", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=(0, 8))
        if _THEME_ON:
            txt = tk.Text(frame, height=height, font=_theme.MONO, wrap=tk.WORD,
                          bg=_theme.CREAM, fg=_theme.INK, relief="flat",
                          highlightthickness=1, highlightbackground=_theme.CYAN)
        else:
            txt = tk.Text(frame, height=height, font=("Consolas", 10), wrap=tk.WORD, bg="#f8f9fa")
        txt.pack(fill=tk.X)
        txt.insert("1.0", content)
        txt.config(state=tk.DISABLED)
        return txt

    # 等宽字体字符步进（em，DejaVu Sans Mono 实测 ~0.59），取 0.62 留余量
    _MONO_CHAR_EM = 0.62
    # 全角字符（CJK 等）在等宽字体里占两格
    _WIDE_CHAR_CP = 0x2E80

    def _wrap_monospace(self, text, fontsize, x):
        """按估算的列宽手工折行。

        ``fig.text`` 画等宽文本时不会自动折行，超宽的行会画到画布外、在 PDF 里
        被裁掉右半截（窗口里的 ``tk.Text`` 是 ``wrap=tk.WORD``，不会）。
        """
        usable_in = 15.0 * (1.0 - x - 0.03)
        width = max(24, int(usable_in * 72.0 / (fontsize * self._MONO_CHAR_EM)))
        out = []
        for line in text.split("\n"):
            if sum(2 if ord(c) >= self._WIDE_CHAR_CP else 1 for c in line) <= width:
                out.append(line)
                continue
            chunks, cur, cur_w = [], [], 0
            for ch in line:
                w = 2 if ord(ch) >= self._WIDE_CHAR_CP else 1
                if cur_w + w > width:
                    chunks.append("".join(cur))
                    cur, cur_w = [], 0
                cur.append(ch)
                cur_w += w
            chunks.append("".join(cur))
            out.extend(chunks)
        return "\n".join(out)

    def _create_text_figure(self, title, params_text, fontsize=10, x=0.03):
        """一整页纯文本（参数面板 / 预设溯源）的 figure。

        画布高度随行数增长（原先只有溯源页这么做），否则行数一多就会画到画布
        外面、PDF 里被裁掉底部。
        """
        wrapped = self._wrap_monospace(params_text, fontsize, x)
        n_lines = wrapped.count("\n") + 1
        height = max(6.0, min(60.0, 1.6 + 0.22 * n_lines))
        fig = Figure(figsize=(15, height), dpi=100)
        tcol = "black"
        if _THEME_ON:
            fig.patch.set_facecolor(_theme.CREAM); tcol = _theme.INK
        else:
            fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111); ax.axis("off")
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20, color=tcol)
        fig.text(x, 0.94, wrapped, fontsize=fontsize, family="monospace",
                 verticalalignment="top", color=tcol)
        return fig

    def _create_params_figure(self, title, params_text):
        return self._create_text_figure(title, params_text, fontsize=11, x=0.05)

    def _create_provenance_figure(self, title, params_text):
        return self._create_text_figure(title, params_text)

    # ---------------- Tab 1 ----------------
    def _build_tab1(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Step 1: Wavelength Cal.")
        if not self.cache1:
            ttk.Label(tab, text="Please complete Step 1.", font=("Arial", 12)).pack(pady=50)
            return
        sc, inner = self._create_scrollable_frame(tab)
        ch = self._ch_colors(); ink = self._ink_line()

        self.fig1 = Figure(figsize=(15, 25), dpi=100)
        rgb = self.cache1["rgb"]; x0, x1 = self.cache1["x_roi"]

        ax01 = self.fig1.add_subplot(3, 1, 1)
        ax01.imshow(normalize_for_display(rgb))
        ax01.add_patch(Rectangle((x0, 0), x1 - x0, rgb.shape[0], fill=False,
                                 linewidth=2, edgecolor='yellow', label='Auto X ROI'))
        for p in self.cache1["peaks"]:
            ax01.axhline(p.y_centroid, linestyle="--", linewidth=1.5,
                         label=f"{p.name} ({p.wavelength_nm}nm)")
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
        ax03.plot(lam_grid, v.slope_k * lam_grid + v.intercept_b, color=ink, linewidth=2,
                  label=f"Fit: p = {v.slope_k:.4f}λ + {v.intercept_b:.2f}")
        ax03.scatter([462.5, 622.5],
                     [self.cache1["peaks"][2].y_centroid, self.cache1["peaks"][0].y_centroid],
                     color=ch["blue"], s=100, label="Train (B, R)", zorder=5)
        ax03.scatter([522.5], [self.cache1["peaks"][1].y_centroid],
                     marker='x', s=150, color=ch["red"], label="Validate (G)", zorder=5)
        ax03.plot([522.5, 522.5], [v.predicted_y, self.cache1["peaks"][1].y_centroid],
                  '--', color=ch["red"], linewidth=2, label=f"Error: {v.wavelength_error_nm:+.2f} nm")
        ax03.set_title("Figure 3: Two-point Fit (B+R) with Third-point Validation (G)", fontsize=14)
        ax03.set_xlabel("Wavelength (nm)", fontsize=12); ax03.set_ylabel("Y Pixel", fontsize=12)
        ax03.invert_yaxis(); ax03.legend(fontsize=9); ax03.grid(True, linestyle=':', alpha=0.5)

        self.fig1.tight_layout()
        self._embed_figure(self.fig1, sc, inner)

        quality = self.cache1.get("quality")
        tier = step1_quality_tier(quality)
        # 这一段会渲染进报告 PDF 的参数图（等宽字体缺中文字形），所以只用英文
        verdict = {
            "PASS": "usable",
            "WARNING": "biased - review downstream quantification",
            "FAILED": "unreliable - redo Step 1",
        }[tier]
        params = (
            f"[Calibration Quality] {tier}   <- {verdict}\n"
            f"  G-point error {v.wavelength_error_nm:+.3f} nm | thresholds: "
            f"PASS <= {QUALITY_PASS_NM} nm, WARNING <= {QUALITY_WARNING_NM} nm, else FAILED\n"
            f"  Source tag: {quality if quality else 'unknown'}\n\n"
            f"[Wavelength Mapping Equation]\n"
            f"  Pixel (p) = {v.slope_k:.6f} * Wavelength(λ) + {v.intercept_b:.4f}\n\n"
            f"[Calibration Metadata]\n"
            f"  Exposure Time: {self.cache1.get('exp_time', float('nan')):.4f} s\n"
            f"  Auto-detected X ROI: [{x0}, {x1}] (Width: {x1 - x0} px)\n"
            f"  Positioning Shots: {self.cache1.get('n_pos', 'n/a')}\n\n"
            f"[Validation Results]\n"
            f"  Training Points: B (462.5nm), R (622.5nm)\n"
            f"  Validation Point: G (522.5nm)\n"
            f"  G-point Residual: {v.residual_px:+.2f} pixels\n"
            f"  G-point Wavelength Error: {v.wavelength_error_nm:+.3f} nm"
        )
        self._add_param_panel(tab, params)
        self.fig1_params = self._create_params_figure("Step 1: Wavelength Calibration Parameters", params)

    # ---------------- Tab 2: Figure 4a-4d (FIXED: numerator = INDEPENDENT positioning LEDs) ----------------
    def _build_tab2(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Step 2: SPD Response")
        if not self.cache2:
            ttk.Label(tab, text="Please complete Step 2.", font=("Arial", 12)).pack(pady=50)
            return

        sc, inner = self._create_scrollable_frame(tab)
        ch = self._ch_colors(); ink = self._ink_line()

        wls = np.asarray(self.cache2["wavelengths"], dtype=float)
        r_resp = np.asarray(self.cache2["R"], dtype=float)
        g_resp = np.asarray(self.cache2["G"], dtype=float)
        b_resp = np.asarray(self.cache2["B"], dtype=float)

        pos_R = self.cache2.get("pos_R"); pos_G = self.cache2.get("pos_G"); pos_B = self.cache2.get("pos_B")
        pos_avail = all(x is not None for x in (pos_R, pos_G, pos_B)) and len(pos_R) == len(wls)
        if pos_avail:
            pos_R = np.asarray(pos_R, float); pos_G = np.asarray(pos_G, float); pos_B = np.asarray(pos_B, float)

        if wls[0] > wls[-1]:
            wls = wls[::-1]; r_resp = r_resp[::-1]; g_resp = g_resp[::-1]; b_resp = b_resp[::-1]
            if pos_avail:
                pos_R, pos_G, pos_B = pos_R[::-1], pos_G[::-1], pos_B[::-1]

        valid = (r_resp > 0) | (g_resp > 0) | (b_resp > 0)
        wv = wls[valid]
        rv, gv, bv = r_resp[valid], g_resp[valid], b_resp[valid]

        # 【修复】校正分子 = 独立定位光源 Pos_c（而非标定灯 Lamp_c），避免循环论证
        if pos_avail:
            cRv = _safe_div(pos_R, r_resp)[valid]
            cGv = _safe_div(pos_G, g_resp)[valid]
            cBv = _safe_div(pos_B, b_resp)[valid]
        else:
            cRv = cGv = cBv = None

        # 交叉点仅作诊断读数输出；融合权重改用响应比例归一 w_c = resp_c / Σresp_c
        h_bg, h_gr = auto_handoffs(wv, bv, gv, rv)
        wB, wG, wR = channel_weights(bv, gv, rv)
        fused = fuse_pointwise((wB, wG, wR), (cBv, cGv, cRv)) if pos_avail else None

        self.fig2 = Figure(figsize=(15, 24), dpi=100)

        # 4a RAW positioning
        ax4a = self.fig2.add_subplot(4, 1, 1)
        if pos_avail:
            ax4a.plot(wv, pos_R[valid], '-', color=ch["red"], lw=1.5, label='R channel')
            ax4a.plot(wv, pos_G[valid], '-', color=ch["green"], lw=1.5, label='G channel')
            ax4a.plot(wv, pos_B[valid], '-', color=ch["blue"], lw=1.5, label='B channel')
            ax4a.set_ylabel("Linear Intensity (a.u./s)", fontsize=11)
        else:
            ax4a.text(0.5, 0.5, "Positioning profiles not cached (loaded from JSON or old cache).\nRe-run Step 1 + Step 2 from DNG.",
                      ha='center', va='center', transform=ax4a.transAxes, color=ink)
        ax4a.set_title("Figure 4a: Positioning Light Source - RAW Measured (per channel, absolute units)", fontsize=13)
        if pos_avail: ax4a.legend(fontsize=9)
        ax4a.grid(True, linestyle=':', alpha=0.5)

        # 4b CORRECTED (thin, should agree) + FUSED (thick, single continuous curve)
        ax4b = self.fig2.add_subplot(4, 1, 2)
        if pos_avail:
            ax4b.plot(wv, cRv, '-', color=ch["red"], lw=1.2, label='R channel')
            ax4b.plot(wv, cGv, '-', color=ch["green"], lw=1.2, label='G channel')
            ax4b.plot(wv, cBv, '-', color=ch["blue"], lw=1.2, label='B channel')
            ax4b.plot(wv, fused, '-', color=ink, lw=3.0, alpha=0.9,
                      label='Fused = w_B·B + w_G·G + w_R·R (continuous)')
            ax4b.set_ylabel("Corrected Intensity (a.u./s)", fontsize=11)
        else:
            ax4b.text(0.5, 0.5, "Corrected positioning spectra not available.\nRe-run Step 1 + Step 2 from DNG.",
                      ha='center', va='center', transform=ax4b.transAxes, color=ink)
        ax4b.set_title("Figure 4b: Positioning Light Source - SPD-CORRECTED + Pointwise Fusion", fontsize=13)
        if pos_avail: ax4b.legend(fontsize=9)
        ax4b.grid(True, linestyle=':', alpha=0.5)

        # 4c RESPONSE
        ax4c = self.fig2.add_subplot(4, 1, 3)
        ax4c.plot(wv, rv, '-', color=ch["red"], lw=2, label='R channel')
        ax4c.plot(wv, gv, '-', color=ch["green"], lw=2, label='G channel')
        ax4c.plot(wv, bv, '-', color=ch["blue"], lw=2, label='B channel')
        ax4c.set_xlabel("Wavelength (nm)", fontsize=12)
        ax4c.set_ylabel("Sensitivity (a.u. / SPD unit)", fontsize=11)
        ax4c.set_title("Figure 4c: Camera Absolute Response = Lamp_c / SPD_true (masked, NO SG, NO normalization)", fontsize=13)
        ax4c.legend(fontsize=10); ax4c.grid(True, linestyle=':', alpha=0.5)

        # 4d WEIGHTS
        ax4d = self.fig2.add_subplot(4, 1, 4)
        ax4d.plot(wv, wB, '-', color=ch["blue"], lw=2.5, label='w_B(λ)')
        ax4d.plot(wv, wG, '-', color=ch["green"], lw=2.5, label='w_G(λ)')
        ax4d.plot(wv, wR, '-', color=ch["red"], lw=2.5, label='w_R(λ)')
        ax4d.axvline(h_bg, ls=':', color=ink, label=f'B→G responsivity crossing {h_bg:.1f} nm')
        ax4d.axvline(h_gr, ls='--', color=ink, alpha=0.7, label=f'G→R responsivity crossing {h_gr:.1f} nm')
        ax4d.set_ylim(-0.02, 1.05)
        ax4d.set_xlabel("Wavelength (nm)", fontsize=12); ax4d.set_ylabel("Weight", fontsize=11)
        ax4d.set_title("Figure 4d: Per-wavelength channel weights (response-proportional, sum = 1)", fontsize=13)
        ax4d.legend(fontsize=9, loc='upper right'); ax4d.grid(True, linestyle=':', alpha=0.5)

        self.fig2.tight_layout()
        self._embed_figure(self.fig2, sc, inner)

        # ---- 参数（含三通道一致性量化） ----
        agree_txt = "n/a"
        fused_pk = fused_wl = float('nan')
        if pos_avail and fused is not None:
            denom = np.mean(fused[fused > 0]) if np.any(fused > 0) else 1.0
            dev = np.mean(np.abs(cBv - cGv) + np.abs(cGv - cRv) + np.abs(cBv - cRv)) / 3.0 / max(denom, 1e-9)
            agree_txt = f"mean pairwise |Δ| = {dev*100:.2f}% of fused mean"
            i = int(np.argmax(fused)); fused_pk = float(fused[i]); fused_wl = float(wv[i])

        params = (
            f"[Method Notes]\n"
            f"  Correction demonstrated on INDEPENDENT data (positioning LEDs), NOT the SPD lamp.\n"
            f"  Numerator = Pos_c (Step 1 profiles); denominator = resp_c (from SPD lamp).\n"
            f"  NO Savitzky-Golay / moving-average: all ops are per-wavelength scalar multiplications.\n\n"
            f"[Formulas]\n"
            f"  resp_c(λ) = Lamp_c(λ) / SPD_true(λ)        (Lamp = SPD calibration source)\n"
            f"  Corrected_c(λ) = Pos_c(λ) / resp_c(λ)      (Pos = positioning LEDs, independent)\n"
            f"  w_c(λ) = resp_c(λ) / (resp_R + resp_G + resp_B)(λ)   -> Σ w_c = 1 pointwise\n"
            f"  Fused(λ) = w_B·Corrected_B + w_G·Corrected_G + w_R·Corrected_R\n\n"
            f"[Effective Band] {wv[0]:.1f} - {wv[-1]:.1f} nm | masked pixels: {len(wls)-np.sum(valid)}/{len(wls)}\n"
            f"[SPD Threshold Ratio] {self.cache2.get('spd_threshold_ratio', 0.05)}\n"
            f"[Shots Averaged] SPD lamp: {self.cache2.get('lamp_R_n', 1)} | Positioning: see Step 1\n"
            f"[Responsivity Crossings (diagnostic only)] B→G: {h_bg:.1f} nm | G→R: {h_gr:.1f} nm\n"
            f"[Cross-channel Agreement] {agree_txt}   <- 三通道互相同意的量化证据\n"
            f"[Fused Curve] peak {fused_pk:.4f} a.u./s @ {fused_wl:.1f} nm\n\n"
            f"[Per-channel Response Peaks (absolute)]\n"
            f"  R: {np.max(rv):.0f} a.u./SPD @ {wv[int(np.argmax(rv))]:.1f} nm\n"
            f"  G: {np.max(gv):.0f} a.u./SPD @ {wv[int(np.argmax(gv))]:.1f} nm\n"
            f"  B: {np.max(bv):.0f} a.u./SPD @ {wv[int(np.argmax(bv))]:.1f} nm\n"
        )
        if pos_avail:
            params += (
                f"\n[Per-channel RAW Positioning Peaks]\n"
                f"  R: {np.max(pos_R[valid]):.1f} a.u./s @ {wv[int(np.argmax(pos_R[valid]))]:.1f} nm\n"
                f"  G: {np.max(pos_G[valid]):.1f} a.u./s @ {wv[int(np.argmax(pos_G[valid]))]:.1f} nm\n"
                f"  B: {np.max(pos_B[valid]):.1f} a.u./s @ {wv[int(np.argmax(pos_B[valid]))]:.1f} nm\n"
                f"\n[Per-channel CORRECTED Positioning Peaks]\n"
                f"  R: {np.max(cRv):.4f} a.u./s @ {wv[int(np.argmax(cRv))]:.1f} nm\n"
                f"  G: {np.max(cGv):.4f} a.u./s @ {wv[int(np.argmax(cGv))]:.1f} nm\n"
                f"  B: {np.max(cBv):.4f} a.u./s @ {wv[int(np.argmax(cBv))]:.1f} nm\n"
            )
        self._add_param_panel(tab, params)
        self.fig2_params = self._create_params_figure("Step 2: SPD Response Parameters", params)

    # ---------------- Tab 3 ----------------
    def _build_tab3(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Step 3: Sample Spectrum")
        if not self.cache3:
            ttk.Label(tab, text="Please complete Step 3.", font=("Arial", 12)).pack(pady=50)
            return
        sc, inner = self._create_scrollable_frame(tab)
        ink = self._ink_line(); ch = self._ch_colors()

        self.fig3 = Figure(figsize=(15, 8), dpi=100)
        ax3 = self.fig3.add_subplot(111)
        fluor = self.cache3["fluor"]
        wls, intensity, mask = self.cache3["wls"], self.cache3["intensity"], self.cache3["mask"]
        if wls[0] > wls[-1]:
            wls, intensity, mask = wls[::-1], intensity[::-1], mask[::-1]
        vis = (wls >= VISIBLE_NM[0]) & (wls <= VISIBLE_NM[1])
        if np.any(vis):
            wls, intensity, mask = wls[vis], intensity[vis], mask[vis]

        ax3.plot(wls, intensity, '-', color=ink, linewidth=2, label='Net Spectrum')
        ax3.fill_between(wls, intensity, where=mask, color=fluor['color'], alpha=0.4, label='Integration')
        ax3.axvline(fluor['peak_wl'], linestyle='--', color=ch["red"], label='Target Peak')
        ax3.set_title("Figure: Background-Subtracted & SPD-Corrected Spectrum (Visible Band Only)", fontsize=14)
        ax3.set_xlabel("Wavelength (nm)", fontsize=12); ax3.set_ylabel("True Intensity", fontsize=12)
        ax3.legend(fontsize=10); ax3.grid(True, linestyle=':', alpha=0.5)
        self.fig3.tight_layout()
        self._embed_figure(self.fig3, sc, inner)

        features = self.cache3.get("features", {})
        spd_status = "APPLIED (Absolute Sensitivity)" if self.cache3.get("spd_applied") else "SKIPPED (Raw RGB)"
        fluor_name = self.cache3.get("fluor_name") or \
            ("Turbidity Mode (Green Channel)" if fluor.get("peak_wl") == 550 else "Custom/Unknown")
        params = (
            f"[Measurement Status]\n"
            f"  Target: {fluor_name}\n"
            f"  Channel: {fluor.get('channel', 'n/a')} | Target WL: {fluor.get('peak_wl', 0)} nm "
            f"| FWHM Window: {fluor.get('fwhm', 0)} nm\n"
            f"  SPD Correction: {spd_status}\n"
            f"  Display Band: [{VISIBLE_NM[0]:.0f}, {VISIBLE_NM[1]:.0f}] nm\n\n"
            f"[Microplate Reader Style Parameters]\n"
            f"  Peak Wavelength: {features.get('peak_wl', 0):.2f} nm\n"
            f"  Peak Intensity: {features.get('peak_intensity', 0):.2f} a.u.\n"
            f"  FWHM (Full Width Half Max): {features.get('fwhm', 0):.2f} nm\n"
            f"  AUC (Area Under Curve): {features.get('auc', 0):.2f}\n"
            f"  Centroid (Center of Mass): {features.get('centroid', 0):.2f} nm\n\n"
            f"[Integration]\n"
            f"  Net Area (Mean): {self.cache3.get('mean_area', 0):.2f} | "
            f"Peak in Window: {self.cache3.get('peak', 0):.2f}"
        )
        self._add_param_panel(tab, params)
        self.fig3_params = self._create_params_figure("Step 3: Sample Spectrum Parameters", params)

    # ---------------- Tab 4 ----------------
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
        ax4.errorbar(c4["concentrations"], c4["means"], yerr=c4["stds"], fmt='o',
                     color=pc["std"], capsize=5, label='Standards', ecolor=pc["std_err"])
        x_line = np.linspace(min(c4["concentrations"]) * 0.8, max(c4["concentrations"]) * 1.2, 100)
        ax4.plot(x_line, c4["slope"] * x_line + c4["intercept"], '--', color=pc["fit"],
                 linewidth=2, label=f'Fit (R²={r2:.4f})')
        ax4.scatter([c4["sample_conc"]], [c4["sample_mean_area"]], color=pc["sample"], marker='*',
                    s=150, label=f'Sample (Conc: {c4["sample_conc"]:.3f})', edgecolors=pc["sample_edge"])
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
        params = (
            f"[Regression Analysis]\n  Method: {reg_method}\n"
            f"  Equation: I = {c4['slope']:.6f} * C + ({c4['intercept']:.6f})\n"
            f"  R-squared (R²): {r2:.6f}\n"
        )
        if c4.get("metric_mode"): params += f"  Quantification Metric: {c4['metric_mode'].upper()}\n"
        if c4.get("t_stat") is not None: params += f"  Slope t-statistic: {c4['t_stat']:.2f} (require >= 3)\n"
        params += (f"\n[Sample Result]\n  Sample Net Area: {c4['sample_mean_area']:.2f}\n"
                   f"  PREDICTED CONCENTRATION: {c4['sample_conc']:.4f} (a.u.)\n")
        u = c4.get("unmix")
        if u:
            lo, hi = u["window"]
            params += (f"\n[Spectral Unmixing: sample = α·E + s·Blank + k]\n"
                       f"  Window: [{lo:.1f}, {hi:.1f}] nm | Scatter peak: {c4.get('scatter_peak_wl', float('nan')):.1f} nm\n"
                       f"  Reference E: std conc = {u['ref_conc']} | AUC(E): {u['auc_E']:.2f}\n")
            for pr in u.get("per_row", []):
                params += (f"    C={pr['conc']:<10g} α={pr['alpha']:.4f}  s={pr['s']:.3f}  "
                           f"k={pr['k']:.3f}  R²={pr['r2']:.4f}\n")
            params += f"    SAMPLE     α={u['alpha']:.4f}  s={u['u_s']:.3f}  k={u['u_k']:.3f}\n"
            if u.get("lod_range"):
                params += (f"  LoD window-sensitivity range: [{u['lod_range'][0]:.3g}, {u['lod_range'][1]:.3g}] (a.u.)\n")
        blocks = c4.get("blocks")
        if blocks:
            params += "\n[Exposure-block LoD selection (ICH Q2: n>=3, t>=3)]\n"
            for b in blocks:
                if b["n"] < 3:
                    params += f"  @{b['exp']:.3f}s: n={b['n']} (skipped, <3 levels)\n"
                else:
                    params += (f"  @{b['exp']:.3f}s: n={b['n']} slope={b['slope']:.4g} R²={b['r2']:.4f} "
                               f"Sy/x={b['syx']:.4g} t={b['t']:.1f} LoD={b['lod']:.4g}\n")
        if c4.get("mode") == "lod_iupac" and "lod_conc" in c4:
            params += (f"\n[IUPAC Limit Analysis (headline)]\n"
                       f"  Source: {c4.get('lod_source', 'full-range curve')}\n"
                       f"  LoD (3.3σ): {c4['lod_conc']:.4f} (a.u.) | LoQ (10σ): {c4['loq_conc']:.4f} (a.u.)\n")
            if "lod_fullrange" in c4:
                params += f"  [reference] full-range LoD/LoQ: {c4['lod_fullrange']:.4g} / {c4['loq_fullrange']:.4g}\n"
        self._add_param_panel(tab, params)
        self.fig4_params = self._create_params_figure("Step 4: Concentration Parameters", params)

    # ---------------- Tab 5: 预设溯源（无条件创建） ----------------
    STEP_TITLES = {1: "Wavelength Calibration",
                   2: "SPD Response Calibration",
                   3: "Sample Spectrum Analysis",
                   4: "Standard Curve / Concentration"}

    def _provenance_text(self):
        """把 self.preset_provenance 渲染成纯文本；缺失的步骤也必须显式列出。"""
        lines = [
            "[Preset Provenance - where the key parameters of each step came from]",
            "  Records, for Step 1-4, whether the key parameters were imported from a",
            "  preset file or produced by the live calibration / fit of this session.",
            "",
        ]
        for step in (1, 2, 3, 4):
            entry = (self.preset_provenance or {}).get(f"step{step}")
            lines.append(f"[Step {step}: {self.STEP_TITLES[step]}]")
            if not isinstance(entry, dict):
                lines.append("  Source      : Not completed / not imported")
                lines.append("  This step was neither completed nor imported -> no provenance to show.")
                lines.append("")
                continue
            kind = entry.get("kind")
            if kind == "preset":
                source = f"Imported preset ({entry.get('preset_type') or 'UNKNOWN'})"
            elif kind == "live":
                source = "Live calibration / fit"
            else:
                source = f"Not completed / not imported (kind={kind})"
            lines.append(f"  Source      : {source}")
            lines.append(f"  Preset Name : {entry.get('preset_name') or 'n/a'}")
            lines.append(f"  File Name   : {entry.get('file_name') or 'n/a'}")
            lines.append(f"  File Path   : {entry.get('file') or 'n/a'}")
            lines.append(f"  Imported At : {entry.get('imported_at') or 'n/a'}")
            params = entry.get("params") or {}
            if params:
                lines.append("  Key Parameters:")
                for key in params:
                    lines.append(f"    - {key}: {params[key]}")
            else:
                lines.append("  Key Parameters: (none recorded)")
            lines.append("")
        return "\n".join(lines)

    def _build_tab5(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="数据来源 / Provenance")
        params = self._provenance_text()
        sc, inner = self._create_scrollable_frame(tab)
        self.fig5 = self._create_provenance_figure("Preset Provenance (Data Source)", params)
        self._embed_figure(self.fig5, sc, inner)
        self._add_param_panel(tab, params,
                              title="Preset Provenance — 预设来源 / 导入文件 / 关键参数",
                              height=22)

    # ---------------- PDF ----------------
    def export_pdf_report(self, figures):
        figures = [f for f in (figures or []) if f is not None]
        if not figures:
            return
        file_path = filedialog.asksaveasfilename(title="Export PDF Report", defaultextension=".pdf",
                                                   filetypes=[("PDF Document", "*.pdf")])
        if file_path:
            try:
                with PdfPages(file_path) as pdf:
                    for fig in figures:
                        pdf.savefig(fig, bbox_inches='tight')
                messagebox.showinfo("Success", f"Full Report (Charts + Parameters) saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", str(e))