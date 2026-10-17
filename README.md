# Fungal-Sentinel-Spectral-Analyzer
Open-source, offline-first spectral analysis software for a low-cost portable fungal fluorescence detector. Smartphone DNG + grating spectroscopy → quantitative fluorescence & turbidity. (iGEM 2026)

**FSSA (Fungal Sentinel Spectral Analyzer) — v1.5.0** · Python desktop reference implementation
Android counterpart: **MycoSpec v2.2.6** (https://github.com/LittleSky-CN/Mycospec)

### 【v1.5.0】

* **Step 3 新增荧光素选项**：`Fluorescein (sodium salt)` —— 发射 520 nm / 激发 494 nm /
  `green` 通道 / FWHM 25（`config.py` 的 `FLUOROPHORES`）。
* **荧光素标识统一为 canonical ID**：`FLUOROPHORES` 字典键改为
  `YPET / EGFP / FLUORESCEIN / MCHERRY / CFP / MTURQUOISE2`，与 Android 端 enum name
  对齐；新增 `canonical_fluorophore()` 做归一化解析（大小写、空格、历史 display_name
  写法都能落回 canonical）。preset JSON 一律存 canonical，存量 preset 仍可读，
  从而打通两端曲线预设互通。

All three preset types can be saved and reused across sessions:

* **WAVELENGTH** (Step 1) — the pixel↔nm mapping plus ROI, peaks, G-point residual
  and capture metadata (`wavelength_calibration.json`).
* **SPD** (Step 2) — the per-channel response *and* the wavelength mapping it was
  computed with, plus the raw lamp profiles and the raw reference-SPD grid. A preset
  loaded under a drifted mapping is recomputed (Mode B), so reusing it is equivalent
  to re-running Step 2 with the current mapping. Drift itself is only ever a warning:
  ≤2 nm silent, 2–10 nm yellow, >10 nm orange with a confirmation — never a rejection.
* **CURVE** (Step 4) — standard curves and their fitted slope/intercept. Unmixing
  presets store their reference spectrum against wavelength (nm), not sensor pixel
  positions, so the saved curve can be applied to a new Step 3 blank/sample pair.

Channel fusion uses response-proportional normalisation,
`w_c(λ) = resp_c(λ) / Σ_c resp_c(λ)` with `Σ_c w_c(λ) = 1` pointwise
(`core/fusion.py`). This is numerically identical to the Android build, so a
given DNG yields the same fused spectrum on desktop and on phone.
