# Fungal-Sentinel-Spectral-Analyzer
Open-source, offline-first spectral analysis software for a low-cost portable fungal fluorescence detector. Smartphone DNG + grating spectroscopy → quantitative fluorescence & turbidity. (iGEM 2026)

**FSSA (Fungal Sentinel Spectral Analyzer) — v1.4.8** · Python desktop reference implementation
Android counterpart: **MycoSpec v2.2.6** (https://github.com/LittleSky-CN/Mycospec)

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
