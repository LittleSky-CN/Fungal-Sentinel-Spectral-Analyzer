# Fungal-Sentinel-Spectral-Analyzer
Open-source, offline-first spectral analysis software for a low-cost portable fungal fluorescence detector. Smartphone DNG + grating spectroscopy → quantitative fluorescence & turbidity. (iGEM 2026)

**FSSA (Fungal Sentinel Spectral Analyzer) — v1.4.5** · Python desktop reference implementation
Android counterpart: **MycoSpec v2.2.6** (https://github.com/LittleSky-CN/Mycospec)

Step 4 concentration curves can be saved and loaded as JSON presets. Unmixing presets
store their reference spectrum against wavelength (nm), not sensor pixel positions,
so the saved curve can be applied to a new Step 3 blank/sample pair.

Channel fusion uses response-proportional normalisation,
`w_c(λ) = resp_c(λ) / Σ_c resp_c(λ)` with `Σ_c w_c(λ) = 1` pointwise
(`core/fusion.py`). This is numerically identical to the Android build, so a
given DNG yields the same fused spectrum on desktop and on phone.
