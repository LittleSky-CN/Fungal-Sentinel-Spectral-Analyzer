# Changelog — FSSA (Fungal Sentinel Spectral Analyzer)

Python reference implementation of the Fungal Sentinel spectral pipeline.
All dates are commit dates; every entry is sourced from repository history, and
every released version maps to a Git tag. Short commit hashes are given so each
claim can be checked against the tree.

Current released version: **v1.6.0** (2026-10-21), tagged `v1.6.0`. Version
references were bumped in `config.py`, `core/calibration.py`,
`core/wavelength_preset.py`, `main.py`, `gui/main_window.py`,
`gui/details_window.py`, `gui/app_icon.py`, `pyproject.toml`,
`requirements.txt`, `README.md` and `CONTRIBUTING.md`; the 【v1.5.0】 historical
notes in the file headers are retained.

Test baseline at every point is quoted from `README.md` / the suites themselves.

---

## v1.6.0 — 2026-10-21

### Added

- **SPD preset payload schema v3** (`f7f98f4`). The Step 2 result is stored on a
  canonical wavelength grid — **380–780 nm at 1 nm, 401 points** — instead of on
  sensor pixel indices. A preset no longer depends on the sensor width or on the
  pixel↔nm mapping it was recorded with. New payload keys: `axis`,
  `grid_points`, `mask`, `effective_band`, `masked_count`, `app_version`,
  `created_at`, `device_model`, `preset_type`; array keys are uniform snake_case
  (`response_r`, `response_g`, `response_b`).
- **Step 1 quality tiers** (`decc744`). `step1_quality_tier()` normalises the two
  sides' vocabularies (live: PASS / WARNING / FAILED; preset-side: GOOD /
  SUSPECT / FAILED / PRESET) into one tier. Unknown or missing labels are graded
  WARNING rather than silently passing.
- **Honest Step 1 status line** (`decc744`). Green `Done! Exp: … | G-err: ±X.XX nm
  (PASS)`; orange `Completed with WARNING — … G-err ±X.XX nm exceeds 2.5 nm;
  mapping may be biased.`; red `FAILED — … G-err ±X.XX nm exceeds 10 nm; mapping
  unreliable.`. A FAILED calibration still fills `self.mapping` (no hard block,
  matching `core.calibrate_wavelength` and Android) but additionally raises a
  warning dialog. Loading a non-PASS preset no longer opens a dialog titled
  "Success".
- **`[Calibration Quality]` line in the report** (`decc744`). The Step 1 parameter
  panel — including the exported PDF — now prints the tier, a plain-language
  conclusion, and the 2.5 / 10 nm thresholds it came from.
- **`csv_data_cleaner/` ships inside this repository** (`f3659cc`): the cleaning
  script, a README describing the steps, and the example raw dataset used during
  development. Paths are command-line arguments, the wavelength sort is stable,
  and the validation figure is written next to the CSV. Generated files are
  gitignored; pandas was not added to the dependency set.
- **`CONTRIBUTING.md`, `CLAUDE.md`, `.claude/RESPONSIBLE_AI_USE.md`** (`b88d908`):
  whether the project takes contributions, what a patch has to satisfy (offline
  by contract, pinned dependencies, English-only source, and the cross-platform
  parity gate with MycoSpec), how to run the tests, and how to report a
  reproducible failure.
- **Test suite restored and published** (`0f38189`): numerical, preset and
  cross-platform parity tests are part of the deliverable again —
  **279 passed, 15 skipped, 0 failed**. `FSSA_E2E_DATA` lets the opt-in
  end-to-end tests point at any local copy of the 10.15 DNG set. GUI tests now
  install the tkinter stubs unconditionally, so headless and on-display runs
  behave identically.
- **Own GitLab CI pipeline** (`b2878a3`): pinned dependencies installable, every
  module compiles, core modules import without a display. The iGEM runner has no
  Docker daemon, so Auto DevOps would fail permanently; shipping
  `.gitlab-ci.yml` stops it from running.

### Changed

- **Licence MIT → Apache License 2.0** (`fa42558`).
- **Dependencies are pinned** (`fa42558`): `numpy==2.4.2`, `scipy==1.17.0`,
  `matplotlib==3.10.8`, `tifffile==2026.7.31`, as a lockfile reproducing the
  verified environment. Auto DevOps disabled by shipping our own pipeline.
- **Comments, docstrings and README in English** (`7ce6ca2`, `0f38189`).
  33 user-facing GUI strings deliberately kept Chinese — that is product copy, not
  documentation.
- The Step 1 / Step 2 cache now carries what actually exists (`f7f98f4`):
  positioning profiles and `measured_total` are resampled onto the same
  wavelength grid as the response arrays, and exposure time / frame count are
  persisted with the preset.

### Removed

- **`opencv-python-headless`** (`fa42558`) — never imported anywhere.
- **Mode B recompute** (`f7f98f4`): `_recompute_from_source()`,
  `RECOMPUTE_MIN_DRIFT_NM`, `_sig6`, and the "auto-recompute SPD after loading a
  WAVELENGTH preset" coupling (a v3 preset carries its own wavelength axis).
- Payload keys `version`, `response_R/G/B`, `lamp_*`, `spd_reference`,
  `smooth_window` (`f7f98f4`). `slope` / `intercept` / `sensor_height` stay in
  the payload as pure diagnostics and never take part in reconstructing a curve.
- Not part of running `main.py`, so no longer shipped: `tools/`,
  `analysis_output/` (recoverable from tag `backup/pre-cleanup`) (`fa42558`).

### Fixed

- **A descending wavelength axis was accepted** (`f7f98f4`). `np.interp` in
  `gui._interp_response` then computed silently wrong values; the L1 check now
  requires a strictly increasing axis.
- **Step 1 quality could be painted green even when the calibration had failed**
  (`decc744`): the old status line always read "Done!". Covered by 32 new test
  cases (`tests/test_step1_quality_status.py`) and 6 new PDF-parity cases, all
  mutation-verified.
- **README claimed a saturation gate that does not exist** (`f8ebf1d`): it said
  "ROI saturation ≥ 1 % rejects the capture", but no Python code path checks
  saturation. The limitation now reads honestly — MycoSpec enforces the check at
  capture time; FSSA does not, so inspect the RAW preview before trusting a batch.

### Breaking change

An SPD preset written by v1.5.0 or earlier can no longer be loaded: the schema
gate refuses anything whose `schema_version` is not exactly 3, even with
`strict=False`, and tells you to re-run Step 2. Re-exporting with this version
restores it.

### Test baseline

279 passed, 15 skipped, 0 failed (`python -m pytest tests -q`).
The 15 skipped are the end-to-end tests that need real DNG frames (`FSSA_E2E=1`).

---

## Older versions

---

## v1.5.0 — 2026-10-17

### Added

- **Fluorescein (sodium salt)** fluorescence option: emission 520 nm, excitation
  494 nm, green channel, FWHM 25 nm.
- **Canonical fluorophore identifiers.** `FLUOROPHORES` keys became
  `YPET / EGFP / FLUORESCEIN / MCHERRY / CFP / MTURQUOISE2`, identical to the
  MycoSpec enum names, plus `canonical_fluorophore()` with an explicit alias
  table and case / space / hyphen normalisation — no fuzzy matching, to avoid
  false positives. Preset JSON always stores the canonical ID while the UI keeps
  showing the legacy `display_name`. Curve presets now interoperate between the
  desktop app and the Android app.

### Changed

- Version number unified to v1.5.0 across `config.py`, `core/calibration.py`,
  `core/wavelength_preset.py`, `pyproject.toml`, `requirements.txt`, the window
  titles, the Windows AppUserModelID and `README.md`. The 【v1.4.x】 historical
  notes are intentionally retained.

### Test baseline

204 passed, 15 skipped — same baseline as v1.4.8.

---

## v1.4.8 — 2026-10-17

### Added

- Step 3 shows **Analysis channel / Peak / FWHM** under the dye dropdown,
  updating as dyes switch; the run log gains an "Analysis channel" line
  (turbidity mode reports GREEN, its fixed channel) (`f914c6b`).

### Changed

- **Report PDF matches the report window** (`f914c6b`). All four Step parameter
  figures share one rendering routine: the canvas grows with the line count and
  text wraps at the monospace column width (CJK counted as two cells), matching
  the window's `wrap=WORD` pane. The exported PDF no longer relies on
  `bbox_inches='tight'` to drag overflowing text onto the page.

### Fixed

- **Step 3 could not select a fluorophore** (`b90fda9`). `load_curve_preset()`
  disabled the dye dropdown (added in v1.4.6 to prevent dye/curve mismatches),
  but `calculate_with_curve_preset()` recomputes the sample area from the
  curve's own `preset["fluorophore"]`, so what Step 3 showed was irrelevant to
  the result — it only looked broken. The dropdown now syncs to the preset's dye
  and stays selectable.
- **Mouse wheel over the dye dropdown crashed** (`b90fda9`):
  `AttributeError: 'str' object has no attribute 'master'`. `bind_all` also
  receives events from the Combobox popdown, whose `event.widget` is a raw path
  string; `_wheel_target_in_region()` now lets those through.

### Test baseline

169 passed, 15 skipped. Two new regression cases, each proven to fail on the
unpatched code (one reproduces the exact traceback).

---

## v1.4.7 — 2026-10-17

### Added

- **Detailed Analysis Report opens automatically** after a successful
  calculation, rebuilt so it always shows the latest run (`fb02e03`).
- **Provenance report page**, included in the PDF export (`fb02e03`): for each of
  Steps 1–4, the origin type, preset type (`WAVELENGTH` / `SPD` / `CURVE`), file
  name and absolute path, preset name, import timestamp and key parameters.
- Preset provenance in `core/` for all four steps (`fb02e03`).
- Step 4 preset buttons are hidden in turbidity mode, which has three steps
  (`fb02e03`).

### Changed

- Importing a Step 4 curve preset now **continues straight into the
  concentration calculation** when Step 3 sample and blank are present, and
  reports exactly which of Step 1 / Step 2 is missing otherwise. A live curve
  built earlier survives a run that cannot proceed (`fb02e03`).
- Provenance is cleared when a preset is cleared or an SPD preset is rejected for
  severe drift (`fb02e03`).

### Test baseline

152 passed, 15 skipped.

---

## v1.4.6 — 2026-10-17

### Added

- **Step 1 wavelength presets** (`4c2c313`): validate / save / load for the Step 1
  mapping, with flat keys aligned to Android `PresetEntity` plus a nested group.
- **Fully CFA-aware RAW decoding** (`4c2c313`): the CFA pattern is read from DNG
  tag 33422 with DNG/EXIF convention normalisation, demosaiced with a
  CFA-aware bilinear scheme identical to Android's
  `FrameProfileExtractor.bilinearInterpolate` (fallback RGGB), column energy is
  taken on raw data, ROI endpoints aligned, and crosstalk correction applied.

### Changed

- **SPD presets are drift-proof** (`4c2c313`): the lamp and reference SPD are
  stored with the preset and the response is recomputed against the current
  mapping on load (Mode B). L1 / L2 problems are hard-rejected; L3 drift only
  warns.
- `load_spd_params()` commits state only after validation succeeds, instead of
  polluting state before validation (`4c2c313`).

### Test baseline

117 passed, 15 skipped; with real DNGs (`FSSA_E2E=1`) 132 passed.

---

## v1.4.5 — 2026-10-16

### Changed

- **Response-proportional channel fusion replaces the logistic handoff relay**
  (`3c0c9c9`). Step 2 previously cross-faded channels over a 15 nm ramp around
  the response-curve crossings; weights are now the response proportions:

  ```
  w_c(λ) = resp_c(λ) / (resp_R + resp_G + resp_B + eps),   Σ w_c = 1
  ```

  The same rule and the same `eps` produce identical fused spectra on desktop and
  on phone, matching MycoSpec v2.2.6 exactly. `auto_handoffs()` survives as a
  diagnostic readout of equal-responsivity wavelengths.
- Figure 4d, its legend, the `[Formulas]` block and the crossing lines in the
  parameter panel were relabelled to match.
- Version references unified to v1.4.5 across the app.

### Added

- `tests/test_fusion_reference.py` locks the weight rule and `Σ w = 1` against a
  numeric reference vector shared with Android's `AlgorithmParityTest`.

---

## Before v1.4.5 — 2026-10-01 … 2026-10-16

No release tags exist for this window; these commits are recorded here because
they changed behaviour, with only the evidence available in history:

- **2026-10-01 — first published tree, v1.4.2** (`b24fcb5`): `core/`
  (`calibration`, `fusion`, `image_processing`, `quantification`,
  `spectral_features`, `turbidity`) and `gui/` (`main_window`,
  `details_window`, `turbidity_window`, `widgets`), 1 703 lines. Nothing is
  recorded about what preceded it; versions ≤ v1.4.1 predate this repository's
  history and are not listed.
- **2026-10-11** (`264308a`, +340 / −287): SPD calibration reworked in
  `core/fusion.py` and `gui/details_window.py`.
- **2026-10-11** (`00ee815`, +1 270 / −481): UI revision towards the layout used
  by the Android client — new `gui/theme.py`, rewritten main and details windows,
  new widgets.
- **2026-10-16** (`1584d9f`, +465): `core/unmixing.py` introduced, calibration and
  metadata handling extended, with three new test modules
  (`test_calibration_response`, `test_image_processing_metadata`,
  `test_unmixing_reference`). This is where the spectral-unmixing metric landed.
- **2026-10-16** (`8bee466`, +342): further algorithm work across
  `core/image_processing.py` and `core/spectral_features.py`, application icon
  and bundled `assets/fssa.ico`.

