# FSSA — Fungal Sentinel Spectral Analyzer

Open-source, offline-first spectral analysis software for a low-cost portable fungal
fluorescence detector. Smartphone DNG + grating spectroscopy → quantitative
fluorescence and turbidity.

**FSSA v1.6.0** · Python desktop reference implementation · iGEM 2026, team CAU-China

Android counterpart: **MycoSpec** — <https://github.com/LittleSky-CN/Mycospec>

---

## What it does

FSSA turns RAW (DNG) frames from an ordinary phone camera into calibrated spectra and,
from those, into concentrations. The phone sits behind a fixed **slit–grating–sample-holder**
rig; the grating spreads the sample's emission across the sensor, so sensor position maps
to wavelength. FSSA performs that mapping and everything downstream — entirely offline,
with no network calls and no cloud processing.

The pipeline is four steps, each producing an artefact the next step depends on:

| Step | Input | Output |
|---|---|---|
| **1 · Wavelength calibration** | 1–5 R/G/B positioning frames | Pixel↔nm mapping (two-point R/B fit), auto X-ROI, G-point validation, quality grade |
| **2 · Spectral response (SPD)** | 1–5 frames of a characterised source + its reference SPD curve | Exposure-normalised per-channel response coefficients |
| **3 · Sample analysis** | 1–5 blank + 1–5 sample frames | Blank-subtracted band integral in the fluorophore window, replicate SD, corrected spectrum |
| **4 · Standard curve** | 2–10 concentration groups (blank + sample each), or a saved curve preset | OLS regression (slope, intercept, R²), predicted concentration, quality warnings |

Channel fusion uses response-proportional normalisation,
`w_c(λ) = resp_c(λ) / Σ_c resp_c(λ)` with `Σ_c w_c(λ) = 1` pointwise (`core/fusion.py`).
This is numerically identical to the Android build, so a given DNG yields the same fused
spectrum on desktop and on phone.

Fluorophores are identified by **canonical IDs** (`YPET`, `EGFP`, `FLUORESCEIN`,
`MCHERRY`, `CFP`, `MTURQUOISE2`) — the same identifiers MycoSpec uses, so presets and
curves travel between the two code bases.

## Who it is for

- **iGEM teams and teaching labs** building a grating-based fluorescence detector who need
  real spectral analysis without buying a laboratory spectrometer.
- **Anyone holding DNG/RAW frames** from such a rig who wants a reproducible pixel↔nm
  mapping and a defensible concentration number.
- **Developers maintaining MycoSpec (Android).** FSSA is the reference implementation:
  the two are expected to agree numerically on the same input, and FSSA is where a
  disagreement is investigated first.

It is **not** for clinical, diagnostic or otherwise regulated measurement. It is a
research and teaching tool, and its numbers are only as good as the optics, the
characterised source and the blanks you feed it (see *Limitations*).

## Install

Requires **Python ≥ 3.10** (developed and verified on CPython 3.13.12) with **tkinter**
for the GUI.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # pinned versions — see "Dependencies" below
```

On Debian/Ubuntu the GUI needs the tkinter package, which is not always bundled:

```bash
sudo apt-get install python3-tk
```

Dependencies are pinned in `requirements.txt` (`numpy==2.4.2`, `scipy==1.17.0`,
`matplotlib==3.10.8`, `tifffile==2026.7.31`) so that an install reproduces the exact
environment the software was verified in.

## Run

```bash
python main.py
```

Then work through the four tabs in order (Step 1 → Step 4). Each step can save its
calibration as a reusable preset, so a later session can skip straight to Step 3.

Preset types:

- **WAVELENGTH** (Step 1) — pixel↔nm mapping plus ROI, peaks and G-point residual.
- **SPD** (Step 2) — per-channel response *and* the wavelength mapping it was computed
  with. A preset loaded under a drifted mapping is recomputed, so reusing it is
  equivalent to re-running Step 2 with the current mapping. Drift is only ever a
  warning: ≤ 2 nm silent, 2–10 nm yellow, > 10 nm orange with confirmation — never a
  silent rejection.
- **CURVE** (Step 4) — standard curves and their fitted slope/intercept. Unmixing
  presets store the reference spectrum against wavelength (nm), not sensor pixel
  positions, so a saved curve applies to a new Step 3 blank/sample pair.

## Reproducing the main results

**The measurement dataset is not in this repository.** Raw DNG frames are large binary
artefacts; per iGEM guidance they are archived separately and referenced from here
rather than committed.

1. Download the **10.15 dataset** (the run every number below comes from):
   `positioning.dng`, `spd.dng`, `true_spd.csv`, `B0.dng` (blank), `B1.dng` (100 mg/L),
   `B2.dng` (80 mg/L), `B3.dng` (20 mg/L), `S.dng` (unknown).
   **Zenodo record: _pending publication — DOI will be listed here._**

2. Run the four steps on it: Step 1 ← `positioning.dng`; Step 2 ← `spd.dng` +
   `true_spd.csv`; Step 3 ← `B0.dng` as blank with `B1/B2/B3/S` as samples; Step 4 ←
   the 0/20/80/100 mg/L groups.

3. Compare against the values from the archived FSSA v1.5.0 run on that dataset:

   | Quantity | Reference value |
   |---|---|
   | Step 1 mapping | `p = −2.739832 · λ + 3198.8985` px |
   | Step 1 G-point residual | **+5.57 nm** → quality grade `WARNING` |
   | Step 3 net area, YPET window 517–537 nm | B1 = 0.267636, B2 = 0.293935, B3 = 0.257038, S = 0.254361 |
   | Step 4 R², YPET (OLS) | 0.638 (weighted `1/y²`: 0.678) |

   Step 4 on this dataset is **weak on purpose**: with only four concentration groups the
   linear fit reaches R² ≈ 0.29–0.68 depending on fluorophore, and alternative
   treatments of the 0 mg/L point fit better (log₁₀(C+1) vs area: R² = 0.945). This is a
   property of the dataset, not of the pipeline, and it is the main open question the
   project is still working on — see *Limitations*.

## Tests

The suite runs on **synthetic input only** — no device, no DNG files, no network:

```bash
pip install pytest              # dev-only: not a runtime dependency
python -m pytest tests -q
```

Verified result on CPython 3.13.12 with the pinned dependencies above:
**279 passed, 15 skipped, 0 failed**.

- **15 skipped** are the end-to-end tests against real DNG frames. They need the
  10.15 dataset, which is archived rather than committed; enable them with
  `FSSA_E2E=1` and point `FSSA_E2E_DATA` at your copy of the frames. Details in
  [`CONTRIBUTING.md`](CONTRIBUTING.md#4-running-the-tests).
- GUI tests stub tkinter unconditionally, so the suite behaves identically on a
  headless CI runner and on a laptop with a display.

Any change to an algorithm shared with the Android side must also keep
MycoSpec's suite green — that parity gate is described in
[`CONTRIBUTING.md`](CONTRIBUTING.md#5-the-parity-gate--read-this-before-touching-a-shared-algorithm).

## Large files and datasets

This repository contains **source only** — no binaries, no datasets, no model weights.
Packed size is well under 1 MB. The 10.15 DNG dataset lives on Zenodo (link above once
published); the Android APK builds live on GitHub Releases.

## Dependencies

Runtime dependencies are pinned in `requirements.txt`. `pyproject.toml` declares the
same set as ranges for library-style installation.

OpenCV (`opencv-python-headless`) was previously declared but is **not imported anywhere**
in the code base; it has been removed from both files.

## AI use

**The software does not use AI.** FSSA contains no model inference, no training code and
no calls to any commercial or open-weight model API — every number it produces comes from
deterministic numerical code (`numpy` / `scipy` linear algebra and regression). There is
no model to document, no training data and no evaluation set.

**During development**, the team used AI coding assistants (conversation-based code
generation) for refactoring, test writing and documentation drafts. Those assistants
contributed code that a human reviewed, ran and, where it was wrong, rewrote. No
scientific result in this repository was produced by an AI model, and no AI-generated
figure is presented as measured data.

## Limitations

- Quantification assumes **fixed optics**, known R/G/B positioning wavelengths, a
  characterised source whose SPD matches the one you supply, and proper blanks and
  standards. Break any of these and the numbers degrade silently.
- On the 10.15 dataset, Step 1 grades `WARNING` (G-point residual 5.57 nm, above the
  2.5 nm `PASS` threshold). Downstream results inherit that uncertainty.
- Step 4 fits are only as good as the concentration series behind them; with four groups
  the R² values are low (see the table above).
- Camera-auto behaviour (tone mapping, denoising) is bypassed by reading RAW, but sensor
  saturation still invalidates a frame. MycoSpec enforces this at capture time (a frame
  with ≥ 1 % saturated pixels inside the ROI is rejected); FSSA has no automatic
  saturation gate — inspect the RAW preview before trusting a batch.

## Digitising a lamp datasheet (the Step 2 input)

Step 2 needs the reference SPD of your characterised source, and manufacturers
publish that as a plot rather than as data. The [`csv_data_cleaner/`](csv_data_cleaner)
utility in this repository converts points exported from a plot digitizer into the
`true_spd.csv` Step 2 expects (401 rows, one per nanometre from 350 to 750 nm,
peak normalised to 1.0), using shape-preserving PCHIP interpolation so the resampled
curve cannot gain peaks the datasheet does not show.

```bash
pip install pandas                      # dev/dataset dependency, not a runtime one
python csv_data_cleaner/csv_data_cleaner.py my_points.csv true_spd.csv
```

See [`csv_data_cleaner/README.md`](csv_data_cleaner/README.md) for the full procedure.

## Contributing

**Yes — we are open to contributions**, and we most want the ones that help the
next team rebuild this instrument: porting the pipeline to another platform,
adding a fluorophore to the canonical ID set, rebuilding the OptiX optics, or
simply telling us that a step in the procedure did not reproduce on your
hardware. A reproducible failure reported back is worth more here than another
feature.

What we ask: live entirely offline, keep numerics in `core/` and presentation in
`gui/`, pin any new dependency, write English comments and commit messages, and
run **both** the Python and the Android test suites when you touch a shared
algorithm.

Full policy, setup steps, the parity gate and the bug-report template are in
[`CONTRIBUTING.md`](CONTRIBUTING.md). Using an AI assistant? Read
[`.claude/RESPONSIBLE_AI_USE.md`](.claude/RESPONSIBLE_AI_USE.md) first.

## License

**Apache License 2.0** — see [`LICENSE`](LICENSE).

Copyright 2026 CAU-China (iGEM Fungal Sentinel team). Apache-2.0 is OSI-approved and was
chosen for its explicit patent grant and clean attribution on forks.

