# Contributing to FSSA

FSSA is the Python reference implementation of the Fungal Sentinel measurement
pipeline. It is written by **team CAU-China (iGEM 2026)** and is published for
reuse: everything here exists so that **a team with a 3D printer, a phone and
these instructions can rebuild the instrument and its software.**

- Repository: <https://gitlab.igem.org/2026/software/cau-china/fungal-sentinel-spectral-analyzer>
  (mirror: <https://github.com/LittleSky-CN/Fungal-Sentinel-Spectral-Analyzer>)
- Android sibling: **MycoSpec** — <https://gitlab.igem.org/2026/software/cau-china/mycospec>
- Latest: **FSSA v1.5.0** · License: **Apache-2.0** (see [LICENSE](LICENSE))

This document answers three questions: *what do we accept*, *how do you get set
up*, and *what has to be true before a change lands*.

---

## 1. We are open to contributions

Bug reports, patches, ports, hardware reports and documentation are all welcome.
Ranked by how much they help the next team:

| Contribution | Why it matters here |
|---|---|
| **A reproducible failure** on your hardware or your dataset | Highest value. It tells the next team where the procedure breaks — see §6. |
| Rebuilding the OptiX optics and reporting what differed | The hardware is only reusable if someone outside the team can build it. |
| Porting the pipeline to another platform or language | The pipeline is specification-first (physical units, written down once). A third implementation strengthens the parity gate. |
| New canonical fluorophore IDs / emission windows | Must land on **both** sides — see §5. |
| Troubleshooting notes, workflow docs, UI text in another language | iGEM's Contribution criterion explicitly rewards documented troubleshooting that future teams can reuse. |
| GUI / UX / performance improvements | Include screenshots and the before/after behaviour. |

### What we ask before you start

1. Open an issue or a merge request **before** starting anything large, so we can
   say whether the change belongs here or on the Android side.
2. Say which implementation(s) your change touches. Anything that touches a
   **shared algorithm** must clear the parity gate in §5.
3. Keep the repository **source-only**: no executables, APKs, `.pyc` trees,
   datasets or build output. Large or raw data goes to Zenodo and is referenced
   from `README.md` — iGEM expects a repository that another team can carry.
4. **Commit history is part of the deliverable.** Write descriptive English
   commit messages; iGEM asks us to leave a history another team can resume from.
5. **Every number you write must be reproducible.** State the version, the commit
   and the command you ran. Do not quote a figure you did not produce, and do not
   present a placeholder figure as a measurement.

---

## 2. Ground rules for anything that lands

- **Offline by contract.** Neither FSSA nor MycoSpec requests network access, and
  neither ships telemetry or analytics. A patch that adds an HTTP call, a
  remote model API or cloud processing will be rejected no matter how useful it
  looks — for a project that handles biological samples this removes the
  third-party-processing question rather than answering it with a policy.
- **English-only source.** Comments, docstrings, identifiers and the README are
  English. The only Chinese left in the project is the Android app's
  `values-zh` UI locale, which is data, not code.
- **Numerical code lives in `core/`, presentation in `gui/`.** Anything that
  changes a number must be testable without a display.
- **Dependencies are pinned.** `requirements.txt` records the exact versions FSSA
  was last verified with. Adding a dependency means adding it there at an exact
  version, explaining why it is needed, and confirming a clean-environment
  install still works.
- **Do not "fix" the test stubs.** The GUI tests stub tkinter unconditionally so
  that the suite behaves identically on a headless CI runner and on a laptop with
  a display. Letting real tkinter back in breaks every widget-building test —
  see the comment in `tests/_install_tk_stubs()`.
- **Update `README.md`** whenever behaviour, requirements or a reference number
  changes. It is the part judges and future teams read.

---

## 3. Development setup

Requires **Python ≥ 3.10 with tkinter**; developed and verified on CPython
3.13.12 (Windows/macOS/Linux).

```bash
git clone https://gitlab.igem.org/2026/software/cau-china/fungal-sentinel-spectral-analyzer.git
cd fungal-sentinel-spectral-analyzer

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # pinned runtime deps (numpy, scipy, matplotlib, tifffile)
pip install pytest                   # test runner — dev-only, deliberately not pinned with runtime deps
```

On Debian/Ubuntu the GUI needs the tkinter package (`sudo apt-get install
python3-tk`). Run the application with `python main.py`.

The numbers quoted below were produced with **pytest 9.1.1**; any current pytest
behaves the same, but please quote your own pytest version when reporting results.

---

## 4. Running the tests

```bash
python -m pytest tests -q
```

Expected result on the pinned dependency set:

```
279 passed, 15 skipped in ~20 s
```

The suite runs on **synthetic input only**: no device, no DNG files, no network.
What it pins is the part a reuser cares about — the two-point wavelength fit and
its G-point residual, response-proportional channel fusion, preset reuse under a
drifted mapping, spectral unmixing, replicate clipping, the six-part provenance
record and its PDF export, and the Step-1 quality grading.

### The 15 skipped tests

They are the end-to-end tests against real DNG frames. They need the **10.15
dataset** (`positioning.dng`, `spd.dng`, `true_spd.csv`, `B0.dng`–`B3.dng`,
`S.dng`), which is large and therefore archived rather than committed (Zenodo —
see `README.md`). To run them:

```bash
# Windows PowerShell
$env:FSSA_E2E = "1"
$env:FSSA_E2E_DATA = "C:/path/to/10.15"
python -m pytest tests -q

# bash / zsh
FSSA_E2E=1 FSSA_E2E_DATA=/path/to/10.15 python -m pytest tests -q
```

Decoding seven 25 MB DNGs takes a couple of minutes, which is exactly why they
are opt-in. If you see a failure **here** — on real frames but not on synthetic
ones — please report it even if you cannot fix it (§6).

---

## 5. The parity gate — read this before touching a shared algorithm

The reusable idea in this project is not either program. It is that the pipeline
is **written down once in physical units, implemented twice** (Python here,
Kotlin in MycoSpec) and tested for agreement. Where the two implementations
overlap, their tests carry identical expectations:

| Expectation | Meaning |
|---|---|
| Spectral unmixing coefficients are recovered to a tight tolerance | The reference is taken as *(highest standard − its blank)*, with blank and constant offset fitted out by least squares. |
| Response calibration does not smooth an isolated spike | The per-channel response is kept unsmoothed. |
| A broad Gaussian peak survives spike removal, an isolated one does not | Local median-absolute-deviation test with a 2 % amplitude floor. |
| Fusion weights equal the response proportions and sum to one | `w_c(λ) = resp_c(λ) / Σ_c resp_c(λ)`, `Σ_c w_c(λ) = 1` pointwise. |

**Running both suites is the gate for any change to a shared algorithm.**

```bash
# Python side (this repository)
python -m pytest tests -q                 # ≥ 279 passed

# Android side
cd /path/to/mycospec
./gradlew :app:testDebugUnitTest          # ≥ 86 tests across 9 classes
```

A merge request that changes a shared algorithm and reports only one side will
not be merged. If the numbers legitimately have to change, change the expectation
in **both** suites in the same MR and say why.

`tests/test_image_parity_android.py` is the file that pins the cross-platform
behaviour directly; untouched numbers there mean one side has drifted.

---

## 6. Reporting something that does not reproduce

A reproducible failure reported back is worth more to the next team than another
feature. Open an issue on iGEM GitLab with:

```
Version / commit:      FSSA v1.5.0 @ <commit sha>
Platform:              OS, Python version, tkinter available? yes/no
Input:                 which DNGs or synthetic fixture (attach if small)
Steps 1-4 inputs:      positioning frames, characterised source + its SPD CSV, blanks, standards
Expected:              the number from README.md §"Reproducing the main results"
Observed:              your number, plus Step 1 quality grade
What you ruled out:     e.g. "same result with a fresh venv", "S-frame alone also fails"
```

Hardware notes belong here too: phone model and whether it exposes Camera2 RAW
at all (some Huawei-class devices do not; the workaround is to import DNG frames
from the gallery — see the Android README).

---

## 7. Using AI assistants

This repository ships configuration for AI coding assistants. If you use one,
read **[.claude/RESPONSIBLE_AI_USE.md](.claude/RESPONSIBLE_AI_USE.md)** first.
In short: never fabricate data, measurements, citations or attributions; keep a
human in the loop who reviewed and ran every line; commit no secrets; and say in
the merge request description which parts were AI-drafted.

---

## 8. Repository layout

```
config.py              constants, thresholds and canonical fluorophore IDs
main.py                entry point -> GUI
core/                  the numerics: calibration, image_processing, fusion,
                       quantification, unmixing, spectral_features, turbidity,
                       and the three preset types (wavelength / spd / curve)
gui/                   tkinter presentation only (main_window, details_window,
                       turbidity_window, widgets, theme)
csv_data_cleaner/      turns a datasheet SPD plot into the true_spd.csv Step 2
                       consumes (needs pandas, which is not a runtime dependency)
tests/                 synthetic-input suite + opt-in real-DNG end-to-end tests
```

Nothing here phones home, nothing here needs a network, and nothing here needs
anything you cannot install offline from `requirements.txt`.

Copyright 2026 CAU-China (iGEM Fungal Sentinel team). Contributions are accepted
under the same **Apache License 2.0** as the rest of the repository.
