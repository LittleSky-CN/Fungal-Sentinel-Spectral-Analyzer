# CSV Data Cleaner

Turns a spectral-power-distribution **plot** from a manufacturer datasheet into the
`true_spd.csv` that **Step 2 (Spectral response / SPD)** of the FSSA and MycoSpec
pipeline consumes.

Datasheets publish these curves as images, not as data. This is the step that gets
the numbers out of the image and onto a defensible grid.

## Install

The utility is standalone and needs **pandas** in addition to FSSA's runtime
dependencies (pandas is deliberately not one of them):

```bash
pip install pandas      # numpy / scipy / matplotlib already come with FSSA
```

## Usage

```bash
# clean our example input — writes true_spd.csv next to it
python csv_data_cleaner.py

# clean your own export
python csv_data_cleaner.py my_datasheet_points.csv true_spd.csv

# skip the comparison figure
python csv_data_cleaner.py my_datasheet_points.csv true_spd.csv --no-plot
```

Input format: a **headerless two-column CSV**, one point per line —
`wavelength_nm, intensity`. That is what a plot digitizer such as
WebPlotDigitizer exports by default.

Output: `true_spd.csv` — 401 rows, `Wavelength_nm,Relative_Intensity`, one row per
nanometre from 350 to 750 nm, peak normalised to 1.0. Drop it straight into
Step 2.

Running it also saves `spectrum_cleaning_validation.png` beside the output: raw
digitised points against the cleaned curve, so you can check by eye that nothing
was invented.

## What it does — and why in that order

| # | Step | Why |
|---|---|---|
| 1 | Read the headerless CSV | Matches the standard plot-digitizer export. |
| 2 | Clip negative values to 0 | Negative readings are baseline noise from digitising, not signal. |
| 3 | Sort strictly by wavelength | Interpolation needs a monotone x-axis. The sort is stable, so repeated wavelengths keep their file order and results are reproducible run to run. |
| 4 | Drop duplicate wavelengths (keep first) | A zero-length x-interval would break PCHIP. |
| 5 | Normalise to the peak = 1.0 | Step 2 only needs the shape; absolute radiometric units belong to the characterisation, not to this table. |
| 6 | Resample to 350–750 nm at 1 nm with **PCHIP** | Shape-preserving monotone cubic interpolation (Fritsch–Carlson, 1980). |
| 7 | Write the CSV + validation figure | The figure is how you notice a bad digitisation before it reaches Step 2. |

**Why PCHIP and not a natural cubic spline.** Between sparse digitised points a
natural spline overshoots and produces emission peaks that are not in the
datasheet. PCHIP stays within the sampled values, so the reference spectrum cannot
gain structure the manufacturer never published.

Points outside the digitised range get **0**, not an extrapolated tail: there is
no evidence out there, and a made-up tail becomes a made-up response correction.

## Files

```
csv_data_cleaner.py            the utility
example_led_dataset_raw.csv    553 raw points digitised from a white-LED
                               datasheet plot (blue chip region near 450 nm,
                               phosphor band near 550 nm); used during
                               development, useful as a smoke test input
```

`true_spd.csv` and `spectrum_cleaning_validation.png` are regenerated on every run
and are not kept in version control.

## Reference

Fritsch, F. N. & Carlson, R. E. Monotone piecewise cubic interpolation.
*SIAM Journal on Numerical Analysis* **17**(2), 238–246 (1980).
