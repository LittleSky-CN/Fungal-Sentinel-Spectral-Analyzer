"""Turn a digitised datasheet spectrum into the true_spd.csv Step 2 consumes.

Why this exists
---------------
Manufacturer datasheets publish spectral power distributions as *plots*, not as
data. This utility converts points exported from a plot digitizer
(WebPlotDigitizer was used while developing this) into the clean, evenly spaced
table that Step 2 of the FSSA / MycoSpec pipeline expects.

What it does, in order
----------------------
1. reads a two-column CSV (wavelength, intensity) with no header;
2. clips negative values to zero -- negative readings are baseline noise left
   over from digitising, not signal;
3. sorts strictly by wavelength and drops duplicate wavelength points;
4. normalises so the peak intensity is 1.0;
5. resamples onto a 350-750 nm grid at 1 nm with **PCHIP**
   (Fritsch-Carlson monotone cubic interpolation), so sparse digitised points
   are joined without inventing peaks;
6. writes `Wavelength_nm,Relative_Intensity` and saves a validation plot.

**PCHIP, not a natural cubic spline.** A natural spline overshoots between
sparse points and creates emission structure that is not in the datasheet. PCHIP
keeps the curve between neighbouring samples, which is exactly what we want when
the curve later becomes the reference SPD for a response calibration.

Reproducing an extraction that is already known to be bad is worse than useless:
if you digitise a different lamp, re-run this and keep whatever it prints.

Usage
-----
    python csv_data_cleaner.py                       # Default Dataset.csv -> true_spd.csv
    python csv_data_cleaner.py INPUT.csv OUTPUT.csv
    python csv_data_cleaner.py --no-plot INPUT.csv

Requires pandas, numpy, scipy and matplotlib. pandas is not a dependency of FSSA
itself -- install it with:  pip install pandas
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import PchipInterpolator

GRID_START_NM = 350
GRID_STOP_NM = 750
GRID_STEP_NM = 1


def read_raw_points(path):
    """Read a headerless two-column CSV as wavelength / intensity."""
    if not os.path.isfile(path):
        print(f"ERROR: input file not found: {path}")
        sys.exit(1)

    frame = pd.read_csv(path, header=None, names=["wl", "intensity"])
    frame = frame.dropna()
    print(f"Read {len(frame)} raw points from {path}.")
    return frame


def clean(frame):
    """Clip the baseline, order the points, drop duplicates and normalise."""
    frame = frame.copy()
    frame["intensity"] = frame["intensity"].clip(lower=0)
    frame = frame.sort_values("wl", kind="mergesort")
    before = len(frame)
    frame = frame.drop_duplicates(subset=["wl"], keep="first")

    if len(frame) < 2:
        print("ERROR: fewer than two usable points remain after cleaning.")
        sys.exit(1)

    peak = frame["intensity"].max()
    if peak <= 0:
        print("ERROR: the input has no positive intensity anywhere.")
        sys.exit(1)

    frame["intensity"] = frame["intensity"] / peak
    peak_wl = frame.loc[frame["intensity"].idxmax(), "wl"]
    print(f"Cleaned: clipped negatives, sorted, removed {before - len(frame)} "
          f"duplicate point(s), normalised (peak at {peak_wl:.1f} nm).")
    return frame


def resample(frame):
    """Mono-cubic resampling onto the evenly spaced output grid."""
    target_wl = np.arange(GRID_START_NM, GRID_STOP_NM + 1, GRID_STEP_NM)

    # extrapolate=False: outside the digitised range there is no evidence, so
    # those bins stay NaN and become 0 rather than a made-up tail.
    interpolator = PchipInterpolator(
        frame["wl"].to_numpy(dtype=float),
        frame["intensity"].to_numpy(dtype=float),
        extrapolate=False,
    )
    target_int = interpolator(target_wl)
    target_int = np.nan_to_num(target_int, nan=0.0)
    target_int = np.clip(target_int, 0.0, 1.0)

    return pd.DataFrame({
        "Wavelength_nm": target_wl,
        "Relative_Intensity": np.round(target_int, 5),
    })


def write_validation_plot(raw, cleaned, path):
    """Comparison figure so the cleaning can be checked by eye."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(raw["wl"], raw["intensity"], s=15, color="red", alpha=0.6,
               label="Raw digitised points (sorted & normalised)", zorder=3)
    ax.plot(cleaned["Wavelength_nm"], cleaned["Relative_Intensity"], "k-",
            linewidth=2, label="Cleaned & resampled (PCHIP)", zorder=2)

    ax.axvline(450, color="blue", linestyle=":", alpha=0.5, label="Blue chip region")
    ax.axvline(550, color="orange", linestyle=":", alpha=0.5, label="Phosphor region")

    ax.set_title("White LED spectrum: raw digitised vs cleaned physical model")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Relative intensity (normalised)")
    ax.set_xlim(380, 750)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Validation plot written to {path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", nargs="?",
                        default=os.path.join(here, "example_led_dataset_raw.csv"),
                        help="digitised two-column CSV (wavelength, intensity)")
    parser.add_argument("output", nargs="?", default=None,
                        help="destination file for Step 2 "
                             "(default: true_spd.csv beside the input file)")
    parser.add_argument("--no-plot", action="store_true",
                        help="skip the validation figure")
    args = parser.parse_args()

    output = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.input)), "true_spd.csv")

    raw = clean(read_raw_points(args.input))
    cleaned = resample(raw)

    cleaned.to_csv(output, index=False)
    print(f"Done: {len(cleaned)} rows written to {output} "
          f"({GRID_START_NM}-{GRID_STOP_NM} nm at {GRID_STEP_NM} nm).")

    if not args.no_plot:
        # Put the figure next to the CSV it documents, whichever directory that
        # happens to be, so a stray copy never lands in the repository root.
        plot_path = os.path.join(
            os.path.dirname(os.path.abspath(output)),
            "spectrum_cleaning_validation.png",
        )
        write_validation_plot(raw, cleaned, plot_path)


if __name__ == "__main__":
    main()
