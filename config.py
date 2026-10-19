# config.py
"""FSSA v1.5.0 Global Configuration.

[v1.5.0] Added the Fluorescein (sodium salt) fluorophore option (emission 520 nm /
excitation 494 nm / green channel / FWHM 25); fluorophore identifiers are unified as
canonical IDs (the FLUOROPHORES dict keys are now YPET/EGFP/FLUORESCEIN/MCHERRY/CFP/MTURQUOISE2,
and canonical_fluorophore() normalises them) so curve presets interoperate with the Android app.
"""

# =============================================================================
# Fluorophore / target database
# =============================================================================
# key is the canonical ID (identical to the Android MycoSpec enum name); display_name is the
# legacy wording shown to the user by the Python side and written into existing presets. Preset JSON always stores canonical.
FLUOROPHORES = {
    "YPET":         {"display_name": "Ypet (Venus variant)",      "peak_wl": 527, "channel": "green", "fwhm": 20, "color": "green"},
    "EGFP":         {"display_name": "EGFP",                      "peak_wl": 509, "channel": "green", "fwhm": 20, "color": "green"},
    # Fluorescein sodium: emission ~520 nm (green channel), excitation ~494 nm. fwhm is the full
    # width of the integration window (peak +/- fwhm/2); the real emission peak is broader (~40 nm),
    # so 25 is a conservative value that keeps it in the same class as the other options - raise this one number if you need a wider integration window.
    "FLUORESCEIN": {
        "display_name": "Fluorescein (sodium salt)",
        "peak_wl": 520, "channel": "green", "fwhm": 25, "color": "green",
        "excitation_nm": 494,   # metadata: the UI has no excitation-wavelength field, so it is recorded with the config for now
    },
    "MCHERRY":      {"display_name": "mCherry",                   "peak_wl": 610, "channel": "red",   "fwhm": 25, "color": "red"},
    "CFP":          {"display_name": "CFP",                       "peak_wl": 475, "channel": "blue",  "fwhm": 20, "color": "blue"},
    "MTURQUOISE2":  {"display_name": "mTurquoise2",               "peak_wl": 482, "channel": "blue",  "fwhm": 20, "color": "cyan"},
}

# Explicit alias table: canonical ID -> every known spelling (case / space / hyphen differences are
# absorbed by normalisation, but "which strings count as the same fluorophore" must be listed exhaustively - no fuzzy matching, to avoid false positives).
_FLUOROPHORE_ALIASES = {
    "YPET": (
        "YPET", "ypet", "Ypet", "YPet",
        "Ypet (Venus variant)", "Ypet (venus variant)", "YPET (Venus variant)",
    ),
    "EGFP": (
        "EGFP", "egfp", "EGFP ", "EGFP (enhanced GFP)", "enhanced GFP", "Enhanced GFP",
    ),
    "FLUORESCEIN": (
        "FLUORESCEIN", "fluorescein", "Fluorescein",
        "Fluorescein (sodium salt)", "fluorescein (sodium salt)",
        "Fluorescein sodium salt", "fluorescein sodium salt",
        "sodium fluorescein", "Sodium fluorescein", "Fluorescein Na",
    ),
    "MCHERRY": (
        "MCHERRY", "mCherry", "mcherry", "m-cherry", "m-Cherry", "m Cherry", "mCherry ",
    ),
    "CFP": (
        "CFP", "cfp", "Cfp", "CFP (cyan fluorescent protein)", "cyan fluorescent protein",
    ),
    "MTURQUOISE2": (
        "MTURQUOISE2", "mturquoise2", "mTurquoise2", "MTurquoise2",
        "m-turquoise2", "mTurquoise-2", "mTurquoise 2", "mTurquoise2 ",
    ),
}


def _normalize_fluorophore_name(name: str) -> str:
    """Normalise: lowercase + strip spaces/hyphens/underscores, for case- and separator-insensitive comparison."""
    return "".join(ch for ch in str(name).strip().lower() if ch not in " \t-_")


# normalised alias -> canonical ID. Canonical keys and display_name are both registered, so legacy
# spellings such as the old Python display names inside existing presets still resolve.
_FLUOROPHORE_LOOKUP = {}
for _cid, _aliases in _FLUOROPHORE_ALIASES.items():
    for _alias in (*_aliases, _cid, FLUOROPHORES[_cid]["display_name"]):
        _FLUOROPHORE_LOOKUP[_normalize_fluorophore_name(_alias)] = _cid


def canonical_fluorophore(name: str | None) -> str | None:
    """Resolve any spelling to a canonical ID; returns None when it cannot be recognised.

        Resolution order: exact canonical match -> normalised match (alias table + display_name).
        Case, spaces and hyphen/underscore differences are all insensitive.
    """
    if name is None or not isinstance(name, str):
        return None
    if name in FLUOROPHORES:
        return name
    key = _normalize_fluorophore_name(name)
    return _FLUOROPHORE_LOOKUP.get(key) if key else None


def fluorophore_display_name(canonical_id: str | None) -> str | None:
    """canonical ID -> localised display name; unknown IDs are returned unchanged (the UI prefers showing the raw string over crashing)."""
    if canonical_id is None:
        return None
    return FLUOROPHORES.get(canonical_id, {}).get("display_name", canonical_id)

# =============================================================================
# Core algorithm default parameters (read by core/calibration.py and others)
# =============================================================================
DEFAULT_PARAMS = {
    "black_level": 64.0,                 # fallback value when the DNG carries no black-level metadata
    "display_percentile": 99.7,          # preview normalisation percentile
    "spd_threshold_ratio": 0.1,          # SPD credibility threshold ratio (consistent with full_report)
    "valid_wavelength_range": (430.0, 660.0),  # hard band mask (consistent with the report's Effective Band)
    "smooth_window": 21,                 # smoothing window (set to 1 in the v1.4.5 flow to disable smoothing entirely)
}

# =============================================================================
# UI theme configuration
# =============================================================================
THEME_ENABLED = True

PIXEL_FONT = {
    "path": "assets/pixel.ttf",
    "family": "Press Start 2P",
}