# core/quantification.py
"""Concentration quantification algorithms (Legacy & IUPAC LoD/LoQ)."""
import numpy as np
from scipy.stats import linregress
from typing import Dict, List, Optional

def calculate_concentration(
    concentrations: List[float], 
    means: List[float], 
    sample_mean_area: float, 
    mode: str = "lod_iupac",
    regression_method: str = "ols",
    weights: Optional[np.ndarray] = None
) -> Dict:
    concentrations = np.array(concentrations)
    means = np.array(means)
    
    if regression_method == "ols":
        w = None
    elif regression_method == "weighted_1y":
        w = 1.0 / np.where(means == 0, 1e-6, means)
    elif regression_method == "weighted_1y2":
        w = 1.0 / np.where(means == 0, 1e-6, means) ** 2
    elif regression_method == "custom":
        if weights is None: raise ValueError("weights must be provided")
        w = np.array(weights)
    else:
        raise ValueError(f"Unknown regression method: {regression_method}")
    
    if w is not None:
        w_sum = np.sum(w)
        wx_sum = np.sum(w * concentrations)
        wy_sum = np.sum(w * means)
        wxx_sum = np.sum(w * concentrations ** 2)
        wxy_sum = np.sum(w * concentrations * means)
        
        denominator = w_sum * wxx_sum - wx_sum ** 2
        if denominator == 0: raise ValueError("Singular matrix")
        
        slope = (w_sum * wxy_sum - wx_sum * wy_sum) / denominator
        intercept = (wxx_sum * wy_sum - wx_sum * wxy_sum) / denominator
        
        y_pred = slope * concentrations + intercept
        ss_res = np.sum(w * (means - y_pred) ** 2)
        ss_tot = np.sum(w * (means - np.average(means, weights=w)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        r_value = np.sqrt(r_squared) if r_squared >= 0 else -np.sqrt(-r_squared)
    else:
        res = linregress(concentrations, means)
        slope, intercept, r_value = res.slope, res.intercept, res.rvalue
        r_squared = r_value ** 2
    
    sample_conc = (sample_mean_area - intercept) / slope if slope != 0 else 0.0
    
    result = {
        "slope": slope, "intercept": intercept, "r_value": r_value,
        "r_squared": r_squared, "sample_conc": sample_conc,
        "sample_mean_area": sample_mean_area, "mode": mode,
        "regression_method": regression_method
    }

    if mode == "lod_iupac" and len(concentrations) > 2:
        n = len(concentrations)
        y_pred = slope * concentrations + intercept
        residuals = means - y_pred
        
        if w is not None:
            syx = np.sqrt(np.sum(w * residuals ** 2) / (n - 2))
        else:
            syx = np.sqrt(np.sum(residuals ** 2) / (n - 2))
        
        result["syx"] = syx
        result["lod_conc"] = (3.3 * syx / abs(slope)) if slope != 0 else 0.0
        result["loq_conc"] = (10.0 * syx / abs(slope)) if slope != 0 else 0.0
        result["lod_signal"] = slope * result["lod_conc"] + intercept
        result["loq_signal"] = slope * result["loq_conc"] + intercept

    return result