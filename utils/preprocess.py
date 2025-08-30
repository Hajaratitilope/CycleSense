import pandas as pd
import numpy as np

def mean_std_cv_from_wide(Xr):
    """
    Given a wide-format single-user dataframe (3 cycles x 4 features),
    compute mean, std, cv for cycle features.
    Returns: DataFrame with feature engineering results.
    """
    out = {}
    for feature in ["LengthofCycle", "EstimatedDayofOvulation",
                    "LengthofLutealPhase", "LengthofMenses"]:
        cols = [c for c in Xr.columns if c.startswith(feature)]
        vals = Xr[cols].values.flatten()
        out[f"{feature}_mean"] = vals.mean()
        out[f"{feature}_std"]  = vals.std(ddof=0)
        out[f"{feature}_cv"]   = (vals.std(ddof=0) / vals.mean()) if vals.mean() > 0 else 0.0
    return pd.DataFrame([out])
