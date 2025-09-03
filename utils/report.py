import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from utils.profiles import short_profile_map, explain_profile, ttc_inference, clinical_inference
from utils.artifacts import (
    cluster_avgs, raw_eval, var_eval,
    raw_name_map, var_name_map,
    raw_centroids, var_centroids
)


# --- Shared helpers ---
def lookup_cluster_stats(logical: str):
    """Retrieve cluster averages for a given logical profile (robust to df or dict)."""
    age_avg = bmi_avg = preg_avg = comp_rate = None

    if isinstance(cluster_avgs, pd.DataFrame):
        df = cluster_avgs
        if "logical_cluster_name" in df.columns and df.index.name != "logical_cluster_name":
            df = df.set_index("logical_cluster_name")
        try:
            row = df.loc[logical]
            age_avg = float(row.get("Age_mean", None))
            bmi_avg = float(row.get("BMI_mean", None))
            preg_avg = float(row.get("Numberpreg_mean", None))
            comp_rate_val = float(row.get("complication_rate", 0.0))
            comp_rate = comp_rate_val / 100.0 if comp_rate_val > 1 else comp_rate_val
        except KeyError:
            pass
    else:
        row = cluster_avgs.get(logical, {})
        age_avg = row.get("Age_mean")
        bmi_avg = row.get("BMI_mean")
        preg_avg = row.get("Numberpreg_mean")
        comp_rate_val = row.get("complication_rate")
        if comp_rate_val is not None:
            comp_rate = comp_rate_val / 100.0 if comp_rate_val > 1 else comp_rate_val

    return age_avg, bmi_avg, preg_avg, comp_rate


def fmt_num(x, fallback):
    return f"{x:.1f}" if x is not None and pd.notna(x) else f"{fallback:.1f}"

def fmt_rate(x):
    return f"{x:.0%}" if x is not None and pd.notna(x) else "N/A"


# --- Reports ---
def make_ttc_report(name, age, bmi, num_preg, complications, logical, cluster_label):
    """Generate TTC (Trying to Conceive) user-friendly report."""
    profile_short = short_profile_map.get(logical, "Cycle profile description not available.")
    ttc_note = ttc_inference(logical, age, bmi, num_preg, complications)

    return f"""Hi {name}, here’s your CycleSense summary:

**Your cycle profile**: {logical}.

**Cycle profile description**: {profile_short}

**TTC Note**: {ttc_note}

**Your stats**: Age {age}, BMI {bmi:.1f}, {num_preg} pregnancies, {"with complications" if complications else "no complications"}.
"""


def make_clinician_report(name, age, bmi, num_preg, complications, logical, cluster_label):
    """Generate clinician-oriented report with structured insights."""
    profile_summary = explain_profile(logical)
    clinical_note = clinical_inference(logical, age, bmi, num_preg, complications)

    age_avg, bmi_avg, preg_avg, comp_rate = lookup_cluster_stats(logical)

    return f"""**Patient**: {name}, {age}y

**Profile**: {logical}

**Cluster**: {profile_summary}

**Demographics**:
- Age {age} (cluster avg {fmt_num(age_avg, age)})
- BMI {bmi:.1f} (cluster avg {fmt_num(bmi_avg, bmi)})
- Pregnancies: {num_preg} (cluster avg {fmt_num(preg_avg, num_preg)})
- Complications: {"Yes" if complications else "No"} (cluster rate {fmt_rate(comp_rate)})

**Interpretation**: {clinical_note}
"""


def make_technical_report():
    """Streamlit report for researchers/engineers: clustering evaluation."""
    st.header("Clustering Evaluation Report")

    st.subheader("📂 Dataset Overview")
    st.markdown(
        "**Dataset overview (n=114):** Women aged 21–43, varying BMI, "
        "0–10 pregnancies, with/without cycle-related complications."
    )

    st.divider()
    st.subheader("📊 Evaluation Metrics")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Raw Features Clustering**")
        st.json({
            "Silhouette": raw_eval["silhouette"],
            "Calinski-Harabasz": raw_eval["calinski_harabasz"],
            "Davies-Bouldin": raw_eval["davies_bouldin"],
        })
        st.markdown("**Cluster Size Distribution (%)**")
        st.write(raw_eval["sizes_%"])

    with col2:
        st.markdown("**Variability Features Clustering**")
        st.json({
            "Silhouette": var_eval["silhouette"],
            "Calinski-Harabasz": var_eval["calinski_harabasz"],
            "Davies-Bouldin": var_eval["davies_bouldin"],
        })
        st.markdown("**Cluster Size Distribution (%)**")
        st.write(var_eval["sizes_%"])

    st.divider()
    st.subheader("📈 Elbow & Silhouette Analysis")
    st.image("figures/Elbow & Silhouette Scores.png", caption="Elbow Plot and Silhouette Scores (Variability Features)")

    st.divider()
    st.subheader("🔍 Cluster Centroids")
    st.markdown("**Raw Features Centroids**")
    st.image("figures/centroid_raw_heatmap.png")

    st.markdown("**Variability Features Centroids**")
    st.image("figures/centroid_var_heatmap.png")

    st.divider()
    st.subheader("🧩 Cluster Naming Maps")
    st.markdown("**Raw Feature Clusters → Semantic Names**")
    st.json(raw_name_map)

    st.markdown("**Variability Feature Clusters → Semantic Names**")
    st.json(var_name_map)

    st.success("This page is for researchers/engineers only (transparency & evaluation).")
