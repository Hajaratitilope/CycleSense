import pandas as pd
from utils.artifacts import (
    pipe_raw, pipe_var,
    raw_name_map, var_name_map,
    logical_map, cluster_avgs
)
from utils.preprocess import mean_std_cv_from_wide


# =========================================================
# Profile Assignment
# =========================================================
def assign_user_profile(user_wide_row: pd.DataFrame):
    """
    Assigns a cycle profile for a single user row based on raw features and variability features.
    Returns:
        raw_name (str): Label for cycle length group.
        var_name (str): Label for variability group.
        combined (str): Raw + variability label.
        logical (str): Simplified logical profile.
    """
    # Predict raw group
    Xr = user_wide_row[pipe_raw.feature_names_in_]
    raw_label = pipe_raw.predict(Xr)[0]
    raw_name = raw_name_map[raw_label]

    # Predict variability group
    tmp = mean_std_cv_from_wide(Xr)[pipe_var.feature_names_in_]
    var_label = pipe_var.predict(tmp)[0]
    var_name = var_name_map[var_label]

    # Combine into logical profile
    combined = f"{raw_name} + {var_name}"
    logical = logical_map.get(combined, combined)

    return raw_name, var_name, combined, logical


# =========================================================
# Descriptions: Research / User / Clinician
# =========================================================
def explain_profile(logical_name: str) -> str:
    """
    Provides a research-oriented descriptive explanation for each logical profile.
    """
    notes = {
        # Stable
        "Stable-Compact": "Short, predictable cycles; avg. age ~32, BMI ~25.7. ~3 pregnancies. ~26% complications — reliable but not risk-free.",
        "Stable-Balanced": "Consistently regular cycles; avg. age ~31, BMI ~23.4. ~3–4 pregnancies. ~30% complications despite steady rhythm.",
        "Stable-Delayed": "Slightly longer but steady cycles; avg. age ~30, BMI ~24.1. ~2.5 pregnancies. No complications observed (tiny cluster, interpret cautiously).",
        "Stable-Extended": "Longest predictable cycles; younger (~29), BMI ~24.3. ~2 pregnancies. No complications seen.",

        # Somewhat irregular
        "Mostly Steady-Balanced": "Balanced cycles with mild irregularity; avg. age ~30, BMI ~24.5. ~2 pregnancies. ~27% complications — early warning cluster.",
        "Somewhat Irregular-Compact": "Short but mildly irregular; older (~33–34), BMI ~25.9. ~2 pregnancies. ~25% complication rate.",
        "Somewhat Irregular-Delayed": "Long, mildly irregular cycles; avg. age ~31, BMI ~23.7. ~2–3 pregnancies. ~17% complication rate.",
        "Somewhat Irregular-Extended": "Very long, mildly irregular; avg. age ~30, leaner (BMI ~21). ~2 pregnancies. ~10% complication rate.",

        # Unstable
        "Unstable-Compact": "Short, highly variable cycles; oldest group (~38), BMI ~26.2. ~5 pregnancies. 100% complications — very high risk cluster.",
        "Unstable-Balanced": "Average-length but highly variable; avg. age ~32, BMI ~26.4. ~3–4 pregnancies. ~33% complication rate.",
        "Unstable-Delayed": "Long and highly variable; younger (~28), BMI ~28.0. ~2 pregnancies. ~33% complication rate.",
        "Unstable-Extended": "Very long and highly variable; avg. age ~32, BMI ~29.1. ~3 pregnancies. ~40% complication rate.",

        # Rare
        "Critical-Extended": "Extended but unstable; very rare (n=1). Age ~25, BMI ~25.1, no pregnancies or complications. Interpret individually."
    }
    return notes.get(logical_name, "Cycle profile combining length pattern and stability characteristics.")


def ttc_inference(logical_name: str, age: float, bmi: float, preg: int, complications: int) -> str:
    """
    TTC (Trying to Conceive) implications based on cycle profile, age, BMI, pregnancy history, and complications.
    Overlays add nuance depending on individual context.
    """
    notes = {
        # Stable
        "Stable-Compact": "Generally favorable for TTC. Predictable cycles help timing. Watch for luteal sufficiency.",
        "Stable-Balanced": "Most fertile baseline group. If there is difficulty conceiving, causes may lie outside cycle rhythm.",
        "Stable-Delayed": "Later ovulation reduces the number of fertile windows per year. TTC may take longer despite stable cycles.",
        "Stable-Extended": "Predictable but late ovulation. TTC might require patience. Monitor luteal adequacy.",

        # Somewhat irregular
        "Mostly Steady-Balanced": "Mild irregularity can delay TTC. Tracking is still useful, though timing may be less precise.",
        "Somewhat Irregular-Compact": "Older age combined with irregularity raises TTC challenges. May signal declining ovarian reserve.",
        "Somewhat Irregular-Delayed": "Longer cycles reduce conception opportunities. TTC delay is possible but not prohibitive.",
        "Somewhat Irregular-Extended": "Very long cycles make conception windows sparse. Early assessment may be warranted.",

        # Unstable
        "Unstable-Compact": "Highly irregular cycles in advanced age with complications. TTC prognosis is guarded. Seek evaluation.",
        "Unstable-Balanced": "Cycles are unpredictable. Conception is possible but erratic. Consider ovulation testing.",
        "Unstable-Delayed": "Metabolic risk profile may be present. TTC may be impaired by anovulation. Lifestyle support can help.",
        "Unstable-Extended": "Highly irregular cycles with high BMI create a double barrier for TTC. Referral for PCOS or endocrine evaluation is likely.",

        # Rare
        "Critical-Extended": "Very rare pattern. Individualized TTC approach is recommended."
    }
    base_note = notes.get(logical_name, "Cycle implications not fully mapped yet.")

    # --- Conditional overlays ---
    if age > 35:
        base_note += " Advanced maternal age: TTC urgency is higher due to declining ovarian reserve."
    elif age < 25:
        base_note += " Younger age is generally protective for TTC potential."

    if bmi >= 30:
        base_note += " Elevated BMI may reduce ovulatory efficiency and implantation."
    elif bmi < 18.5:
        base_note += " Very low BMI may impair ovulation or luteal function."

    cluster_stats = cluster_avgs.get(logical_name, {})
    cluster_mean_preg = cluster_stats.get("Numberpreg_mean", None)
    if cluster_mean_preg is not None:
        if preg > cluster_mean_preg:
            base_note += " History of multiple pregnancies suggests proven fertility."
        elif preg == 0:
            base_note += " No prior pregnancies: TTC monitoring may help detect early issues."

    if complications:
        base_note += " User has reproductive complications; additional monitoring recommended."

    return base_note


def clinical_inference(logical_name: str, age: float, bmi: float, preg: int, complications: int) -> str:
    """
    Clinician-facing interpretation of cycle profile, with overlays only when warranted.
    Reassuring profiles suppress unnecessary warnings.
    """
    notes = {
        # Stable
        "Stable-Compact": "Predictable, short cycles. Usually ovulatory. Monitor for luteal phase adequacy.",
        "Stable-Balanced": "Normal ovulatory pattern. No immediate cycle-related red flags.",
        "Stable-Delayed": "Later ovulation. May warrant luteal monitoring.",
        "Stable-Extended": "Late but regular ovulation. Keep in mind risk of subfertility if luteal phase is short.",

        # Somewhat irregular
        "Mostly Steady-Balanced": "Mild irregularity. Could be early ovulatory dysfunction. Watch metabolic or endocrine markers.",
        "Somewhat Irregular-Compact": "Short but irregular cycles in an older age group. Possible diminished ovarian reserve.",
        "Somewhat Irregular-Delayed": "Long but mildly irregular cycles. Check for anovulation or thyroid dysfunction.",
        "Somewhat Irregular-Extended": "Very long cycles in a leaner profile. Possible hypothalamic dysfunction.",

        # Unstable
        "Unstable-Compact": "Highly irregular cycles in older age. Consistent with perimenopausal transition. High complication risk.",
        "Unstable-Balanced": "Irregular cycles with average cycle length. May reflect subclinical ovulatory dysfunction.",
        "Unstable-Delayed": "Long, variable cycles with higher BMI. Screen for PCOS or metabolic syndrome.",
        "Unstable-Extended": "Very long, highly irregular cycles with high BMI. Strong PCOS suspicion. Evaluate endocrine profile.",

        # Rare
        "Critical-Extended": "Rare presentation. Interpret with caution. Individualized evaluation required."
    }

    base_note = notes.get(logical_name, "Cycle implications not fully mapped.")
    reassuring_profiles = {"Stable-Balanced", "Stable-Compact", "Stable-Delayed", "Stable-Extended"}

    if logical_name not in reassuring_profiles:
        if age > 35:
            base_note += " Advanced reproductive age: increased risk of anovulation, miscarriage."
        elif age < 20:
            base_note += " Younger age: irregularity may reflect hypothalamic immaturity."

        if bmi >= 30:
            base_note += " Elevated BMI: consider metabolic/endocrine assessment."
        elif bmi < 18.5:
            base_note += " Underweight: possible hypothalamic anovulation."

    if complications:
        base_note += " History of reproductive complications: requires close follow-up."

    return base_note


# =========================================================
# User-friendly short descriptions (for TTC/general users)
# =========================================================
short_profile_map = {
    "Stable-Balanced": "Your cycles are quite regular and steady; a reliable rhythm.",
    "Stable-Compact": "Your cycles are short but consistent.",
    "Stable-Delayed": "Your cycles are a bit longer than average but still predictable.",
    "Stable-Extended": "Your cycles run longer than usual but remain steady.",
    "Somewhat Irregular-Compact": "Your cycles are shorter but sometimes unpredictable.",
    "Somewhat Irregular-Delayed": "Your cycles can run long and vary more than usual.",
    "Somewhat Irregular-Extended": "Your cycles are extended and less predictable.",
    "Unstable-Balanced": "Your cycles show ups and downs despite some balance.",
    "Unstable-Compact": "Your cycles are short and quite erratic.",
    "Unstable-Delayed": "Your cycles are long and irregular.",
    "Unstable-Extended": "Your cycles are extended and very inconsistent.",
    "Critical-Extended": "Your cycles are severely prolonged and need close medical attention."
}
