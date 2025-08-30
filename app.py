import streamlit as st
import pandas as pd

from utils.report import (
    make_ttc_report,
    make_clinician_report,
    make_technical_report
)
from utils.profiles import assign_user_profile


# =========================================================
# APP HEADER & SIDEBAR
# =========================================================
st.set_page_config(
    page_title="CycleSense",
    page_icon="🌸",
    layout="centered"
)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["User Reports", "Technical Report"])


# =========================================================
# USER REPORTS PAGE
# =========================================================
if page == "User Reports":
    st.title("🌸 CycleSense")
    st.markdown("**Personalized Menstrual Cycle Analysis and TTC Inference**")
    st.caption("This application operates on information from the **last 3 recorded cycles**.")

    # --- User Demographic Inputs ---
    st.subheader("Enter Your Details")
    name = st.text_input("Name")
    age = st.number_input("Age (years)", min_value=18, max_value=60)
    height_m = st.number_input("Height (m)", min_value=1.0, max_value=2.2, step=0.01)
    weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, step=0.1)
    num_preg = st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1)
    complications = st.radio("History of reproductive complications?", ["No", "Yes"])
    complications = 1 if complications == "Yes" else 0

    # --- Cycle Data Input ---
    st.subheader("Cycle Data (last 3 cycles)")
    cycle_data = []
    for i in range(1, 4):
        st.markdown(f"**Cycle {i}**")
        length_cycle = st.number_input(
            f"Length of Cycle {i} (days)", 
            min_value=15, max_value=60, value=28, key=f"cyc_len_{i}"
        )
        length_menses = st.number_input(
            f"Length of Menses {i} (days)", 
            min_value=2, max_value=10, value=5, key=f"menses_{i}"
        )
        ovulation_day = st.number_input(
            f"Estimated Ovulation Day {i}", 
            min_value=10, max_value=30, value=14, key=f"ovu_{i}"
        )
        cycle_data.append({
            "LengthofCycle": length_cycle,
            "LengthofMenses": length_menses,
            "EstimatedDayofOvulation": ovulation_day
        })

    # --- Report Type Selection ---
    report_type = st.selectbox("Select report type:", ["TTC User", "Clinician"])

    # --- Generate Report Button ---
    if st.button("Generate Report"):
        # Flatten cycle data into wide format
        user_wide = {}
        for i, cycle in enumerate(cycle_data, 1):
            user_wide[f"LengthofCycle_cycle{i}"] = cycle["LengthofCycle"]
            user_wide[f"LengthofMenses_cycle{i}"] = cycle["LengthofMenses"]
            user_wide[f"EstimatedDayofOvulation_cycle{i}"] = cycle["EstimatedDayofOvulation"]
            user_wide[f"LengthofLutealPhase_cycle{i}"] = (
                cycle["LengthofCycle"] - cycle["LengthofMenses"] - cycle["EstimatedDayofOvulation"]
            )

        user_wide_df = pd.DataFrame([user_wide])

        # Assign user profile
        _, _, combined, logical = assign_user_profile(user_wide_df)

        # Compute BMI
        bmi = weight_kg / (height_m ** 2)

        # Build user info dictionary
        user_info = {
            "name": name,
            "age": age,
            "bmi": bmi,
            "num_preg": num_preg,
            "complications": complications,
            "logical": logical,
            "cluster_label": combined
        }

        # Generate appropriate report
        if report_type == "TTC User":
            report = make_ttc_report(**user_info)
        elif report_type == "Clinician":
            report = make_clinician_report(**user_info)

        # Display report
        st.subheader(f"{report_type} Report")
        st.text(report)


# =========================================================
# TECHNICAL REPORT PAGE
# =========================================================
elif page == "Technical Report":
    st.title("CycleSense Technical Report")
    make_technical_report()
