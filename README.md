# CycleSense  

## Overview  
**CycleSense** is an end-to-end analytical framework for menstrual cycle classification and fertility insights.  
It addresses the gap in fertility planning and Trying-to-Conceive (TTC) journeys where existing apps often stop at surface-level insights (period tracking, ovulation windows) without leveraging deeper data patterns.  

CycleSense applies **clustering and demographic overlays** to classify menstrual cycles into meaningful categories, quantify stability/variability, and contextualize these insights for:  
- **Personal users** (women TTC).  
- **Clinicians** (decision support & counseling).  
- **Researchers** (population-level reproductive health studies).  

---

## Dataset  
- **Source:** [Kaggle – Menstrual Cycle Data](https://www.kaggle.com/datasets/nikitabisht/menstrual-cycle-data/data).  
- **Origin:** Traced to *Marquette University, Natural Family Planning (NFP) Data Repository* (2020).  
- **Structure:**  
  - Original data in **long format** (multiple cycles per user).  
  - Transformed into **wide format** (user represented once, with cycle features suffixed by cycle number).  
- **Inclusion criteria:**  
  - Users with at least 3 recorded cycles retained.  
  - Users with fewer dropped; users with >3 cycles truncated to first 3.  
  - Final dataset: **114 users** (from 139).  

---

## Methodology  

### Feature Engineering  
- **Cycle features:** Length of menses, estimated day of ovulation, luteal phase length, cycle length.  
- **Demographics:** Age, height, weight, BMI (recalculated where needed).  
- **Reproductive history:** Pregnancies, abortions, miscarriages.  
- **Variability metrics** (from 3 cycles per user): mean, standard deviation, coefficient of variation.  

### Clustering Approach  
- Two clustering pipelines:  
  1. **Raw cycle features** (direct measures).  
  2. **Variability features** (stability across cycles).  
- Optimal number of clusters: **k = 4** (elbow + silhouette analysis).  

### Cluster Interpretation  
- **Raw Cycle Features:** Prolonged Delayed, Compact Cycle, Extended Cycle, Balanced Cycle.  
- **Variability Features:** Highly Variable, Predictable, Needs Monitoring, Mildly Irregular.  
- **Composite profiles** created by combining both (e.g. *Compact + Predictable → Stable Compact*).  

### Evaluation Metrics  
Clustering performance was assessed using:  
- **Silhouette Score**  
- **Calinski-Harabasz Index**  
- **Davies–Bouldin Index**  

---

## Application  
CycleSense outputs were layered with demographic overlays (age, BMI, reproductive history) to contextualize insights.  

**Use cases:**  
- **Personal users:** Understand cycle stability, TTC insights, early warnings.  
- **Clinicians:** Stratify patients, provide data-driven fertility counseling.  
- **Researchers:** Aggregate demographic and cycle data for public health insights.  

---

## Demo  
🔗 **Live Streamlit App:** [https://cyclesense.streamlit.app/](https://cyclesense.streamlit.app/)  

The app allows users to:  
- Input demographic + cycle details.  
- Receive cycle classification + demographic overlay.  
- Visualize fertility insights interactively.  

---

## Project Structure  

```bash
CycleSense/
│
├── figures/ # visual outputs (png files: centroid_raw_heatmap, centroid_var_heatmap, elbow_silhouette_scores)
├── models/ # serialized artifacts (.pkl files: centroids, names, kmeans models, evals, logical_map)
├── utils/ # core python modules (init.py, artifacts.py, preprocess.py, profiles.py, report.py)
├── app.py # Streamlit application entry point
├── requirements.txt # dependencies
└── README.md
```


---

## Tech Stack  
- **Language:** Python 3.x  
- **Core libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn  
- **App framework:** Streamlit  
- **Deployment:** Streamlit Cloud  

---

## Installation & Usage  

Clone repo:  
```bash
git clone https://github.com/<your-username>/CycleSense.git
cd CycleSense
```

Create environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Run the streamlit app:
```bash
streamlit run app.py
```
Access at: http://localhost:8501

---

## Limitations
- **Not medical advice**. Educational & research purposes only.
- Predictions depend on historical cycle data; accuracy declines for irregular or sparse data.
- Dataset limited to specific population sample (Marquette NFP users).

---

## Acknowledgements
- **Dataset:** Marquette University, Natural Family Planning Data Repository (via Kaggle).
- **Fellowship:** Developed as part of the Women in AI Fellowship (Webfala Digital Skills for All Initiative, 2025).

---

## Team Flow Finders:
  - Hajarat Titilope Olufade
  - Kafayat Abdulwahab
  - Saudat Abubakar
  - Adijat Atanda

---

