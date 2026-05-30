# 🎵 MxMH — Music × Mental Health

> **Does music improve, worsen, or have no effect on mental health?**  
> An end-to-end ML analysis with SHAP explainability, KMeans clustering, and a live Streamlit app.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-link.streamlit.app)

---

## What this project does

Analyses the [MxMH survey dataset](https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results) (736 respondents) to predict whether music improves, worsens, or has no effect on mental health — and explains *why* using SHAP.

| Component | What it does |
|---|---|
| **4 Classifiers** | Random Forest, Gradient Boosting, XGBoost, Logistic Regression |
| **SHAP explainability** | Global feature importance + per-prediction waterfall charts |
| **KMeans clustering** | Identifies distinct listener personas (e.g. "High-stress Heavy Listeners") |
| **Streamlit app** | Interactive prediction with live SHAP explanation |

---

## Quickstart

### 1. Clone
```bash
git clone https://github.com/Sukriti-124/MxMH.git
cd MxMH
```

### 2. Install dependencies
```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Get the data (2 minutes)

1. Go to [Kaggle dataset](https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results)
2. Click **Download** → extract the zip
3. Place `mxmh_survey_results.csv` inside the `data/` folder

```
MxMH_project/
└── data/
    └── mxmh_survey_results.csv   ← here
```

### 4. Run full pipeline
```bash
python main.py
```

### 5. Launch Streamlit app
```bash
streamlit run app.py
```

---

## Project structure

```
MxMH/
├── src/
│   ├── data_loader.py      # Kaggle download + CSV loading
│   ├── preprocessing.py    # Cleaning, encoding, feature engineering
│   ├── models.py           # 4 classifiers + evaluation
│   ├── shap_analysis.py    # SHAP global + per-prediction explanation
│   └── clustering.py       # KMeans listener personas + PCA
├── outputs/                # Generated plots and CSVs (gitignored)
├── models/                 # Saved best model pkl (gitignored)
├── data/                   # Dataset CSV (gitignored)
├── main.py                 # End-to-end pipeline runner
├── app.py                  # Streamlit app
└── requirements.txt
```

---

## Key findings

- **Mental health score** (combined anxiety + depression + insomnia + OCD) is the strongest predictor of perceived music effects
- **Music engagement** (instrumentalist + composer + exploratory listener) correlates with music being perceived as beneficial
- **KMeans identified 4–5 distinct listener personas** ranging from "Engaged Low-anxiety Musicians" to "High-stress Heavy Listeners"
- Models achieve ~74% accuracy on a highly imbalanced dataset (Improves: 74%, No effect: 16%, Worsens: <1%)

---

## Dataset

- **Source**: [MxMH Survey Results — Kaggle](https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results)
- **Size**: 736 respondents, 33 features
- **Collected**: August–November 2022
- **Target**: `Music effects` (Improves / No effect / Worsens)

---

## Tech stack

`Python` · `Scikit-learn` · `XGBoost` · `SHAP` · `Streamlit` · `Plotly` · `Pandas` · `Seaborn`
