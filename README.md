# MxMH — Music × Mental Health

> **Does music improve, worsen, or have no effect on mental health?**
> End-to-end ML system with ensemble prediction, SHAP explainability, KMeans listener personas, and an interactive Streamlit app — built on 736 real survey responses.
---

## Live demos

| | Link |
|---|---|
| **Streamlit app** — Interactive prediction + SHAP | [Launch app](https://mxmh-music-and-mental-health-aqzvpkdui6thkhebthdbqc.streamlit.app/) |
| **Tableau dashboard** — EDA + Data Storytelling | [View dashboard](https://public.tableau.com/app/profile/sukriti.srivastava5327/viz/MusicMentalHealthDashboard_17736861353150/Dashboard1) |

---

## What this project does

Most ML projects stop at model accuracy. This one goes further:

- **Predicts** whether music improves, worsens, or has no effect on a person's mental health based on their listening profile
- **Explains** every prediction using SHAP waterfall charts — not just *what* the model predicted, but *why*
- **Clusters** 736 respondents into distinct listener personas using KMeans + PCA
- **Visualises** the full data story across 5 interactive Tableau views

---

## Key findings

- People who report music **worsening** their mental health have significantly higher anxiety and depression scores than those who report improvement
- **Music engagement** (playing instruments, composing, exploring new genres) is a stronger predictor of perceived benefit than listening hours alone
- **Higher listening hours alone do not improve mental health** — context (while working, genre choice, engagement level) matters more
- KMeans identified **4 distinct listener personas** ranging from "Engaged Low-anxiety Musicians" to "High-stress Heavy Listeners"
- Gradient Boosting achieved best overall performance: **76% accuracy, 0.69 F1 (weighted), 0.60 ROC-AUC** on a heavily imbalanced dataset

---

## ML pipeline

```
Raw survey data (736 rows, 33 features)
        ↓
Preprocessing — missing value handling, outlier capping,
                ordinal encoding, 3 engineered features
        ↓
4 Classifiers — Random Forest · Gradient Boosting · XGBoost · Logistic Regression
        ↓
SHAP explainability — global feature importance + per-prediction waterfall
        ↓
KMeans clustering — elbow method, silhouette scoring, persona naming
        ↓
Streamlit app — live prediction, SHAP explanation, cluster assignment
```

---

## Feature engineering

Three new features derived from existing columns:

| Feature | Formula | What it captures |
|---|---|---|
| `mental_health_score` | anxiety + depression + insomnia + ocd | Overall mental health burden (0–40) |
| `music_engagement` | instrumentalist + composer + exploratory + foreign_languages | Depth of music involvement (0–4) |
| `listening_intensity` | hours_per_day × (while_working + 1) | Active vs passive listening behaviour |

---

## Model performance

| Model | Accuracy | F1 (Weighted) | F1 (Macro) | ROC-AUC |
|---|---|---|---|---|
| Gradient Boosting | 0.762 | 0.694 | 0.344 | 0.601 |
| Random Forest | 0.760 | 0.688 | 0.353 | 0.634 |
| XGBoost | 0.740 | 0.682 | 0.356 | 0.632 |
| Logistic Regression | 0.548 | 0.601 | 0.383 | 0.659 |

> Note: High class imbalance (Improve: 74%, No effect: 23%, Worsen: 3%) means accuracy alone is misleading — F1 macro and ROC-AUC are the meaningful metrics here.

---

## Project structure

```
MxMH-Music-and-Mental-Health/
├── src/
│   ├── data_loader.py       # CSV loading with clear setup instructions
│   ├── preprocessing.py     # Cleaning, encoding, feature engineering
│   ├── models.py            # 4 classifiers, evaluation, plots
│   ├── shap_analysis.py     # SHAP global + per-prediction explanation
│   └── clustering.py        # KMeans personas + PCA + elbow method
├── .streamlit/
│   └── config.toml          # Purple theme configuration
├── outputs/                 # Auto-generated plots + CSVs (gitignored)
├── data/                    # Dataset CSV (gitignored — see setup)
├── main.py                  # Run full pipeline end-to-end
├── app.py                   # Streamlit app (4 tabs)
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone
```bash
git clone https://github.com/Sukriti-124/MxMH-Music-and-Mental-Health.git
cd MxMH-Music-and-Mental-Health
```

### 2. Set up environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Get the data
1. Download from [Kaggle — MxMH Survey Results](https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results)
2. Extract and place `mxmh_survey_results.csv` in the `data/` folder

### 4. Run pipeline + launch app
```bash
python main.py          # trains models, runs SHAP + clustering (~3 min)
streamlit run app.py    # opens app at localhost:8501
```

---

## App features

| Tab | What you see |
|---|---|
| **Predict** | Fill your music profile → ensemble prediction across all 4 models + probability bars |
| **SHAP explanation** | Waterfall chart showing exactly which features drove your prediction |
| **Your persona** | Which of the 4 listener clusters you belong to, plotted on PCA space |
| **Model insights** | Confusion matrices, learning curves, model comparison table |

---

## Dataset

- **Source**: [MxMH Survey Results](https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results)
- **Size**: 736 respondents · 33 features
- **Collected**: August–November 2022
- **Target variable**: `Music effects` — Improves / No effect / Worsens

---

## Tech stack

| Layer | Tools |
|---|---|
| **ML** | Scikit-learn · XGBoost · SHAP |
| **Data** | Pandas · NumPy |
| **Visualisation** | Matplotlib · Seaborn · Plotly · Tableau |
| **App** | Streamlit |
| **Dev** | Python 3.10+ · Git · VS Code |

---

## Author

**Sukriti Srivastava**
MS Data Science · University of Maryland, College Park
[LinkedIn](https://linkedin.com/in/sukriti-124-srivastava) · [GitHub](https://github.com/Sukriti-124)
