# MxMH-Music-and-Mental-Health

An interactive data analysis project that explores the relationship between music listening habits and mental health conditions such as anxiety, depression, insomnia, and OCD using survey data.

Tableau Dashboard : [Music x Mental Health](https://public.tableau.com/app/profile/sukriti.srivastava5327/viz/MusicMentalHealthDashboard_17736861353150/Dashboard1)
((Interactive dashboard built using Tableau Public for exploring music habits and mental health trends))

## 📌 Research Question

Music is widely used as a tool for relaxation, productivity, and emotional regulation. However, the relationship between music consumption patterns and mental health is still an area of ongoing exploration.
This project analyzes survey responses to investigate whether listening habits, music genre preferences, and listening context (such as listening while working) are associated with self-reported mental health indicators.

## 📊 Dataset

- Dataset Source: Kaggle Music & Mental Health Survey Results - [Music x Mental Health_Dataset] (https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results)
-  It captures preference across different music genres and self-reported mental health conditions — specifically anxiety, depression, insomnia, and OCD. 
- Shape - 736 rows (survey respondents) x 33 columns (covering music listening habits, mental health, and perosnal information)
- The 33 columns break into 5 groups:
  - Personal information (3)
    - Age: Respondent's age
    - Instrumentalist: Does the respondent play an instrument regularly?
    - Composer: Does the respondent compose music?
  - Listening habits (23)
    - Primary Streaming Device: Respondent's primary streaming service
    - Hours per day: Number of hours the respondent listens to music per day
    - While working: Does the respondent listen to music while studying/working?
    - Fav genre: Respondent's favorite or top genre
    - Exploratory: Does the respondent actively explore new artists/genres?
    - Foreign languages: Does the respondent regularly listen to music with lyrics in a language they are not fluent in?
    - BPM: Beats per minute of favorite genre
    - Frequency of 16 different music genres: How frequently the respondent listens to each genre (Classical · Country · EDM · Folk · Gospel · Hip hop ·     Jazz · K-pop · Latin · Lo-fi · Metal · Pop · R&B · Rap · Rock · Video game music)
  - Mental health (4)
    - Anxiety: Self-reported anxiety, on a scale of 0-10
    - Depression: Self-reported depression, on a scale of 0-10
    - Insomnia: Self-reported insomnia, on a scale of 0-10
    - OCD: Self-reported OCD, on a scale of 0-10
  - Outcome (1)
    - Music effects: Does music improve/worsen respondent's mental health conditions?
  - Survey data (2)
    - Timestamp: Date and time when form was submitted
    - Permissions: Permissions to publicize data

## 🧠 Approach

1. Loaded and explored the dataset to understand feature distributions.  
2. Handled missing values and performed any necessary preprocessing.  
3. Conducted Exploratory Data Analysis (EDA) to identify patterns between music listening habits and mental health indicators.
4. Analyzed relationships between music genres, listening hours, and mental health scores using statistical summaries and visualizations.
5. Performed feature preprocessing, including encoding categorical variables and scaling features where required.
6. Built and trained machine learning models to predict mental health indicators based on music listening behavior. 
7. Evaluated models using metrics such as accuracy, precision, recall, and F1-score.  
8. Developed an interactive Tableau dashboard to present insights in a user-friendly format.


## 🏆 Evaluation & Results
Best Model: XGB
- **Accuracy:** 74%
- **ROC-AUC:** 68.3%

## 🛠️ Project Files
- `mxmh_analysis.ipynb` – Main analysis and model training notebook
- `README.md` – Project overview and instructions


## 🧰 Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib/Seaborn, Tableau, Jupyter Notebook, Google Colab
