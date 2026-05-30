"""
app.py
Streamlit app for MxMH — Music × Mental Health Predictor

Features:
    - User inputs their music profile via sidebar
    - Ensemble prediction across all 4 models
    - SHAP waterfall explanation for best model
    - Cluster persona assignment
    - Model comparison tab
"""

import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader    import load_raw
from src.preprocessing  import preprocess
from src.models         import train_and_evaluate, MODELS_DIR
from src.shap_analysis  import run_shap_analysis, explain_single
from src.clustering     import run_clustering

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MxMH — Music × Mental Health",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cache heavy computation ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models and data...")
def load_pipeline():
    raw_df  = load_raw()
    proc    = preprocess(raw_df)
    results = train_and_evaluate(proc["X"], proc["y"], list(proc["label_encoder"].classes_))

    shap_res = run_shap_analysis(
        model=results["_best_model"],
        X_train=results["_X_train"],
        X_test=results["_X_test"],
        class_names=list(proc["label_encoder"].classes_),
        feature_names=proc["feature_names"],
    )
    cluster_res = run_clustering(proc["df_clean"])

    return proc, results, shap_res, cluster_res


proc, results, shap_res, cluster_res = load_pipeline()

CLASS_NAMES    = list(proc["label_encoder"].classes_)
FEATURE_NAMES  = proc["feature_names"]
BEST_MODEL     = results["_best_model"]
X_TRAIN        = results["_X_train"]
LABEL_ENCODER  = proc["label_encoder"]


# ── Helpers ───────────────────────────────────────────────────────────────
EFFECT_EMOJI = {
    "Improves": "😊 Improves",
    "No effect": "😐 No effect",
    "Worsens": "😟 Worsens",
}
EFFECT_COLOR = {
    "Improves": "normal",
    "No effect": "off",
    "Worsens": "inverse",
}

GENRE_OPTIONS = [
    "Classical", "Country", "EDM", "Folk", "Gospel",
    "Hip hop", "Jazz", "K pop", "Latin", "Lofi",
    "Metal", "Pop", "R&B", "Rap", "Rock", "Video game music",
]

STREAM_OPTIONS = [
    "Spotify", "Apple Music", "YouTube Music",
    "Pandora", "Amazon Music", "Other",
]

FREQ_OPTIONS = ["Never", "Rarely", "Sometimes", "Very frequently"]


def build_input_row(inputs: dict) -> pd.DataFrame:
    """Convert sidebar inputs to a single-row DataFrame matching training features."""
    freq_map = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Very frequently": 3}
    bin_map  = {"Yes": 1, "No": 0}

    from sklearn.preprocessing import LabelEncoder

    # Encode fav_genre + primary_streaming_service the same way as training
    genre_le   = proc["encoders"].get("fav_genre")
    stream_le  = proc["encoders"].get("primary_streaming_service")

    fav_genre_enc = (
        genre_le.transform([inputs["fav_genre"]])[0]
        if genre_le and inputs["fav_genre"] in genre_le.classes_
        else 0
    )
    stream_enc = (
        stream_le.transform([inputs["streaming_service"]])[0]
        if stream_le and inputs["streaming_service"] in stream_le.classes_
        else 0
    )

    freq_genres = [
        "classical","country","edm","folk","gospel","hip_hop",
        "jazz","k_pop","latin","lofi","metal","pop",
        "rnb","rap","rock","video_game_music"
    ]

    row = {
        "age":                        inputs["age"],
        "primary_streaming_service":  stream_enc,
        "hours_per_day":              inputs["hours_per_day"],
        "listening_while_working":    bin_map[inputs["while_working"]],
        "instrumentalist":            bin_map[inputs["instrumentalist"]],
        "composer":                   bin_map[inputs["composer"]],
        "fav_genre":                  fav_genre_enc,
        "exploratory":                bin_map[inputs["exploratory"]],
        "foreign_languages":          bin_map[inputs["foreign_languages"]],
        "bpm_fav_genre":              inputs["bpm"],
        "anxiety":                    inputs["anxiety"],
        "depression":                 inputs["depression"],
        "insomnia":                   inputs["insomnia"],
        "ocd":                        inputs["ocd"],
    }

    for g in freq_genres:
        row[f"frequency_{g}"] = freq_map.get(inputs.get(f"freq_{g}", "Never"), 0)

    # Engineered features
    row["mental_health_score"]  = inputs["anxiety"] + inputs["depression"] + inputs["insomnia"] + inputs["ocd"]
    row["music_engagement"]     = (
        bin_map[inputs["instrumentalist"]] + bin_map[inputs["composer"]] +
        bin_map[inputs["exploratory"]]     + bin_map[inputs["foreign_languages"]]
    )
    row["listening_intensity"] = inputs["hours_per_day"] * (bin_map[inputs["while_working"]] + 1)

    df_row = pd.DataFrame([row])

    # Align to training feature order, fill any missing with 0
    for col in FEATURE_NAMES:
        if col not in df_row.columns:
            df_row[col] = 0
    df_row = df_row[FEATURE_NAMES]

    return df_row


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR — User inputs
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🎵 Your Music Profile")

    st.subheader("Personal")
    age              = st.slider("Age", 10, 89, 22)
    streaming_service= st.selectbox("Primary streaming service", STREAM_OPTIONS)

    st.subheader("Listening Habits")
    hours_per_day  = st.slider("Hours listening per day", 0.0, 20.0, 2.0, step=0.5)
    while_working  = st.radio("Listen while working?", ["Yes", "No"], horizontal=True)
    instrumentalist= st.radio("Play an instrument?",   ["Yes", "No"], horizontal=True)
    composer       = st.radio("Compose music?",         ["Yes", "No"], horizontal=True)
    exploratory    = st.radio("Explore new artists?",   ["Yes", "No"], horizontal=True)
    foreign_lang   = st.radio("Listen in foreign languages?", ["Yes", "No"], horizontal=True)
    fav_genre      = st.selectbox("Favourite genre", GENRE_OPTIONS, index=11)
    bpm            = st.slider("BPM of favourite genre", 60, 200, 120)

    with st.expander("Genre listening frequencies"):
        freq_inputs = {}
        for g in GENRE_OPTIONS:
            key = g.lower().replace(" ", "_").replace("&", "").replace("/", "_")
            freq_inputs[f"freq_{key}"] = st.select_slider(g, FREQ_OPTIONS, "Sometimes")

    st.subheader("Mental Health (self-reported 0–10)")
    anxiety    = st.slider("Anxiety",    0, 10, 4)
    depression = st.slider("Depression", 0, 10, 3)
    insomnia   = st.slider("Insomnia",   0, 10, 2)
    ocd        = st.slider("OCD",        0, 10, 1)

    predict_btn = st.button("🔍 Predict", use_container_width=True, type="primary")


# ── Collect all inputs ───────────────────────────────────────────────────
user_inputs = {
    "age": age,
    "streaming_service": streaming_service,
    "hours_per_day": hours_per_day,
    "while_working": while_working,
    "instrumentalist": instrumentalist,
    "composer": composer,
    "exploratory": exploratory,
    "foreign_languages": foreign_lang,
    "fav_genre": fav_genre,
    "bpm": bpm,
    "anxiety": anxiety,
    "depression": depression,
    "insomnia": insomnia,
    "ocd": ocd,
    **freq_inputs,
}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT — Tabs
# ═══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Predict", "🔍 SHAP Explanation", "👥 Your Persona", "📊 Model Insights"
])


with tab1:
    st.title("Music × Mental Health Predictor")
    st.caption(
        "Based on 736 survey responses — predicts whether music is likely to "
        "improve, have no effect, or worsen mental health."
    )

    if predict_btn or True:   # always show after first render
        X_row = build_input_row(user_inputs)

        # Ensemble prediction across all 4 models
        model_preds  = {}
        model_probas = {}

        for name, res in results.items():
            if name.startswith("_"):
                continue
            model   = res["model"]
            proba   = model.predict_proba(X_row)[0]
            pred_idx= int(np.argmax(proba))
            model_preds[name]  = CLASS_NAMES[pred_idx]
            model_probas[name] = proba

        # Ensemble: average probabilities
        avg_proba = np.mean(list(model_probas.values()), axis=0)
        ensemble_idx = int(np.argmax(avg_proba))
        ensemble_label = CLASS_NAMES[ensemble_idx]
        confidence = avg_proba[ensemble_idx]

        # Display result
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric(
                label="Ensemble Prediction",
                value=EFFECT_EMOJI.get(ensemble_label, ensemble_label),
                delta=f"{confidence*100:.1f}% confidence",
            )

        st.markdown("---")

        # Per-model breakdown
        st.subheader("Per-model predictions")
        cols = st.columns(4)
        for i, (name, pred) in enumerate(model_preds.items()):
            proba = model_probas[name]
            with cols[i % 4]:
                st.metric(name, EFFECT_EMOJI.get(pred, pred))
                for j, cn in enumerate(CLASS_NAMES):
                    st.progress(float(proba[j]), text=f"{cn}: {proba[j]*100:.1f}%")

        # Probability bar chart
        st.subheader("Ensemble probability breakdown")
        fig, ax = plt.subplots(figsize=(6, 2.5))
        colors = ["#2ecc71", "#95a5a6", "#e74c3c"]
        bars = ax.barh(CLASS_NAMES, avg_proba, color=colors[:len(CLASS_NAMES)], height=0.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        for bar, val in zip(bars, avg_proba):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{val*100:.1f}%", va="center", fontsize=10)
        ax.set_title("Ensemble Prediction Probabilities")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Store for other tabs
        st.session_state["X_row"]        = X_row
        st.session_state["ensemble_idx"] = ensemble_idx
        st.session_state["ensemble_label"] = ensemble_label


with tab2:
    st.title("🔍 Why This Prediction?")
    st.caption(
        "SHAP (SHapley Additive exPlanations) shows exactly which features "
        "pushed your prediction up or down. Red = pushed towards this outcome, "
        "Blue = pushed away."
    )

    if "X_row" in st.session_state:
        X_row       = st.session_state["X_row"]
        class_idx   = st.session_state["ensemble_idx"]
        class_label = st.session_state["ensemble_label"]

        st.markdown(f"**Explaining prediction: `{class_label}`**")

        with st.spinner("Computing SHAP values..."):
            exp = explain_single(
                model=BEST_MODEL,
                X_row=X_row,
                X_background=X_TRAIN.sample(min(100, len(X_TRAIN)), random_state=42),
                class_names=CLASS_NAMES,
                feature_names=FEATURE_NAMES,
                class_idx=class_idx,
            )

        # Waterfall plot
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(9, 8))
        fig.patch.set_facecolor("#13111f")
        shap.plots.waterfall(exp, max_display=12, show=False)
        plt.gcf().set_facecolor("#13111f")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        plt.style.use("default")

        # Plain English top factors
        sv = exp.values
        sorted_idx = np.argsort(np.abs(sv))[::-1][:5]
        st.subheader("Top 5 factors for your result")
        for i in sorted_idx:
            feat = FEATURE_NAMES[i]
            val  = float(X_row.iloc[0, i])
            sv_i = float(sv[i])
            direction = "⬆️ increased" if sv_i > 0 else "⬇️ decreased"
            st.markdown(
                f"- **{feat}** = `{val:.2f}` → {direction} '{class_label}' likelihood "
                f"by `{abs(sv_i):.3f}`"
            )

        # Global importance
        st.subheader("Global feature importance (all predictions)")
        importance_df = shap_res["importance_df"].head(15)
        plt.style.use("dark_background")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        fig2.patch.set_facecolor("#13111f")
        ax2.set_facecolor("#13111f")
        ax2.barh(
            importance_df["feature"][::-1],
            importance_df["mean_abs_shap"][::-1],
            color="#4C72B0"
        )
        ax2.set_xlabel("Mean |SHAP Value|")
        ax2.set_title("Top 15 Features — All Test Predictions")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.style.use("default")
        plt.close(fig2)

    else:
        st.info("👈 Fill in your profile and click Predict first.")


with tab3:
    st.title("👥 Your Listener Persona")
    st.caption("KMeans clustering identified distinct listener types in the data.")

    if "X_row" in st.session_state:
        X_row = st.session_state["X_row"]

        persona_names = cluster_res["persona_names"]
        kmeans        = cluster_res["kmeans"]
        pca           = cluster_res["pca"]
        scaler        = cluster_res["scaler"]
        features_used = cluster_res["features_used"]

        # Assign user to a cluster
        user_features = X_row[[c for c in features_used if c in X_row.columns]].copy()
        # Fill missing cluster features with 0
        for f in features_used:
            if f not in user_features.columns:
                user_features[f] = 0
        user_features = user_features[features_used]

        user_scaled = scaler.transform(user_features)
        user_pca    = pca.transform(user_scaled)
        user_cluster= int(kmeans.predict(user_pca)[0])
        user_persona= persona_names.get(user_cluster, f"Group {user_cluster}")

        st.markdown("---")
        st.metric("Your Listener Persona", user_persona)
        st.markdown("---")

        # Cluster descriptions
        st.subheader(f"About this cluster")
        profiles = cluster_res["profiles"]
        if user_cluster in profiles.index:
            profile = profiles.loc[user_cluster]
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg hours/day",       f"{profile.get('hours_per_day', 0):.1f}")
            c2.metric("Avg mental health score", f"{profile.get('mental_health_score', 0):.1f}/40")
            c3.metric("Avg music engagement", f"{profile.get('music_engagement', 0):.1f}/4")

        # PCA scatter with user highlighted
        st.subheader("All listeners — PCA space")
        X_pca    = cluster_res["X_pca"]
        labels   = cluster_res["labels"]
        k        = cluster_res["k"]

        import seaborn as sns
        fig, ax = plt.subplots(figsize=(8, 5))
        palette = sns.color_palette("Set2", k)
        for cid in range(k):
            mask = labels == cid
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       s=25, alpha=0.5, color=palette[cid],
                       label=persona_names.get(cid, f"Cluster {cid}"))

        # Plot user
        ax.scatter(user_pca[0, 0], user_pca[0, 1],
                   s=200, marker="*", color="gold",
                   edgecolors="black", linewidths=1.5,
                   zorder=10, label="You ⭐")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Listener Clusters — You are the ⭐")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    else:
        st.info("👈 Fill in your profile and click Predict first.")


with tab4:
    st.title("📊 Model Insights")

    # Model comparison table
    st.subheader("Model comparison")
    summary = results["_summary"]
    st.dataframe(summary.style.highlight_max(
    axis=0,
    props="color: #13111f; background-color: #7f77dd; font-weight: 500;"
    ), use_container_width=True)
    # Plots
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("SHAP summary")
        shap_img = Path("outputs/shap_summary.png")
        if shap_img.exists():
            st.image(str(shap_img))

    with col2:
        st.subheader("Cluster heatmap")
        cluster_img = Path("outputs/cluster_heatmap.png")
        if cluster_img.exists():
            st.image(str(cluster_img))

    # Confusion matrices
    st.subheader("Confusion matrices")
    cols = st.columns(2)
    for i, name in enumerate(["Random Forest", "Gradient Boosting", "XGBoost", "Logistic Regression"]):
        img_path = Path(f"outputs/cm_{name.lower().replace(' ', '_')}.png")
        if img_path.exists():
            with cols[i % 2]:
                st.image(str(img_path), caption=name)
