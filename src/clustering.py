"""
clustering.py
Unsupervised learning on MxMH — finds listener personas via KMeans.

Steps:
    1. Scale features
    2. PCA for dimensionality reduction (visualisation + noise removal)
    3. Elbow method to find optimal k
    4. KMeans clustering
    5. Persona profiling — describe each cluster in plain English
    6. Visualise clusters in PCA space

Outputs:
    outputs/elbow_curve.png
    outputs/cluster_pca.png
    outputs/cluster_heatmap.png
    outputs/cluster_profiles.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
K_RANGE      = range(2, 9)

# Features most relevant for listener persona clustering
CLUSTER_FEATURES = [
    "hours_per_day",
    "listening_while_working",
    "bpm_fav_genre",
    "mental_health_score",
    "music_engagement",
    "listening_intensity",
    "frequency_classical", "frequency_edm",  "frequency_hip_hop",
    "frequency_jazz",      "frequency_lofi", "frequency_metal",
    "frequency_pop",       "frequency_rock",
    "anxiety", "depression", "insomnia", "ocd",
]


def _elbow_plot(inertias: list, silhouettes: list):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(list(K_RANGE), inertias, "bo-", linewidth=2, markersize=7)
    ax1.set_title("Elbow Method — Inertia")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia")

    ax2.plot(list(K_RANGE), silhouettes, "ro-", linewidth=2, markersize=7)
    ax2.set_title("Silhouette Score")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")

    plt.suptitle("Optimal k Selection", y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / "elbow_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _pca_cluster_plot(X_pca: np.ndarray, labels: np.ndarray, k: int, variance: list):
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("Set2", k)

    for cluster_id in range(k):
        mask = labels == cluster_id
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            s=40, alpha=0.7,
            color=palette[cluster_id],
            label=f"Cluster {cluster_id}",
        )

    ax.set_xlabel(f"PC1 ({variance[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({variance[1]:.1f}% variance)")
    ax.set_title(f"KMeans Clusters (k={k}) — PCA Space")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / "cluster_pca.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _profile_heatmap(profiles: pd.DataFrame):
    """Z-scored heatmap so features with different scales are comparable."""
    z = (profiles - profiles.mean()) / (profiles.std() + 1e-9)
    fig, ax = plt.subplots(figsize=(14, max(4, len(profiles.columns) * 0.4)))
    sns.heatmap(
        z.T,
        annot=profiles.T.round(2),
        fmt="g",
        cmap="RdYlGn",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Z-score"},
    )
    ax.set_title("Cluster Profiles (z-scored, annotated with raw means)")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / "cluster_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _name_clusters(profiles: pd.DataFrame) -> dict:
    """
    Heuristic rule-based persona naming.
    Works on the raw mean profile values.
    """
    names = {}
    for cluster_id, row in profiles.iterrows():
        mh   = row.get("mental_health_score", 0)
        hrs  = row.get("hours_per_day", 0)
        eng  = row.get("music_engagement", 0)
        bpm  = row.get("bpm_fav_genre", 120)
        anx  = row.get("anxiety", 0)
        dep  = row.get("depression", 0)
        work = row.get("listening_while_working", 0)

        if mh > 18 and hrs > 3:
            name = "🎧 High-stress Heavy Listeners"
        elif mh < 8 and eng >= 2:
            name = "🎵 Engaged Low-anxiety Musicians"
        elif work > 0.6 and hrs > 2:
            name = "💼 Focus & Flow Workers"
        elif bpm > 140 and eng < 1:
            name = "⚡ High-energy Casual Listeners"
        elif anx > 6 or dep > 6:
            name = "🌧️ Music as Coping — High Distress"
        elif hrs < 1.5 and mh < 10:
            name = "😌 Casual Low-distress Listeners"
        else:
            name = f"🎼 Mixed Profile Group {cluster_id}"

        names[cluster_id] = name

    return names


def run_clustering(df: pd.DataFrame) -> dict:
    """
    Full unsupervised clustering pipeline.

    Args:
        df — cleaned DataFrame from preprocessing (contains raw feature columns)

    Returns dict with:
        labels, profiles, persona_names, k, scaler, pca, kmeans
    """
    print("\n[clustering] Starting KMeans persona analysis...")

    # Select available features
    available = [c for c in CLUSTER_FEATURES if c in df.columns]
    X_raw = df[available].copy().fillna(df[available].median())

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # PCA — keep enough components for 85% variance
    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.argmax(cumvar >= 0.85)) + 1
    n_components = max(n_components, 2)   # at least 2 for visualisation

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    variance_pct = pca.explained_variance_ratio_ * 100
    print(f"  PCA: {n_components} components → {cumvar[n_components-1]*100:.1f}% variance retained")

    # Elbow + silhouette
    inertias, silhouettes = [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels_k = km.fit_predict(X_pca)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_pca, labels_k))

    _elbow_plot(inertias, silhouettes)

    # Best k = highest silhouette
    best_k = list(K_RANGE)[int(np.argmax(silhouettes))]
    print(f"  Best k = {best_k}  (silhouette={max(silhouettes):.3f})")

    # Final KMeans
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_pca)

    _pca_cluster_plot(X_pca, labels, best_k, variance_pct)

    # Cluster profiles — mean of original (unscaled) features per cluster
    df_with_labels = df[available].copy()
    df_with_labels["cluster"] = labels

    # Also append music_effects if present for enriched profiling
    if "music_effects" in df.columns:
        df_with_labels["music_effects"] = df["music_effects"].values

    profiles = df_with_labels.groupby("cluster")[available].mean().round(2)

    # Music effects distribution per cluster
    if "music_effects" in df_with_labels.columns:
        effects_dist = (
            df_with_labels.groupby("cluster")["music_effects"]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
            .round(3)
        )
        print("\n  Music effects distribution per cluster:")
        print(effects_dist.to_string())
        effects_dist.to_csv(OUTPUTS_DIR / "cluster_effects.csv")

    _profile_heatmap(profiles)
    profiles.to_csv(OUTPUTS_DIR / "cluster_profiles.csv")

    persona_names = _name_clusters(profiles)

    print("\n  Cluster personas identified:")
    for cid, name in persona_names.items():
        size = (labels == cid).sum()
        pct  = size / len(labels) * 100
        print(f"    Cluster {cid} ({size} people, {pct:.1f}%): {name}")

    return {
        "labels":        labels,
        "profiles":      profiles,
        "persona_names": persona_names,
        "k":             best_k,
        "scaler":        scaler,
        "pca":           pca,
        "kmeans":        kmeans,
        "X_pca":         X_pca,
        "features_used": available,
    }
