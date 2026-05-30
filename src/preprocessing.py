"""
preprocessing.py
Cleans the raw MxMH dataset and engineers features for modelling.

Steps:
    1. Rename columns (snake_case, remove brackets)
    2. Handle missing values
    3. Handle outliers
    4. Encode binary + frequency columns
    5. Engineer mental_health_score and music_engagement
    6. Label-encode remaining categoricals
    7. Return X, y and the cleaned DataFrame
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ── Column rename map ─────────────────────────────────────────────────────
FREQUENCY_COLS = [
    "frequency_classical", "frequency_country", "frequency_edm",
    "frequency_folk",      "frequency_gospel",  "frequency_hip_hop",
    "frequency_jazz",      "frequency_k_pop",   "frequency_latin",
    "frequency_lofi",      "frequency_metal",   "frequency_pop",
    "frequency_rnb",       "frequency_rap",     "frequency_rock",
    "frequency_video_game_music",
]

BINARY_COLS = [
    "listening_while_working", "instrumentalist",
    "composer", "exploratory", "foreign_languages",
]

MENTAL_HEALTH_COLS = ["anxiety", "depression", "insomnia", "ocd"]

FREQ_MAP = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Very frequently": 3}
BIN_MAP  = {"Yes": 1, "No": 0}

TARGET   = "music_effects"
DROP_COLS = ["timestamp", "permissions"]


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
    )
    df.rename(columns={
        "while_working":   "listening_while_working",
        "bpm":             "bpm_fav_genre",
        "frequency_r&b":   "frequency_rnb",
        "music_effects":   "music_effects",
    }, inplace=True)
    return df


def _handle_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        df_clean   — rows where music_effects is known
        df_no_resp — rows where music_effects was missing (labelled 'Not responded')
    """
    df = df.copy()

    # MCAR columns — small counts, fill with median/mode
    df["age"] = df["age"].fillna(df["age"].median())
    for col in ["primary_streaming_service", "listening_while_working",
                "instrumentalist", "composer", "foreign_languages"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    # BPM — MAR, fill with genre median; cap out-of-range values first
    df.loc[(df["bpm_fav_genre"] < 30) | (df["bpm_fav_genre"] > 400), "bpm_fav_genre"] = np.nan
    df["bpm_fav_genre"] = (
        df.groupby("fav_genre")["bpm_fav_genre"]
        .transform(lambda x: x.fillna(x.median()))
    )

    # music_effects — MNAR, separate out rather than impute
    df_no_resp = df[df[TARGET].isnull()].copy()
    df_no_resp[TARGET] = "Not responded"
    df_clean = df[df[TARGET].notna()].copy()

    return df_clean, df_no_resp


def _handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Cap listening hours > 20 at 20 (24 h/day is impossible)
    df["hours_per_day"] = df["hours_per_day"].clip(upper=20)
    return df


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Binary columns
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map(BIN_MAP)

    # Ordinal frequency columns
    for col in FREQUENCY_COLS:
        if col in df.columns:
            df[col] = df[col].map(FREQ_MAP)

    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Overall mental health burden (0–40)
    df["mental_health_score"] = df[MENTAL_HEALTH_COLS].sum(axis=1)

    # Music engagement level (0–4)
    engagement_cols = ["instrumentalist", "composer", "exploratory", "foreign_languages"]
    df["music_engagement"] = df[engagement_cols].sum(axis=1)

    # Listening intensity (hours × whether they listen while working)
    df["listening_intensity"] = df["hours_per_day"] * (df["listening_while_working"] + 1)

    return df


def _label_encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    encoders = {}
    for col in ["primary_streaming_service", "fav_genre"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


def preprocess(raw_df: pd.DataFrame) -> dict:
    """
    Full preprocessing pipeline.

    Returns a dict with keys:
        df_clean      — cleaned DataFrame (all columns, music_effects as strings)
        df_no_resp    — rows with no music_effects response
        X             — feature matrix (DataFrame)
        y             — target series (string labels)
        feature_names — list of feature column names
        encoders      — dict of LabelEncoders for cat columns
        label_encoder — LabelEncoder fitted on y
    """
    df = _rename_columns(raw_df)
    df, df_no_resp = _handle_missing(df)
    df = _handle_outliers(df)
    df = _encode(df)
    df = _engineer_features(df)
    df, encoders = _label_encode_categoricals(df)

    # Drop irrelevant columns
    drop = [c for c in DROP_COLS if c in df.columns]
    df_model = df.drop(columns=drop)

    # Separate X and y
    X = df_model.drop(columns=[TARGET])
    y_raw = df_model[TARGET]

    # Encode target
    le_target = LabelEncoder()
    y = pd.Series(le_target.fit_transform(y_raw), name=TARGET)

    print(f"[preprocessing] X: {X.shape}  |  Classes: {list(le_target.classes_)}")
    print(f"[preprocessing] Class distribution:\n{y_raw.value_counts().to_string()}\n")

    return {
        "df_clean":      df,
        "df_no_resp":    df_no_resp,
        "X":             X,
        "y":             y,
        "y_raw":         y_raw,
        "feature_names": list(X.columns),
        "encoders":      encoders,
        "label_encoder": le_target,
    }
