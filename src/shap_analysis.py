"""
shap_analysis.py
SHAP explainability for the MxMH best model.

Generates:
    outputs/shap_summary.png       — global feature importance (all classes)
    outputs/shap_beeswarm_<cls>.png — beeswarm per class
    outputs/shap_bar.png            — mean |SHAP| bar chart
    outputs/shap_values.csv         — raw SHAP values (mean abs per feature)

Also exposes:
    explain_single(model, X_row, feature_names, class_names)
    → returns shap Explanation object for one row (used in Streamlit app)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from sklearn.pipeline import Pipeline

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def _get_raw_model(model):
    """Unwrap Pipeline to get the actual estimator."""
    if isinstance(model, Pipeline):
        return model[-1]
    return model


def _get_explainer(model, X_background: pd.DataFrame):
    """Build the right SHAP explainer for the model type."""
    raw = _get_raw_model(model)
    model_type = type(raw).__name__

    # TreeExplainer: RF and XGB support multiclass, GradientBoosting does NOT
    if model_type in ("RandomForestClassifier", "XGBClassifier",
                      "DecisionTreeClassifier", "ExtraTreesClassifier"):
        return shap.TreeExplainer(raw)

    if model_type in ("LogisticRegression", "LinearSVC", "SGDClassifier"):
        if isinstance(model, Pipeline):
            X_bg_transformed = model[:-1].transform(X_background)
        else:
            X_bg_transformed = X_background
        return shap.LinearExplainer(raw, X_bg_transformed)

    # GradientBoosting + anything else → KernelExplainer (slower but universal)
    print(f"  [shap] Using KernelExplainer for {model_type}...")
    pred_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
    bg = shap.sample(X_background, 50)
    return shap.KernelExplainer(pred_fn, bg)


def _transform_X(model, X: pd.DataFrame) -> np.ndarray:
    """Apply pipeline transforms (e.g. StandardScaler) if present."""
    if isinstance(model, Pipeline) and len(model) > 1:
        return model[:-1].transform(X)
    return X.values if hasattr(X, "values") else X


def run_shap_analysis(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    class_names: list,
    feature_names: list,
    n_background: int = 200,
) -> dict:
    """
    Full SHAP analysis — generates all plots and returns shap_values + explainer.
    """
    print("\n[shap] Computing SHAP values...")

    # Background dataset for explainer (sample of training data)
    bg_idx = np.random.choice(len(X_train), size=min(n_background, len(X_train)), replace=False)
    X_background = X_train.iloc[bg_idx]
    X_explain    = X_test

    explainer  = _get_explainer(model, X_background)
    X_exp_vals = _transform_X(model, X_explain)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_exp_vals)

    # Normalise to list-of-arrays format (one per class)
    # TreeExplainer for multiclass returns list; LinearExplainer may return 2D array
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # shape (n_samples, n_features, n_classes) → list of (n_samples, n_features)
        shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
        # Binary or single output — wrap in list
        shap_values = [shap_values]

    n_classes = len(shap_values)
    feature_df = pd.DataFrame(X_exp_vals, columns=feature_names)

    # ── 1. Summary plot (all classes stacked) ────────────────────────────
    print("  [shap] Generating summary plot...")
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(
        shap_values, feature_df,
        class_names=class_names,
        show=False,
        max_display=15,
    )
    plt.title("SHAP Summary — All Classes", pad=12)
    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / "shap_summary.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── 2. Beeswarm per class ─────────────────────────────────────────────
    for i, cls in enumerate(class_names):
        if i >= len(shap_values):
            break
        print(f"  [shap] Beeswarm for class '{cls}'...")
        fig, ax = plt.subplots(figsize=(9, 6))
        shap.summary_plot(
            shap_values[i], feature_df,
            plot_type="dot", show=False, max_display=15,
        )
        plt.title(f"SHAP Beeswarm — '{cls}'", pad=12)
        plt.tight_layout()
        fname = OUTPUTS_DIR / f"shap_beeswarm_{cls.lower().replace(' ', '_')}.png"
        fig.savefig(fname, dpi=130, bbox_inches="tight")
        plt.close(fig)

    # ── 3. Mean |SHAP| bar chart (averaged across classes) ───────────────
    print("  [shap] Generating bar chart...")
    mean_abs = np.mean([np.abs(sv).mean(0) for sv in shap_values], axis=0)
    importance_df = pd.DataFrame({
        "feature":    feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    top15 = importance_df.head(15)
    ax.barh(top15["feature"][::-1], top15["mean_abs_shap"][::-1], color="#4C72B0")
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("Top 15 Features by Mean |SHAP| (All Classes)")
    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / "shap_bar.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Save raw importance CSV
    importance_df.to_csv(OUTPUTS_DIR / "shap_values.csv", index=False)

    print(f"\n  Top 5 features by SHAP importance:")
    for _, row in importance_df.head(5).iterrows():
        print(f"    {row['feature']:<35} {row['mean_abs_shap']:.4f}")

    return {
        "explainer":      explainer,
        "shap_values":    shap_values,
        "importance_df":  importance_df,
        "feature_df":     feature_df,
        "X_background":   X_background,
    }


def explain_single(
    model,
    X_row: pd.DataFrame,
    X_background: pd.DataFrame,
    class_names: list,
    feature_names: list,
    class_idx: int = 0,
) -> shap.Explanation:
    """
    Returns a shap.Explanation for a single row.
    Used by the Streamlit app to show per-prediction waterfall charts.

    Args:
        model        — trained model (or Pipeline)
        X_row        — single-row DataFrame (same columns as training X)
        X_background — sample of training X for explainer background
        class_names  — list of class label strings
        feature_names— list of feature names
        class_idx    — which class to explain (0=first class alphabetically)
    """
    explainer = _get_explainer(model, X_background)

    X_row_vals = _transform_X(model, X_row)
    X_bg_vals  = _transform_X(model, X_background)

    shap_vals = explainer.shap_values(X_row_vals)

    # Normalise
    if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        shap_vals = [shap_vals[:, :, i] for i in range(shap_vals.shape[2])]
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2:
        shap_vals = [shap_vals]

    sv_class = shap_vals[class_idx][0]

    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        base = ev[class_idx]
    else:
        base = ev

    exp = shap.Explanation(
        values=sv_class,
        base_values=base,
        data=X_row_vals[0] if hasattr(X_row_vals, "__getitem__") else X_row_vals,
        feature_names=feature_names,
    )
    return exp
