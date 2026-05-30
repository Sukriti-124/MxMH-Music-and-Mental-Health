"""
models.py
Trains and evaluates 4 classifiers on the MxMH dataset:
    Random Forest, Gradient Boosting, XGBoost, Logistic Regression

Outputs:
    - Per-model metrics (accuracy, F1, ROC-AUC)
    - Confusion matrices saved to outputs/
    - Learning curves saved to outputs/
    - Best model + scaler saved to models/
    - Summary comparison DataFrame
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, f1_score
)
from xgboost import XGBClassifier

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
MODELS_DIR  = Path(__file__).parent.parent / "models"
OUTPUTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.2


def _build_models() -> dict:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            objective="multi:softmax",
            num_class=3,
            eval_metric="mlogloss",
            max_depth=6,
            n_estimators=200,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                solver="lbfgs",
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


def _plot_confusion_matrix(cm, class_names, model_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    fname = OUTPUTS_DIR / f"cm_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def _plot_learning_curve(model, X_train, y_train, model_name):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 6),
        scoring="f1_weighted",
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_sizes, train_scores.mean(1), label="Train F1")
    ax.plot(train_sizes, val_scores.mean(1),   label="Validation F1")
    ax.fill_between(train_sizes, train_scores.mean(1) - train_scores.std(1),
                                  train_scores.mean(1) + train_scores.std(1), alpha=0.15)
    ax.fill_between(train_sizes, val_scores.mean(1) - val_scores.std(1),
                                  val_scores.mean(1) + val_scores.std(1), alpha=0.15)
    ax.set_title(f"Learning Curve — {model_name}")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("F1 (weighted)")
    ax.legend()
    plt.tight_layout()
    fname = OUTPUTS_DIR / f"lc_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, class_names: list) -> dict:
    """
    Train all models, evaluate, save plots and best model.

    Returns:
        results dict with keys = model names, values = metric dicts
        Also saves best_model.pkl and best_model_scaler.pkl to models/
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = _build_models()
    results = {}
    best_f1 = -1
    best_name = None
    best_model = None

    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        acc  = accuracy_score(y_test, y_pred)
        f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_m = f1_score(y_test, y_pred, average="macro",    zero_division=0)

        # ROC-AUC (one-vs-rest, weighted)
        if y_proba is not None:
            try:
                auc = roc_auc_score(
                    label_binarize(y_test, classes=sorted(y.unique())),
                    y_proba, average="weighted", multi_class="ovr"
                )
            except Exception:
                auc = None
        else:
            auc = None

        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred,
                                    target_names=class_names, zero_division=0)

        _plot_confusion_matrix(cm, class_names, name)
        _plot_learning_curve(model, X_train, y_train, name)

        results[name] = {
            "model":      model,
            "accuracy":   round(acc,  4),
            "f1_weighted":round(f1_w, 4),
            "f1_macro":   round(f1_m, 4),
            "roc_auc":    round(auc,  4) if auc else None,
            "conf_matrix": cm,
            "class_report": cr,
            "X_train": X_train,
            "X_test":  X_test,
            "y_train": y_train,
            "y_test":  y_test,
            "y_pred":  y_pred,
        }

        auc_str = f"{auc:.3f}" if auc else "n/a"
        print(f"    accuracy={acc:.3f}  F1(w)={f1_w:.3f}  ROC-AUC={auc_str}")

        shap_preferred = name in ("Random Forest", "XGBoost")
        if f1_w > best_f1 or (shap_preferred and f1_w >= best_f1 - 0.01):
            best_f1   = f1_w
            best_name = name
            best_model = model


    # Save best model
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    print(f"\n  Best model: {best_name} (F1 weighted={best_f1:.4f})")
    print(f"  Saved → models/best_model.pkl")

    # Comparison table
    summary = pd.DataFrame([
        {
            "Model":        n,
            "Accuracy":     r["accuracy"],
            "F1 (Weighted)":r["f1_weighted"],
            "F1 (Macro)":   r["f1_macro"],
            "ROC-AUC":      r["roc_auc"],
        }
        for n, r in results.items()
    ]).set_index("Model").sort_values("F1 (Weighted)", ascending=False)

    print("\n── Model Comparison ─────────────────────────────")
    print(summary.to_string())
    summary.to_csv(OUTPUTS_DIR / "model_comparison.csv")

    # Save summary plot
    fig, ax = plt.subplots(figsize=(8, 4))
    summary[["Accuracy", "F1 (Weighted)", "ROC-AUC"]].plot(
        kind="bar", ax=ax, colormap="Set2", edgecolor="white"
    )
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / "model_comparison.png", dpi=120)
    plt.close(fig)

    results["_summary"]    = summary
    results["_best_name"]  = best_name
    results["_best_model"] = best_model
    results["_X_train"]    = X_train
    results["_X_test"]     = X_test
    results["_y_train"]    = y_train
    results["_y_test"]     = y_test

    return results
