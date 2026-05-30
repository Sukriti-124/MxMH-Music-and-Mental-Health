"""
main.py
Run the full MxMH analysis pipeline end-to-end.

Usage:
    python main.py                  # full pipeline
    python main.py --skip-shap      # skip SHAP (faster)
    python main.py --skip-cluster   # skip clustering
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader   import load_raw
from src.preprocessing import preprocess
from src.models        import train_and_evaluate
from src.shap_analysis import run_shap_analysis
from src.clustering    import run_clustering


def parse_args():
    p = argparse.ArgumentParser(description="MxMH Mental Health + Music Analysis")
    p.add_argument("--skip-shap",    action="store_true", help="Skip SHAP analysis")
    p.add_argument("--skip-cluster", action="store_true", help="Skip clustering")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 55)
    print("  MxMH: Music × Mental Health — Full Pipeline")
    print("=" * 55)

    # ── 1. Load ───────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    raw_df = load_raw()

    # ── 2. Preprocess ─────────────────────────────────────────
    print("\n[2/4] Preprocessing...")
    proc = preprocess(raw_df)
    X            = proc["X"]
    y            = proc["y"]
    class_names  = list(proc["label_encoder"].classes_)
    feature_names= proc["feature_names"]
    df_clean     = proc["df_clean"]

    # ── 3. Train & evaluate models ───────────────────────────
    print("\n[3/4] Training models...")
    results = train_and_evaluate(X, y, class_names)

    best_model = results["_best_model"]
    X_train    = results["_X_train"]
    X_test     = results["_X_test"]

    # ── 4a. SHAP ─────────────────────────────────────────────
    if not args.skip_shap:
        print("\n[4a/4] SHAP explainability...")
        shap_results = run_shap_analysis(
            model=best_model,
            X_train=X_train,
            X_test=X_test,
            class_names=class_names,
            feature_names=feature_names,
        )
    else:
        print("\n[4a/4] SHAP skipped.")

    # ── 4b. Clustering ───────────────────────────────────────
    if not args.skip_cluster:
        print("\n[4b/4] Unsupervised clustering...")
        cluster_results = run_clustering(df_clean)
    else:
        print("\n[4b/4] Clustering skipped.")

    print("\n" + "=" * 55)
    print("  Pipeline complete! Check the outputs/ folder.")
    print("=" * 55)

    print("\nGenerated files:")
    for f in sorted(Path("outputs").glob("*")):
        print(f"  outputs/{f.name}")


if __name__ == "__main__":
    main()
