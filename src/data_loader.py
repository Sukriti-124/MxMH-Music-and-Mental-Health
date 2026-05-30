"""
data_loader.py
Loads the MxMH dataset from the local data/ folder.

Setup (one-time, 2 minutes):
    1. Go to: https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results
    2. Click Download → extract the zip
    3. Place  mxmh_survey_results.csv  inside the  data/  folder of this project
    4. Run the project normally — this file handles the rest automatically
"""

import sys
import pandas as pd
from pathlib import Path

DATA_DIR  = Path(__file__).parent.parent / "data"
DATA_FILE = DATA_DIR / "mxmh_survey_results.csv"
KAGGLE_URL = "https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results"


def _missing_data_instructions():
    print(
        "\n╔══════════════════════════════════════════════════════════╗\n"
        "║              Dataset not found — quick fix               ║\n"
        "╠══════════════════════════════════════════════════════════╣\n"
        f"║  1. Go to: {KAGGLE_URL}\n"
        "║  2. Click the Download button (top-right)\n"
        "║  3. Extract the zip — get mxmh_survey_results.csv\n"
        "║  4. Move it into:  data/mxmh_survey_results.csv\n"
        "║  5. Run again — you're all set!\n"
        "╚══════════════════════════════════════════════════════════╝\n"
    )
    sys.exit(1)


def load_raw() -> pd.DataFrame:
    """
    Load the raw MxMH CSV.
    Prints clear instructions and exits if the file is not found.
    """
    DATA_DIR.mkdir(exist_ok=True)

    if not DATA_FILE.exists():
        print(f"\n[data_loader] ✗ File not found: {DATA_FILE}")
        _missing_data_instructions()

    df = pd.read_csv(DATA_FILE)
    print(f"[data_loader] ✓ Loaded {df.shape[0]} rows × {df.shape[1]} columns  ({DATA_FILE.name})")
    return df


if __name__ == "__main__":
    df = load_raw()
    print(df.head(3))
