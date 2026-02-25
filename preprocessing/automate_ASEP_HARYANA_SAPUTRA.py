"""
automate_ASEP_HARYANA_SAPUTRA.py
Automation script for Poker Hand dataset preprocessing.
Mirrors the logic from Eksperimen_ASEP_HARYANA_SAPUTRA.ipynb exactly.

Usage:
    python automate_ASEP_HARYANA_SAPUTRA.py
    python automate_ASEP_HARYANA_SAPUTRA.py --raw-dir ../pokerhand_raw --out-dir pokerhand_preprocessing
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COLUMNS = [
    "S1", "C1",
    "S2", "C2",
    "S3", "C3",
    "S4", "C4",
    "S5", "C5",
    "CLASS",
]

SUIT_COLS = ["S1", "S2", "S3", "S4", "S5"]
RANK_COLS = ["C1", "C2", "C3", "C4", "C5"]
DERIVED_COLS = ["all_same_suit", "n_unique_suits", "n_unique_ranks", "rank_range"]

CLASS_NAMES = {
    0: "Nothing",
    1: "One Pair",
    2: "Two Pairs",
    3: "Three of a Kind",
    4: "Straight",
    5: "Flush",
    6: "Full House",
    7: "Four of a Kind",
    8: "Straight Flush",
    9: "Royal Flush",
}

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Load data
# ---------------------------------------------------------------------------
def load_data(raw_dir: str | Path) -> pd.DataFrame:
    """
    Load the Poker Hand training dataset from raw .data file.
    Assigns column names and returns a combined DataFrame.

    Parameters
    ----------
    raw_dir : path to directory containing poker-hand-training-true.data

    Returns
    -------
    pd.DataFrame with COLUMNS headers
    """
    raw_dir = Path(raw_dir)
    train_path = raw_dir / "poker-hand-training-true.data"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            "Ensure the dataset has been extracted from poker+hand.zip."
        )

    log.info("Loading training data from %s", train_path)
    df = pd.read_csv(train_path, header=None, names=COLUMNS)
    log.info("Loaded %d rows × %d cols", *df.shape)

    # Basic validation
    assert list(df.columns) == COLUMNS, "Column mismatch after load"
    assert df.dtypes.nunique() == 1, "Expected all int64 dtypes"

    return df


# ---------------------------------------------------------------------------
# Step 2: Clean data
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Verify and log missing values (expected: zero).
    - Remove exact duplicate rows.

    Parameters
    ----------
    df : raw DataFrame from load_data()

    Returns
    -------
    Cleaned pd.DataFrame
    """
    # Missing value check
    missing_total = df.isnull().sum().sum()
    if missing_total > 0:
        log.warning("Found %d missing values — filling with column medians", missing_total)
        df = df.fillna(df.median(numeric_only=True))
    else:
        log.info("No missing values found.")

    # Duplicate removal
    n_before = len(df)
    df = df.drop_duplicates()
    n_after = len(df)
    log.info("Duplicates removed: %d (%.2f%%)", n_before - n_after, (n_before - n_after) / n_before * 100)

    # Range validation (suit: 1-4, rank: 1-13, class: 0-9)
    for col in SUIT_COLS:
        assert df[col].between(1, 4).all(), f"Out-of-range values in {col}"
    for col in RANK_COLS:
        assert df[col].between(1, 13).all(), f"Out-of-range values in {col}"
    assert df["CLASS"].between(0, 9).all(), "Out-of-range values in CLASS"

    log.info("Data validation passed. Clean shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Step 3: Feature engineering + encoding + SMOTE
# ---------------------------------------------------------------------------
def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add poker-relevant derived features identical to notebook logic.
    These features have strong correlation with the target CLASS.
    """
    d = df.copy()
    d["all_same_suit"] = (d[SUIT_COLS].nunique(axis=1) == 1).astype(int)
    d["n_unique_suits"] = d[SUIT_COLS].nunique(axis=1)
    d["n_unique_ranks"] = d[RANK_COLS].nunique(axis=1)
    d["rank_range"] = d[RANK_COLS].max(axis=1) - d[RANK_COLS].min(axis=1)
    return d


def preprocess_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    apply_smote: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, list[str]]:
    """
    Full preprocessing pipeline (mirrors notebook):
    1. Feature engineering (derived features)
    2. Train-test split (stratified)
    3. OneHotEncoder on suit cols, StandardScaler on rank + derived cols
    4. SMOTE balancing on training set only

    Parameters
    ----------
    df            : cleaned DataFrame
    test_size     : fraction of data for test set
    random_state  : reproducibility seed
    apply_smote   : whether to apply SMOTE (disabled for inference)

    Returns
    -------
    X_train, X_test, y_train, y_test, fitted_preprocessor, feature_names
    """
    log.info("Adding derived features …")
    df_feat = _add_derived_features(df)

    X = df_feat.drop(columns=["CLASS"])
    y = df_feat["CLASS"]

    # 80/20 stratified split
    log.info("Splitting data (test_size=%.2f, stratify=True) …", test_size)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Preprocessor: OHE for suits, StandardScaler for ranks + derived
    preprocessor = ColumnTransformer(
        transformers=[
            ("suit_ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), SUIT_COLS),
            ("rank_scale", StandardScaler(), RANK_COLS + DERIVED_COLS),
        ],
        remainder="drop",
    )

    log.info("Fitting preprocessor on training split only …")
    X_train_enc = preprocessor.fit_transform(X_train_raw)
    X_test_enc  = preprocessor.transform(X_test_raw)

    # Derive output feature names
    ohe_names   = preprocessor.named_transformers_["suit_ohe"].get_feature_names_out(SUIT_COLS).tolist()
    scale_names = RANK_COLS + DERIVED_COLS
    feature_names = ohe_names + scale_names

    log.info("Encoded feature count: %d", len(feature_names))
    log.info("X_train before SMOTE: %s  class dist: %s", X_train_enc.shape, Counter(y_train))

    if apply_smote:
        log.info("Applying SMOTE (k_neighbors=3) …")
        smote = SMOTE(random_state=random_state, k_neighbors=3)
        X_train_enc, y_train = smote.fit_resample(X_train_enc, y_train)
        log.info("X_train after SMOTE : %s  class dist: %s", X_train_enc.shape, Counter(y_train))

    return X_train_enc, X_test_enc, y_train, y_test, preprocessor, feature_names


# ---------------------------------------------------------------------------
# Step 4: Save processed data
# ---------------------------------------------------------------------------
def save_processed_data(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray | pd.Series,
    y_test:  np.ndarray | pd.Series,
    preprocessor: ColumnTransformer,
    feature_names: list[str],
    out_dir: str | Path,
) -> None:
    """
    Persist train/test splits and preprocessor to `out_dir`.

    Outputs
    -------
    out_dir/
        X_train.csv
        X_test.csv
        y_train.csv
        y_test.csv
        preprocessor.joblib
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(X_train, columns=feature_names).to_csv(out_dir / "X_train.csv", index=False)
    pd.DataFrame(X_test,  columns=feature_names).to_csv(out_dir / "X_test.csv",  index=False)
    pd.DataFrame({"CLASS": y_train}).to_csv(out_dir / "y_train.csv", index=False)
    pd.DataFrame({"CLASS": y_test }).to_csv(out_dir / "y_test.csv",  index=False)
    joblib.dump(preprocessor, out_dir / "preprocessor.joblib")

    log.info("Saved preprocessed data to: %s", out_dir)
    for file in sorted(out_dir.iterdir()):
        log.info("  %s  (%.1f KB)", file.name, file.stat().st_size / 1024)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Poker Hand Preprocessing Pipeline — ASEP HARYANA SAPUTRA"
    )
    parser.add_argument(
        "--raw-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "pokerhand_raw"),
        help="Directory containing raw .data files (default: ../pokerhand_raw)",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "pokerhand_preprocessing"),
        help="Output directory for processed CSVs (default: ./pokerhand_preprocessing)",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data for the test split (default: 0.2)",
    )
    parser.add_argument(
        "--no-smote", action="store_true",
        help="Disable SMOTE oversampling (useful for quick sanity checks)",
    )
    args = parser.parse_args()

    log.info("="*60)
    log.info("Poker Hand Preprocessing — ASEP HARYANA SAPUTRA")
    log.info("="*60)

    df_raw    = load_data(raw_dir=args.raw_dir)
    df_clean  = clean_data(df_raw)

    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(
        df=df_clean,
        test_size=args.test_size,
        random_state=42,
        apply_smote=not args.no_smote,
    )

    save_processed_data(
        X_train, X_test, y_train, y_test,
        preprocessor, feature_names,
        out_dir=args.out_dir,
    )

    log.info("="*60)
    log.info("Pipeline complete. Dataset ready for training.")
    log.info("  X_train: %s | X_test: %s", X_train.shape, X_test.shape)
    log.info("="*60)


if __name__ == "__main__":
    main()
