"""
Preprocessing shared by all training/inference scripts (Sec. VI-A/B):
  - MICE imputation (fit on train split only)
  - drop features missing in >40% of stays, replaced by train-split median
    (documented in the paper as a deliberate simplification vs. full MICE)
  - standardize, then clip to [-1, 1] (matches the Chebyshev calibration
    domain used by the Encryption/Inference Agents)
  - 80/20 stratified split, seed = 42
"""
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split

MAX_MISSING_RATE = 0.40


def load_and_preprocess(csv_path, seed=42, test_size=0.2):
    df = pd.read_csv(csv_path)
    y = df["mortality"].values.astype(np.float32)
    X_df = df.drop(columns=["mortality"])
    feature_names = list(X_df.columns)

    miss_rate = X_df.isna().mean()
    high_missing = miss_rate[miss_rate > MAX_MISSING_RATE].index.tolist()

    X_tr_df, X_te_df, y_tr, y_te = train_test_split(
        X_df, y, test_size=test_size, random_state=seed, stratify=y
    )

    # High-missingness columns: train-median impute (both splits use the
    # TRAIN median only, to avoid test leakage).
    for col in high_missing:
        med = X_tr_df[col].median()
        X_tr_df[col] = X_tr_df[col].fillna(med)
        X_te_df[col] = X_te_df[col].fillna(med)

    # Remaining columns: MICE (IterativeImputer), fit on train only.
    remaining = [c for c in feature_names if c not in high_missing]
    imputer = IterativeImputer(random_state=seed, max_iter=10, sample_posterior=False)
    X_tr_df[remaining] = imputer.fit_transform(X_tr_df[remaining])
    X_te_df[remaining] = imputer.transform(X_te_df[remaining])

    # Standardize on train stats, then clip to [-1, 1].
    mu = X_tr_df.mean()
    sigma = X_tr_df.std().replace(0, 1.0)
    X_tr = ((X_tr_df - mu) / sigma).clip(-1, 1).values.astype(np.float32)
    X_te = ((X_te_df - mu) / sigma).clip(-1, 1).values.astype(np.float32)

    stats = {
        "feature_names": feature_names,
        "high_missing_cols": high_missing,
        "mu": mu.to_dict(),
        "sigma": sigma.to_dict(),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "prevalence_train": float(y_tr.mean()),
        "prevalence_test": float(y_te.mean()),
    }
    return X_tr, X_te, y_tr, y_te, stats


if __name__ == "__main__":
    import json
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/cohort.csv"
    X_tr, X_te, y_tr, y_te, stats = load_and_preprocess(csv_path)
    print(f"train={X_tr.shape} test={X_te.shape}")
    print(f"prevalence train={stats['prevalence_train']:.4f} test={stats['prevalence_test']:.4f}")
    print(f"high-missing cols (median-imputed): {stats['high_missing_cols']}")
    print(f"value range after clip: [{X_tr.min():.3f}, {X_tr.max():.3f}]")
