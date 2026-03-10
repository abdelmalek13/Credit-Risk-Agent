#!/usr/bin/env python3
"""One-time preprocessing: build feature-enriched dataset and pre-compute analytics.

Usage:
    python preprocess.py
"""

import pathlib
import time

from src.feature_engineering import build_full_dataset
from src.precomputed import (
    compute_default_rate_segments,
    compute_feature_importance,
    compute_summary_statistics,
    compute_target_correlations,
    save_precomputed,
)

DATA_DIR = pathlib.Path(__file__).resolve().parent
CACHE_DIR = DATA_DIR / "cache"


def main():
    start = time.time()

    df = build_full_dataset(DATA_DIR)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = CACHE_DIR / "app_with_features.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Saved enriched dataset to {parquet_path}  ({len(df)} rows, {len(df.columns)} cols)")

    print("\nComputing feature importance (LightGBM) ...")
    importance = compute_feature_importance(df)
    print(f"  Model AUC: {importance['model_auc']}")

    print("Computing target correlations ...")
    correlations = compute_target_correlations(df)

    print("Computing summary statistics ...")
    summary = compute_summary_statistics(df)

    print("Computing default-rate segments ...")
    segments = compute_default_rate_segments(df)

    artifacts = {
        "feature_importance": importance,
        "target_correlations": correlations,
        "summary_statistics": summary,
        "default_rate_segments": segments,
        "dataset_info": {
            "rows": len(df),
            "columns": len(df.columns),
            "default_rate": round(float(df["TARGET"].mean()), 4),
            "column_names": list(df.columns),
        },
    }

    save_precomputed(artifacts, CACHE_DIR)

    elapsed = time.time() - start
    print(f"\nPreprocessing complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
