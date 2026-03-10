"""Pre-compute analytics artifacts: feature importance, correlations, summary stats."""

import json
import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CACHE_DIR = pathlib.Path(__file__).resolve().parent.parent / "cache"


def compute_feature_importance(df: pd.DataFrame, top_n: int = 30) -> dict:
    """Train a LightGBM model and extract feature importances."""
    import lightgbm as lgb

    target = "TARGET"
    exclude = {"SK_ID_CURR", "TARGET", "index"}
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]

    X = df[numeric_cols].copy()
    y = df[target]

    X = X.fillna(-999)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 7,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    importance = dict(zip(X.columns, model.feature_importance(importance_type="gain")))
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]

    auc = model.best_score["valid_0"]["auc"]
    return {
        "feature_importance": [{"feature": f, "importance": float(v)} for f, v in sorted_imp],
        "model_auc": round(auc, 4),
        "num_features_used": len(numeric_cols),
    }


def compute_target_correlations(df: pd.DataFrame, top_n: int = 30) -> list[dict]:
    """Compute correlations of numeric columns with TARGET."""
    target = "TARGET"
    exclude = {"SK_ID_CURR", "TARGET", "index"}
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]

    correlations = df[numeric_cols + [target]].corr()[target].drop(target)
    top = correlations.abs().sort_values(ascending=False).head(top_n)

    return [
        {"feature": feat, "correlation": round(float(correlations[feat]), 4)}
        for feat in top.index
    ]


def compute_summary_statistics(df: pd.DataFrame) -> dict:
    """Compute per-column summary statistics."""
    stats = {}
    for col in df.columns:
        info = {
            "dtype": str(df[col].dtype),
            "null_pct": round(float(df[col].isna().mean() * 100), 2),
            "nunique": int(df[col].nunique()),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe()
            info.update({
                "mean": round(float(desc.get("mean", 0)), 4),
                "std": round(float(desc.get("std", 0)), 4),
                "min": float(desc.get("min", 0)),
                "max": float(desc.get("max", 0)),
                "median": round(float(desc.get("50%", 0)), 4),
            })
        else:
            top_values = df[col].value_counts().head(5).to_dict()
            info["top_values"] = {str(k): int(v) for k, v in top_values.items()}
        stats[col] = info
    return stats


def compute_default_rate_segments(df: pd.DataFrame) -> dict:
    """Compute default rate broken down by key categorical segments."""
    segments = {}
    cat_columns = [
        "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE", "CODE_GENDER", "NAME_CONTRACT_TYPE",
        "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "REGION_RATING_CLIENT",
    ]

    for col in cat_columns:
        if col not in df.columns:
            continue
        seg = (
            df.groupby(col)["TARGET"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "default_rate", "count": "sample_size"})
            .sort_values("default_rate", ascending=False)
        )
        seg["default_rate"] = seg["default_rate"].round(4)
        segments[col] = seg.reset_index().to_dict(orient="records")

    return segments


def save_precomputed(artifacts: dict, cache_dir: pathlib.Path = CACHE_DIR):
    """Save all pre-computed artifacts as JSON."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "precomputed_analytics.json"
    with open(path, "w") as f:
        json.dump(artifacts, f, indent=2, default=str)
    print(f"Saved precomputed analytics to {path}")


def load_precomputed(cache_dir: pathlib.Path = CACHE_DIR) -> dict:
    """Load pre-computed artifacts from cache, with convenience aliases."""
    path = cache_dir / "precomputed_analytics.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
    else:
        data = {}

    ds = data.get("dataset_info", {})
    if "default_rate" in ds:
        data["overall_default_rate"] = ds["default_rate"]
    if "rows" in ds:
        data["num_rows"] = ds["rows"]
    if "columns" in ds:
        data["num_columns"] = ds["columns"]

    if "dataset_info" in data and "default_rate" in data["dataset_info"]:
        data["dataset_info"]["overall_default_rate"] = data["dataset_info"]["default_rate"]

    return data
