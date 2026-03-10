"""Load and serve the preprocessed dataset and raw CSVs via DuckDB."""

import pathlib
import duckdb
import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent
CACHE_DIR = DATA_DIR / "cache"


def load_main_dataset() -> pd.DataFrame:
    """Load the preprocessed parquet (with engineered features).

    Falls back to raw application_train.csv if cache doesn't exist yet.
    """
    parquet_path = CACHE_DIR / "app_with_features.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    csv_path = DATA_DIR / "application_train.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)

    raise FileNotFoundError(
        "No dataset found. Place application_train.csv in the project root "
        "or run preprocess.py first."
    )


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection pre-registered with all CSV files as views."""
    con = duckdb.connect(database=":memory:")

    csv_files = {
        "application_train": "application_train.csv",
        "application_test": "application_test.csv",
        "bureau": "bureau.csv",
        "bureau_balance": "bureau_balance.csv",
        "credit_card_balance": "credit_card_balance.csv",
        "installments_payments": "installments_payments.csv",
        "pos_cash_balance": "POS_CASH_balance.csv",
        "previous_application": "previous_application.csv",
    }

    for view_name, filename in csv_files.items():
        filepath = DATA_DIR / filename
        if filepath.exists():
            con.execute(
                f"CREATE VIEW {view_name} AS SELECT * FROM read_csv_auto('{filepath}')"
            )

    return con


def load_column_descriptions() -> pd.DataFrame:
    """Load the column-descriptions metadata file."""
    path = DATA_DIR / "HomeCredit_columns_description.csv"
    if path.exists():
        return pd.read_csv(path, encoding="latin-1")
    return pd.DataFrame(columns=["Table", "Row", "Description", "Special"])


def get_schema_summary(df: pd.DataFrame) -> str:
    """Build a compact schema string for the LLM system prompt."""
    lines = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        nunique = df[col].nunique()
        null_pct = df[col].isna().mean() * 100
        if pd.api.types.is_numeric_dtype(df[col]):
            sample = f"range [{df[col].min():.2f}, {df[col].max():.2f}]"
        else:
            top_vals = df[col].dropna().value_counts().head(3).index.tolist()
            sample = f"values: {top_vals}"
        lines.append(
            f"  - {col} ({dtype}, {nunique} unique, {null_pct:.1f}% null): {sample}"
        )
    return "\n".join(lines)
