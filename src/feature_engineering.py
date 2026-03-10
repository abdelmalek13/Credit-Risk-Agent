"""Aggregate auxiliary tables and engineer features for the main dataset."""

import numpy as np
import pandas as pd
import pathlib

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent


def clean_application(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the application_train dataframe."""
    df = df.copy()

    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365.25).round(1)
    df["EMPLOYMENT_YEARS"] = (-df["DAYS_EMPLOYED"] / 365.25).round(1)
    df["INCOME_TO_CREDIT_RATIO"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"].replace(0, np.nan)
    df["ANNUITY_TO_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["CREDIT_TO_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"].replace(0, np.nan)
    df["INCOME_PER_FAMILY_MEMBER"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)

    return df


def aggregate_bureau(data_dir: pathlib.Path = DATA_DIR) -> pd.DataFrame:
    """Aggregate bureau.csv into per-client features."""
    path = data_dir / "bureau.csv"
    if not path.exists():
        return pd.DataFrame(columns=["SK_ID_CURR"])

    bureau = pd.read_csv(path)

    agg = bureau.groupby("SK_ID_CURR").agg(
        BUREAU_CREDIT_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_COUNT=("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
        BUREAU_CLOSED_COUNT=("CREDIT_ACTIVE", lambda x: (x == "Closed").sum()),
        BUREAU_AVG_DAYS_CREDIT=("DAYS_CREDIT", "mean"),
        BUREAU_MAX_OVERDUE=("AMT_CREDIT_MAX_OVERDUE", "max"),
        BUREAU_AVG_CREDIT_SUM=("AMT_CREDIT_SUM", "mean"),
        BUREAU_TOTAL_DEBT=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_AVG_CREDIT_DAY_OVERDUE=("CREDIT_DAY_OVERDUE", "mean"),
    ).reset_index()

    return agg


def aggregate_previous_application(data_dir: pathlib.Path = DATA_DIR) -> pd.DataFrame:
    """Aggregate previous_application.csv into per-client features."""
    path = data_dir / "previous_application.csv"
    if not path.exists():
        return pd.DataFrame(columns=["SK_ID_CURR"])

    prev = pd.read_csv(path)

    agg = prev.groupby("SK_ID_CURR").agg(
        PREV_APP_COUNT=("SK_ID_PREV", "count"),
        PREV_APPROVED_COUNT=("NAME_CONTRACT_STATUS", lambda x: (x == "Approved").sum()),
        PREV_REFUSED_COUNT=("NAME_CONTRACT_STATUS", lambda x: (x == "Refused").sum()),
        PREV_AVG_AMT_CREDIT=("AMT_CREDIT", "mean"),
        PREV_AVG_AMT_ANNUITY=("AMT_ANNUITY", "mean"),
        PREV_AVG_DAYS_DECISION=("DAYS_DECISION", "mean"),
    ).reset_index()

    agg["PREV_APPROVAL_RATE"] = agg["PREV_APPROVED_COUNT"] / agg["PREV_APP_COUNT"].replace(0, np.nan)

    return agg


def aggregate_installments(data_dir: pathlib.Path = DATA_DIR) -> pd.DataFrame:
    """Aggregate installments_payments.csv into per-client features."""
    path = data_dir / "installments_payments.csv"
    if not path.exists():
        return pd.DataFrame(columns=["SK_ID_CURR"])

    inst = pd.read_csv(path)

    inst["PAYMENT_DIFF"] = inst["AMT_PAYMENT"] - inst["AMT_INSTALMENT"]
    inst["DAYS_LATE"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
    inst["IS_LATE"] = (inst["DAYS_LATE"] > 0).astype(int)

    agg = inst.groupby("SK_ID_CURR").agg(
        INST_COUNT=("NUM_INSTALMENT_NUMBER", "count"),
        INST_AVG_DAYS_LATE=("DAYS_LATE", "mean"),
        INST_MAX_DAYS_LATE=("DAYS_LATE", "max"),
        INST_LATE_RATIO=("IS_LATE", "mean"),
        INST_AVG_PAYMENT_DIFF=("PAYMENT_DIFF", "mean"),
    ).reset_index()

    return agg


def aggregate_credit_card(data_dir: pathlib.Path = DATA_DIR) -> pd.DataFrame:
    """Aggregate credit_card_balance.csv into per-client features."""
    path = data_dir / "credit_card_balance.csv"
    if not path.exists():
        return pd.DataFrame(columns=["SK_ID_CURR"])

    cc = pd.read_csv(path)

    agg = cc.groupby("SK_ID_CURR").agg(
        CC_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
        CC_AVG_BALANCE=("AMT_BALANCE", "mean"),
        CC_MAX_BALANCE=("AMT_BALANCE", "max"),
        CC_AVG_DRAWINGS=("AMT_DRAWINGS_CURRENT", "mean"),
        CC_AVG_PAYMENT=("AMT_PAYMENT_CURRENT", "mean"),
        CC_MAX_DPD=("SK_DPD", "max"),
    ).reset_index()

    return agg


def aggregate_pos_cash(data_dir: pathlib.Path = DATA_DIR) -> pd.DataFrame:
    """Aggregate POS_CASH_balance.csv into per-client features."""
    path = data_dir / "POS_CASH_balance.csv"
    if not path.exists():
        return pd.DataFrame(columns=["SK_ID_CURR"])

    pos = pd.read_csv(path)

    agg = pos.groupby("SK_ID_CURR").agg(
        POS_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
        POS_MAX_DPD=("SK_DPD", "max"),
        POS_AVG_DPD=("SK_DPD", "mean"),
        POS_COMPLETED_COUNT=("NAME_CONTRACT_STATUS", lambda x: (x == "Completed").sum()),
    ).reset_index()

    return agg


def build_full_dataset(data_dir: pathlib.Path = DATA_DIR) -> pd.DataFrame:
    """Load application_train, clean it, and merge all aggregated features."""
    print("Loading application_train.csv ...")
    app = pd.read_csv(data_dir / "application_train.csv")
    app = clean_application(app)

    print("Aggregating bureau data ...")
    bureau_agg = aggregate_bureau(data_dir)
    app = app.merge(bureau_agg, on="SK_ID_CURR", how="left")

    print("Aggregating previous applications ...")
    prev_agg = aggregate_previous_application(data_dir)
    app = app.merge(prev_agg, on="SK_ID_CURR", how="left")

    print("Aggregating installment payments ...")
    inst_agg = aggregate_installments(data_dir)
    app = app.merge(inst_agg, on="SK_ID_CURR", how="left")

    print("Aggregating credit card balances ...")
    cc_agg = aggregate_credit_card(data_dir)
    app = app.merge(cc_agg, on="SK_ID_CURR", how="left")

    print("Aggregating POS/cash balances ...")
    pos_agg = aggregate_pos_cash(data_dir)
    app = app.merge(pos_agg, on="SK_ID_CURR", how="left")

    print(f"Final dataset shape: {app.shape}")
    return app
