from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def ensure_customer_ids(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "customer_id" in dataframe.columns:
        return dataframe

    enriched = dataframe.copy()
    enriched.insert(0, "customer_id", [f"CUST-{index + 1:04d}" for index in range(len(enriched))])
    return enriched


def metrics_to_frame(metrics: dict) -> pd.DataFrame:
    report = metrics["classification_report"]
    rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            row = {"label": label}
            row.update({key: round(float(value), 4) for key, value in values.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def model_path() -> Path:
    return Path("models/churn_model.joblib")


def db_path() -> Path:
    return Path("data/churnguard_history.db")

