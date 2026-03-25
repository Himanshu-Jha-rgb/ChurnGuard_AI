from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from churnguard_ai.database import PredictionDatabase, PredictionRecord
from churnguard_ai.modeling import (
    TARGET_COLUMN,
    load_artifacts,
    predict_churn,
    train_model,
    validate_dataset,
    save_artifacts,
)
from churnguard_ai.ui_helpers import (
    db_path,
    ensure_customer_ids,
    metrics_to_frame,
    model_path,
    read_csv,
)


st.set_page_config(page_title="ChurnGuard AI", page_icon=":bar_chart:", layout="wide")


def render_header() -> None:
    st.title("ChurnGuard AI")
    st.caption(
        "Customer churn prediction system with preprocessing, model training, live inference, and SQL-backed history."
    )


def render_training_tab() -> None:
    st.subheader("1. Train a Churn Model")
    training_file = st.file_uploader(
        "Upload a training CSV file",
        type=["csv"],
        key="training_uploader",
        help="The training file must include 'customer_id' and 'churn' columns.",
    )

    if not training_file:
        st.info(
            "Upload the sample dataset from `data/sample_customer_churn.csv` or your own dataset to start training."
        )
        return

    training_df = ensure_customer_ids(read_csv(training_file))
    st.write("Preview of uploaded training data")
    st.dataframe(training_df.head(), use_container_width=True)

    validation_errors = validate_dataset(training_df)
    if validation_errors:
        for error in validation_errors:
            st.error(error)
        return

    if st.button("Train model", type="primary"):
        with st.spinner("Training and tuning classification models..."):
            artifacts = train_model(training_df)
            save_artifacts(artifacts, str(model_path()))

        metrics = artifacts.metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Best model", artifacts.best_model_name)
        col2.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        col3.metric("ROC-AUC", f"{metrics['roc_auc']:.2f}")

        st.success(f"Model trained and saved to {model_path()}")
        st.write("Detailed validation report")
        st.dataframe(metrics_to_frame(metrics), use_container_width=True)


def render_prediction_tab(database: PredictionDatabase) -> None:
    st.subheader("2. Run Churn Predictions")
    current_model_path = model_path()
    if not current_model_path.exists():
        st.warning("Train a model first so predictions can be generated.")
        return

    inference_file = st.file_uploader(
        "Upload a CSV file for inference",
        type=["csv"],
        key="inference_uploader",
        help="The uploaded file should include the same feature columns used during training.",
    )
    if not inference_file:
        return

    inference_df = ensure_customer_ids(read_csv(inference_file))
    st.write("Preview of uploaded inference data")
    st.dataframe(inference_df.head(), use_container_width=True)

    if st.button("Generate predictions", type="primary"):
        artifacts = load_artifacts(str(current_model_path))
        results = predict_churn(artifacts, inference_df)

        database.insert_predictions(
            [
                PredictionRecord(
                    customer_id=str(row["customer_id"]),
                    churn_prediction=int(row["churn_prediction"]),
                    churn_probability=float(row["churn_probability"]),
                    model_name=artifacts.best_model_name,
                )
                for _, row in results.iterrows()
            ]
        )

        st.success("Predictions generated and stored in the SQL history database.")
        st.dataframe(results, use_container_width=True)
        st.bar_chart(
            results["churn_prediction"]
            .value_counts()
            .rename(index={0: "Retained", 1: "Churned"})
        )


def render_history_tab(database: PredictionDatabase) -> None:
    st.subheader("3. Prediction History")
    history = database.fetch_recent_predictions()
    if not history:
        st.info("No saved predictions yet.")
        return

    history_df = pd.DataFrame(history)
    st.dataframe(history_df, use_container_width=True)


def main() -> None:
    Path("models").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)

    render_header()
    database = PredictionDatabase(db_path())

    train_tab, predict_tab, history_tab = st.tabs(
        ["Model Training", "Predictions", "History"]
    )

    with train_tab:
        render_training_tab()

    with predict_tab:
        render_prediction_tab(database)

    with history_tab:
        render_history_tab(database)


if __name__ == "__main__":
    main()
