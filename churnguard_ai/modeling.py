from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


TARGET_COLUMN = "churn"
IDENTIFIER_COLUMN = "customer_id"


@dataclass(slots=True)
class TrainingArtifacts:
    model: Pipeline
    metrics: dict[str, Any]
    feature_columns: list[str]
    best_model_name: str


def validate_dataset(dataframe: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    required_columns = {TARGET_COLUMN, IDENTIFIER_COLUMN}

    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        errors.append(
            "Missing required columns: " + ", ".join(sorted(missing_columns))
        )

    if TARGET_COLUMN in dataframe.columns:
        unique_target_values = set(dataframe[TARGET_COLUMN].dropna().astype(str).str.lower())
        valid_target_values = {"0", "1", "yes", "no", "true", "false"}
        if not unique_target_values:
            errors.append("The target column 'churn' is empty.")
        elif not unique_target_values.issubset(valid_target_values):
            errors.append(
                "The target column 'churn' must contain binary values like 0/1, yes/no, or true/false."
            )

    if len(dataframe) < 10:
        errors.append("The dataset must contain at least 10 rows for training and validation.")

    return errors


def normalize_target(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "1": 1,
        "yes": 1,
        "true": 1,
        "0": 0,
        "no": 0,
        "false": 0,
    }
    return normalized.map(mapping).astype(int)


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def build_model_candidates(preprocessor: ColumnTransformer) -> list[tuple[str, GridSearchCV]]:
    logistic_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )
    random_forest_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    return [
        (
            "Logistic Regression",
            GridSearchCV(
                estimator=logistic_pipeline,
                param_grid={
                    "classifier__C": [0.1, 1.0, 10.0],
                    "classifier__solver": ["liblinear", "lbfgs"],
                },
                scoring="roc_auc",
                cv=3,
                n_jobs=1,
            ),
        ),
        (
            "Random Forest",
            GridSearchCV(
                estimator=random_forest_pipeline,
                param_grid={
                    "classifier__n_estimators": [100, 200],
                    "classifier__max_depth": [None, 6, 12],
                },
                scoring="roc_auc",
                cv=3,
                n_jobs=1,
            ),
        ),
    ]


def train_model(dataframe: pd.DataFrame) -> TrainingArtifacts:
    errors = validate_dataset(dataframe)
    if errors:
        raise ValueError(" ".join(errors))

    prepared = dataframe.copy()
    prepared[TARGET_COLUMN] = normalize_target(prepared[TARGET_COLUMN])

    features = prepared.drop(columns=[TARGET_COLUMN, IDENTIFIER_COLUMN])
    target = prepared[TARGET_COLUMN]
    feature_columns = features.columns.tolist()

    if target.nunique() < 2:
        raise ValueError("The target column must contain both churn and non-churn examples.")

    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    best_score = -1.0
    best_model_name = ""
    best_estimator: Pipeline | None = None
    best_predictions: pd.Series | None = None
    best_probabilities: pd.Series | None = None

    for model_name, grid_search in build_model_candidates(build_preprocessor(features)):
        grid_search.fit(x_train, y_train)
        candidate_estimator = grid_search.best_estimator_
        candidate_predictions = candidate_estimator.predict(x_valid)
        candidate_probabilities = candidate_estimator.predict_proba(x_valid)[:, 1]
        candidate_score = roc_auc_score(y_valid, candidate_probabilities)

        if candidate_score > best_score:
            best_score = candidate_score
            best_model_name = model_name
            best_estimator = candidate_estimator
            best_predictions = candidate_predictions
            best_probabilities = candidate_probabilities

    if best_estimator is None or best_predictions is None or best_probabilities is None:
        raise RuntimeError("Model training did not produce a valid estimator.")

    metrics = {
        "accuracy": round(float(accuracy_score(y_valid, best_predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_valid, best_probabilities)), 4),
        "classification_report": classification_report(
            y_valid, best_predictions, output_dict=True
        ),
    }

    return TrainingArtifacts(
        model=best_estimator,
        metrics=metrics,
        feature_columns=feature_columns,
        best_model_name=best_model_name,
    )


def save_artifacts(artifacts: TrainingArtifacts, output_path: str) -> None:
    joblib.dump(artifacts, output_path)


def load_artifacts(input_path: str) -> TrainingArtifacts:
    return joblib.load(input_path)


def predict_churn(
    artifacts: TrainingArtifacts,
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    missing_columns = set(artifacts.feature_columns).difference(dataframe.columns)
    if missing_columns:
        raise ValueError(
            "Missing required feature columns: " + ", ".join(sorted(missing_columns))
        )

    inference_frame = dataframe[artifacts.feature_columns].copy()
    probabilities = artifacts.model.predict_proba(inference_frame)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    results = dataframe.copy()
    results["churn_prediction"] = predictions
    results["churn_probability"] = probabilities.round(4)
    return results
