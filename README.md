# ChurnGuard AI

ChurnGuard AI is a customer churn prediction system built with Python, Streamlit, Scikit-learn, Pandas, and SQLite. It trains classification models on structured CSV datasets, evaluates validation performance, runs real-time inference on new customer records, and stores prediction history in a SQL database for later analysis.

## What this project includes

- Data validation and preprocessing for mixed numeric and categorical customer data
- Feature engineering through imputation, one-hot encoding, and scaling
- Model training with hyperparameter tuning across Logistic Regression and Random Forest
- Validation metrics including accuracy, ROC-AUC, and a classification report
- Interactive Streamlit UI for training, inference, and prediction history review
- SQLite-backed persistence for structured prediction records

## Project structure

```text
ChurnGuard-AI/
├── app.py
├── churnguard_ai/
│   ├── database.py
│   ├── modeling.py
│   └── ui_helpers.py
├── data/
│   ├── sample_customer_churn.csv
│   └── sample_inference_customers.csv
├── models/
├── pyproject.toml
└── README.md
```

## Dataset requirements

The training CSV should include:

- `customer_id`
- `churn`
- Any number of feature columns such as tenure, contract type, monthly charges, support tickets, and payment method

The `churn` column must use binary values such as `0/1`, `yes/no`, or `true/false`.

## Local setup with uv

```bash
uv sync
```

## Run the Streamlit app

```bash
uv run streamlit run app.py
```

## How to use

1. Open the Streamlit app.
2. In the training tab, upload `data/sample_customer_churn.csv` or your own training dataset.
3. Train the model and review the validation metrics.
4. In the predictions tab, upload `data/sample_inference_customers.csv` or another inference dataset.
5. Generate predictions and review the saved SQL history in the history tab.

## Notes

- The trained model is saved to `models/churn_model.joblib`
- The SQLite database is saved to `data/churnguard_history.db`
- The current implementation uses SQLite from Python's standard library for the SQL storage layer
- `uv sync` will create and manage the local `.venv` for this project
