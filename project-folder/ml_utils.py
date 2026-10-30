"""Shared, serializable preprocessing and evaluation for training and Streamlit."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
CATEGORY_ORDER = ["Budget", "Mid-Range", "Premium", "Flagship"]
EXCLUDED_COLUMNS = ["price", "price_category", "model"]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "svm.pkl",
}
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
BOOLEAN_VALUES = {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0,
                  "1.0": 1.0, "0.0": 0.0}


def identify_features(features):
    """Infer feature groups from training rows, not from held-out data."""
    boolean_columns = []
    for column in features.columns:
        values = features[column].dropna().astype(str).str.strip().str.lower()
        is_flag = column.startswith("has_") or column.endswith("_available")
        if pd.api.types.is_bool_dtype(features[column]) or (
            is_flag and set(values).issubset(BOOLEAN_VALUES)
        ):
            boolean_columns.append(column)
    numeric_columns = [
        column for column in features.select_dtypes(include=np.number).columns
        if column not in boolean_columns
    ]
    categorical_columns = [
        column for column in features.columns
        if column not in numeric_columns + boolean_columns
    ]
    return numeric_columns, categorical_columns, boolean_columns


class RawFeatureNormalizer(TransformerMixin, BaseEstimator):
    """Keep raw CSV boolean and missing-value handling inside saved pipelines."""

    def __init__(self, numeric_columns, categorical_columns, boolean_columns):
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.boolean_columns = boolean_columns

    def fit(self, features, target=None):
        self.feature_names_in_ = np.asarray(features.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, features):
        missing = set(self.feature_names_in_) - set(features.columns)
        if missing:
            raise ValueError("Missing input columns: " + ", ".join(sorted(missing)))
        result = features.loc[:, self.feature_names_in_].copy()
        for column in self.boolean_columns:
            values = result[column].astype("string").str.strip().str.lower()
            invalid = values.notna() & ~values.isin(BOOLEAN_VALUES)
            if invalid.any():
                raise ValueError(f"{column} must contain TRUE/FALSE or 0/1 values.")
            # Preserve missing flags for training-fitted most-frequent imputation.
            result[column] = values.map(BOOLEAN_VALUES).astype(float)
        for column in self.numeric_columns:
            result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
            if np.isinf(result[column]).any():
                raise ValueError(f"{column} contains an infinite numeric value.")
        for column in self.categorical_columns:
            result[column] = result[column].astype(object).where(result[column].notna(), np.nan)
        return result


def build_preprocessor(numeric_columns, categorical_columns, boolean_columns, scale=True):
    """Create a fresh, unfitted preprocessor for each classifier."""
    # Medians resist outliers and retain incomplete rows. Scaling prevents large
    # units (storage, battery) from dominating distance/margin-based classifiers.
    numeric_steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    # One-hot encoding avoids imposing an artificial order on brands and OS names.
    # Dense output is small for this dataset and is directly usable by GaussianNB.
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_columns),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical_columns),
            ("boolean", SimpleImputer(strategy="most_frequent", keep_empty_features=True),
             boolean_columns),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def calculate_metrics(actual, predicted, probabilities, classes):
    """Return the six required metrics and an explicit AUC availability message."""
    # Weighted metrics account for the unequal number of phones in each category.
    metrics = {
        "Accuracy": accuracy_score(actual, predicted),
        "AUC": np.nan,
        "Precision": precision_score(actual, predicted, average="weighted", zero_division=0),
        "Recall": recall_score(actual, predicted, average="weighted", zero_division=0),
        "F1 Score": f1_score(actual, predicted, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(actual, predicted),
    }
    classes = np.asarray(classes)
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(actual), len(classes)):
        raise ValueError("Probability columns do not match the saved classifier classes.")
    if set(actual) != set(classes):
        return metrics, "AUC is unavailable: evaluation data must include every trained category."
    # sklearn expects alphabetically sorted labels. Reorder probability columns
    # using classes_, independently of the business order used for plots.
    order = np.argsort(classes)
    try:
        metrics["AUC"] = roc_auc_score(
            actual, probabilities[:, order], labels=classes[order],
            multi_class="ovr", average="weighted",
        )
    except ValueError as error:
        return metrics, f"AUC is unavailable for this evaluation: {error}"
    return metrics, None


def plot_confusion_matrix(actual, predicted, model_name):
    matrix = confusion_matrix(actual, predicted, labels=CATEGORY_ORDER)
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CATEGORY_ORDER, yticklabels=CATEGORY_ORDER, ax=axis)
    axis.set(title=f"{model_name} - Confusion Matrix",
             xlabel="Predicted Price Category", ylabel="Actual Price Category")
    axis.tick_params(axis="y", labelrotation=0)
    figure.tight_layout()
    return figure