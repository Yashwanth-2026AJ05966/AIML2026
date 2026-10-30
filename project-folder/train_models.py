"""Train and compare five smartphone price-category classifiers."""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml_utils import (
    BASE_DIR, CATEGORY_ORDER, EXCLUDED_COLUMNS, METRIC_NAMES, MODEL_FILES,
    RawFeatureNormalizer, build_preprocessor, calculate_metrics,
    identify_features, plot_confusion_matrix,
)


def load_data(path):
    if not path.is_file():
        raise FileNotFoundError(f"Place smartphones_cleaned_v6.csv in {BASE_DIR}")
    data = pd.read_csv(path)
    if data.empty or "price" not in data.columns:
        raise ValueError("The source CSV must contain smartphone records and a price column.")
    print(f"Dataset: {data.shape[0]} rows, {data.shape[1]} columns")
    data.info()
    print("\nMissing values per column:\n", data.isna().sum().to_string())
    print(f"\nDuplicate records: {data.duplicated().sum()} (reported; not silently removed)")
    prices = pd.to_numeric(data["price"], errors="raise")
    if prices.isna().any() or not np.isfinite(prices).all() or (prices < 0).any():
        raise ValueError("Every training record must have a finite, non-negative price.")
    data["price"] = prices
    print("\nOriginal price statistics:\n", prices.describe().to_string())
    return data


def create_target(data):
    data = data.copy()
    # Left-closed intervals put exactly 15000, 30000 and 50000 in the next tier.
    data["price_category"] = pd.cut(
        data["price"], bins=[-np.inf, 15000, 30000, 50000, np.inf],
        labels=CATEGORY_ORDER, right=False,
    ).astype(str)
    print("\nPrice-category distribution:\n",
          data["price_category"].value_counts().reindex(CATEGORY_ORDER, fill_value=0).to_string())
    return data


def prepare_features(data):
    # Target leakage means learning information that reveals the answer. Price
    # defines the target directly; the model name is a high-cardinality identifier.
    features = data.drop(columns=EXCLUDED_COLUMNS, errors="ignore")
    if features.empty:
        raise ValueError("No specification columns remain after excluding target/identifiers.")
    assert not set(EXCLUDED_COLUMNS).intersection(features.columns)
    return features, data["price_category"]


def save_models(pipeline, filename):
    joblib.dump(pipeline, BASE_DIR / "model" / filename)


def train_models():
    data = create_target(load_data(BASE_DIR / "smartphones_cleaned_v6.csv"))
    features, target = prepare_features(data)
    counts = target.value_counts().reindex(CATEGORY_ORDER, fill_value=0)
    if (counts < 5).any():
        raise ValueError("At least five examples per category are required for this stratified split.")
    # Stratification preserves minority categories in the reproducible 80/20 split.
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=42, stratify=target,
    )
    # Export RAW held-out rows before any preprocessing; these rows never fit a model.
    test_data = x_test.copy()
    test_data["price_category"] = y_test
    test_data.to_csv(BASE_DIR / "test_data.csv", index=False)
    print(f"\nTraining rows: {len(x_train)}; held-out test rows: {len(x_test)}")
    print("Training feature columns:", list(x_train.columns))
    numeric_columns, categorical_columns, boolean_columns = identify_features(x_train)
    print("Numeric features:", numeric_columns)
    print("Categorical features:", categorical_columns)
    print("Boolean features:", boolean_columns)
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(probability=True, random_state=42),
    }
    (BASE_DIR / "model").mkdir(exist_ok=True)
    (BASE_DIR / "confusion_matrices").mkdir(exist_ok=True)
    results = []
    for name, classifier in classifiers.items():
        print(f"\nTraining {name}...", flush=True)
        pipeline = Pipeline([
            ("normalize", RawFeatureNormalizer(numeric_columns, categorical_columns, boolean_columns)),
            ("preprocessor", build_preprocessor(
                numeric_columns, categorical_columns, boolean_columns,
                scale=name != "Decision Tree",
            )),
            ("classifier", classifier),
        ])
        # All learned imputation, scaling, and category vocabularies see training rows only.
        pipeline.fit(x_train, y_train)
        predicted = pipeline.predict(x_test)
        probabilities = pipeline.predict_proba(x_test)
        metrics, auc_message = calculate_metrics(y_test, predicted, probabilities, pipeline.classes_)
        if auc_message:
            print(auc_message)
        results.append({"Model": name, **metrics})
        save_models(pipeline, MODEL_FILES[name])
        figure = plot_confusion_matrix(y_test, predicted, name)
        figure.savefig(BASE_DIR / "confusion_matrices" / (MODEL_FILES[name].replace(".pkl", ".png")), dpi=150)
        plt.close(figure)
    comparison = pd.DataFrame(results, columns=["Model", *METRIC_NAMES])
    comparison.to_csv(BASE_DIR / "model_comparison.csv", index=False)
    print("\nModel Comparison (held-out test data):")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    best = comparison.sort_values(["F1 Score", "Accuracy"], ascending=False, kind="stable").iloc[0]
    print(f"\nBest model by weighted F1 Score (accuracy breaks ties): {best['Model']}")
    print(f"Weighted F1 Score: {best['F1 Score']:.4f}")
    print("This is a held-out comparison, not a guarantee of future performance.")
    return comparison


if __name__ == "__main__":
    train_models()