"""Focused assignment checks. Run after training: python -m unittest -v test_project."""

import ast
from contextlib import redirect_stdout
from io import BytesIO, StringIO
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from streamlit.testing.v1 import AppTest

from ml_utils import (
    BASE_DIR, CATEGORY_ORDER, EXCLUDED_COLUMNS, METRIC_NAMES,
    MODEL_FILES, calculate_metrics,
)
from train_models import create_target, prepare_features


class AssignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with redirect_stdout(StringIO()):
            data = create_target(pd.read_csv(BASE_DIR / "smartphones_cleaned_v6.csv"))
        features, target = prepare_features(data)
        cls.x_train, cls.x_test, cls.y_train, cls.y_test = train_test_split(
            features, target, test_size=0.20, random_state=42, stratify=target,
        )
        cls.export = pd.read_csv(BASE_DIR / "test_data.csv")
        cls.models = {name: joblib.load(BASE_DIR / "model" / filename)
                      for name, filename in MODEL_FILES.items()}

    def test_exact_price_boundaries(self):
        prices = pd.DataFrame({"price": [0, 14999, 15000, 29999, 30000, 49999, 50000, 650000]})
        with redirect_stdout(StringIO()):
            actual = create_target(prices)["price_category"].tolist()
        self.assertEqual(actual, ["Budget", "Budget", "Mid-Range", "Mid-Range",
                                  "Premium", "Premium", "Flagship", "Flagship"])

    def test_raw_test_split_and_no_leakage(self):
        self.assertEqual(len(self.export), 196)
        self.assertFalse(set(EXCLUDED_COLUMNS).intersection(self.x_train.columns))
        self.assertFalse({"price", "model"}.intersection(self.export.columns))
        expected = self.x_test.copy()
        expected["price_category"] = self.y_test
        pd.testing.assert_frame_equal(self.export, expected.reset_index(drop=True), check_dtype=False)
        self.assertTrue(set(self.x_train.index).isdisjoint(self.x_test.index))

    def test_five_saved_models_and_comparison(self):
        comparison = pd.read_csv(BASE_DIR / "model_comparison.csv")
        self.assertEqual(list(comparison.columns), ["Model", *METRIC_NAMES])
        self.assertEqual(set(comparison["Model"]), set(MODEL_FILES))
        self.assertEqual(len(comparison), 5)
        self.assertTrue(np.isfinite(comparison[METRIC_NAMES].to_numpy()).all())
        features = self.export.drop(columns="price_category")
        for name, pipeline in self.models.items():
            with self.subTest(model=name):
                self.assertEqual(list(pipeline.feature_names_in_), list(self.x_train.columns))
                self.assertEqual(set(pipeline.classes_), set(CATEGORY_ORDER))
                predicted = pipeline.predict(features)
                probabilities = pipeline.predict_proba(features)
                np.testing.assert_allclose(probabilities.sum(axis=1), 1)
                metrics, message = calculate_metrics(self.export["price_category"], predicted,
                                                      probabilities, pipeline.classes_)
                self.assertIsNone(message)
                np.testing.assert_allclose(comparison.set_index("Model").loc[name, METRIC_NAMES].astype(float),
                                           [metrics[metric] for metric in METRIC_NAMES])
                image = BASE_DIR / "confusion_matrices" / MODEL_FILES[name].replace(".pkl", ".png")
                self.assertEqual(image.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_preprocessing_fitted_only_on_training(self):
        for name, pipeline in self.models.items():
            with self.subTest(model=name):
                normalizer = pipeline.named_steps["normalize"]
                normalized = normalizer.transform(self.x_train)
                preprocessor = pipeline.named_steps["preprocessor"]
                numeric = normalizer.numeric_columns
                numeric_pipeline = preprocessor.named_transformers_["numeric"]
                np.testing.assert_allclose(numeric_pipeline.named_steps["imputer"].statistics_,
                                           normalized[numeric].median().to_numpy())
                if "scaler" in numeric_pipeline.named_steps:
                    imputed = numeric_pipeline.named_steps["imputer"].transform(normalized[numeric])
                    np.testing.assert_allclose(numeric_pipeline.named_steps["scaler"].mean_, imputed.mean(axis=0))
                category_pipeline = preprocessor.named_transformers_["categorical"]
                imputed_categories = category_pipeline.named_steps["imputer"].transform(
                    normalized[normalizer.categorical_columns])
                for index, categories in enumerate(category_pipeline.named_steps["encoder"].categories_):
                    self.assertEqual(set(categories), set(imputed_categories[:, index]))
                transformed = preprocessor.transform(normalizer.transform(self.x_test))
                self.assertIsInstance(transformed, np.ndarray)
                self.assertTrue(np.isfinite(transformed).all())

    def test_unknown_categories_and_missing_values(self):
        for name, pipeline in self.models.items():
            with self.subTest(model=name):
                sample = self.x_test.head(4).astype(object)
                normalizer = pipeline.named_steps["normalize"]
                sample.loc[sample.index[0], normalizer.numeric_columns] = np.nan
                sample.loc[sample.index[1], normalizer.categorical_columns] = np.nan
                sample.loc[sample.index[2], normalizer.boolean_columns] = np.nan
                sample.loc[sample.index[3], normalizer.categorical_columns] = "unseen-category"
                for column in normalizer.boolean_columns:
                    sample.loc[sample.index[0], column] = "TRUE"
                    sample.loc[sample.index[1], column] = "0"
                self.assertEqual(len(pipeline.predict(sample)), 4)
                self.assertTrue(np.isfinite(pipeline.predict_proba(sample)).all())

    def test_probability_column_order_and_auc_edge_case(self):
        pipeline = next(iter(self.models.values()))
        predicted = pipeline.predict(self.x_test)
        probability = pipeline.predict_proba(self.x_test)
        expected, _ = calculate_metrics(self.y_test, predicted, probability, pipeline.classes_)
        actual, message = calculate_metrics(self.y_test, predicted, probability[:, ::-1], pipeline.classes_[::-1])
        self.assertIsNone(message)
        self.assertAlmostEqual(expected["AUC"], actual["AUC"])
        mask = self.y_test == "Budget"
        subset, message = calculate_metrics(self.y_test[mask], predicted[mask], probability[mask], pipeline.classes_)
        self.assertTrue(np.isnan(subset["AUC"]))
        self.assertIn("every trained category", message)

    def test_app_never_trains(self):
        tree = ast.parse((BASE_DIR / "app.py").read_text(encoding="utf-8"))
        training_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                          and isinstance(node.func, ast.Attribute)
                          and node.func.attr in {"fit", "fit_transform", "partial_fit"}]
        self.assertEqual(training_calls, [])
        self.assertIn("joblib.load", (BASE_DIR / "app.py").read_text(encoding="utf-8"))

    def test_streamlit_each_model_with_uploaded_test_csv(self):
        for name in MODEL_FILES:
            with self.subTest(model=name):
                app = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30).run()
                self.assertFalse(app.exception)
                app.sidebar.selectbox[0].select(name)
                with patch("streamlit.file_uploader", return_value=BytesIO(self.export.to_csv(index=False).encode())):
                    app.run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertEqual([metric.label for metric in app.metric], METRIC_NAMES)

    def test_streamlit_invalid_uploads(self):
        cases = {
            "missing target": (self.export.drop(columns="price_category").to_csv(index=False), "Missing target"),
            "missing feature": (self.export.drop(columns="brand_name").to_csv(index=False), "features are missing"),
            "empty file": ("", "could not be read"),
            "no rows": (",".join(self.export.columns) + "\n", "no smartphone records"),
            "invalid target": (self.export.assign(price_category="Invalid").to_csv(index=False), "must contain only"),
        }
        for name, (text, message) in cases.items():
            with self.subTest(case=name):
                with patch("streamlit.file_uploader", return_value=BytesIO(text.encode())):
                    app = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30).run()
                self.assertFalse(app.exception)
                self.assertTrue(app.error)
                self.assertIn(message, app.error[0].value)


if __name__ == "__main__":
    unittest.main()