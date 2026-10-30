"""Evaluate raw smartphone test CSVs using existing trained pipelines only."""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from ml_utils import (
    BASE_DIR, CATEGORY_ORDER, METRIC_NAMES, MODEL_FILES,
    calculate_metrics, plot_confusion_matrix,
)


@st.cache_resource
def load_saved_model(path_string, modified_time):
    # Only load trusted local artifacts; uploaded CSVs are never unpickled.
    return joblib.load(path_string)


def main():
    st.set_page_config(page_title="Smartphone Price Category Classification", layout="wide")
    st.title("Smartphone Price Category Classification")
    st.write("Predict and evaluate smartphone price categories using pre-trained Machine Learning models.")
    selected_name = st.sidebar.selectbox("Classifier", list(MODEL_FILES))
    st.subheader(f"Selected classifier: {selected_name}")
    uploaded_file = st.file_uploader("Upload test_data.csv", type=["csv"])
    if uploaded_file is None:
        return
    try:
        data = pd.read_csv(uploaded_file)
    except (ValueError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
        st.error("The uploaded file could not be read as a CSV. Please upload test_data.csv.")
        return
    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head(10), hide_index=True)
    if data.empty:
        st.error("The uploaded CSV contains no smartphone records.")
        return
    if "price_category" not in data.columns:
        st.error("Missing target column: price_category. Upload the generated test_data.csv.")
        return
    if data["price_category"].isna().any() or not data["price_category"].isin(CATEGORY_ORDER).all():
        st.error("price_category must contain only Budget, Mid-Range, Premium, or Flagship, without missing labels.")
        return
    path = BASE_DIR / "model" / MODEL_FILES[selected_name]
    if not path.is_file():
        st.error(f"Saved model not found: model/{path.name}. Run python train_models.py first.")
        return
    try:
        # Training is deliberately separate: the app only loads a fitted pipeline.
        pipeline = load_saved_model(str(path), path.stat().st_mtime_ns)
        required_columns = list(pipeline.feature_names_in_)
        missing = sorted(set(required_columns) - set(data.columns))
        if missing:
            st.error("Required smartphone features are missing: " + ", ".join(missing))
            return
        # Select the saved feature schema, never price, target, model, or extra columns.
        features = data.loc[:, required_columns]
        actual = data["price_category"]
        predicted = pipeline.predict(features)
        probabilities = pipeline.predict_proba(features)
        metrics, auc_message = calculate_metrics(actual, predicted, probabilities, pipeline.classes_)
    except Exception as error:
        st.error(f"Could not load or evaluate the saved model. Check CSV values and installed package versions. Details: {error}")
        return
    st.subheader("Evaluation Metrics")
    for column, metric_name in zip(st.columns(6), METRIC_NAMES):
        value = metrics[metric_name]
        column.metric(metric_name, "N/A" if pd.isna(value) else f"{value:.4f}")
    if auc_message:
        st.warning(auc_message)
    figure = plot_confusion_matrix(actual, predicted, selected_name)
    st.pyplot(figure)
    plt.close(figure)
    st.subheader("Actual and Predicted Categories")
    comparison = pd.DataFrame({
        "Actual Price Category": actual.to_numpy(),
        "Predicted Price Category": predicted,
    })
    st.dataframe(comparison.head(30), hide_index=True)
    st.download_button("Download Predictions", comparison.to_csv(index=False),
                       file_name="predictions.csv", mime="text/csv")


if __name__ == "__main__":
    main()