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


def evaluate_saved_model(data, model_name):
    path = BASE_DIR / "model" / MODEL_FILES[model_name]
    if not path.is_file():
        raise FileNotFoundError(f"Saved model not found: model/{path.name}. Run python train_models.py first.")
    pipeline = load_saved_model(str(path), path.stat().st_mtime_ns)
    required_columns = list(pipeline.feature_names_in_)
    missing = sorted(set(required_columns) - set(data.columns))
    if missing:
        raise ValueError("Required smartphone features are missing: " + ", ".join(missing))
    # Use only the saved feature schema, never price, target, model, or extra columns.
    features = data.loc[:, required_columns]
    predicted = pipeline.predict(features)
    probabilities = pipeline.predict_proba(features)
    metrics, auc_message = calculate_metrics(
        data["price_category"], predicted, probabilities, pipeline.classes_,
    )
    return pipeline, predicted, probabilities, metrics, auc_message


def show_classifier():
    selected = st.session_state.get("classifier")
    if selected in MODEL_FILES:
        st.session_state["active_classifier"] = selected
        st.session_state["view"] = "classifier"
        st.session_state["classifier_selection_error"] = False
    else:
        st.session_state["classifier_selection_error"] = True


def show_model_comparison():
    st.session_state["view"] = "comparison"
    st.session_state["classifier"] = None
    st.session_state["classifier_selection_error"] = False


def main():
    st.set_page_config(page_title="Smartphone Price Category Classification", layout="wide")
    st.title("Smartphone Price Category Classification")
    st.write("Predict and evaluate smartphone price categories using pre-trained Machine Learning models.")
    st.session_state.setdefault("view", "classifier")
    st.session_state.setdefault("active_classifier", None)
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload test_data.csv", type=["csv"])
        st.divider()
        st.subheader("Classifier Selection")
        with st.form("classifier_selection", border=False):
            st.radio("Classifier", list(MODEL_FILES), index=None, key="classifier")
            st.form_submit_button("Generate", on_click=show_classifier, key="generate_classifier",
                                   icon=":material/play_arrow:", use_container_width=True)
        if st.session_state.get("classifier_selection_error"):
            st.warning("Select a classifier before clicking Generate.")
        st.divider()
        st.button("All Models' Comparison Table", on_click=show_model_comparison,
                    key="show_model_comparison", icon=":material/compare_arrows:", use_container_width=True,
                    help="Compare all five pre-trained models on your uploaded test CSV using "
                        "Accuracy, AUC, Precision, Recall, F1 Score, and MCC. "
                        "View the results in a separate table and download them as a CSV.")
    selected_name = st.session_state["active_classifier"]
    comparison_view = st.session_state["view"] == "comparison"
    if uploaded_file is None:
        return
    try:
        data = pd.read_csv(uploaded_file)
    except (ValueError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
        st.error("The uploaded file could not be read as a CSV. Please upload test_data.csv.")
        return
    if not comparison_view:
        if selected_name is not None:
            st.subheader(f"Selected classifier: {selected_name}")
        st.subheader("Uploaded Data")
        st.caption(f"{len(data):,} smartphone records (CSV header excluded)")
        st.dataframe(data, hide_index=True, height=400)
    if data.empty:
        st.error("The uploaded CSV contains no smartphone records.")
        return
    if "price_category" not in data.columns:
        st.error("Missing target column: price_category. Upload the generated test_data.csv.")
        return
    if data["price_category"].isna().any() or not data["price_category"].isin(CATEGORY_ORDER).all():
        st.error("price_category must contain only Budget, Mid-Range, Premium, or Flagship, without missing labels.")
        return
    if not comparison_view and selected_name is None:
        return
    evaluations = {}
    failures = {}
    comparison_rows = []
    auc_messages = set()
    # Compare pre-trained pipelines on this upload, not the training comparison CSV.
    model_names = MODEL_FILES if comparison_view else [selected_name]
    for model_name in model_names:
        try:
            evaluation = evaluate_saved_model(data, model_name)
            evaluations[model_name] = evaluation
            comparison_rows.append({"Model": model_name, **evaluation[3]})
            if evaluation[4]:
                auc_messages.add(evaluation[4])
        except Exception as error:
            failures[model_name] = str(error)
            comparison_rows.append({"Model": model_name, **dict.fromkeys(METRIC_NAMES)})
    if comparison_view:
        st.subheader("Model Comparison on Uploaded Data")
        model_comparison = pd.DataFrame(comparison_rows, columns=["Model", *METRIC_NAMES])
        st.dataframe(model_comparison, hide_index=True, height=240, column_config={
            metric: st.column_config.NumberColumn(format="%.4f") for metric in METRIC_NAMES
        })
        st.download_button("Download Model Comparison", model_comparison.to_csv(index=False),
                           file_name="uploaded_model_comparison.csv", mime="text/csv")
    for message in sorted(auc_messages):
        st.warning(message)
    for model_name, message in failures.items():
        if comparison_view:
            st.warning(f"{model_name} could not be evaluated: {message}")
    if comparison_view:
        return
    if selected_name in failures:
        st.error(f"{selected_name} could not be evaluated: {failures[selected_name]}")
        return
    pipeline, predicted, probabilities, metrics, _ = evaluations[selected_name]
    actual = data["price_category"]
    st.subheader(f"Evaluation Metrics: {selected_name}")
    for column, metric_name in zip(st.columns(6), METRIC_NAMES):
        value = metrics[metric_name]
        column.metric(metric_name, "N/A" if pd.isna(value) else f"{value:.4f}")
    figure = plot_confusion_matrix(actual, predicted, selected_name)
    st.pyplot(figure)
    plt.close(figure)
    st.subheader("Actual and Predicted Categories")
    comparison = pd.DataFrame({
        "CSV Row": range(2, len(data) + 2),
        "Actual Price Category": actual.to_numpy(),
        "Predicted Price Category": predicted,
    })
    for source, label in [("model", "Phone Model"), ("brand_name", "Brand"),
                          ("ram_capacity", "RAM (GB)"), ("internal_memory", "Storage (GB)")]:
        if source in data.columns:
            comparison.insert(len(comparison.columns) - 2, label, data[source].to_numpy())
    comparison["Result"] = ["Correct" if expected == prediction else "Incorrect"
                            for expected, prediction in zip(actual, predicted)]
    class_indices = {category: index for index, category in enumerate(pipeline.classes_)}
    comparison["Predicted Category Probability (%)"] = [
        100 * probabilities[index, class_indices[category]]
        for index, category in enumerate(predicted)
    ]
    correct_count = int((comparison["Result"] == "Correct").sum())
    st.caption(f"{len(comparison):,} records | {correct_count:,} correct | "
               f"{len(comparison) - correct_count:,} incorrect")
    result_filter = st.radio("Prediction Results", ["All rows", "Incorrect only", "Correct only"],
                             horizontal=True, key="prediction_results")
    displayed = comparison
    if result_filter != "All rows":
        displayed = comparison.loc[comparison["Result"] == result_filter.split()[0]]
    st.dataframe(displayed, hide_index=True, height=400, column_config={
        "CSV Row": st.column_config.NumberColumn(format="%d"),
        "Predicted Category Probability (%)": st.column_config.NumberColumn(format="%.1f"),
    })
    st.download_button("Download All Predictions", comparison.to_csv(index=False),
                       file_name="predictions.csv", mime="text/csv")


if __name__ == "__main__":
    main()