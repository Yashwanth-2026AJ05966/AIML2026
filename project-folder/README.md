# Smartphone Price Category Classification

## 1. Problem Statement

**Smartphone Price Category Classification Using Machine Learning** compares
five multiclass classifiers to determine how effectively smartphone specifications
can predict a device's price segment. The objective is to assign each smartphone
to one of four categories: Budget, Mid-Range, Premium, or Flagship. This is a
classification task, not an attempt to predict an exact selling price.

The solution uses specifications such as brand, processor, RAM, internal storage,
battery capacity, display characteristics, camera capabilities, and connectivity.
Price is used to construct the correct category labels but is excluded from the
prediction features. The workflow handles missing values and mixed feature types,
trains Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Gaussian
Naive Bayes, and Support Vector Machine (SVM), and compares their performance
using six evaluation metrics on a common held-out test set.

The completed project provides five reusable trained pipelines and a Streamlit
application for uploading labelled test data, viewing predictions, inspecting
confusion matrices, and comparing models. On the 196 held-out smartphones, SVM
achieved the best overall results among the five models, with 81.12% accuracy
and a weighted F1 score of 0.8089. These results demonstrate the feasibility of
specification-based price-category classification on this dataset, while also
showing that performance depends on the chosen learning algorithm.

## 2. Dataset Description

The dataset `smartphones_cleaned_v6.csv` was downloaded from the
[Smartphones Dataset on Kaggle](https://www.kaggle.com/datasets/informrohit1/smartphones-dataset?resource=download),
as specified in the assignment handout PDF. It contains 980 records and 26 columns.
It has missing specification values and no exact duplicate records. The training
program prints dimensions, data types, missing counts, duplicates, and original
price statistics. No incomplete specification rows are discarded.

The downloaded CSV is included unchanged in the project. No synthetic records
or additional external datasets were used.

## 3. Target Variable

`price_category` is created from `price` before splitting. `price` is used ONLY
to create the target during training and is removed from model input features.
Records with missing, negative, nonnumeric, or infinite prices cause an explicit
error because they cannot be reliably labelled.

## 4. Price Categories

| Category | Exact price rule | Records |
| --- | --- | ---: |
| Budget | price < 15000 | 336 |
| Mid-Range | 15000 <= price < 30000 | 350 |
| Premium | 30000 <= price < 50000 | 133 |
| Flagship | price >= 50000 | 161 |

Price values are interpreted in the supplied dataset's units (assumed INR).

## 5. Dataset Features

The 24 input columns, verified against the supplied CSV, are:

- Categorical: `brand_name`, `processor_brand`, `os`.
- Boolean/binary flags: `has_5g`, `has_nfc`, `has_ir_blaster`,
  `fast_charging_available`, `extended_memory_available`.
- Numeric: `rating`, `num_cores`, `processor_speed`, `battery_capacity`,
  `fast_charging`, `ram_capacity`, `internal_memory`, `screen_size`,
  `refresh_rate`, `num_rear_cameras`, `num_front_cameras`,
  `primary_camera_rear`, `primary_camera_front`, `extended_upto`,
  `resolution_width`, `resolution_height`.

`model` is excluded as a high-cardinality identifier. `price` and
`price_category` are never input features. Feature groups are inferred using
training rows and printed, rather than blindly assuming column types.

## 6. Data Preprocessing

The reproducible split uses `test_size=0.20`, `random_state=42`, and `stratify=y`.
Stratification preserves representation of the less frequent classes. There are
784 training rows and 196 held-out rows.

Each saved pipeline includes raw-value normalization, a `ColumnTransformer`,
and its classifier. All fitted preprocessing sees training rows only:

- Numeric missing values: median imputation, robust to large price/specification
  outliers. `StandardScaler` is used except for the scale-insensitive tree.
- Categorical missing values: most-frequent imputation, followed by one-hot
  encoding so category names do not acquire a false ordinal relationship.
  Unseen categories are accepted with `handle_unknown="ignore"`.
- Boolean flags: TRUE/FALSE or 0/1 become 1/0 inside the pipeline. Missing flags
  use training-set most-frequent imputation; invalid flag strings are rejected.
- Dense one-hot output ensures Gaussian Naive Bayes receives a dense numeric
  array. This is practical for this small dataset.

The serializable normalizer is defined in `ml_utils.py`, not in a script's
`__main__` namespace, so models load correctly in Streamlit and other processes.
Keep that module with the model files.

## 7. Machine Learning Models

Exactly these five scikit-learn models are trained without hyperparameter search:

1. Logistic Regression (`max_iter=2000`, `random_state=42`).
2. Decision Tree (`random_state=42`).
3. K-Nearest Neighbors (default five neighbors).
4. Gaussian Naive Bayes.
5. Support Vector Machine (`probability=True`, `random_state=42`).

Other estimator parameters use scikit-learn defaults. `random_state=42` is used
where the estimator supports it. Every classifier has its own fitted pipeline.

## 8. Evaluation Metrics

Exactly six metrics are reported: Accuracy, AUC, Precision, Recall, F1 Score,
and Matthews Correlation Coefficient (MCC).

Precision, recall, and F1 use `average="weighted"` to account for unequal class
support, with `zero_division=0`. AUC uses `predict_proba`, weighted one-vs-rest
multiclass ROC AUC, and probability columns aligned to the classifier's
`classes_`. It is never calculated from predicted labels. If an upload lacks
any trained category, AUC is explicitly marked unavailable; other metrics and
the confusion matrix are still displayed.

Confusion matrices always use Budget, Mid-Range, Premium, Flagship in that order.

## 9. Project Structure

The existing assignment directory `project-folder` is used instead of creating
a second unrelated assignment folder.

```text
project-folder/
|-- app.py
|-- train_models.py
|-- ml_utils.py
|-- test_project.py
|-- smartphones_cleaned_v6.csv
|-- test_data.csv
|-- model_comparison.csv
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- model/
|   |-- logistic_regression.pkl
|   |-- decision_tree.pkl
|   |-- knn.pkl
|   |-- naive_bayes.pkl
|   `-- svm.pkl
`-- confusion_matrices/
    |-- logistic_regression.png
    |-- decision_tree.png
    |-- knn.png
    |-- naive_bayes.png
    `-- svm.png
```

The local `.venv` is not intended for version control. The required model and
evaluation artifacts are not ignored. No PDF report is generated.

## 10. Training Instructions

Run from this project directory after installing dependencies:

```bash
python train_models.py
```

The program reads ONLY `smartphones_cleaned_v6.csv`, creates the stratified split,
saves raw held-out rows, trains all five pipelines on the training split, and
evaluates them on the test split. It saves all models, five confusion matrices,
and `model_comparison.csv`, then prints the comparison and best weighted-F1
model (accuracy breaks ties). Scores and the winning model are never hardcoded.
Training overwrites generated artifacts reproducibly.

## 11. Streamlit Application

```bash
python -m streamlit run app.py
```

Upload `test_data.csv` using the uploader at the top of the sidebar, then select
a classifier using the radio buttons and click **Generate**. Horizontal dividers
separate the upload, classifier selection, and all-model comparison sections.
Choosing a radio option alone does not refresh the right pane; Generate applies
the selection. Until the first Generate, only the uploaded data preview is shown.
Generated results include six metrics, a confusion matrix, and
actual-versus-predicted categories. A pending selection leaves the previously
generated results unchanged.

Click **All Models' Comparison Table** in the sidebar to switch to a separate comparison-only
view, which evaluates all five saved local pipelines on the uploaded records.
The comparison table shows
Model, Accuracy, AUC, Precision, Recall, F1 Score, and MCC, with a CSV download.
These scores are calculated from the current upload, not read from the training
comparison file. Missing models are reported explicitly and their scores remain
unavailable; other models can still be compared.

To return from comparison, select a classifier and click **Generate**.
Comparison mode clears the previous radio selection, so the same classifier
can be selected again. Generate without a selection shows a sidebar warning
and leaves the current results unchanged.
The uploaded file remains available when switching views.
Both record tables in the classifier view include every record in a
fixed-height scrollable view. The uploaded record count excludes the CSV header.

Prediction results include the CSV row number (the first record is row 2),
brand, RAM, storage, actual and predicted categories, a correct/incorrect result,
and the predicted category's probability. Phone model names are also displayed
when supplied as an optional extra column; they never become model features.
Results can be filtered to incorrect or correct predictions. Summary counts
refer to the entire upload, and the download always includes all predictions,
regardless of the current filter. Probabilities are model estimates, not
guarantees of correctness.

**The app never trains or fits models.** Training is separate to preserve the
held-out evaluation and avoid repeating training on each Streamlit interaction.
Extra uploaded columns are ignored; only the saved feature schema is used.
Missing models, invalid CSVs, invalid values, and missing columns produce errors.
Only trusted local joblib files are loaded; never replace these with untrusted
pickle files. Use the same package versions for training and app execution.

## 12. How to Run Locally

Python 3.11 or newer is recommended; validation was performed with Python 3.13.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_models.py
.\.venv\Scripts\python.exe -m unittest -v test_project
.\.venv\Scripts\python.exe -m py_compile app.py
.\.venv\Scripts\python.exe -m py_compile train_models.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train_models.py
python -m unittest -v test_project
python -m streamlit run app.py
```

Saved paths are based on `pathlib.Path(__file__)`, not the launch directory.
The automated checks verify price boundaries, test-split integrity, training-only
imputation/scaling, raw-CSV model loading, unseen/missing inputs, AUC ordering,
model artifacts, and the no-training rule in the app.

## 13. Example Test Data

`test_data.csv` is the actual untouched 20% held-out split, exported before
imputation, normalization, scaling, or encoding. It contains the 24 raw input
columns plus `price_category`, and excludes `price` and `model`. Missing feature
values remain missing in this file. It is never read for fitting.

Upload the entire file for a four-class AUC calculation. A subset may omit a
class, in which case AUC is unavailable rather than misleadingly recalculated
as a different classification task. Uploading other labelled data is supported,
but evaluation is meaningful only if those records were not used for training.

## 14. Model Comparison

The following results were obtained by evaluating the five saved pipelines on
all 196 held-out records in `test_data.csv`. They match the saved evaluation
results in `model_comparison.csv` and are rounded to four decimal places.
Precision, Recall, and F1 are weighted averages across the four categories;
AUC is weighted multiclass one-vs-rest AUC. Accuracy is shown as a proportion.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7857 | 0.9384 | 0.7870 | 0.7857 | 0.7836 | 0.6966 |
| Decision Tree | 0.7449 | 0.8162 | 0.7475 | 0.7449 | 0.7436 | 0.6392 |
| KNN | 0.8010 | 0.9209 | 0.7982 | 0.8010 | 0.7989 | 0.7188 |
| Naive Bayes | 0.2602 | 0.7693 | 0.4898 | 0.2602 | 0.2544 | 0.1002 |
| Support Vector Machine | 0.8112 | 0.9506 | 0.8103 | 0.8112 | 0.8089 | 0.7325 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | A strong baseline with 78.57% accuracy and weighted F1 of 0.7836. Its AUC of 0.9384 is second only to SVM, indicating strong class-ranking performance even though its final classification accuracy is below KNN and SVM. |
| Decision Tree | Achieved 74.49% accuracy and weighted F1 of 0.7436, below Logistic Regression, KNN, and SVM. Its AUC of 0.8162 also trails those three models. The test results alone do not establish whether the difference is caused by overfitting. |
| KNN | The second-best model by accuracy, weighted F1, and MCC, with 80.10% accuracy and weighted F1 of 0.7989. This suggests that similarity between preprocessed smartphone specifications is useful for category prediction. Its AUC of 0.9209 is below Logistic Regression despite higher classification accuracy. |
| Naive Bayes | The weakest model by accuracy, weighted F1, and MCC, with 26.02% accuracy, weighted F1 of 0.2544, and MCC of 0.1002. Its AUC of 0.7693 indicates some ranking ability, but its final category predictions are poor. Correlated specifications and departures from Gaussian assumptions are possible explanations, not conclusions established by these metrics. |
| Support Vector Machine | The strongest model on this test split, leading all six reported metrics. It achieved 81.12% accuracy, AUC of 0.9506, weighted F1 of 0.8089, and MCC of 0.7325. Its default RBF kernel can represent nonlinear relationships between specifications and price categories. |

**Overall winner for this dataset: Support Vector Machine (SVM).** It has the
highest weighted F1 score, the project's primary comparison criterion, and also
the highest accuracy, AUC, precision, recall, and MCC. KNN is the closest
alternative by accuracy and weighted F1.

This is a simple academic comparison, without extensive tuning or resampling.
The winner is specific to this held-out split; the small difference between
SVM and KNN is not proof of statistically significant superiority. Choosing a
winner on this test split is descriptive; an additional independent dataset
would be needed for an unbiased final assessment of that selection.

## 15. No Data Leakage in the Implemented Pipeline

**There is no target leakage or train-test preprocessing leakage in the
implemented training and evaluation pipeline.** The following safeguards keep
the prediction features and held-out evaluation separate:

- `price` is used only to create category labels with fixed thresholds.
  Both `price` and `price_category`, along with the phone model identifier,
  are removed from input features before the train-test split.
- Training labels are passed separately as `y_train` in
  `pipeline.fit(x_train, y_train)`. This is required for supervised learning
  and does not make the target a prediction feature.
- Training and test rows are disjoint. Feature grouping, imputation, scaling,
  and one-hot encoding are fitted using training rows only.
- The held-out test rows are used only for prediction and evaluation, never
  for fitting preprocessing or classifiers.
- The Streamlit application loads the saved pipelines and predicts from their
  feature columns only. Uploaded target labels are used to calculate metrics
  and compare actual versus predicted categories, not to generate predictions.

These safeguards are checked by the automated tests for split integrity,
feature exclusion, and training-only preprocessing. The model-selection
limitation described in Section 14 is separate from target or preprocessing
leakage.

## 16. Source Code and Application Hosting

**GitHub source code:**

[AIML2026 - Smartphone Price Category Classification](https://github.com/Yashwanth-2026AJ05966/AIML2026/tree/main/project-folder)

The repository contains the application source, training code, dataset, saved
models, evaluation artifacts, dependency requirements, tests, and documentation.

**Live Streamlit application:**

[Smartphone Price Category Classification](https://2026aj05966.streamlit.app/)

Open the hosted application, upload `test_data.csv`, and select a classifier
and click **Generate**, or click **All Models' Comparison Table** to compare
all five models. The application can also be run locally using the instructions
in Section 12.