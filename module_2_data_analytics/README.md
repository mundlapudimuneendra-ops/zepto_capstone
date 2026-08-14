# 🚢 Zepto Data Analytics Capstone — Module 2

## 📖 Project Overview

This project performs **Exploratory Data Analysis (EDA)** and **Predictive Modeling** on the classic Titanic dataset using Python, Pandas, Seaborn, Matplotlib, Scikit-learn, and imbalanced-learn.

The workflow covers loading the dataset through `seaborn`, persisting it to CSV, full EDA (univariate, bivariate, multivariate), feature engineering, classification with class-imbalance handling, hyperparameter tuning, a regression task on `fare`, and finally saving the trained pipelines for re-use.

---

## 🎯 Objectives

- Load the Titanic dataset once via `sns.load_dataset("titanic")` and persist it to `titanic.csv`.
- Perform complete EDA: missing-value analysis, distribution analysis, correlation, survival analysis, and multivariate visualizations.
- Train three classifiers (Logistic Regression, Decision Tree, Random Forest) on a stratified train/test split.
- Compare class-imbalance strategies: baseline, `class_weight='balanced'`, and SMOTE.
- Tune the Random Forest with `GridSearchCV` and report an OOB score for an internal validation estimate.
- Train a `LinearRegression` model to predict `fare` and report MAE / RMSE / R² / Adjusted R².
- Save the best classifier and regression pipelines with `joblib`, then reload them and run a raw-input prediction.
- Produce a final cross-task comparison and a deployment recommendation.

---

## 🛠️ Technologies Used

- Python 3
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn
- imbalanced-learn (SMOTE)
- joblib
- Jupyter Notebook
- VS Code

---

## 📂 Project Structure

```text
module_2_data_analytics/
│
├── charts/                       # generated PNGs
│   ├── correlation_heatmap.png
│   ├── multivariate_pairplot.png
│   ├── multivariate_facet_age.png
│   ├── multivariate_violin_fare.png
│   ├── multivariate_survival_by_embark.png
│   ├── modeling_confusion_matrices.png
│   ├── modeling_roc_curves.png
│   ├── modeling_decision_tree.png
│   ├── modeling_imbalance_confusion.png
│   └── modeling_residual_plot.png
│
├── models/                       # serialized pipelines
│   ├── best_classifier.joblib    # tuned Random Forest (GridSearchCV best)
│   └── fare_regressor.joblib     # LinearRegression on fare
│
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   └── 02_modeling.ipynb         # Classification + Regression + Persistence
│
├── src/                          # Authentication & REST API application
│   ├── __init__.py
│   ├── auth.py                   # PBKDF2 hashing, salt generation, HMAC tokens
│   ├── database.py               # SQLite user storage & schema management
│   └── app.py                    # AppService & REST HTTP endpoints
│
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_auth.py              # Unit tests for auth & database
│   └── test_api.py               # Integration tests for protected endpoints
│
├── titanic.csv                   # single source of truth (output of 01_eda.ipynb)
├── README.md
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Workflow

### 1. Exploratory Data Analysis

Open and run:

```
notebooks/01_eda.ipynb
```

The notebook:

- Loads the dataset once with `sns.load_dataset("titanic")` and writes it to `titanic.csv`.
- Renames `pclass → ticket_class` and `sibsp → sublings`.
- Computes missing-value percentages and applies the rules: drop `deck` (>75% missing), impute `age` with median, drop 2 rows with missing `embarked`.
- Visualizes `age`, `fare`, survival by sex, `ticket_class`, and the joint `sex × ticket_class` group via boolean masking.
- Computes mean / median / mode / skewness for `fare`.
- Builds a 6×6 correlation matrix over `survived, ticket_class, age, sublings, parch, fare` (parch substituted for `purchus_ticket`, which does not exist in the dataset) and renders an annotated heatmap.
- Identifies the two strongest off-diagonal correlations programmatically.
- Produces four distinct multivariate charts (pairplot, FacetGrid, split violin, catplot) with markdown interpretations.
- Runs a z-score standardization sanity check on `age` and `fare`.

### 2. Modeling

Open and run:

```
notebooks/02_modeling.ipynb
```

The notebook:

- Loads `titanic.csv` via `pd.read_csv`. No `sns.load_dataset` call.
- Performs a stratified 80/20 train/test split with `random_state=42`.
- Defines a `Pipeline` per model wrapping a `ColumnTransformer` (numeric: median impute + `StandardScaler`; categorical `sex`/`embarked`: most-frequent impute + `OneHotEncoder`).
- Trains three classifiers — Logistic Regression, Decision Tree, Random Forest — and reports confusion matrix, accuracy, precision, recall, F1, ROC, AUC.
- Compares class-imbalance strategies for Logistic Regression: baseline, `class_weight='balanced'`, and SMOTE (applied via `imblearn.pipeline.Pipeline` so the resampler never touches the test set).
- Tunes the Random Forest with `GridSearchCV` (5-fold, ROC-AUC) and reports a separate forest with `oob_score=True`.
- Trains a `LinearRegression` model to predict `fare` and reports MAE / RMSE / R² / Adjusted R²; produces a residual plot.
- Writes a final cross-task comparison table and a final recommendation.
- Persists the best classifier and the regression pipeline to `models/` with `joblib`, reloads them, and runs a raw-input prediction to demonstrate end-to-end usability.

---

## 📊 Dataset

The Titanic dataset contains demographic and ticket information for 891 passengers, including:

| Column        | Description |
|---------------|-------------|
| survived      | 0 = No, 1 = Yes |
| pclass        | Ticket class (1, 2, 3) |
| sex           | Male / Female |
| age           | Age in years |
| sibsp         | Number of siblings/spouses aboard |
| parch         | Number of parents/children aboard |
| fare          | Passenger fare |
| embarked      | Port of embarkation (C, Q, S) |
| class         | String form of `pclass` |
| who           | man / woman / child |
| adult_male    | Boolean |
| deck          | Deck (A–G) — 77% missing, dropped |
| embark_town   | String form of `embarked` |
| alive         | String form of `survived` |
| alone         | Boolean derived from `sibsp` + `parch` |

Target distribution: **survived = 0 → 549 (61.6%)**, **survived = 1 → 342 (38.4%)** — moderately imbalanced.

---

## 📈 Key Findings

- **Sex** is the strongest single predictor of survival: female ≈ 74%, male ≈ 19%.
- **Ticket class** strongly affects survival: 1st ≈ 63%, 2nd ≈ 47%, 3rd ≈ 24%.
- Strongest correlations (Pearson, on the 6×6 matrix):
  - `ticket_class` ↔ `fare` = **−0.548** (higher class = lower fare).
  - `sublings` ↔ `parch` = **+0.415** (families travel together).
- Class-imbalance strategies shift performance toward higher recall: `class_weight='balanced'` and SMOTE both raise recall to ≈ 0.78 without sacrificing AUC.
- Best classifier by test ROC-AUC on this run: **LR balanced** (AUC ≈ 0.846).
- Random Forest tuning picked `n_estimators=100, max_depth=10, min_samples_split=5`; the OOB score (0.80) is close to the 5-fold CV ROC-AUC (0.87), suggesting the RF generalizes reasonably.
- Linear Regression on `fare` achieved **R² ≈ 0.495** (Adjusted R² ≈ 0.477), MAE ≈ 21.8, RMSE ≈ 32.0; the residual plot shows the model under-predicts high fares (expected for a linear model on a right-skewed target).

---

## 🚀 How to Run

### Clone the repository

```bash
git clone <your-github-repository-url>
cd Zepto_capstone/module_2_data_analytics
```

### Create a virtual environment

```bash
python -m venv venv
```

### Activate the environment

Windows

```bash
venv\Scripts\activate
```

macOS / Linux

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the notebooks

```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_modeling.ipynb
```

### Run the authentication server

```bash
python -m src.app
```

The REST API server will run on `http://127.0.0.1:8000`. Available endpoints:
- `POST /api/register`: `{"username": "...", "email": "...", "password": "..."}`
- `POST /api/login`: `{"username": "...", "password": "..."}` (Returns Bearer token)
- `GET /api/profile`: Protected profile check (`Authorization: Bearer <token>`)
- `POST /api/predict/survival`: Protected ML survival prediction (`Authorization: Bearer <token>`)
- `POST /api/predict/fare`: Protected ML fare regression prediction (`Authorization: Bearer <token>`)

### Run the unit and integration tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Or using `pytest`:

```bash
python -m pytest tests/
```

---

## 🎓 Learning Outcomes

This project demonstrates:

- Exploratory Data Analysis (EDA)
- Data Cleaning and Missing-Value Treatment
- Univariate, Bivariate, and Multivariate Visualization
- Correlation Analysis and Heatmap Interpretation
- Train/Test Split Stratification and Cross-Validation
- Scikit-learn `Pipeline` and `ColumnTransformer` workflows
- Class-Imbalance Handling (`class_weight`, SMOTE)
- Hyperparameter Tuning with `GridSearchCV`
- Out-of-Bag (OOB) Score for Internal Validation
- Linear Regression with MAE, RMSE, R², Adjusted R²
- Residual Plot Interpretation
- Model Serialization with `joblib`
- End-to-End Reload + Raw-Input Prediction
- Reproducible Analytics Workflows

---

## 📊 Artifacts & Model Outputs

1. **`titanic.csv`**: Single source of truth dataset extracted via Seaborn and preprocessed for downstream modeling.
2. **`charts/`**: 10 exported visualization figures covering correlation heatmaps, pairplots, ROC curves, confusion matrices, decision tree diagrams, and regression residual plots.
3. **`models/best_classifier.joblib`**: Serialized Scikit-learn Random Forest classification pipeline.
4. **`models/fare_regressor.joblib`**: Serialized Scikit-learn LinearRegression fare regression pipeline.

---

## 🛠️ Troubleshooting & Debugging

- **`ModuleNotFoundError: No module named 'src'` when running tests**: Ensure you execute pytest or unittest commands from inside `module_2_data_analytics` or pass `PYTHONPATH=.` so python can locate the `src` package.
- **Joblib Numpy Array Shape Deprecation Warnings**: Harmless warning during joblib unpickling on modern NumPy versions. Models load and predict correctly.
- **REST API HTTP 401 Unauthorized Errors**: Protected endpoints (`/api/profile`, `/api/predict/survival`, `/api/predict/fare`) require an `Authorization: Bearer <token>` header returned from a successful `POST /api/login` response.

---

## 👨‍💻 Author

**Mundlapudi Muneendra**

AI & Machine Learning Student

GitHub:
https://github.com/mundlapudimuneendra-ops

---

## 📄 License

This project was developed for educational purposes as part of the Zepto Data Analytics Capstone.

