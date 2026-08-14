# 🚀 Zepto Capstone Project — End-to-End AI, ML & Data Engineering Suite

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-orange.svg)](https://www.langchain.com/langgraph)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5%2B-purple.svg)](https://www.trychroma.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Educational-green.svg)](#license)

## 📖 Executive Summary

The **Zepto Capstone Project** is a comprehensive, production-grade artificial intelligence, machine learning, and data engineering repository. It brings together end-to-end data pipelines, statistical exploratory data analysis, machine learning classification & regression with REST API deployment, and an offline/online Retrieval-Augmented Generation (RAG) Customer Support Assistant driven by LangGraph and FastAPI.

The project is modularized into three distinct capstone domains:
1. **Module 1 — Data Pipeline (ETL & SQL Analysis)**: Automated web scraping, data cleaning, currency conversion (GBP to INR), relational database modeling, and analytical SQL querying.
2. **Module 2 — Data Analytics & Machine Learning**: Exploratory data analysis (EDA), class-imbalance strategies (SMOTE, balanced weights), hyperparameter tuning, regression analysis, model serialization, PBKDF2 authentication, and live prediction REST APIs.
3. **Module 3 — Zepto Support Assistant (RAG + LangGraph + FastAPI)**: Policy ingestion into local ChromaDB vector store, 3-node LangGraph execution graph, dynamic dual-mode generation (`MOCK_LLM=1` offline vs `MOCK_LLM=0` live), Pydantic structured output validation with automatic retries, and FastAPI endpoint deployment.

---

## 📂 Repository Directory Structure

```text
Zepto_capstone/
├── module_1_data_pipeline/        # Module 1: Web Scraping, ETL & SQLite Database
│   ├── data/                      # Raw and cleaned CSV datasets
│   │   ├── raw_books.csv
│   │   └── cleaned_books.csv
│   ├── database/                  # Normalized SQLite Database
│   │   └── books.db
│   ├── notebooks/                 # Jupyter Data Cleaning & SQL Verification
│   │   ├── cleaning.ipynb
│   │   └── queries.ipynb
│   ├── src/                       # ETL Source Code
│   │   ├── scraper.py             # BeautifulSoup Web Scraper
│   │   ├── cleaner.py             # Data Transformation Engine
│   │   └── database.py            # SQLite Schema & Ingestion Script
│   ├── README.md                  # Module 1 Detailed Documentation
│   └── requirements.txt
│
├── module_2_data_analytics/       # Module 2: Data Analytics, ML & REST API Server
│   ├── charts/                    # Exported EDA & ML Visualizations
│   ├── models/                    # Serialized ML Pipelines (joblib)
│   │   ├── best_classifier.joblib
│   │   └── fare_regressor.joblib
│   ├── notebooks/                 # Analytics & Modeling Notebooks
│   │   ├── 01_eda.ipynb           # Missing value analysis, EDA & visualizations
│   │   └── 02_modeling.ipynb      # Imbalanced ML, Tuning & Regression
│   ├── src/                       # REST API & Auth Subsystem
│   │   ├── app.py                 # FastAPI/AppService HTTP Endpoints
│   │   ├── auth.py                # PBKDF2 Hashing, Salt & HMAC Token Security
│   │   └── database.py            # SQLite User Store & Authentication Schema
│   ├── tests/                     # Unit & Integration Test Suite
│   │   ├── test_auth.py           # Authentication & Password Security Tests
│   │   └── test_api.py            # Protected Endpoint & Prediction Integration Tests
│   ├── titanic.csv                # Single Source of Truth Dataset
│   ├── README.md                  # Module 2 Detailed Documentation
│   └── requirements.txt
│
├── module_3_/                     # Module 3: Zepto Customer Support RAG Assistant
│   └── support_assistant/
│       ├── chroma_store/          # Persistent ChromaDB Vector Store
│       ├── docs/                  # Policy Corpus Files (doc_01.txt .. doc_08.txt)
│       ├── Dockerfile             # Container Deployment Spec
│       ├── graph.py               # LangGraph StateGraph & Node Implementations
│       ├── ingest.py              # Ingestion, Verbatim Chunking & Vector Search
│       ├── main.py                # FastAPI HTTP Server (POST /ask)
│       ├── prompts.py             # LLM Prompt Templates & System Rules
│       ├── schema.py              # Pydantic Schemas & State Interfaces
│       ├── README.md              # Module 3 Detailed Documentation
│       └── requirements.txt
│
├── .gitignore                     # Root Git Ignore Configuration
├── README.md                      # Global Capstone Project Documentation
└── requirements.txt               # Unified Root Python Dependencies
```

---

## 🛠️ Unified Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Language & Runtimes** | Python 3.9+, Jupyter Notebook |
| **Data Engineering & Scraping** | BeautifulSoup4, Requests, LXML, Pandas, NumPy, SQLite3 |
| **Data Visualization** | Seaborn, Matplotlib |
| **Machine Learning & Modeling** | Scikit-learn, Imbalanced-learn (SMOTE), Joblib |
| **RAG & Agentic Workflows** | LangGraph, LangChain Core, ChromaDB, Sentence-Transformers (`all-MiniLM-L6-v2`) |
| **Web Services & APIs** | FastAPI, Uvicorn, Pydantic (v2) |
| **Security & Auth** | PBKDF2 Password Hashing, Salt Generation, HMAC Bearer Tokens |
| **DevOps & Testing** | Docker, Pytest, Unittest |

---

## 🧩 Module Overview & Key Highlights

### 🔹 Module 1: Zepto Data Pipeline (ETL & SQL)
- **Extraction**: Web scraper extracts book inventory data from *Books to Scrape*.
- **Transformation**: Cleans text, parses star ratings into integers, converts currency from **GBP to INR**, standardizes stock availability flags, and removes duplicates.
- **Loading**: Persists processed data into a normalized SQLite database (`books.db`) with `Categories` and `Books` tables linked via foreign key constraints.
- **SQL Analysis**: Executes analytical queries in Jupyter to determine category volume, price distribution, and top-rated titles.

### 🔹 Module 2: Data Analytics & ML Modeling Subsystem
- **Exploratory Data Analysis (EDA)**: Imputes missing values (`age`, `embarked`), handles high-missingness columns (`deck`), evaluates univariate/bivariate distributions, and computes annotated correlation matrices.
- **Classification Modeling**: Builds stratified train/test splits, compares Logistic Regression, Decision Trees, and Random Forest models across baseline, `class_weight='balanced'`, and **SMOTE** techniques to mitigate class imbalance.
- **Hyperparameter Tuning**: Optimizes Random Forest parameters with `GridSearchCV` (5-fold CV) and validates with Out-of-Bag (OOB) scoring.
- **Regression Modeling**: Trains a `LinearRegression` model to predict passenger `fare`, evaluating MAE, RMSE, R², and Adjusted R² alongside residual plots.
- **Production REST API**: Serves endpoints (`POST /api/register`, `POST /api/login`, `POST /api/predict/survival`, `POST /api/predict/fare`) secured with custom PBKDF2 salted password hashing and HMAC token authentication.

### 🔹 Module 3: Zepto Support Assistant (RAG + LangGraph)
- **Ingestion & Vector Indexing**: Loads Zepto policy documents, chunks text verbatim to preserve full semantic context, and indexes vectors into a local ChromaDB collection using `sentence-transformers/all-MiniLM-L6-v2`.
- **LangGraph Orchestration**: Implements a 3-node `StateGraph`:
  - `classify_intent`: Classifies queries into `policy_question` vs `general_question`.
  - `retrieve_and_answer`: Fetches top-3 matching chunks via cosine similarity and constructs grounded answers with citation sources.
  - `direct_answer`: Gracefully handles out-of-domain general queries.
- **Dynamic Dual-Mode**:
  - `MOCK_LLM=1` (Default): 100% offline, deterministic execution with zero external network dependencies.
  - `MOCK_LLM=0`: Live LLM mode with structured Pydantic `SupportResponse` validation and automatic retries (up to 3 attempts).
- **FastAPI Endpoint**: Exposes `POST /ask` with self-healing startup index verification and containerized Docker support.

---

## ⚙️ Quickstart & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/mundlapudimuneendra-ops/zepto_capstone.git
cd zepto_capstone
```

### 2. Create & Activate Virtual Environment
* **Windows (PowerShell)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install All Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Execution Instructions by Module

### 🏃 Running Module 1 (Data Pipeline)
```bash
cd module_1_data_pipeline

# 1. Run web scraper
python src/scraper.py

# 2. Build and populate SQLite database
python src/database.py

# 3. Launch analysis notebooks
jupyter notebook notebooks/cleaning.ipynb
jupyter notebook notebooks/queries.ipynb
```

---

### 🏃 Running Module 2 (Data Analytics & ML REST API)
```bash
cd module_2_data_analytics

# 1. Execute EDA and ML modeling notebooks
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_modeling.ipynb

# 2. Start the REST API Authentication & ML Inference Server
python -m src.app

# 3. Run unit and integration tests
python -m unittest discover -s tests -p "test_*.py"
# or with pytest:
pytest tests/
```

**Available REST Endpoints** (`http://127.0.0.1:8000`):
- `POST /api/register` — Register new user account
- `POST /api/login` — Authenticate and receive Bearer Token
- `GET /api/profile` — Fetch protected profile information
- `POST /api/predict/survival` — Run ML survival classification
- `POST /api/predict/fare` — Run ML fare regression model

---

### 🏃 Running Module 3 (Zepto Support Assistant)
```bash
cd module_3_/support_assistant

# 1. Ingest corpus and build ChromaDB index
python ingest.py

# 2. Run CLI demo
python main.py

# 3. Launch FastAPI server
uvicorn main:app --host 0.0.0.0 --port 7860
```

**Test `/ask` Endpoint**:
```bash
curl -X POST "http://127.0.0.1:7860/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the return policy for damaged items?"}'
```

**Docker Container Execution**:
```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```

---

## 🧪 Verification & Testing Summary

- **Module 1**: Verified web scraper outputs `raw_books.csv`, cleaner exports `cleaned_books.csv`, and database script creates valid relational `books.db` schema.
- **Module 2**: Unit tests pass for PBKDF2 authentication, salt generation, user registration, token verification, and model prediction endpoints (`pytest tests/`).
- **Module 3**: Verified deterministic response generation, ChromaDB vector retrieval, intent classification routing, FastAPI `/ask` HTTP responses, and Pydantic validation rules.

---

## 📊 Module Results Summary

| Module | Primary Outputs & Artifacts | Performance & Key Metrics |
| :--- | :--- | :--- |
| **Module 1: Data Pipeline** | `raw_books.csv`, `cleaned_books.csv`, `books.db` | 80 books scraped & normalized across 4 categories; price conversion GBP -> INR (105x); 100% relational integrity. |
| **Module 2: Analytics & ML** | `charts/*.png`, `best_classifier.joblib`, `fare_regressor.joblib` | Best Classifier test ROC-AUC ≈ 0.846; Random Forest CV ROC-AUC ≈ 0.87 (OOB score 0.80); Fare Regression R² ≈ 0.495; 12/12 unit tests passing. |
| **Module 3: Support Assistant** | `chroma_store/`, `POST /ask` FastAPI API, Docker container | Top-3 cosine similarity retrieval across 8 policy chunks; 100% offline mock execution; dynamic Pydantic schema validation. |

---

## 🔮 Future Improvements

- **Module 1 (Data Pipeline)**:
  - Automate scheduled ETL runs using Apache Airflow or GitHub Actions cron workflows.
  - Integrate live exchange rate APIs for real-time multi-currency conversions.
  - Expand normalized database schema to track historical price changes and book reviews.

- **Module 2 (Analytics & ML)**:
  - Experiment with gradient boosting models (XGBoost, LightGBM, CatBoost) for improved classification performance.
  - Transition from in-memory token authentication to standard OAuth2 with JWT refresh tokens and expiration windows.
  - Add Prometheus / Grafana metrics monitoring to track REST API latency and model prediction drift.

- **Module 3 (Support Assistant)**:
  - Implement hybrid search combining BM25 keyword matching with dense vector retrieval in ChromaDB.
  - Integrate multi-turn conversational memory into the LangGraph state to support follow-up questions.
  - Add streaming text generation endpoints (SSE / WebSockets) for real-time LLM token streaming in UI.

---

## 👨‍💻 Author

**Muneendra Mundlapudi**  
*AI & Machine Learning Specialist*  
GitHub: [mundlapudimuneendra-ops](https://github.com/mundlapudimuneendra-ops)

---

## 📄 License

This repository is developed for educational and assessment purposes as part of the **Zepto AI/ML Capstone Project**. All rights reserved.

