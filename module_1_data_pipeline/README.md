# 📚 Zepto Data Pipeline Capstone

## 📖 Project Overview

This project implements a complete **Extract, Transform, Load (ETL)** pipeline using Python, Pandas, BeautifulSoup, and SQLite.

The pipeline extracts book information from **Books to Scrape**, cleans and transforms the data, loads it into a normalized SQLite database, and performs SQL analysis using Jupyter Notebook.

---

## 🎯 Objectives

- Scrape book data from Books to Scrape
- Clean and preprocess the scraped dataset
- Convert book prices from GBP to INR
- Store data in a normalized SQLite database
- Execute SQL queries for data analysis
- Build a reusable ETL pipeline

---

## 🛠️ Technologies Used

- Python 3
- Requests
- BeautifulSoup4
- Pandas
- SQLite3
- Jupyter Notebook
- VS Code

---

## 📂 Project Structure

```text
data_pipeline/
│
├── data/
│   ├── raw_books.csv
│   └── cleaned_books.csv
│
├── database/
│   └── books.db
│
├── notebooks/
│   ├── cleaning.ipynb
│   └── queries.ipynb
│
├── src/
│   ├── scraper.py
│   ├── cleaner.py
│   └── database.py
│
├── README.md
├── requirements.txt
```

---

## ⚙️ Workflow

### 1. Extract

Run the scraper to collect book data.

```bash
python src/scraper.py
```

Output:

```
data/raw_books.csv
```

---

### 2. Transform

Open and run:

```
notebooks/cleaning.ipynb
```

The cleaning process:

- Removes unwanted characters
- Converts prices to numeric values
- Converts GBP to INR
- Converts ratings to integers
- Converts stock availability to Boolean
- Removes duplicate records

Output:

```
data/cleaned_books.csv
```

---

### 3. Load

Run:

```bash
python src/database.py
```

This creates:

- SQLite database
- Categories table
- Books table
- Foreign key relationship

Database:

```
database/books.db
```

---

### 4. Verify

Open:

```
notebooks/queries.ipynb
```

The notebook executes SQL queries using SQLite and Pandas to verify and analyze the data.

---

## 🗄️ Database Schema

### Categories

| Column | Type |
|---------|------|
| category_id | INTEGER PRIMARY KEY |
| category_name | TEXT UNIQUE |

### Books

| Column | Type |
|---------|------|
| book_id | INTEGER PRIMARY KEY |
| title | TEXT |
| price_inr | REAL |
| rating | INTEGER |
| in_stock | INTEGER |
| category_id | INTEGER (Foreign Key) |

---

## 📊 Dataset

Each book contains:

- Title
- Price (GBP)
- Price (INR)
- Rating
- Availability
- Category

---

## 📈 SQL Queries

The project includes SQL queries such as:

- Display all books
- Display books with ratings above a threshold
- Count books by category
- Find the most expensive books
- Join books with categories
- Sort books by price

All queries are executed in:

```
notebooks/queries.ipynb
```

using:

- sqlite3
- pandas.read_sql_query()

---

## 🚀 How to Run

### Clone the repository

```bash
git clone <your-github-repository-url>
cd data_pipeline
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

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the ETL pipeline

```bash
python src/scraper.py
python src/database.py
```

Run the notebooks:

- `notebooks/cleaning.ipynb`
- `notebooks/queries.ipynb`

---

## 🎓 Learning Outcomes

This project demonstrates:

- Web Scraping
- Data Cleaning
- Data Transformation
- CSV Processing
- SQLite Database Design
- SQL Queries
- Pandas Data Analysis
- ETL Pipeline Development

---

## 📊 Pipeline Outputs & Verification

1. **`data/raw_books.csv`**: Contains raw scraped fields (`title`, `price`, `star_rating`, `availability`, `category`).
2. **`data/cleaned_books.csv`**: Cleaned dataset containing 80 rows across 4 categories (Travel, Mystery, Historical Fiction, Classics) with price converted from GBP to INR (conversion multiplier: 105.0) and title deduplication.
3. **`database/books.db`**: Relational SQLite database with normalized schema (`categories` table linked to `books` table via Foreign Key `category_id`).

---

## 🛠️ Troubleshooting

- **HTTP 429 / Rate Limiting during Web Scraping**: `scraper.py` uses a custom `requests.Session` with a Chrome User-Agent header, automatic 4-attempt exponential backoff retries via `urllib3.util.retry.Retry`, and polite request delays (`0.2s`) to prevent blocking.
- **Database Duplicate Ingestion**: `database.py` enforces `UNIQUE` constraints on category names and book titles (`INSERT OR IGNORE`) and includes `fresh_database()` helper to maintain idempotency across pipeline runs.
- **Mojibake Character Decoding**: `cleaner.py` uses explicit Latin-1 to UTF-8 re-encoding (`_fix_mojibake()`) to resolve currency symbol corruption (`Â£` -> `£` / `INR`).

---

## 👨‍💻 Author

**Mundlapudi Muneendra**

AI & Machine Learning Student

GitHub:
https://github.com/mundlapudimuneendra-ops

---

## 📄 License

This project was developed for educational purposes as part of the Zepto Data Pipeline Capstone.