"""
database.py
-----------
Loads data/cleaned_books.csv into a normalised SQLite database.

Schema (kept identical to the original capstone spec so the existing
notebooks / reports still work):

  categories(category_id PK, category_name UNIQUE)
  books(
      book_id PK,
      title TEXT UNIQUE,            <- UNIQUE so duplicates can't sneak in
      price_inr REAL,
      rating INTEGER,
      in_stock INTEGER (0/1),
      category_id FK -> categories
  )

Why a UNIQUE constraint on `title`?
  The previous version of the loader inserted every row of the CSV on
  every run, which is what produced the 400-row "duplicate" database. The
  UNIQUE + INSERT OR IGNORE combination makes the loader idempotent: run
  it twice and the second run inserts zero new rows.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Tuple, Dict

import pandas as pd


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------
SCHEMA_SQL: Tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS categories (
        category_id   INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS books (
        book_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        title       TEXT    UNIQUE NOT NULL,
        price_inr   REAL,
        rating      INTEGER,
        in_stock    INTEGER,
        category_id INTEGER NOT NULL,
        FOREIGN KEY (category_id) REFERENCES categories(category_id)
    )
    """,
)


def fresh_database(db_path: str) -> None:
    """
    Wipe and recreate the database file.

    This is the cleanest way to remove the duplicate rows from the previous
    version. If you'd rather *keep* historical data and only skip duplicates
    on future runs, delete this call and the UNIQUE constraint will still
    protect you.
    """
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed old database: {db_path}")


def init_schema(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    for ddl in SCHEMA_SQL:
        cursor.execute(ddl)
    conn.commit()


def insert_categories(conn: sqlite3.Connection, categories: pd.Series) -> Dict[str, int]:
    """Insert unique category names and return a {name: id} map."""
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT OR IGNORE INTO categories(category_name) VALUES (?)",
        [(name,) for name in sorted(categories.unique())],
    )
    conn.commit()

    cursor.execute("SELECT category_id, category_name FROM categories")
    return {name: cid for cid, name in cursor.fetchall()}


def insert_books(conn: sqlite3.Connection, df: pd.DataFrame, category_map: Dict[str, int]) -> int:
    """
    Insert all books, skipping any title that already exists.

    Returns the total number of rows in the books table after the insert.
    """
    cursor = conn.cursor()
    rows = [
        (
            row["title"],
            float(row["price_inr"]),
            int(row["star_rating"]),
            int(bool(row["availability"])),
            category_map[row["category"]],
        )
        for _, row in df.iterrows()
    ]
    cursor.executemany(
        """
        INSERT OR IGNORE INTO books
            (title, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return cursor.execute("SELECT COUNT(*) FROM books").fetchone()[0]


def load(csv_path: str, db_path: str) -> None:
    print(f"Loading cleaned CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    fresh_database(db_path)
    conn = sqlite3.connect(db_path)
    try:
        init_schema(conn)
        category_map = insert_categories(conn, df["category"])
        insert_books(conn, df, category_map)

        cursor = conn.cursor()
        n_categories = cursor.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
        n_books = cursor.execute("SELECT COUNT(*) FROM books").fetchone()[0]
        n_unique = cursor.execute("SELECT COUNT(DISTINCT title) FROM books").fetchone()[0]

        print("\nData loaded successfully")
        print(f"  Categories: {n_categories}")
        print(f"  Books (rows): {n_books}")
        print(f"  Unique titles: {n_unique}")
        if n_books != n_unique:
            print("  WARNING: duplicate titles still present!")
    finally:
        conn.close()


def main() -> None:
    # Project root is the directory containing this script.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    db_dir = os.path.join(base_dir, "database")
    os.makedirs(db_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "cleaned_books.csv")
    db_path = os.path.join(db_dir, "books.db")
    load(csv_path, db_path)


if __name__ == "__main__":
    main()
