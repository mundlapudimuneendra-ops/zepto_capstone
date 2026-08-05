"""
cleaner.py
----------
Reads data/raw_books.csv produced by scraper.py and writes
data/cleaned_books.csv with a tidy, typed schema.

Transformations applied (in order):
  1. Fix the mojibake that the scraper inherits from the site ("Â£" -> "£",
     smart quotes etc.) by re-decoding the column as UTF-8.
  2. Strip the "Â£" prefix and cast price to float.
  3. Map textual ratings ("One".."Five") to integers 1..5.
  4. Convert "in_stock" availability to a boolean.
  5. Convert GBP -> INR and rename the column to price_inr.
  6. Drop duplicate titles so the loader can never insert a duplicate.
"""

from __future__ import annotations

import os

import pandas as pd

# Fixed conversion rate. Pinned at the top of the file so it's easy to find.
GBP_TO_INR = 105

# Rating text -> integer. The scraper hands us the second class on the
# star-rating <p>, which is always one of these five words.
RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def _fix_mojibake(series: pd.Series) -> pd.Series:
    """
    books.toscrape.com is served as Latin-1/Windows-1252 but the text was
    actually written in UTF-8, so the pound sign and curly quotes come out
    as 'Ã', 'Â', 'â\x80\x99' etc. This re-decodes them so the CSV is
    human-readable.
    """
    return series.str.encode("latin-1", errors="ignore").str.decode("utf-8", errors="ignore")


def clean(raw_csv: str, cleaned_csv: str) -> pd.DataFrame:
    """Run the full cleaning pipeline; return the resulting DataFrame."""
    print(f"Loading raw data from: {raw_csv}")
    df = pd.read_csv(raw_csv)

    # 1. Mojibake cleanup
    for col in ("title", "price", "availability", "category"):
        df[col] = _fix_mojibake(df[col])

    # 2. Price -> float (still in GBP at this point)
    df["price"] = (
        df["price"]
        .str.replace("£", "", regex=False)
        .str.strip()
        .astype(float)
    )

    # 3. Rating -> int
    df["star_rating"] = df["star_rating"].map(RATING_MAP).astype("Int64")

    # 4. Availability -> bool
    df["availability"] = (
        df["availability"]
        .str.strip()
        .str.contains("in stock", case=False)
    )

    # 5. GBP -> INR
    df["price_inr"] = (df["price"] * GBP_TO_INR).round(2)
    df = df.drop(columns=["price"])

    # 6. Defensive dedupe — same title can sneak in if the scraper is re-run
    #    after the site adds a new book. Keep the first occurrence.
    before = len(df)
    df = df.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"  Dropped {before - after} duplicate title(s).")

    # Final column order matches what the database loader expects.
    df = df[["title", "price_inr", "star_rating", "availability", "category"]]

    print(f"Cleaned rows: {len(df)}")
    print("Categories found:", sorted(df["category"].unique().tolist()))
    print(f"Saving cleaned data to: {cleaned_csv}")
    df.to_csv(cleaned_csv, index=False, encoding="utf-8")
    return df


def main() -> None:
    # Project root is the directory containing this script.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    raw_csv = os.path.join(data_dir, "raw_books.csv")
    cleaned_csv = os.path.join(data_dir, "cleaned_books.csv")
    clean(raw_csv, cleaned_csv)


if __name__ == "__main__":
    main()
