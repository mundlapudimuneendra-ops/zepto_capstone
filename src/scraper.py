"""
scraper.py
----------
Scrapes real book data from https://books.toscrape.com.

How it works (read this once and you'll understand every line):
  1. Hit the homepage. The sidebar contains 50 category links such as
     /catalogue/category/books/travel_2/index.html.
  2. Pick a configurable list of categories by SLUG (see CATEGORIES below).
  3. For each category, walk its first N pages and extract every book card.
  4. Tag every row with the real category name so the database stays normalised.
  5. Save the result to data/raw_books.csv.

Why a requests.Session + User-Agent + retries?
  books.toscrape.com is a sandbox; it occasionally resets idle TLS
  connections from generic Python clients. A session that reuses one
  connection plus a browser User-Agent and urllib3 retries makes scraping
  stable without us adding sleep-based hacks.
"""

from __future__ import annotations

import os
import time
from typing import List, Dict

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = "https://books.toscrape.com/"
CATALOGUE_URL = "https://books.toscrape.com/catalogue/"

# Pick the categories you want. Slugs come straight from the side-nav URLs.
# To scrape a different set, just edit this list — the rest of the script
# stays the same.
CATEGORIES: List[Dict[str, str]] = [
    {"slug": "travel_2",                "name": "Travel"},
    {"slug": "mystery_3",               "name": "Mystery"},
    {"slug": "historical-fiction_4",    "name": "Historical Fiction"},
    {"slug": "classics_6",              "name": "Classics"},
]

# How many listing pages to walk inside each category. Each page holds 20 books.
PAGES_PER_CATEGORY = 2

# Tiny pause between requests — be polite to the sandbox.
REQUEST_DELAY_SEC = 0.2


# ---------------------------------------------------------------------------
# HTTP session with retries
# ---------------------------------------------------------------------------
def make_session() -> requests.Session:
    """Return a requests.Session that retries on transient errors."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        }
    )
    retry = Retry(
        total=4,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ---------------------------------------------------------------------------
# Scraping helpers
# ---------------------------------------------------------------------------
def discover_categories(session: requests.Session) -> Dict[str, str]:
    """
    Read the homepage side-nav and return a {slug: display_name} map.

    Even though we already declared CATEGORIES above, this is useful as a
    sanity check — if books.toscrape.com renames a category, we'll see it
    here before we waste requests.
    """
    response = session.get(BASE_URL, timeout=20)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    found: Dict[str, str] = {}
    for anchor in soup.select("div.side_categories ul li ul li a"):
        name = anchor.text.strip()
        href = anchor.get("href", "")  # e.g. "catalogue/category/books/travel_2/index.html"
        # Pull just the slug (e.g. "travel_2") so we can match it to CATEGORIES.
        parts = href.strip("/").split("/")
        if len(parts) >= 2:
            slug = parts[-2]
            found[slug] = name
    return found


def parse_book_cards(soup: BeautifulSoup) -> List[Dict[str, str]]:
    """Pull every <article class='product_pod'> on a page into a dict."""
    rows: List[Dict[str, str]] = []
    for book in soup.find_all("article", class_="product_pod"):
        title = book.h3.a["title"]
        price = book.find("p", class_="price_color").text.strip()
        # The star rating lives in a class like "star-rating Three" — index [1]
        rating = book.p["class"][1]
        availability = book.find("p", class_="instock availability").text.strip()
        rows.append(
            {
                "title": title,
                "price": price,
                "star_rating": rating,
                "availability": availability,
            }
        )
    return rows


def scrape_category(
    session: requests.Session,
    category_name: str,
    slug: str,
    pages: int,
) -> List[Dict[str, str]]:
    """Scrape `pages` listing pages inside one category and tag each row."""
    rows: List[Dict[str, str]] = []
    for page in range(1, pages + 1):
        # books.toscrape.com quirk: page 1 lives at /index.html, page 2+ at
        # /page-N.html. A 404 means the category is shorter than PAGES_PER_CATEGORY
        # (e.g. Travel only has 11 books) — that's fine, just stop early.
        if page == 1:
            url = f"{CATALOGUE_URL}category/books/{slug}/index.html"
        else:
            url = f"{CATALOGUE_URL}category/books/{slug}/page-{page}.html"
        print(f"  -> {category_name} page {page}: {url}")
        response = session.get(url, timeout=20)
        if response.status_code != 200:
            print(f"     ! HTTP {response.status_code} — no more pages in this category.")
            break
        rows.extend(parse_book_cards(BeautifulSoup(response.text, "html.parser")))
        time.sleep(REQUEST_DELAY_SEC)
    # Stamp the real category onto every row.
    for row in rows:
        row["category"] = category_name
    return rows


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    session = make_session()

    print("Discovering categories on the homepage...")
    discovered = discover_categories(session)
    print(f"  Found {len(discovered)} categories on the site.")

    all_rows: List[Dict[str, str]] = []
    for category in CATEGORIES:
        slug, name = category["slug"], category["name"]
        if slug not in discovered:
            print(f"  ! Skipping '{name}' — slug '{slug}' not on the site anymore.")
            continue
        print(f"Scraping category: {name} ({slug})")
        all_rows.extend(scrape_category(session, name, slug, PAGES_PER_CATEGORY))

    df = pd.DataFrame(all_rows)
    print(f"\nTotal books scraped: {len(df)}")
    print("Books per category:")
    print(df["category"].value_counts().to_string())

    # Resolve <project_root>/data/raw_books.csv no matter where we run from.
    # Project root is the directory containing this script.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, "raw_books.csv")
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"\nSaved {len(df)} rows to: {output_path}")


if __name__ == "__main__":
    main()
