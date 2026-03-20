import pandas as pd
import sqlite3
import re

CSV_PATH = "amazon.csv"
DB_PATH = "amazon.db"

def clean_price(val):
    if pd.isna(val):
        return None
    return float(re.sub(r"[^\d.]", "", str(val)))

def clean_percentage(val):
    if pd.isna(val):
        return None
    return float(re.sub(r"[^\d.]", "", str(val)))

def clean_rating_count(val):
    if pd.isna(val):
        return None
    return int(re.sub(r"[^\d]", "", str(val)))

def load():
    df = pd.read_csv(CSV_PATH)

    # Keep only KPI-relevant columns
    df = df[[
        "product_id", "product_name", "category",
        "discounted_price", "actual_price", "discount_percentage",
        "rating", "rating_count"
    ]]

    # Clean price columns (remove ₹ and commas)
    df["discounted_price"] = df["discounted_price"].apply(clean_price)
    df["actual_price"]     = df["actual_price"].apply(clean_price)
    df["discount_percentage"] = df["discount_percentage"].apply(clean_percentage)
    df["rating_count"]     = df["rating_count"].apply(clean_rating_count)

    # Extract top-level category (e.g. "Electronics" from long pipe-separated string)
    df["main_category"] = df["category"].apply(
        lambda x: str(x).split("|")[0] if pd.notna(x) else None
    )

    # Load into SQLite
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("products", conn, if_exists="replace", index=False)
    conn.close()

    print(f"Loaded {len(df)} rows into {DB_PATH} — table: products")
    print("\nColumns available for querying:")
    for col, dtype in df.dtypes.items():
        print(f"  {col:25s} ({dtype})")

if __name__ == "__main__":
    load()