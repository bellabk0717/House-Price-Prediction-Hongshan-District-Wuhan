import pandas as pd
import numpy as np
import re

def load_raw_excel(path: str) -> pd.DataFrame:
    """Load raw Excel dataset."""
    return pd.read_excel(path)

def clean_price(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Chinese string price format into numeric."""
    df["price"] = (
        df["price"]
        .str.replace("万", "", regex=False)
        .astype(float) * 10000
    )
    return df

def clean_area_and_rooms(df: pd.DataFrame) -> pd.DataFrame:
    """Extract area, rooms, halls."""
    df["area"] = df["area"].str.replace("㎡", "", regex=False).astype(float)

    df["rooms"] = df["info"].str.extract(r"(\d+)室").astype(float)
    df["halls"] = df["info"].str.extract(r"(\d+)厅").astype(float)

    return df

def clean_floor(df: pd.DataFrame) -> pd.DataFrame:
    """Extract floor level (高/中/低)."""
    df["floor_level"] = df["info"].str.extract(r"(高|中|低)(?=楼层)")
    df["total_floor"] = df["info"].str.extract(r"共(\d+)层").astype(float)
    return df

def clean_subdistrict(df: pd.DataFrame) -> pd.DataFrame:
    """Split address into community name and subdistrict."""
    df[["community", "subdistrict"]] = df["location"].str.split(" ", 1, expand=True)
    return df

def clean_dataset(path: str) -> pd.DataFrame:
    """Full cleaning pipeline."""
    df = load_raw_excel(path)
    df = clean_price(df)
    df = clean_area_and_rooms(df)
    df = clean_floor(df)
    df = clean_subdistrict(df)

    df.drop(columns=["info", "Unnamed: 0"], errors="ignore", inplace=True)
    df = df.dropna(subset=["area", "price"])
    return df
