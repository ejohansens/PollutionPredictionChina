from pathlib import Path
from typing import List

import pandas as pd

NUMERIC_COLUMNS = [
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
]

DATA_FOLDER_NAME = "PRSA_Data_20130301-20170228"

# Find all station CSV files in the data directory.
def _find_data_files(data_root: Path) -> List[Path]:
    if not data_root.exists() or not data_root.is_dir():
        raise FileNotFoundError(f"Data folder not found: {data_root}")

    files = sorted(data_root.glob("PRSA_Data_*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_root}")

    return files


# Clean column labels so names are safe and consistent.
def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda x: x.strip().strip('"'))
    df.columns = [col.replace(" ", "") for col in df.columns]
    return df

# Build a timestamp column from year, month, day, and hour.
def _parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(
        df.assign(
            year=df["year"].astype("Int64"),
            month=df["month"].astype("Int64"),
            day=df["day"].astype("Int64"),
            hour=df["hour"].astype("Int64"),
        ).loc[:, ["year", "month", "day", "hour"]],
        format="%Y-%m-%d %H",
        errors="coerce",
    )
    return df

# Convert sensor columns to numeric values and coerce bad values to NaN.
def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in NUMERIC_COLUMNS + ["No"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df

# Trim and normalize wind direction and station string values.
def _clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "wd" in df.columns:
        df["wd"] = df["wd"].astype(str).str.strip().replace({"nan": None})
    if "station" in df.columns:
        df["station"] = df["station"].astype(str).str.strip().str.replace('"', "")
    return df

# Apply all cleaning steps to a single DataFrame.
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = _standardize_columns(df)
    df = _convert_numeric_columns(df)
    df = _clean_text_columns(df)
    df = _parse_datetime(df)
    df = df.drop_duplicates()
    df = df.loc[df["date"].notna()].reset_index(drop=True)
    return df


# Load one CSV file and return a cleaned DataFrame.
def load_csv_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["NA", "nan", "NaN", ""], keep_default_na=True)
    df = clean_data(df)
    df["source_file"] = path.name
    return df

# Load all station CSV files and combine them into one DataFrame.
def load_all_data(data_root: Path) -> pd.DataFrame:
    files = _find_data_files(data_root)
    frames = [load_csv_file(path) for path in files]
    combined = pd.concat(frames, ignore_index=True)
    combined["station"] = combined["station"].astype(str)
    return combined
