import pandas as pd
from pathlib import Path


# --------------------------------------------------
# FILE PATH
# --------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

file_path = (
    project_root
    / "data"
    / "processed"
    / "premier_league_2022_2027.parquet"
)


# --------------------------------------------------
# CHECK FILE EXISTS
# --------------------------------------------------

print("FILE EXISTS:", file_path.exists())

print("\nFile path:")
print(file_path)


# --------------------------------------------------
# LOAD PARQUET
# --------------------------------------------------

df = pd.read_parquet(file_path)


# --------------------------------------------------
# BASIC INFORMATION
# --------------------------------------------------

print("\n")
print("=" * 60)
print("PARQUET VALIDATION")
print("=" * 60)

print("\nRows:")
print(len(df))

print("\nColumns:")
print(df.columns.tolist())


# --------------------------------------------------
# SEASONS
# --------------------------------------------------

print("\nEvents by season:")
print(
    df["season"].value_counts().sort_index()
)


# --------------------------------------------------
# DUPLICATES
# --------------------------------------------------

print("\nDuplicate event IDs:")
print(
    df["event_id"].duplicated().sum()
)


# --------------------------------------------------
# MISSING VALUES
# --------------------------------------------------

print("\nMissing values:")
print(
    df.isnull().sum()
)


# --------------------------------------------------
# DATE RANGE
# --------------------------------------------------

print("\nDate range:")
print(
    df["start_date"].min(),
    "to",
    df["start_date"].max()
)


# --------------------------------------------------
# DATA TYPES
# --------------------------------------------------

print("\nData types:")
print(df.dtypes)


# --------------------------------------------------
# SAMPLE
# --------------------------------------------------

print("\nFirst 5 rows:")
print(
    df.head().to_string(index=False)
)


print("\n")
print("=" * 60)
print("PARQUET VALIDATION COMPLETE")
print("=" * 60)