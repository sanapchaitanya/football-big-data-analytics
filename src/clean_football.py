import pandas as pd
from pathlib import Path


# --------------------------------------------------
# PATHS
# --------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

input_path = (
    project_root
    / "data"
    / "processed"
    / "premier_league_2022_2027.parquet"
)

output_path = (
    project_root
    / "data"
    / "processed"
    / "premier_league_2022_2027_clean.parquet"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_parquet(input_path)


print("=" * 60)
print("FOOTBALL DATA CLEANING")
print("=" * 60)

print("\nRows before cleaning:")
print(len(df))


# --------------------------------------------------
# STANDARDIZE DATE
# --------------------------------------------------

df["start_date"] = pd.to_datetime(
    df["start_date"],
    utc=True
)


# --------------------------------------------------
# RENAME SPORT-SPECIFIC COLUMNS
# TO COMMON PARTICIPANT COLUMNS
# --------------------------------------------------

df = df.rename(
    columns={
        "home_team": "participant_1",
        "away_team": "participant_2"
    }
)


# --------------------------------------------------
# STANDARDIZE TEXT COLUMNS
# --------------------------------------------------

text_columns = [
    "event_id",
    "sport_type",
    "country",
    "country_code",
    "tournament",
    "season",
    "match",
    "participant_1",
    "participant_2",
    "event_status",
    "result",
    "source"
]

for column in text_columns:

    df[column] = df[column].astype("string").str.strip()


# --------------------------------------------------
# REMOVE DUPLICATES
# --------------------------------------------------

df = df.drop_duplicates(
    subset=["event_id"]
)


# --------------------------------------------------
# SORT DATA
# --------------------------------------------------

df = df.sort_values(
    "start_date"
).reset_index(drop=True)


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

print("\nRows after cleaning:")
print(len(df))

print("\nDuplicate event IDs:")
print(
    df["event_id"].duplicated().sum()
)

print("\nMissing values:")
print(
    df.isnull().sum()
)

print("\nData types:")
print(
    df.dtypes
)


# --------------------------------------------------
# SAVE CLEAN DATASET
# --------------------------------------------------

df.to_parquet(
    output_path,
    index=False
)


print("\n")
print("=" * 60)
print("CLEAN DATASET SAVED")
print("=" * 60)

print("\nFile:")
print(output_path)

print("\nRows saved:")
print(len(df))