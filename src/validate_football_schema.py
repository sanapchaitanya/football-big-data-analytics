import pandas as pd
from pathlib import Path

from sports_schema import validate_schema


# --------------------------------------------------
# FILE PATH
# --------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

file_path = (
    project_root
    / "data"
    / "processed"
    / "premier_league_2022_2027_clean.parquet"
)


# --------------------------------------------------
# LOAD CLEAN FOOTBALL DATA
# --------------------------------------------------

df = pd.read_parquet(file_path)


print("=" * 60)
print("VALIDATING PREMIER LEAGUE DATASET")
print("=" * 60)

print("\nDataset rows:")
print(len(df))


# --------------------------------------------------
# VALIDATE COMMON SCHEMA
# --------------------------------------------------

missing_columns, extra_columns = validate_schema(df)


# --------------------------------------------------
# ADDITIONAL CHECK
# --------------------------------------------------

print("\n")
print("=" * 60)
print("FINAL RESULT")
print("=" * 60)


if not missing_columns and not extra_columns:

    print("\nPremier League dataset matches")
    print("the common sports schema.")

else:

    print("\nPremier League dataset does NOT")
    print("fully match the common sports schema.")


print("\nValidation complete.")