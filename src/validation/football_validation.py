from __future__ import annotations

import pandas as pd

from features.football_features import FEATURE_COLUMNS, TARGET_ENCODING


def validate_modeling_dataset(modeling_df: pd.DataFrame) -> list[str]:
    errors: list[str] = []

    expected_targets = set(TARGET_ENCODING)
    actual_targets = set(modeling_df["target"].dropna().unique())
    if actual_targets != expected_targets:
        errors.append(
            f"Unexpected target classes. Expected {expected_targets}, got {actual_targets}."
        )

    if modeling_df[FEATURE_COLUMNS].isna().any().any():
        null_counts = modeling_df[FEATURE_COLUMNS].isna().sum()
        errors.append(
            "Required feature columns contain null values: "
            f"{null_counts[null_counts > 0].to_dict()}"
        )

    numeric_check = modeling_df[FEATURE_COLUMNS].apply(
        lambda column: pd.api.types.is_numeric_dtype(column)
    )
    non_numeric = numeric_check[~numeric_check].index.tolist()
    if non_numeric:
        errors.append(f"Feature columns must be numeric: {non_numeric}")

    if not modeling_df["start_date"].is_monotonic_increasing:
        errors.append("Modeling dataset is not chronologically ordered by start_date.")

    split_ranges = {
        "train": ["2022-23", "2023-24", "2024-25"],
        "validation": ["2025-26"],
        "test": ["2026-27"],
    }
    for split_name, seasons in split_ranges.items():
        split_df = modeling_df[modeling_df["season"].isin(seasons)]
        if not split_df.empty and not split_df["start_date"].is_monotonic_increasing:
            errors.append(f"{split_name} split is not chronologically ordered.")

    if not set(modeling_df["event_status"].unique()).issubset({"Full Time"}):
        errors.append("Modeling dataset contains non-Full Time matches.")

    if modeling_df["result"].isna().any():
        errors.append("Modeling dataset contains null result values.")

    if (modeling_df["home_matches_played_before"] < 0).any():
        errors.append("home_matches_played_before contains negative values.")

    if (modeling_df["away_matches_played_before"] < 0).any():
        errors.append("away_matches_played_before contains negative values.")

    if not (modeling_df["season_match_number_home"] >= 1).all():
        errors.append("season_match_number_home contains impossible values.")

    if not (modeling_df["season_match_number_away"] >= 1).all():
        errors.append("season_match_number_away contains impossible values.")

    return errors


def validate_no_future_information(modeling_df: pd.DataFrame) -> list[str]:
    """Validate structural conditions that support leakage-safe feature generation."""
    errors: list[str] = []

    first_matches = modeling_df.groupby(["season", "home_team"]).head(1)
    impossible_home = first_matches[
        first_matches["home_matches_played_before"]
        >= first_matches["season_match_number_home"]
    ]
    if not impossible_home.empty:
        errors.append("Home match-number consistency check failed.")

    first_away_matches = modeling_df.groupby(["season", "away_team"]).head(1)
    impossible_away = first_away_matches[
        first_away_matches["away_matches_played_before"]
        >= first_away_matches["season_match_number_away"]
    ]
    if not impossible_away.empty:
        errors.append("Away match-number consistency check failed.")

    same_time_duplicates = modeling_df.duplicated(
        subset=["start_date", "home_team", "away_team"]
    ).sum()
    if same_time_duplicates:
        errors.append("Duplicate same-time fixture rows detected.")

    return errors
