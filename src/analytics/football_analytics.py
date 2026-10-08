from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "football_espn_clean.parquet"
)


def load_data(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the cleaned ESPN football dataset.
    """

    file_path = path or DATA_PATH

    df = pd.read_parquet(file_path)

    df["start_date"] = pd.to_datetime(
        df["start_date"],
        utc=True
    )

    # Only completed matches for historical analytics
    df = df[
        (df["event_status"] == "Full Time")
        & df["result"].notna()
    ].copy()

    # Extract goals
    goals = df["result"].str.split("-", expand=True)

    df["home_goals"] = pd.to_numeric(
        goals[0],
        errors="coerce"
    )

    df["away_goals"] = pd.to_numeric(
        goals[1],
        errors="coerce"
    )

    df["total_goals"] = (
        df["home_goals"]
        + df["away_goals"]
    )

    df["result_type"] = df.apply(
        lambda row: (
            "Home Win"
            if row["home_goals"] > row["away_goals"]
            else "Away Win"
            if row["home_goals"] < row["away_goals"]
            else "Draw"
        ),
        axis=1,
    )

    return df.sort_values(
        "start_date"
    ).reset_index(drop=True)


# ---------------------------------------------------------
# LEAGUE OVERVIEW
# ---------------------------------------------------------

def league_overview(df: pd.DataFrame) -> dict:

    total_matches = len(df)

    total_goals = int(
        df["total_goals"].sum()
    )

    return {
        "matches": total_matches,
        "goals": total_goals,
        "average_goals_per_match": (
            total_goals / total_matches
            if total_matches
            else 0
        ),
        "home_wins": int(
            (df["result_type"] == "Home Win").sum()
        ),
        "draws": int(
            (df["result_type"] == "Draw").sum()
        ),
        "away_wins": int(
            (df["result_type"] == "Away Win").sum()
        ),
    }


# ---------------------------------------------------------
# SEASON ANALYTICS
# ---------------------------------------------------------

def season_summary(df: pd.DataFrame) -> pd.DataFrame:

    summary = (
        df.groupby("season")
        .agg(
            matches=("event_id", "count"),
            total_goals=("total_goals", "sum"),
            average_goals=("total_goals", "mean"),
            home_wins=(
                "result_type",
                lambda x: (x == "Home Win").sum()
            ),
            draws=(
                "result_type",
                lambda x: (x == "Draw").sum()
            ),
            away_wins=(
                "result_type",
                lambda x: (x == "Away Win").sum()
            ),
        )
        .reset_index()
    )

    return summary


# ---------------------------------------------------------
# TEAM SUMMARY
# ---------------------------------------------------------

def team_summary(
    df: pd.DataFrame,
    team: str
) -> dict:

    home = df[
        df["home_team"] == team
    ].copy()

    away = df[
        df["away_team"] == team
    ].copy()

    matches = len(home) + len(away)

    wins = (
        (home["result_type"] == "Home Win").sum()
        +
        (away["result_type"] == "Away Win").sum()
    )

    draws = (
        (home["result_type"] == "Draw").sum()
        +
        (away["result_type"] == "Draw").sum()
    )

    losses = matches - wins - draws

    goals_scored = (
        home["home_goals"].sum()
        +
        away["away_goals"].sum()
    )

    goals_conceded = (
        home["away_goals"].sum()
        +
        away["home_goals"].sum()
    )

    points = (
        wins * 3
        + draws
    )

    return {
        "team": team,
        "matches": matches,
        "wins": int(wins),
        "draws": int(draws),
        "losses": int(losses),
        "points": int(points),
        "points_per_game": (
            points / matches
            if matches
            else 0
        ),
        "goals_scored": int(goals_scored),
        "goals_conceded": int(goals_conceded),
        "goal_difference": int(
            goals_scored - goals_conceded
        ),
    }


# ---------------------------------------------------------
# LEAGUE TABLE
# ---------------------------------------------------------

def league_table(
    df: pd.DataFrame,
    season: Optional[str] = None
) -> pd.DataFrame:

    data = df.copy()

    if season:
        data = data[
            data["season"] == season
        ]

    teams = sorted(
        set(data["home_team"])
        | set(data["away_team"])
    )

    rows = []

    for team in teams:

        stats = team_summary(
            data,
            team
        )

        rows.append(stats)

    table = pd.DataFrame(rows)

    if not table.empty:
        table = table.sort_values(
            ["points", "goal_difference", "goals_scored"],
            ascending=False
        ).reset_index(drop=True)

        table.insert(
            0,
            "position",
            range(1, len(table) + 1)
        )

    return table


# ---------------------------------------------------------
# HOME / AWAY PERFORMANCE
# ---------------------------------------------------------

def home_away_summary(
    df: pd.DataFrame,
    team: str
) -> pd.DataFrame:

    home = df[
        df["home_team"] == team
    ]

    away = df[
        df["away_team"] == team
    ]

    home_points = (
        (home["result_type"] == "Home Win").sum() * 3
        +
        (home["result_type"] == "Draw").sum()
    )

    away_points = (
        (away["result_type"] == "Away Win").sum() * 3
        +
        (away["result_type"] == "Draw").sum()
    )

    return pd.DataFrame({
        "venue": ["Home", "Away"],
        "matches": [
            len(home),
            len(away)
        ],
        "points": [
            home_points,
            away_points
        ],
        "points_per_game": [
            home_points / len(home)
            if len(home) else 0,

            away_points / len(away)
            if len(away) else 0
        ],
        "goals_scored": [
            home["home_goals"].sum(),
            away["away_goals"].sum()
        ],
        "goals_conceded": [
            home["away_goals"].sum(),
            away["home_goals"].sum()
        ]
    })


# ---------------------------------------------------------
# RECENT FORM
# ---------------------------------------------------------

def recent_form(
    df: pd.DataFrame,
    team: str,
    n: int = 5
) -> pd.DataFrame:

    matches = df[
        (df["home_team"] == team)
        |
        (df["away_team"] == team)
    ].sort_values(
        "start_date",
        ascending=False
    ).head(n).copy()

    rows = []

    for _, match in matches.iterrows():

        if match["home_team"] == team:

            opponent = match["away_team"]

            if match["result_type"] == "Home Win":
                result = "W"
                points = 3
            elif match["result_type"] == "Draw":
                result = "D"
                points = 1
            else:
                result = "L"
                points = 0

            goals_for = match["home_goals"]
            goals_against = match["away_goals"]

        else:

            opponent = match["home_team"]

            if match["result_type"] == "Away Win":
                result = "W"
                points = 3
            elif match["result_type"] == "Draw":
                result = "D"
                points = 1
            else:
                result = "L"
                points = 0

            goals_for = match["away_goals"]
            goals_against = match["home_goals"]

        rows.append({
            "date": match["start_date"],
            "opponent": opponent,
            "result": result,
            "points": points,
            "goals_for": goals_for,
            "goals_against": goals_against,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# HEAD TO HEAD
# ---------------------------------------------------------

def head_to_head(
    df: pd.DataFrame,
    team1: str,
    team2: str
) -> pd.DataFrame:

    h2h = df[
        (
            (df["home_team"] == team1)
            &
            (df["away_team"] == team2)
        )
        |
        (
            (df["home_team"] == team2)
            &
            (df["away_team"] == team1)
        )
    ].sort_values(
        "start_date",
        ascending=False
    ).copy()

    return h2h[
        [
            "start_date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "result_type"
        ]
    ]


# ---------------------------------------------------------
# TEAM COMPARISON
# ---------------------------------------------------------

def compare_teams(
    df: pd.DataFrame,
    team1: str,
    team2: str
) -> pd.DataFrame:

    stats1 = team_summary(df, team1)
    stats2 = team_summary(df, team2)

    comparison = pd.DataFrame({
        "metric": [
            "Matches",
            "Wins",
            "Draws",
            "Losses",
            "Points",
            "Points Per Game",
            "Goals Scored",
            "Goals Conceded",
            "Goal Difference",
        ],
        team1: [
            stats1["matches"],
            stats1["wins"],
            stats1["draws"],
            stats1["losses"],
            stats1["points"],
            stats1["points_per_game"],
            stats1["goals_scored"],
            stats1["goals_conceded"],
            stats1["goal_difference"],
        ],
        team2: [
            stats2["matches"],
            stats2["wins"],
            stats2["draws"],
            stats2["losses"],
            stats2["points"],
            stats2["points_per_game"],
            stats2["goals_scored"],
            stats2["goals_conceded"],
            stats2["goal_difference"],
        ],
    })

    return comparison


# ---------------------------------------------------------
# GOAL ANALYTICS
# ---------------------------------------------------------

def goal_summary(df: pd.DataFrame) -> dict:

    return {
        "total_goals": int(
            df["total_goals"].sum()
        ),
        "average_goals": float(
            df["total_goals"].mean()
        ),
        "highest_scoring_match": int(
            df["total_goals"].max()
        ),
        "home_goals": int(
            df["home_goals"].sum()
        ),
        "away_goals": int(
            df["away_goals"].sum()
        ),
    }