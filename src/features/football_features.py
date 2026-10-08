from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import pandas as pd


TARGET_ENCODING = {
    "Home Win": 0,
    "Draw": 1,
    "Away Win": 2,
}

FEATURE_COLUMNS = [
    "home_matches_played_before",
    "away_matches_played_before",
    "home_points_per_game_before",
    "away_points_per_game_before",
    "home_historical_matches_played_before",
    "away_historical_matches_played_before",
    "home_historical_points_per_game_before",
    "away_historical_points_per_game_before",
    "home_recent_points_per_game_5",
    "away_recent_points_per_game_5",
    "home_goals_scored_per_game_before",
    "away_goals_scored_per_game_before",
    "home_goals_conceded_per_game_before",
    "away_goals_conceded_per_game_before",
    "home_historical_goals_scored_per_game_before",
    "away_historical_goals_scored_per_game_before",
    "home_historical_goals_conceded_per_game_before",
    "away_historical_goals_conceded_per_game_before",
    "home_recent_goals_scored_per_game_5",
    "away_recent_goals_scored_per_game_5",
    "home_recent_goals_conceded_per_game_5",
    "away_recent_goals_conceded_per_game_5",
    "home_goal_difference_per_game_before",
    "away_goal_difference_per_game_before",
    "home_historical_goal_difference_per_game_before",
    "away_historical_goal_difference_per_game_before",
    "home_home_points_per_game_before",
    "away_away_points_per_game_before",
    "home_home_goals_scored_per_game_before",
    "away_away_goals_scored_per_game_before",
    "home_home_goals_conceded_per_game_before",
    "away_away_goals_conceded_per_game_before",
    "home_historical_home_points_per_game_before",
    "away_historical_away_points_per_game_before",
    "home_historical_home_goals_scored_per_game_before",
    "away_historical_away_goals_scored_per_game_before",
    "home_historical_home_goals_conceded_per_game_before",
    "away_historical_away_goals_conceded_per_game_before",
    "h2h_matches_before",
    "h2h_home_team_points_per_game_before",
    "h2h_away_team_points_per_game_before",
    "h2h_home_team_goals_scored_per_game_before",
    "h2h_away_team_goals_scored_per_game_before",
    "points_per_game_diff",
    "recent_points_per_game_5_diff",
    "goals_scored_per_game_diff",
    "goals_conceded_per_game_diff",
    "goal_difference_per_game_diff",
    "historical_points_per_game_diff",
    "historical_goals_scored_per_game_diff",
    "historical_goals_conceded_per_game_diff",
    "historical_goal_difference_per_game_diff",
    "season_match_number_home",
    "season_match_number_away",
]

CONTEXT_COLUMNS = [
    "event_id",
    "start_date",
    "season",
    "match",
    "home_team",
    "away_team",
    "event_status",
    "result",
    "home_goals",
    "away_goals",
    "target",
    "target_encoded",
]


@dataclass
class TeamSeasonState:
    matches: int = 0
    points: int = 0
    goals_for: int = 0
    goals_against: int = 0
    home_matches: int = 0
    home_points: int = 0
    home_goals_for: int = 0
    home_goals_against: int = 0
    away_matches: int = 0
    away_points: int = 0
    away_goals_for: int = 0
    away_goals_against: int = 0
    recent: deque = field(default_factory=lambda: deque(maxlen=5))


@dataclass
class HeadToHeadState:
    matches: int = 0
    points: dict = field(default_factory=lambda: defaultdict(int))
    goals_for: dict = field(default_factory=lambda: defaultdict(int))


def _safe_rate(numerator: float, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _parse_result(result: str) -> tuple[int, int]:
    home_goals, away_goals = str(result).split("-")
    return int(home_goals), int(away_goals)


def _target_from_goals(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "Home Win"
    if home_goals < away_goals:
        return "Away Win"
    return "Draw"


def _points_for(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _recent_rate(state: TeamSeasonState, key: str) -> float:
    if not state.recent:
        return 0.0
    return sum(item[key] for item in state.recent) / len(state.recent)


def _team_features(
    prefix: str,
    season_state: TeamSeasonState,
    historical_state: TeamSeasonState,
    venue: str,
) -> dict[str, float]:
    if venue == "home":
        venue_matches = season_state.home_matches
        venue_points = season_state.home_points
        venue_goals_for = season_state.home_goals_for
        venue_goals_against = season_state.home_goals_against
        historical_venue_matches = historical_state.home_matches
        historical_venue_points = historical_state.home_points
        historical_venue_goals_for = historical_state.home_goals_for
        historical_venue_goals_against = historical_state.home_goals_against
        venue_prefix = "home_home"
        historical_venue_prefix = "home_historical_home"
    else:
        venue_matches = season_state.away_matches
        venue_points = season_state.away_points
        venue_goals_for = season_state.away_goals_for
        venue_goals_against = season_state.away_goals_against
        historical_venue_matches = historical_state.away_matches
        historical_venue_points = historical_state.away_points
        historical_venue_goals_for = historical_state.away_goals_for
        historical_venue_goals_against = historical_state.away_goals_against
        venue_prefix = "away_away"
        historical_venue_prefix = "away_historical_away"

    return {
        f"{prefix}_matches_played_before": season_state.matches,
        f"{prefix}_points_per_game_before": _safe_rate(
            season_state.points, season_state.matches
        ),
        f"{prefix}_historical_matches_played_before": historical_state.matches,
        f"{prefix}_historical_points_per_game_before": _safe_rate(
            historical_state.points, historical_state.matches
        ),
        f"{prefix}_recent_points_per_game_5": _recent_rate(
            historical_state, "points"
        ),
        f"{prefix}_goals_scored_per_game_before": _safe_rate(
            season_state.goals_for, season_state.matches
        ),
        f"{prefix}_goals_conceded_per_game_before": _safe_rate(
            season_state.goals_against, season_state.matches
        ),
        f"{prefix}_historical_goals_scored_per_game_before": _safe_rate(
            historical_state.goals_for, historical_state.matches
        ),
        f"{prefix}_historical_goals_conceded_per_game_before": _safe_rate(
            historical_state.goals_against, historical_state.matches
        ),
        f"{prefix}_recent_goals_scored_per_game_5": _recent_rate(
            historical_state, "goals_for"
        ),
        f"{prefix}_recent_goals_conceded_per_game_5": _recent_rate(
            historical_state, "goals_against"
        ),
        f"{prefix}_goal_difference_per_game_before": _safe_rate(
            season_state.goals_for - season_state.goals_against, season_state.matches
        ),
        f"{prefix}_historical_goal_difference_per_game_before": _safe_rate(
            historical_state.goals_for - historical_state.goals_against,
            historical_state.matches,
        ),
        f"{venue_prefix}_points_per_game_before": _safe_rate(
            venue_points, venue_matches
        ),
        f"{venue_prefix}_goals_scored_per_game_before": _safe_rate(
            venue_goals_for, venue_matches
        ),
        f"{venue_prefix}_goals_conceded_per_game_before": _safe_rate(
            venue_goals_against, venue_matches
        ),
        f"{historical_venue_prefix}_points_per_game_before": _safe_rate(
            historical_venue_points, historical_venue_matches
        ),
        f"{historical_venue_prefix}_goals_scored_per_game_before": _safe_rate(
            historical_venue_goals_for, historical_venue_matches
        ),
        f"{historical_venue_prefix}_goals_conceded_per_game_before": _safe_rate(
            historical_venue_goals_against, historical_venue_matches
        ),
    }


def _h2h_key(team_a: str, team_b: str) -> tuple[str, str]:
    return tuple(sorted([team_a, team_b]))


def _h2h_features(
    home_team: str,
    away_team: str,
    h2h_state: HeadToHeadState,
) -> dict[str, float]:
    return {
        "h2h_matches_before": h2h_state.matches,
        "h2h_home_team_points_per_game_before": _safe_rate(
            h2h_state.points[home_team], h2h_state.matches
        ),
        "h2h_away_team_points_per_game_before": _safe_rate(
            h2h_state.points[away_team], h2h_state.matches
        ),
        "h2h_home_team_goals_scored_per_game_before": _safe_rate(
            h2h_state.goals_for[home_team], h2h_state.matches
        ),
        "h2h_away_team_goals_scored_per_game_before": _safe_rate(
            h2h_state.goals_for[away_team], h2h_state.matches
        ),
    }


def _add_diff_features(row: dict) -> None:
    row["points_per_game_diff"] = (
        row["home_points_per_game_before"] - row["away_points_per_game_before"]
    )
    row["recent_points_per_game_5_diff"] = (
        row["home_recent_points_per_game_5"] - row["away_recent_points_per_game_5"]
    )
    row["goals_scored_per_game_diff"] = (
        row["home_goals_scored_per_game_before"]
        - row["away_goals_scored_per_game_before"]
    )
    row["goals_conceded_per_game_diff"] = (
        row["home_goals_conceded_per_game_before"]
        - row["away_goals_conceded_per_game_before"]
    )
    row["goal_difference_per_game_diff"] = (
        row["home_goal_difference_per_game_before"]
        - row["away_goal_difference_per_game_before"]
    )
    row["historical_points_per_game_diff"] = (
        row["home_historical_points_per_game_before"]
        - row["away_historical_points_per_game_before"]
    )
    row["historical_goals_scored_per_game_diff"] = (
        row["home_historical_goals_scored_per_game_before"]
        - row["away_historical_goals_scored_per_game_before"]
    )
    row["historical_goals_conceded_per_game_diff"] = (
        row["home_historical_goals_conceded_per_game_before"]
        - row["away_historical_goals_conceded_per_game_before"]
    )
    row["historical_goal_difference_per_game_diff"] = (
        row["home_historical_goal_difference_per_game_before"]
        - row["away_historical_goal_difference_per_game_before"]
    )
    row["season_match_number_home"] = row["home_matches_played_before"] + 1
    row["season_match_number_away"] = row["away_matches_played_before"] + 1


def _update_states(
    match: pd.Series,
    season_states: dict[tuple[str, str], TeamSeasonState],
    historical_states: dict[str, TeamSeasonState],
    h2h_states: dict[tuple[str, str], HeadToHeadState],
) -> None:
    season = str(match["season"])
    home_team = str(match["home_team"])
    away_team = str(match["away_team"])
    home_goals = int(match["home_goals"])
    away_goals = int(match["away_goals"])

    home_points = _points_for(home_goals, away_goals)
    away_points = _points_for(away_goals, home_goals)

    home_state = season_states[(season, home_team)]
    away_state = season_states[(season, away_team)]
    home_historical_state = historical_states[home_team]
    away_historical_state = historical_states[away_team]

    for state in (home_state, home_historical_state):
        state.matches += 1
        state.points += home_points
        state.goals_for += home_goals
        state.goals_against += away_goals
        state.home_matches += 1
        state.home_points += home_points
        state.home_goals_for += home_goals
        state.home_goals_against += away_goals
        state.recent.append(
            {
                "points": home_points,
                "goals_for": home_goals,
                "goals_against": away_goals,
            }
        )

    for state in (away_state, away_historical_state):
        state.matches += 1
        state.points += away_points
        state.goals_for += away_goals
        state.goals_against += home_goals
        state.away_matches += 1
        state.away_points += away_points
        state.away_goals_for += away_goals
        state.away_goals_against += home_goals
        state.recent.append(
            {
                "points": away_points,
                "goals_for": away_goals,
                "goals_against": home_goals,
            }
        )

    h2h_state = h2h_states[_h2h_key(home_team, away_team)]
    h2h_state.matches += 1
    h2h_state.points[home_team] += home_points
    h2h_state.points[away_team] += away_points
    h2h_state.goals_for[home_team] += home_goals
    h2h_state.goals_for[away_team] += away_goals


def build_modeling_dataset(clean_df: pd.DataFrame) -> pd.DataFrame:
    completed = clean_df[
        clean_df["is_completed"].eq(True)
        & clean_df["event_status"].eq("Full Time")
        & clean_df["result"].notna()
    ].copy()

    completed[["home_goals", "away_goals"]] = completed["result"].apply(
        lambda result: pd.Series(_parse_result(result))
    )
    completed["target"] = completed.apply(
        lambda row: _target_from_goals(row["home_goals"], row["away_goals"]),
        axis=1,
    )
    completed["target_encoded"] = completed["target"].map(TARGET_ENCODING)
    completed = completed.sort_values(["start_date", "event_id"]).reset_index(drop=True)

    season_states: dict[tuple[str, str], TeamSeasonState] = defaultdict(TeamSeasonState)
    historical_states: dict[str, TeamSeasonState] = defaultdict(TeamSeasonState)
    h2h_states: dict[tuple[str, str], HeadToHeadState] = defaultdict(HeadToHeadState)
    rows = []

    for _, same_time_matches in completed.groupby("start_date", sort=True):
        batch_rows = []

        for _, match in same_time_matches.iterrows():
            season = str(match["season"])
            home_team = str(match["home_team"])
            away_team = str(match["away_team"])
            home_state = season_states[(season, home_team)]
            away_state = season_states[(season, away_team)]
            home_historical_state = historical_states[home_team]
            away_historical_state = historical_states[away_team]
            h2h_state = h2h_states[_h2h_key(home_team, away_team)]

            row = {
                "event_id": match["event_id"],
                "start_date": match["start_date"],
                "season": match["season"],
                "match": match["match"],
                "home_team": home_team,
                "away_team": away_team,
                "event_status": match["event_status"],
                "result": match["result"],
                "home_goals": int(match["home_goals"]),
                "away_goals": int(match["away_goals"]),
                "target": match["target"],
                "target_encoded": int(match["target_encoded"]),
            }

            row.update(
                _team_features("home", home_state, home_historical_state, "home")
            )
            row.update(
                _team_features("away", away_state, away_historical_state, "away")
            )
            row.update(_h2h_features(home_team, away_team, h2h_state))
            _add_diff_features(row)
            batch_rows.append(row)

        rows.extend(batch_rows)

        for _, match in same_time_matches.iterrows():
            _update_states(match, season_states, historical_states, h2h_states)

    modeling_df = pd.DataFrame(rows)
    ordered_columns = CONTEXT_COLUMNS + FEATURE_COLUMNS
    modeling_df = modeling_df[ordered_columns]
    modeling_df[FEATURE_COLUMNS] = modeling_df[FEATURE_COLUMNS].astype(float)
    modeling_df["target_encoded"] = modeling_df["target_encoded"].astype(int)
    return modeling_df.sort_values(["start_date", "event_id"]).reset_index(drop=True)


def chronological_split(
    modeling_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = modeling_df[modeling_df["season"].isin(["2022-23", "2023-24", "2024-25"])]
    validation = modeling_df[modeling_df["season"].eq("2025-26")]
    test = modeling_df[modeling_df["season"].eq("2026-27")]
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )

def build_prediction_features(
    clean_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    season: str,
    prediction_date=None,
) -> pd.DataFrame:
    """
    Build one leakage-safe feature row for a specific fixture.

    Only matches completed BEFORE the prediction date are used.
    This prevents future information from entering the prediction.
    """

    completed = clean_df.copy()

    completed["start_date"] = pd.to_datetime(
        completed["start_date"],
        utc=True,
    )

    completed = completed[
        completed["event_status"].eq("Full Time")
        & completed["result"].notna()
        & completed["season"].notna()
    ].copy()

    # ---------------------------------------------------------
    # DERIVE GOALS FROM ESPN RESULT
    # ---------------------------------------------------------

    goals = completed["result"].str.split(
        "-",
        expand=True,
    )

    completed["home_goals"] = pd.to_numeric(
        goals[0],
        errors="coerce",
    )

    completed["away_goals"] = pd.to_numeric(
        goals[1],
        errors="coerce",
    )

    completed = completed[
        completed["home_goals"].notna()
        & completed["away_goals"].notna()
        ].copy()

    # ---------------------------------------------------------
    # LEAKAGE PREVENTION
    # ---------------------------------------------------------

    if prediction_date is not None:

        prediction_date = pd.Timestamp(prediction_date)

        if prediction_date.tzinfo is None:
            prediction_date = prediction_date.tz_localize("UTC")
        else:
            prediction_date = prediction_date.tz_convert("UTC")

        # STRICTLY before kickoff.
        # Matches on the same kickoff date are not used.
        completed = completed[
            completed["start_date"] < prediction_date
        ].copy()

    # ---------------------------------------------------------
    # SORT CHRONOLOGICALLY
    # ---------------------------------------------------------

    completed = completed.sort_values(
        ["start_date", "event_id"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # CREATE STATES
    # ---------------------------------------------------------

    season_states: dict[
        tuple[str, str],
        TeamSeasonState
    ] = defaultdict(TeamSeasonState)

    historical_states: dict[
        str,
        TeamSeasonState
    ] = defaultdict(TeamSeasonState)

    h2h_states: dict[
        tuple[str, str],
        HeadToHeadState
    ] = defaultdict(HeadToHeadState)

    # ---------------------------------------------------------
    # REBUILD ALL HISTORICAL STATES
    # ---------------------------------------------------------

    for _, match in completed.iterrows():

        _update_states(
            match,
            season_states,
            historical_states,
            h2h_states,
        )

    # ---------------------------------------------------------
    # GET CURRENT STATES
    # ---------------------------------------------------------

    home_state = season_states[
        (season, home_team)
    ]

    away_state = season_states[
        (season, away_team)
    ]

    home_historical_state = historical_states[
        home_team
    ]

    away_historical_state = historical_states[
        away_team
    ]

    h2h_state = h2h_states[
        _h2h_key(home_team, away_team)
    ]

    # ---------------------------------------------------------
    # CREATE PREDICTION ROW
    # ---------------------------------------------------------

    row = {
        "event_id": "prediction",
        "start_date": (
            prediction_date
            if prediction_date is not None
            else pd.Timestamp.now(tz="UTC")
        ),
        "season": season,
        "match": f"{home_team} vs {away_team}",
        "home_team": home_team,
        "away_team": away_team,
        "event_status": "Scheduled",
        "result": None,
        "home_goals": None,
        "away_goals": None,
        "target": None,
        "target_encoded": None,
    }

    # Home team features
    row.update(
        _team_features(
            "home",
            home_state,
            home_historical_state,
            "home",
        )
    )

    # Away team features
    row.update(
        _team_features(
            "away",
            away_state,
            away_historical_state,
            "away",
        )
    )

    # Head-to-head features
    row.update(
        _h2h_features(
            home_team,
            away_team,
            h2h_state,
        )
    )

    # Difference features
    _add_diff_features(row)

    prediction_df = pd.DataFrame([row])

    prediction_df[FEATURE_COLUMNS] = (
        prediction_df[FEATURE_COLUMNS]
        .astype(float)
    )

    return prediction_df

def predict_score_from_features(
    feature_row: pd.DataFrame,
) -> tuple[int, int]:
    """
    Estimate the most likely scoreline using pre-match
    attacking and defensive goal rates.

    XGBoost remains the only ML model in the project.
    This is a statistical scoreline estimation layer.
    """

    row = feature_row.iloc[0]

    # ---------------------------------------------------------
    # HOME EXPECTED GOALS
    # ---------------------------------------------------------

    home_attack = row[
        "home_home_goals_scored_per_game_before"
    ]

    away_defence = row[
        "away_away_goals_conceded_per_game_before"
    ]

    if home_attack <= 0:
        home_attack = row[
            "home_goals_scored_per_game_before"
        ]

    if away_defence <= 0:
        away_defence = row[
            "away_goals_conceded_per_game_before"
        ]

    if home_attack <= 0:
        home_attack = row[
            "home_historical_goals_scored_per_game_before"
        ]

    if away_defence <= 0:
        away_defence = row[
            "away_historical_goals_conceded_per_game_before"
        ]

    # ---------------------------------------------------------
    # AWAY EXPECTED GOALS
    # ---------------------------------------------------------

    away_attack = row[
        "away_away_goals_scored_per_game_before"
    ]

    home_defence = row[
        "home_home_goals_conceded_per_game_before"
    ]

    if away_attack <= 0:
        away_attack = row[
            "away_goals_scored_per_game_before"
        ]

    if home_defence <= 0:
        home_defence = row[
            "home_goals_conceded_per_game_before"
        ]

    if away_attack <= 0:
        away_attack = row[
            "away_historical_goals_scored_per_game_before"
        ]

    if home_defence <= 0:
        home_defence = row[
            "home_historical_goals_conceded_per_game_before"
        ]

    # ---------------------------------------------------------
    # EXPECTED GOALS
    # ---------------------------------------------------------

    expected_home_goals = (
        home_attack + away_defence
    ) / 2

    expected_away_goals = (
        away_attack + home_defence
    ) / 2

    expected_home_goals = max(
        0.05,
        min(expected_home_goals, 5.0),
    )

    expected_away_goals = max(
        0.05,
        min(expected_away_goals, 5.0),
    )

    # ---------------------------------------------------------
    # POISSON SCORELINE
    # ---------------------------------------------------------

    import math

    best_probability = -1
    best_score = (0, 0)

    for home_goals in range(0, 7):

        for away_goals in range(0, 7):

            home_probability = (
                math.exp(-expected_home_goals)
                * expected_home_goals ** home_goals
                / math.factorial(home_goals)
            )

            away_probability = (
                math.exp(-expected_away_goals)
                * expected_away_goals ** away_goals
                / math.factorial(away_goals)
            )

            score_probability = (
                home_probability * away_probability
            )

            if score_probability > best_probability:

                best_probability = score_probability

                best_score = (
                    home_goals,
                    away_goals,
                )

    return best_score