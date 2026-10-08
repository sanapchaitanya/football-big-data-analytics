import pandas as pd
import time
from pathlib import Path
from espn_client import ESPNClient


# --------------------------------------------------
# ESPN SETTINGS
# --------------------------------------------------

LEAGUE = "eng.1"

SEASONS = [2022, 2023, 2024, 2025, 2026]


# --------------------------------------------------
# FUNCTION: GET ONE SEASON
# --------------------------------------------------

def get_season(season):

    print("\n" + "=" * 50)
    print(f"COLLECTING SEASON: {season}-{str(season + 1)[-2:]}")
    print("=" * 50)

    client = ESPNClient()

    # --------------------------------------------------
    # 1. GET SEASON CALENDAR
    # --------------------------------------------------

    calendar_data = client.get_calendar(
        league=LEAGUE,
        season=season
    )

    event_dates = calendar_data["eventDate"]["dates"]

    print(
        f"Number of event dates: {len(event_dates)}"
    )


    # --------------------------------------------------
    # 2. GET EVENTS FOR EACH DATE
    # --------------------------------------------------

    rows = []

    for date in event_dates:

        date_only = date[:10]

        espn_date = date_only.replace("-", "")

        scoreboard_data = client.get_scoreboard(
            sport="soccer",
            league=LEAGUE,
            dates=espn_date
        )

        events = scoreboard_data.get("events", [])

        print(
            f"{date_only} -> {len(events)} events"
        )


        # --------------------------------------------------
        # 3. EXTRACT EVENTS
        # --------------------------------------------------

        for event in events:

            competition = event["competitions"][0]

            competitors = competition["competitors"]

            home_team = None
            away_team = None

            home_score = None
            away_score = None


            for team in competitors:

                if team["homeAway"] == "home":

                    home_team = team["team"]["displayName"]

                    home_score = team.get("score")


                elif team["homeAway"] == "away":

                    away_team = team["team"]["displayName"]

                    away_score = team.get("score")


            status = event["status"]["type"]


            # --------------------------------------------------
            # 4. RESULT
            # --------------------------------------------------

            if (
                home_score is not None
                and away_score is not None
                and status.get("completed") is True
            ):

                result = f"{home_score}-{away_score}"

            else:

                result = None


            # --------------------------------------------------
            # 5. CREATE ROW
            # --------------------------------------------------

            rows.append({

                "event_id": event["id"],

                "start_date": event["date"],

                "sport_type": "Football",

                "country": "England",

                "country_code": "ENG",

                "tournament": "Premier League",

                "season": f"{season}-{str(season + 1)[-2:]}",

                "match": event["name"],

                "home_team": home_team,

                "away_team": away_team,

                "event_status": status.get("description"),

                "result": result,

                "source": "ESPN"

            })


        # Small pause between requests
        time.sleep(0.1)


    # --------------------------------------------------
    # 6. CREATE SEASON DATAFRAME
    # --------------------------------------------------

    season_df = pd.DataFrame(rows)

    return season_df


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

all_seasons = []


for season in SEASONS:

    season_df = get_season(season)

    print(
        f"\nEvents collected for {season}: "
        f"{len(season_df)}"
    )

    all_seasons.append(season_df)


# --------------------------------------------------
# COMBINE ALL SEASONS
# --------------------------------------------------

df = pd.concat(
    all_seasons,
    ignore_index=True
)


# --------------------------------------------------
# CONVERT DATE
# --------------------------------------------------

df["start_date"] = pd.to_datetime(
    df["start_date"],
    utc=True
)


# --------------------------------------------------
# REMOVE DUPLICATES
# --------------------------------------------------

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=["event_id"]
)

after_duplicates = len(df)


# --------------------------------------------------
# SORT
# --------------------------------------------------

df = df.sort_values(
    "start_date"
).reset_index(drop=True)


# --------------------------------------------------
# FINAL VALIDATION
# --------------------------------------------------

print("\n")
print("=" * 60)
print("FINAL ESPN PREMIER LEAGUE DATASET")
print("=" * 60)


print("\nTotal events:")
print(len(df))


print("\nEvents by season:")
print(
    df["season"].value_counts().sort_index()
)


print("\nDuplicate event IDs:")
print(
    df["event_id"].duplicated().sum()
)


print("\nDuplicates removed:")
print(
    before_duplicates - after_duplicates
)


print("\nMissing values:")
print(
    df.isnull().sum()
)


print("\nDate range:")
print(
    df["start_date"].min(),
    "to",
    df["start_date"].max()
)


print("\nEvent status:")
print(
    df["event_status"].value_counts(dropna=False)
)


print("\nFirst 10 rows:")
print(
    df.head(10).to_string(index=False)

)

# --------------------------------------------------
# SAVE DATASET
# --------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

output_dir = project_root / "data" / "processed"

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

output_path = (
    output_dir /
    "premier_league_2022_2027.parquet"
)

df.to_parquet(
    output_path,
    index=False
)

print("\n")
print("=" * 60)
print("DATASET SAVED SUCCESSFULLY")
print("=" * 60)

print("\nFile:")
print(output_path)

print("\nRows saved:")
print(len(df))

print("\nColumns saved:")
print(df.columns.tolist())