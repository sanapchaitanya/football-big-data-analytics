import hashlib
import requests
import pandas as pd


# --------------------------------------------------
# PROJECT SETTINGS
# --------------------------------------------------

URL = "https://raw.githubusercontent.com/openfootball/football.json/master/2025-26/en.1.json"

SPORT_TYPE = "Fotbal"
COUNTRY = "England"
COUNTRY_CODE = "ENG"
TOURNAMENT = "Premier League"
SEASON = "2025-26"
SOURCE = "openfootball"

SNAPSHOT_DATE = pd.Timestamp.today().normalize()


# --------------------------------------------------
# 1. DOWNLOAD THE DATA
# --------------------------------------------------

response = requests.get(URL)

response.raise_for_status()

data = response.json()


# --------------------------------------------------
# 2. GET THE MATCHES
# --------------------------------------------------

matches = data["matches"]


# --------------------------------------------------
# 3. CREATE ROWS
# --------------------------------------------------

rows = []

for match in matches:

    # Get score
    score = match.get("score")

    if isinstance(score, dict):
        score = score.get("ft")

    # Determine result and status
    if isinstance(score, list) and len(score) == 2:

        result = f"{score[0]}-{score[1]}"
        event_status = "Finished"

    else:

        result = None
        event_status = "Scheduled"

    # Get teams
    home_team = match.get("team1")
    away_team = match.get("team2")

    # Create a stable ID
    id_string = (
        f"{TOURNAMENT}|"
        f"{SEASON}|"
        f"{home_team}|"
        f"{away_team}"
    )

    event_id = hashlib.md5(
        id_string.encode("utf-8")
    ).hexdigest()[:12]

    # Create row
    row = {

        "event_id": event_id,

        "start_date": match.get("date"),

        "sport_type": SPORT_TYPE,

        "country": COUNTRY,

        "country_code": COUNTRY_CODE,

        "tournament": TOURNAMENT,

        "season": SEASON,

        "match": f"{home_team} vs {away_team}",

        "home_team": home_team,

        "away_team": away_team,

        "snapshot_date": SNAPSHOT_DATE,

        "event_status": event_status,

        "result": result,

        "source": SOURCE
    }

    rows.append(row)


# --------------------------------------------------
# 4. CREATE DATAFRAME
# --------------------------------------------------

df = pd.DataFrame(rows)


# --------------------------------------------------
# 5. CONVERT DATE
# --------------------------------------------------

df["start_date"] = pd.to_datetime(df["start_date"])


# --------------------------------------------------
# 6. SORT DATA
# --------------------------------------------------

df = df.sort_values(
    ["start_date", "tournament", "match"]
).reset_index(drop=True)


# --------------------------------------------------
# 7. DISPLAY DATA
# --------------------------------------------------

print("\nDATAFRAME CREATED SUCCESSFULLY!\n")

print(df.head(10).to_string(index=False))


print("\n\nCOLUMNS:")
print(df.columns.tolist())


print("\n\nNUMBER OF MATCHES:")
print(len(df))


print("\n\nDATA TYPES:")
print(df.dtypes)


print("\n\nMISSING VALUES:")
print(df.isnull().sum())


print("\n\nDATE RANGE:")
print(
    df["start_date"].min(),
    "to",
    df["start_date"].max()
)

# --------------------------------------------------
# 8. SAVE AS PARQUET
# --------------------------------------------------

output_path = "../data/processed/premier_league_2025_26.parquet"

df.to_parquet(
    output_path,
    engine="pyarrow",
    index=False
)

print("\nPARQUET FILE SAVED:")
print(output_path)