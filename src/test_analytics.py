from analytics.football_analytics import (
    load_data,
    league_overview,
    season_summary,
    league_table,
    team_summary,
    recent_form,
    head_to_head,
    compare_teams,
    goal_summary,
)


df = load_data()

print("\n" + "=" * 70)
print("FOOTBALL ANALYTICS TEST")
print("=" * 70)

print("\nDataset:")
print(df.shape)

print("\nLeague Overview:")
print(league_overview(df))

print("\nSeason Summary:")
print(
    season_summary(df).to_string(index=False)
)

print("\nLeague Table:")
print(
    league_table(df).head(10).to_string(index=False)
)

teams = sorted(
    set(df["home_team"]) |
    set(df["away_team"])
)

team1 = teams[0]
team2 = teams[1]

print(f"\nTeam 1: {team1}")
print(team_summary(df, team1))

print(f"\nRecent form: {team1}")
print(
    recent_form(df, team1).to_string(index=False)
)

print(f"\nTeam comparison: {team1} vs {team2}")
print(
    compare_teams(
        df,
        team1,
        team2
    ).to_string(index=False)
)

print("\nGoal Summary:")
print(goal_summary(df))

print("\nAnalytics engine working successfully.")