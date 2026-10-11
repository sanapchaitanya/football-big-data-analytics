#!/usr/bin/env python3
import csv
import sys
import re

reader = csv.DictReader(sys.stdin)

for row in reader:
    if row.get("is_completed", "").strip().lower() != "true":
        continue

    season = row.get("season", "").strip()
    home = row.get("home_team", "").strip()
    away = row.get("away_team", "").strip()
    result = row.get("result", "").strip()

    score = re.fullmatch(r"\s*(\d+)\s*[-–]\s*(\d+)\s*", result)

    if not season or not home or not away or not score:
        continue

    hg, ag = map(int, score.groups())

    if hg > ag:
        home_stats = [1, 1, 0, 0, hg, ag, 3]
        away_stats = [1, 0, 0, 1, ag, hg, 0]
    elif hg < ag:
        home_stats = [1, 0, 0, 1, hg, ag, 0]
        away_stats = [1, 1, 0, 0, ag, hg, 3]
    else:
        home_stats = [1, 0, 1, 0, hg, ag, 1]
        away_stats = [1, 0, 1, 0, ag, hg, 1]

    for team, stats in [(home, home_stats), (away, away_stats)]:
        key = f"{season}|{team}"
        print(f"{key}\t{','.join(map(str, stats))}")
