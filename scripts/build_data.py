"""Audit the supplied files and save reproducible pregame tables."""

import gzip
import hashlib
import json

from src.data import ROOT, load_data, validate_data
from src.features import build_matchups, build_roster_history, build_team_history

TABLES = ROOT / "artifacts/tables"


def save_table(frame, name):
    path = TABLES / name
    compression = {"method": "gzip", "mtime": 0} if name.endswith(".gz") else None
    frame.to_csv(path, index=False, float_format="%.10g", compression=compression)


def main():
    team, player = load_data()
    games, report, contradictions, mismatches = validate_data(team, player)
    if len(mismatches):
        raise ValueError("Player points do not reconcile with team totals")
    report["source_sha256"] = {}
    for name in ["team", "player"]:
        with gzip.open(ROOT / f"Data/{name}_game_data.csv.gz", "rb") as handle:
            report["source_sha256"][name] = hashlib.sha256(handle.read()).hexdigest()
    print("Building team histories", flush=True)
    history = build_team_history(games)
    print("Building prior player roles", flush=True)
    roster, roles = build_roster_history(history, player)
    history = history.merge(roster, on=["nbagameid", "off_team"], validate="one_to_one")
    matchups = build_matchups(history)
    coverage = matchups.groupby("season").agg(
        games=("nbagameid", "size"),
        eligible=("eligible", "sum"),
        start=("gamedate", "min"),
        end=("gamedate", "max"),
    )
    report["regular_games"] = len(matchups)
    report["eligible_games"] = int(matchups.eligible.sum())
    report["test_season"] = 2023
    save_table(history, "team_history.csv.gz")
    save_table(roles, "player_roles.csv.gz")
    save_table(matchups, "matchups.csv.gz")
    save_table(coverage.reset_index(), "coverage.csv")
    save_table(contradictions, "missed_contradictions.csv")
    (TABLES / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(coverage.to_string(), flush=True)


if __name__ == "__main__":
    main()
