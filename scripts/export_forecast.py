"""Export a published historical forecast without starting Streamlit."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data import ROOT
from src.game_report import forecast_record, resolve_game_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game_id", help="Eligible historical NBA game identifier")
    parser.add_argument(
        "--output", type=Path, help="Write JSON to this path instead of stdout"
    )
    args = parser.parse_args()
    tables = ROOT / "artifacts/tables"
    predictions = pd.read_csv(tables / "record_comparison_games.csv").set_index(
        "nbagameid"
    )
    game_id = resolve_game_id(args.game_id, predictions.index)
    if game_id is None:
        parser.error("The game is not covered by the published 2023–24 predictions")
    matchups = pd.read_csv(tables / "matchups.csv.gz").set_index("nbagameid")
    terms = pd.read_csv(tables / "prediction_contributions.csv.gz")
    release = json.loads((ROOT / "artifacts/models/release.json").read_text())
    report = forecast_record(
        predictions.loc[game_id], matchups.loc[game_id], terms, release
    )
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
