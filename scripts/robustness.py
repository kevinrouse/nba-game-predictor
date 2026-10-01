"""Audit the existing choices and predictions without fitting new candidates."""

import json

import pandas as pd

from scripts.build_data import TABLES, save_table
from src.validation import replay_selection, uncertainty_sensitivity


def main():
    folds = pd.read_csv(TABLES / "cv_folds.csv")
    candidates = pd.read_csv(TABLES / "cv_summary.csv")
    replay = replay_selection(folds, candidates)
    save_table(replay, "selection_replay.csv")
    baselines = folds[folds.name.isin(["Baseline net_rating", "Baseline win_pct"])]
    comparison = pd.concat(
        [
            replay.rename(columns={"selected_name": "name"}).assign(
                procedure="Choose from earlier seasons"
            ),
            baselines.assign(procedure=baselines.name).loc[
                lambda d: d.season.isin(replay.season)
            ],
        ],
        ignore_index=True,
    )
    save_table(
        comparison[
            ["season", "procedure", "name", "log_loss", "brier", "auc", "accuracy", "n"]
        ],
        "selection_replay_comparison.csv",
    )
    predictions = pd.read_csv(TABLES / "record_comparison_games.csv")
    sensitivity = uncertainty_sensitivity(predictions)
    save_table(sensitivity, "block_sensitivity.csv")
    report = dict(
        status="Retrospective diagnostics; original model and test predictions unchanged",
        replay_seasons=replay.season.astype(int).tolist(),
        replay_choices=replay.selected_name.tolist(),
        replay_mean_log_loss=float(replay.log_loss.mean()),
        baseline_mean_log_loss=float(
            comparison.loc[
                comparison.procedure.eq("Baseline net_rating"), "log_loss"
            ].mean()
        ),
        block_lengths=[1, 2, 4],
        caveat="Candidate library was designed during original development; this is not nested independent evaluation",
    )
    (TABLES / "robustness.json").write_text(json.dumps(report, indent=2) + "\n")
    print(replay.to_string(index=False), flush=True)
    print(sensitivity.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
