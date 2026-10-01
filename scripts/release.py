"""Record or verify the exact source data and frozen forecast artifacts.

Use --write deliberately when publishing a reviewed model release. Normal checks
only verify the committed manifest; they never bless changed artifacts silently.
"""

import argparse
import hashlib
import importlib.metadata
import json
import sys

import nbformat

from src.data import ROOT

MANIFEST = ROOT / "artifacts/models/release.json"
RELEASE_FILES = [
    "Data/team_game_data.csv.gz",
    "Data/player_game_data.csv.gz",
    "artifacts/models/selection.json",
    "artifacts/models/frozen_models.joblib",
    "artifacts/tables/matchups.csv.gz",
    "artifacts/tables/team_history.csv.gz",
    "artifacts/tables/player_roles.csv.gz",
    "artifacts/tables/test_predictions.csv",
    "artifacts/tables/record_comparison_games.csv",
    "artifacts/tables/prediction_contributions.csv.gz",
    "artifacts/tables/importance.csv",
    "artifacts/tables/intercept.json",
]


def file_digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_manifest():
    selection = json.loads((ROOT / RELEASE_FILES[2]).read_text())
    hashes = {name: file_digest(ROOT / name) for name in RELEASE_FILES}
    manifest = dict(
        schema_version=1,
        model_id="nba-pregame-" + hashes["artifacts/models/frozen_models.joblib"][:12],
        selected_model=selection["config"]["name"],
        train_seasons=selection["train_seasons"],
        test_season=selection["test_season"],
        features=selection["features"],
        intercept_log_odds=json.loads(
            (ROOT / "artifacts/tables/intercept.json").read_text()
        )["intercept"],
        python=f"{sys.version_info.major}.{sys.version_info.minor}",
        versions={
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "joblib"]
        },
        sha256=hashes,
    )
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_files(manifest, root=ROOT):
    mismatches = [
        name
        for name, expected in manifest["sha256"].items()
        if not (root / name).is_file() or file_digest(root / name) != expected
    ]
    if mismatches:
        raise ValueError(
            "Release files differ from the published manifest: " + ", ".join(mismatches)
        )


def verify_notebooks():
    counts = []
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        cells = [c for c in notebook.cells if c.cell_type == "code"]
        if any(c.execution_count is None for c in cells):
            raise ValueError(f"Unexecuted cells in {path.name}")
        if any(o.output_type == "error" for c in cells for o in c.outputs):
            raise ValueError(f"Saved cell errors in {path.name}")
        counts.append(len(cells))
    if len(counts) != 4:
        raise ValueError("Expected the four analytical notebooks")
    return sum(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="Record a deliberately reviewed release"
    )
    args = parser.parse_args()
    manifest = write_manifest() if args.write else json.loads(MANIFEST.read_text())
    verify_files(manifest)
    code_cells = verify_notebooks()
    print(
        f"Verified {manifest['model_id']}: {len(manifest['sha256'])} files and {code_cells} executed notebook cells"
    )


if __name__ == "__main__":
    main()
