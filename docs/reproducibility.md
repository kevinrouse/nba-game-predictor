# Reproducing the published analysis

There are two different checks: **are these the exact published files?**, and **does rebuilding from the sources recover the same analytical results?** A serialized model can differ between environments while producing numerically equivalent predictions, so the checks stay separate.

## Verify a checkout

Use Python 3.11 and install `requirements-dev.txt`, then run:

```bash
python -m scripts.release
ruff check src scripts tests app.py
python -m pytest -q
```

The release manifest records SHA-256 hashes of the supplied compressed sources, frozen model/selection, pregame tables and prediction explanations. Verification fails on a missing or changed file. It also validates the four notebooks and rejects unexecuted cells or saved cell errors. Do not regenerate the manifest to hide a mismatch; first establish why the file changed.

Tests cover more than table shapes: current/future outcome mutations, future truncation, removal of the current roster, train-only preprocessing, identical benchmark game sets, paired resampling, selection using strictly earlier validation scores, export reconstruction and dashboard navigation. Small artificial histories make boundary failures observable; the published artifacts provide end-to-end numerical checks.

## Rebuild from the sources

```bash
python -m scripts.reproduce
python -m scripts.verify_reproduction
```

The pipeline runs, in order:

1. `build_data`: validate source keys/arithmetic and reconstruct team/player histories.
2. `compare_models`: compare the original 74 configurations over six chronological folds and refit the selected model on development seasons.
3. `evaluate`: reproduce the original final-test predictions and probability diagnostics.
4. `explore` and `story_analysis`: generate explanatory tables and the retrospective record-rule comparison.
5. `robustness`: replay selection from preceding validation seasons and report every specified bootstrap block length.
6. `execute_notebooks`: execute each notebook in a fresh kernel.
7. `export_forecast`: rebuild the published Detroit–Dallas audit example.
8. `pytest`: check the rebuilt analysis and app behavior.

All paths are repository-relative and random seeds are 78. No basketball API or external account is needed. The pipeline does overwrite generated artifacts in that checkout; use a clean clone when comparing a new implementation with the published result.

`verify_reproduction` compares every tracked analytical CSV, including compressed tables, against Git `HEAD`. Identifiers, integer counts, nonnumeric values, column names, row order and the selected configuration must match exactly. A regression test rejects even a one-unit game-ID change; numerical tolerances apply only to floating-point columns. Floating-point comparisons allow absolute tolerance `1e-8` and relative tolerance `1e-7` for platform arithmetic. This is not permission to change a result materially. Serialized-model byte hashes are checked **before** a rebuild; the rebuilt model is checked through its predictions and selection afterward.

To publish a deliberately changed model, review the evaluation design and artifacts first, then explicitly write a new release manifest. The existing test season is already inspected and cannot become a new untouched test simply by rerunning the pipeline.

## Continuous checks

[Project checks](https://github.com/kevinrouse/nba-game-predictor/actions/workflows/checks.yml) runs on pushes to `main`, pull requests and manual dispatch:

- **Validate the published project:** install dependencies, lint, verify exact release files and notebook output status, and run the test suite.
- **Rebuild the analysis from source:** rebuild all models and reports, execute the notebooks, run tests and compare the results with the committed tables and selection.

Validation runs on Linux; strict full reproduction uses macOS 14 on ARM64, matching the original training architecture. Both use Python 3.11 and read-only repository permissions. Action versions are pinned to commit hashes. The rebuild saves notebooks, figures and analytical tables as a seven-day workflow artifact, including diagnostics when verification fails. This makes discrepancies inspectable; it does not commit generated changes back to the repository.

## What the cross-platform check found

A complete Linux rebuild executed all notebooks and passed the tests. Its selected logistic model and published forecast results matched the reference, but refitting the four random-forest candidates produced small development-prediction differences. The largest probability difference was 0.004079 (0.41 percentage points). Two configurations changed one favorite each across all six validation seasons; the other two changed none. The largest change in a forest's mean validation loss was 0.000023. The selected model did not change.

These are measured differences between the original ARM/macOS fit and the Linux/x86 refit, not evidence of a data-boundary failure. They show why a fixed random seed alone is insufficient for strict cross-platform equality of every fitted tree. The strict reproduction job therefore uses the original platform family, while Linux independently verifies release files, predictions, feature boundaries and app behavior. The numerical tolerances have not been expanded to absorb the forest differences. A Linux refit can complete the pipeline yet fail the deliberately strict comparison against the original candidate predictions.

The check also exposed an unrelated presentation issue: equal zero coefficients could appear in a different row order. Coefficient reports now break importance ties by feature name, making their order deterministic without changing any coefficient or probability.

For the interactive app, see the [deployment guide](deployment.md). The app uses cached local tables, invalidating table reads when their file modification times change after a rebuild.
