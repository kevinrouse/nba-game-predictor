# What a forecast means

The forecast is a probability that the **designated home team wins a historical regular-season game**, calculated from information ending strictly before that game's date. It is not a current injury assessment, a betting recommendation or a preseason projection. The app reads published predictions; changing a selector never fits a model.

## Coverage and information boundary

- Both teams must have at least ten previous regular-season games in that season.
- The fitted model uses development seasons ending in 2022–23. Parameters stay fixed throughout the 2023–24 test.
- Team and player histories update after completed games. Current-game points, minutes, starters and participation flags are excluded.
- A previous roster is historical membership evidence; it can miss a trade or arrival between games.
- Dates, rather than tipoff timestamps, define the boundary. Two games for the same team on one date are rejected because their order cannot be established safely.
- Probabilities describe the modeled population. They are not validated for opening games, playoff games, live games or arbitrary hypothetical rosters.

## Audit record

Open a historical game, copy the URL from the address bar, or download its JSON record. A link has the form `?game=22301181`; unknown IDs fall back to the explorer rather than fabricating a prediction. The [Detroit–Dallas example](../artifacts/examples/detroit-at-dallas.json) is an actual export.

The same record is available without the app:

```bash
python -m scripts.export_forecast 22301181 --output forecast.json
```

Omit `--output` to write JSON to standard output. Unknown or uncovered game IDs fail with an explanation; no hypothetical probability is generated.

| Field | Meaning |
|---|---|
| `model_id` | Identifier derived from the SHA-256 of the published fitted-model bundle |
| `model_name` | Selected algorithm, input set and regularization setting |
| `fitted_through_season_start_year` | Starting-year label for the last training season; 2022 means 2022–23 |
| `game` | Game identifier, date and home/away team codes |
| `pregame_forecast` | Probabilities, both winner picks, prior records, history cutoffs and input contributions |
| `observed_result` | Postgame scores and outcome, explicitly separated from prediction inputs |

Every input includes its home value, away value, signed difference and exact contribution to home-win log odds. The contributions are calculated after the fitted median imputation and standardization. A missing raw input is represented by JSON `null`; it is not silently replaced in the exported raw values.

The probability can be reconstructed as:

```text
score = intercept_log_odds + sum(input log_odds_contribution)
home_probability = 1 / (1 + exp(-score))
away_probability = 1 - home_probability
```

Contributions describe how this model computes a forecast. They do not identify a causal reason for the game result, and correlated inputs can redistribute their coefficients. The favorite threshold remains 0.5; exact pregame-record ties favor home in the separate record rule.

## Evidence and limitations

The selected model has 0.6025 log loss and 67.3% winner accuracy on 1,074 eligible test games. It beats the higher-record rule's 64.1% accuracy and the fitted net-rating baseline's 0.6120 loss. The hard record comparison was added retrospectively. Calibration remains imperfect: predicted home wins average 56.1% against 53.7% observed.

One-, two- and four-week bootstrap blocks preserve the direction of the observed gain, but the longest blocks leave only six units to resample. These checks do not establish future-season performance or remove missing current availability information. A new model or calibration change would require a new development decision and later evaluation data; inspecting a test miss does not authorize retuning on it.

## Release identity

[`release.json`](../artifacts/models/release.json) records the source/model artifact hashes, selected inputs and numerical-library versions. [`scripts.release`](../scripts/release.py) verifies those files without loading or replacing a changed model. A deliberately reviewed release can update the manifest with `python -m scripts.release --write`; normal verification never regenerates it automatically. See [reproducibility](reproducibility.md) for rebuilding and comparing numerical outputs.
