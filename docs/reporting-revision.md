# Analysis and presentation revision

The original model selection and 2023–24 predictions were completed before this revision. This revision expands the analytical notebooks and dashboard, adds worked feature calculations, makes the candidate-model comparisons visible, and evaluates a requested basketball decision rule.

## Better-record decision rule

Pick the team with the higher **season-to-date win percentage before tipoff**. Teams can have played different numbers of games, so total wins alone do not define a better record. Break exact percentage ties in favor of the designated home team. The rule has no fitted parameters and does not use the current outcome or final season standings.

The calculation uses integer prior wins and games, comparing cross-products to avoid floating-point tie errors. Evaluate the same eligible games as the frozen model. Compare accuracy, correct picks, agreement/disagreement and a paired calendar-week bootstrap interval for the accuracy difference. This rule chooses a winner; it does not provide a probability, so no artificial 0/1 probabilities or log-loss comparison are assigned to it. The existing win-percentage logistic baseline remains the corresponding probability benchmark.

This rule was added **after the original test results were inspected**. Its comparison is retrospective reporting, not another untouched test or a model-selection opportunity. The tie convention is fixed here before calculating the new baseline. No features, hyperparameters, coefficients, calibration or thresholds of the selected model are changed. All original probability predictions must reproduce exactly within their saved precision.

## Connected analytical story

1. **Data and EDA:** establish the unit of observation and coverage, audit real defects, investigate records, scoring efficiency, rest and player roles on development seasons, and connect each observation to a candidate predictor and a caveat.
2. **Pregame features:** work through an actual game from raw opponent joins to shifted totals, rolling windows, role-weighted player summaries and a signed matchup row. Show boundary tests, missing-history coverage and the train-only preprocessing step.
3. **Modeling:** show the chronological folds, simple decision/probability baselines, exact model configurations, family comparisons, matched-input comparisons, regularization paths, training gaps, season variation and the unchanged selection decision.
4. **Interpretation:** evaluate the frozen model and record rule on identical games, quantify paired uncertainty, assess calibration/ranking, trace a probability into its additive inputs, examine both successful and failed disagreements, and state the remaining limitations.

The Streamlit app follows the same sequence, with links to the corresponding notebook. New exploratory charts use development seasons unless explicitly labeled final-test diagnostics. Descriptive full-season summaries are never used as pregame predictors. Additional plots explain the implemented model; they do not retroactively select new inputs using the test season.
