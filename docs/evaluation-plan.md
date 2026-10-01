# Evaluation plan

Fixed before comparing game models. Season numbers are starting years.

- Use supplied regular-season games (2014–2023). One row per game; target is a designated-home win.
- Require both teams to have ten earlier regular-season games that season. Opening games remain in the explorers and provide history, but are outside prediction coverage.
- First training window: 2014–2016. Validate successively on complete seasons 2017 through 2022, always training on earlier seasons only.
- Reserve the complete 2023–24 season (2023) for one final evaluation of the selected model. No test-season model fitting, feature selection, calibration fitting, or threshold selection.
- Features may update from completed games within a validation/test season. Fitted model parameters stay frozen throughout that season. This simulates a sequence of pregame predictions, not a preseason forecast.
- Compare historical home win rate, win-percentage and net-rating logistic baselines, unregularized/L1/L2 logistic regression, random forest, and gradient boosting.
- Compare season strength, 5/10/20-game recent form, historical roster information, and their combinations. Use identical eligible games for all candidates.
- Primary selection score: mean season validation log loss. For candidates within 0.001 of the best score, prefer fewer features, then logistic regression, then lower validation log loss. Report Brier score, AUC, accuracy, season variation, calibration and training/validation gap as diagnostics.
- Fit all learned preprocessing inside each training fold. Inspect correlations and coefficient stability only on development data. Do not select features from test importance.
- Report the test result alongside all three prespecified baselines, a paired week-block bootstrap interval, and calibration bin counts. One test season cannot establish season-to-season uncertainty.
- Same-game roster, minutes, starter status, production and `missed` are never prediction inputs. No timestamp establishes that same-game `missed` was available before tipoff.
- Every team's feature history must end strictly before its predicted game date. Mutation and truncation tests will verify this for team and player inputs.

No external basketball data, travel information or betting prices are used. Playoff transfer is optional and outside this project's primary evaluation.
