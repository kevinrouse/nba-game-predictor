# Results and review

## Basketball prediction

The project estimates a designated-home win probability before a historical NBA regular-season game. It is separate from the archived playoff-series project. Series length, bracket simulations and a playoff transfer test are outside this implementation's scope.

There are 11,979 regular-season games in the supplied files and 10,418 after requiring ten previous games for both teams. The raw dimensions are 27,144 × 41 for team data and 434,797 × 59 for player data. The final fit uses 9,344 games through 2022–23. The reserved 2023–24 season supplies 1,074 eligible games; 156 opening games are excluded by the fixed warmup rule.

## What survived development

The initial training set is 2014–15 through 2016–17. Six expanding windows validate 2017–18 through 2022–23, always using earlier seasons for training. The 74 candidate configurations and their scores are stored in `artifacts/tables/candidate_configs.json` and `cv_summary.csv`.

Mean validation-season log loss:

| Approach | Loss |
|---|---:|
| Home win rate | 0.684844 |
| Win-percentage baseline | 0.634187 |
| Net-rating baseline | 0.630585 |
| Best team-only model | 0.628126 |
| Team + five-game form | 0.627567 |
| Team + ten-game form | 0.626960 |
| Team + twenty-game form | 0.626720 |
| Team + player histories | 0.624979 |
| Team + twenty-game form + players | 0.624380 |
| Best random forest | 0.630754 |
| Best gradient boosting | 0.628728 |

The selection rule, fixed in `evaluation-plan.md`, prefers fewer inputs within 0.001 of the best score. That selects **L1 team + player histories, C=0.1**, with 22 candidate input differences. The final fit has 19 nonzero coefficients; top-five minute share, top-seven minute share and minute continuity shrink to zero. The extra recent-team-form inputs are excluded because their validation advantage is smaller than the tolerance. The six-input compact candidate is simpler but worse at 0.626962, outside the tolerance. Unregularized team + players is 0.626032; the best L2 version is 0.625605. L1's edge is small and dataset-specific.

The selected settings are L1 ratio 1, C 0.1, liblinear, 3,000 maximum iterations and seed 78. Median imputation and standardization are inside the fold-fitted pipeline. Home court is represented by the intercept. There are no arbitrary interaction terms. The [feature definitions](data-and-features.md) enumerate every measure and the input exclusions.

Player histories improve the matching L1 team-only comparison in five of six validation years. They worsen 2022–23 by 0.00090. Those comparisons were used to make choices, so they do not independently establish the isolated benefit of players in future seasons.

## The reserved season

The added better-record decision rule picks 688 of 1,074 games correctly (64.06%); the frozen model picks 723 (67.32%). Exact percentage ties favor home, affecting 18 games. The rules disagree on 163 games: the model is correct on 99 and the record rule on 64. The paired week-block interval for the 3.26-percentage-point accuracy gain is +1.38 to +5.31 points, using 2,000 draws of the same 22 calendar weeks. This comparison was added after the original evaluation and does not select or modify the model. See the [reporting revision](reporting-revision.md).

The hard winner rule is not a probability estimate. Its comparison is accuracy-only; the original fitted probability baselines remain below.

| Model | Log loss | Brier | AUC | Accuracy |
|---|---:|---:|---:|---:|
| Selected model | 0.602463 | 0.207969 | 0.734480 | 67.32% |
| Home win rate | 0.692934 | 0.249875 | 0.500000 | 53.72% |
| Win-percentage baseline | 0.612245 | 0.212894 | 0.720800 | 64.90% |
| Net-rating baseline | 0.611991 | 0.212711 | 0.720622 | 64.53% |

The selected model's 95% test log-loss interval is [0.575112, 0.628364]. The paired difference from net rating is −0.009528, with interval [−0.015518, −0.003192]. Brier difference is −0.004742, interval [−0.007419, −0.002023]. These use 2,000 paired resamples of 22 calendar-week blocks, recomputing a game-weighted mean within each draw. They capture some local schedule dependence but not dependence across week boundaries, model-selection uncertainty, or between-season variation.

Predicted home wins average 56.08%, versus 53.72% observed. Calibration slope is 1.0583, intercept −0.1259, and ten-bin absolute calibration error 0.0367. Bin counts range from 17 to 199. Wilson intervals are descriptive within-bin intervals, not a complete dependence adjustment. Neither the test diagnostics nor error analysis changes the frozen model.

## Basketball observations

- Development home win rate is 57.16%; it varies across seasons.
- Team-seasons average 60.21% of recent minutes in their top five players. Minnesota 2017–18 has the lowest average minute-share volatility and Boston 2020–21 the highest among development team-seasons. These are data descriptions, not injury narratives.
- Season net rating and prior-ten-game net rating correlate at 0.794. Recent strength is substantially overlapping information.
- The biggest test miss is Detroit at Dallas, April 12, 2024: Dallas had a 91.53% prediction and lost 89–107. Charlotte at Minnesota on January 22 is another major miss, with Minnesota at 91.16% and losing 125–128.
- Toronto won 27.78% of eligible games against a 41.60% predicted rate. The Lakers won 59.15% against 52.20% predicted. Team residuals are descriptive and do not identify the reason for the gap.
- `player_prediction_changes.csv` shows the largest development probability changes between separately fitted team and team+player models. These are not causal player effects.

## Critical review

**Statistical quality.** Inputs end before game dates, mutation/truncation tests verify the boundary, and learned preprocessing uses training folds only. The final season does not select the model. Model-selection scores can be optimistic after trying 74 configurations; the held-out season is the independent result. One season limits generalization claims.

**Predictive quality.** The final model improves on each baseline in the reserved season, and player gains occur in five of six development seasons. It still overestimates home advantage somewhat. A strong forest training score does not translate into better later-season probabilities: the combined forest with a minimum of 20 samples per leaf has training loss 0.5246 versus validation 0.6321. The selected logistic model has 0.6036 versus 0.6250. That supports retaining the simpler fit.

**Code and reproducibility.** Direct functions handle the joins, histories and model comparison; no framework or class hierarchy is needed. Only source data and prior rows determine the features. Compressed sources, derived tables and model artifacts are included, with separate minimal runtime and analysis requirements. Seeds and relative paths support reproduction.

**Communication.** The public story is basketball prediction. The app separates probabilities from results and validation from the final test. Notebook findings report small effects cautiously. Correlated coefficients are not presented as causal effects; weak sign changes are not converted into basketball claims.

The four expanded notebooks now develop one sequence: basketball questions, reconstructing prior information, comparing model families, and interpreting frozen predictions. Worked calculations reconcile team ratios, player contributions and a full predicted probability to the saved artifacts. Model comparison includes the best searched family settings and a common-input comparison, with actual last-fold fits, regularization paths, feature additions and season variation. Error analysis includes successful and unsuccessful departures from the record rule, calibration counts, ROC curves, team residuals and season-stage diagnostics. These additions explain the original fit without changing it.

**Scope.** The site offers histories and held-out game predictions, not a live injury-aware service. There is no evidence to support a 2024 playoff simulation from these files, and the revised project does not need one. Opening games, unobserved current roster information and neutral sites remain limitations.

## Robustness and operational review

A new decision audit selects each upcoming development season using only preceding validation-season scores. It selects L1 team + player histories, C=.1, in all five replays from 2018–19 through 2022–23. Mean next-season loss is 0.628677, versus 0.633794 for net rating on those same seasons; the model leads in all five. The candidate library was designed during the original development process, so this is a retrospective stability check, not independent nested validation.

Paired resampling with one-, two- and four-week nonoverlapping calendar blocks gives accuracy-gain intervals of [1.38, 5.31], [1.18, 5.13] and [1.12, 5.31] percentage points. Log-loss differences also remain negative under all three choices. The longest blocks leave only six resampling units; this does not remove between-season uncertainty. The original headline interval and frozen forecasts are unchanged.

The app now supports direct historical-game URLs and an audit export shared with a command-line interface. Each record has the model identifier, all source-history cutoffs, raw input values, exact log-odds contributions and a separate observed-result field. Invalid game IDs do not generate predictions. The error explorer opens the corresponding game directly. Table caches include modification times so a local rebuild does not leave stale results in the app.

A committed release manifest detects changed or missing source/model artifacts. GitHub Actions validates the published project on Linux and rebuilds the complete analysis on the original macOS/ARM architecture, comparing every numerical CSV and the model selection with Git HEAD. A full Linux experiment passed the tests and matched the selected model, but found small differences in refitted random-forest development predictions; the [reproducibility guide](reproducibility.md) records their size and the reference-platform decision. Coefficient reports now use a deterministic feature-name tie-break for equal weights. Numerical tolerances remain strict; [workflow results](https://github.com/kevinrouse/nba-game-predictor/actions/workflows/checks.yml) provide the current check status.

## Verification

Twenty-three tests pass, covering source reconciliation, unique matchups, strictly earlier dates, current/future mutation invariance, truncation invariance, historical role weights, removal of a current roster, frozen model/preprocessing integrity, probability sums, selection logic, bootstrap pairing, exact record-rule ties and same-game comparisons, reconstruction of probability contributions and worked player inputs, selection replay boundaries, longer resampling blocks, release-file tampering, downloadable forecast reconstruction, game URLs, and all eleven app sections with representative selector combinations. The four notebooks execute with stored outputs and no cell errors. Static checks pass. The local Streamlit app renders on desktop and at a 390-pixel viewport without horizontal body overflow.

A fresh local clone of this revision created a new Python 3.11 environment, installed `requirements-dev.txt`, verified the twelve release files and passed lint checks. The full pipeline rebuilt all 74 candidate configurations, executed all four notebooks and passed all twenty-three tests. `verify_reproduction` matched all 45 published analytical tables and the original model selection with exact identifier/count checks and the documented tolerances for floating-point values. The exported Detroit–Dallas forecast matched the published JSON exactly. The notebooks contain 156 cells, including 58 executed code cells, with no saved cell errors or warnings. Browser navigation through all eleven sections, direct game URLs, forecast downloads and invalid-link recovery produced no JavaScript errors.

Checksums of the original fitted model, selection file and test-prediction CSV remain identical through both reporting reviews. The new narrative and record comparison therefore explain the same frozen predictions rather than a test-driven replacement model.

Local notes, environments, credentials, caches, original R session files, editor files and notebook checkpoints are excluded from Git. The original playoff repository remains unchanged. Cloud deployment still requires an authenticated Streamlit session; see [deployment status](deployment.md).
