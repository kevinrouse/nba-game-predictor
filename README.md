# NBA Game Predictor

[![Project checks](https://github.com/kevinrouse/nba-game-predictor/actions/workflows/checks.yml/badge.svg)](https://github.com/kevinrouse/nba-game-predictor/actions/workflows/checks.yml)

**Can team and player histories improve on picking the team with the better record?**

I built a pregame NBA win-probability model to investigate what records leave out: scoring margins, changing rotations, recent player participation and rest. The project follows those basketball questions through an audited dataset, chronological model comparisons and an app for examining individual predictions.

On **1,074 eligible games in 2023–24**, the frozen model picks **67.3% correctly**, versus **64.1%** for the better-record rule: **35 additional correct picks**. The more useful result is an inspectable probability, with its uncertainty and mistakes visible alongside the headline.

[Read the analysis](#follow-the-analysis) · [Run Streamlit](#run-the-project) · [Inspect a forecast](artifacts/examples/detroit-at-dallas.json) · [Reproduction checks](docs/reproducibility.md)

## The result, and what it means

The final model is **L1 logistic regression, C=0.1**, fitted through 2022–23. Its 22 candidate inputs have 19 nonzero coefficients. All comparisons below use the same games.

| Approach | Winner accuracy ↑ | Log loss ↓ |
|---|---:|---:|
| Historical home-win rate | 53.7% | 0.6929 |
| Higher pregame winning percentage; ties favor home | 64.1% | — |
| Fitted win-percentage logistic baseline | 64.9% | 0.6122 |
| Fitted net-rating logistic baseline | 64.5% | 0.6120 |
| **Team and player histories** | **67.3%** | **0.6025** |

The hard record rule chooses a side without supplying a probability, so it receives no artificial log-loss score. The fitted record baseline separately learns how winning-percentage differences and home advantage translate into probabilities. The selected model's Brier score is **0.2080** and ROC AUC **0.7345**.

The model and record rule disagree on 163 games: the model is correct on 99, the rule on 64. The **3.26-percentage-point accuracy gain** has a paired week-block 95% interval of **+1.38 to +5.31 points**. The log-loss difference against net rating is **−0.0095**, with interval **−0.0155 to −0.0032**. These describe uncertainty within one season, not expected performance across future seasons.

The record rule and subsequent robustness diagnostics were added after the original test evaluation. They are explicitly [retrospective reporting](docs/reporting-revision.md); the original fitted model, favorite threshold and stored probabilities remain unchanged.

## Follow the analysis

| Question | Evidence |
|---|---|
| [What might the record miss?](notebooks/01_data_and_eda.ipynb) | Source-data audit; season strength, shooting, possession control, recent form, player roles and schedule; a basketball reason and caveat for each candidate |
| [Could these inputs have been known before tipoff?](notebooks/02_pregame_features.ipynb) | A worked Clippers–Thunder game: opponent joins, ratios of earlier totals, shifted windows, individual player weights and mutation tests for leakage |
| [Which model earns the added detail?](notebooks/03_modeling.ipynb) | Actual fits of ordinary/L1/L2 logistic regression, random forest and boosting; matched-input comparisons, six chronological validation seasons, regularization and a replay of earlier model choices |
| [What do the predictions get right and miss?](notebooks/04_model_interpretation.ipynb) | Frozen test results, calibration, ranking, paired uncertainty, block-length sensitivity, exact probability contributions and both sides of model–record disagreements |

A concrete place to start is the feature notebook's **Clippers at Thunder, December 15, 2018** example. OKC had 27 earlier games and LAC had 28. The notebook reconstructs the actual team totals and individual player contributions, then reconciles them to the saved model inputs.

## Decisions that shaped the model

- **Respect the information boundary.** Team counts are shifted before cumulative or rolling calculations. Player membership comes from the previous game's roster; role weights come from earlier appearances. Current-game minutes, starters and `missed` flags never enter a forecast. Tests mutate or delete later records and require earlier features to stay fixed.
- **Compare against strong simple explanations.** Record and net rating are competitive baselines. Seventy-four configurations test a limited set of basketball features and model families. Forests improve training fit without improving later-season probabilities; regularized logistic models perform better in this search.
- **Choose on later seasons, not randomly mixed rows.** Initial training covers 2014–15 through 2016–17. Six expanding windows validate 2017–18 through 2022–23; preprocessing is fitted within each training fold. Model selection minimizes mean season log loss, with a fixed simplicity preference within 0.001 of the best score.
- **Keep a small gain in perspective.** Adding player histories lowers validation loss from 0.6281 to 0.6250 and helps five of six seasons. The lowest-loss model adds twenty-game team form, but its 0.00060 advantage is below the tolerance, so the smaller team-plus-player model is selected.
- **Audit the decision as well as its result.** Replaying selection using only preceding validation scores chooses the same L1 model in all five later development seasons. One-, two- and four-week resampling all retain a positive accuracy-gain interval. These are retrospective checks with explicit limits, including the historically developed candidate library and only six four-week blocks.

## Explore the games

Streamlit follows the same sequence as the notebooks, then opens into team, roster and game explorers. Compare season strength with five-, ten- and twenty-game form; inspect historical role weights; trace a prediction into its inputs; or open a confident mistake.

Every historical forecast has a shareable game URL and a downloadable JSON audit record. The record includes the model identifier, prior records, source-history cutoffs, all 22 input contributions and a separately labeled observed outcome. The error explorer links directly to these game views.

![The NBA Game Predictor dashboard](artifacts/figures/dashboard.png)

One revealing miss is **Detroit at Dallas, April 12, 2024**: Dallas was a **91.5%** favorite and lost **89–107**. [Inspect that forecast record](artifacts/examples/detroit-at-dallas.json), or open `http://localhost:8503/?game=22301181` after starting the app below. The histories show how the model made the forecast; they do not establish an injury or tactical explanation for the result.

## Data and limits

The supplied files contain **27,144 team rows × 41 columns** and **434,797 player rows × 59 columns**. There are **11,979 regular-season games**, covering 2014–15 through 2023–24. Each game has two offensive team rows; each player row represents a rostered player, including nonparticipants. Defensive measures come from the opponent's offensive row. Overlapping player on-court totals are not additive team statistics.

Both teams need ten earlier games that season, leaving **10,418 eligible games**: **9,344 development** and **1,074 final-test** games. Opening games contribute history but receive no model prediction. The original CSV bytes are preserved in lossless `.csv.gz` files in `Data/`; the [data and feature audit](docs/data-and-features.md) documents source inconsistencies and the decisions they require. No external basketball data, injury feeds or betting prices are used.

This is historical pregame forecasting, not a live service or preseason forecast. Histories update after completed games; model parameters stay fixed. Current availability and same-day roster changes are unknown. Neutral sites are not separately identified, and clock anomalies prevent a reliable pace measure. The model predicts 56.1% home wins against 53.7% observed, so calibration is imperfect. One test season cannot establish future reliability.

The [forecast contract](docs/forecast-contract.md) states exactly what a prediction contains and excludes. The [evaluation plan](docs/evaluation-plan.md) preserves the original selection protocol.

## Run the project

Python **3.11** is the tested runtime. Source data, derived tables and frozen model artifacts are included; dashboard controls do not retrain models.

```bash
git clone https://github.com/kevinrouse/nba-game-predictor.git
cd nba-game-predictor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.port 8503
```

To verify the published snapshot and rebuild the analysis:

```bash
pip install -r requirements-dev.txt
python -m scripts.release          # Verify committed data/model hashes and notebook outputs
python -m pytest -q               # Information boundaries, selection, exports and app behavior
python -m scripts.reproduce       # Rebuild data, all candidates, reports and executed notebooks
python -m scripts.verify_reproduction  # Compare rebuilt tables and selection with Git HEAD
```

[GitHub Actions](https://github.com/kevinrouse/nba-game-predictor/actions/workflows/checks.yml) runs separate validation and full-reproduction jobs. The [reproducibility guide](docs/reproducibility.md) explains exact file integrity versus numerical reproduction across platforms. [Streamlit deployment instructions](docs/deployment.md) are available; no public Cloud URL is currently verified.

| Location | Contents |
|---|---|
| `src/` | Data validation, pregame histories, models, analytical checks and forecast exports |
| `scripts/` | Reproduction steps and release verification |
| `notebooks/` | Four executed analytical notebooks |
| `artifacts/` | Frozen model, predictions, comparisons, figures and example forecast |
| `Data/` | Losslessly compressed sources and supplied dictionary |
| `docs/` | Definitions, evaluation design, forecast contract, review and deployment |
| `original/` | Archived R Markdown/PDF; outside the Python pipeline |

The code is MIT-licensed. Supplied datasets and original assignment materials retain their existing rights.
