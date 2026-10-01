# Data and pregame histories

## Source audit

The team file has 27,144 × 41 values; the player file has 434,797 × 59. Regular seasons run from 2014–2023, using starting-year labels. Playoffs cover 2004–2022 and do not enter this model. There are no supplied 2024 postseason games.

No duplicate game-team or game-player keys, negative numeric values, shooting-accounting failures or player/team point mismatches were found. Each game has two opposing teams, one designated host and one winner. Player game, season and date fields reconcile with the team records.

Four missing names retain valid player identifiers and zero minutes. Four `missed=1` rows contain positive seconds: participation takes precedence in the historical absence calculation. Two players score with zero seconds; their points are preserved. Thirty-four team-game totals fall below 239 player-minutes, including 218.5 for both sides of game 21400342. Nothing is imputed into the clock; minute shares use recorded totals and pace is omitted. High totals reach 340 minutes and can reflect overtime; the files do not independently explain every clock anomaly.

Both compressed CSV files reproduce the supplied uncompressed bytes exactly. SHA-256 digests appear in `artifacts/tables/audit.json`. The original R Markdown/PDF are archived in `original/`; historical R paths and instructions are not runtime dependencies.

## Team histories

Grouping uses season and team, sorted by date. Counts are shifted by one game before cumulative or rolling sums. The implementation rejects two same-date games for a team because it cannot order their tipoffs safely.

| Measure | Numerator / denominator from prior games |
|---|---|
| Offensive rating | 100 × points / offensive possessions |
| Defensive rating | 100 × opponent points / opponent possessions |
| Net rating | Offensive minus defensive rating |
| Win percentage | Wins / games |
| Effective FG percentage | (FG made + 0.5 × threes made) / FG attempted |
| Opponent effective FG percentage | Same formula from opponent offense |
| Turnover rate | Turnovers / (shot attempts + turnovers) |
| Offensive rebounding | Offensive rebounds / rebound chances |
| Defensive rebounding | Opponent-row defensive rebounds / opponent-row rebound chances |
| Free-throw rate | FT attempted / FG attempted |
| Point differential | (Points − opponent points) / games |

The supplied `rebdefensive` on an offensive row is what the defending opponent collected. Using it as the offense's own defensive rebounds would be wrong. Source `shotattempts` includes shooting-foul attempts and is distinct from FG attempts.

Five-, ten- and twenty-game net rating, shooting and win percentage are computed. The model comparison tests net rating and shooting for each window; recent win percentage is available for exploration but omitted from the candidate models to avoid another closely related strength measure. Offensive/defensive rating and point differential are descriptive; net rating is the selected strength summary.

Rest is days since the previous game minus one, capped to [0, 7]. A back-to-back is a one-day gap. Schedule density counts earlier games within the previous seven calendar days, including the seventh day. All comparisons are home minus away; home status is encoded by orientation and the fitted intercept.

## Player histories

No current-game roster is consulted. All player matrices are sliced strictly before the current game. Columns for players only seen in later games contain no earlier values and carry zero weight; deleting future rows verifies their invariance.

1. Use the previous game's roster as the set of known players.
2. Over the previous twenty team games, compute each known player's mean minutes in games with positive playing time. A player with no prior appearance has zero role weight.
3. Normalize these roles to sum to one.
4. Use previous-five-game records to summarize player scoring, participation and historical absences.
5. Aggregate with the prior role weights, then compare home and away teams.

| Player / rotation measure | Definition |
|---|---|
| Top-five / top-seven minute share | Largest five/seven player totals divided by all recorded minutes in the prior ten team games |
| Rotation size | Average count of players reaching ten minutes per game over the prior ten games |
| Minute continuity | Sum of the smaller share per player between prior games 1–5 and 6–10; 1 means identical minute distributions |
| Starter continuity | Mean fraction of five starting slots retained across consecutive prior games, using up to five transitions |
| Rotation volatility | Mean half-L1 distance between consecutive minute-share distributions over the prior ten games |
| Scoring concentration | Sum of squared player shares of prior-ten-game points |
| Shooting-role concentration | Same concentration using shot attempts + turnovers |
| Role-weighted scoring efficiency | Prior-five-game shot-attempt points / shot attempts per player, weighted by normal role; weights renormalized among players with attempts |
| Role-weighted historical absence | Prior-five-game `missed` fraction among each player's rostered records, weighted by normal role; positive seconds override `missed` |
| Role-weighted participation | Fraction of prior-five-game rostered records with positive seconds, weighted by normal role |

These are participation histories, not confirmed injury status. Previous roster membership can lag a trade or arrival. A newly rostered player without earlier team appearances has no role estimate. On-court team totals in the player file are not added together.

## Coverage and verification

One home/away pair becomes one prediction row. Both teams must have ten prior regular-season games. This covers 10,418 of 11,979 games; 1,074 of 1,230 in the final season. Explorers retain all games. No prediction is silently filled in for an excluded opener.

`history_end` and `player_history_end` must precede `gamedate` for both sides. Tests additionally modify current/future statistics, delete future records, remove the current roster, check unequal prior-role weights, and regenerate published probabilities from the frozen model. Scores and outcomes remain metadata excluded from every candidate input list.

## Worked calculations and reporting rule

[Notebook 2](../notebooks/02_pregame_features.ipynb) reconstructs Clippers at Thunder on December 15, 2018, from earlier raw rows. Its team ratios, rolling windows, weighted player summaries and continuity calculation reconcile to the saved features. The dashboard's Feature workshop exposes the same example and links each candidate to its basketball motivation, definition and caveat.

The added better-record decision rule compares the two pregame win percentages, with home favored on exact ties. Prior integer wins are recovered from win percentage × prior games and compared by cross-multiplication, so rounded CSV values cannot turn equivalent records into false inequalities. It never consults the current outcome or final standings. This is a winner-only reporting benchmark, distinct from the original fitted win-percentage probability baseline; see the [reporting revision](reporting-revision.md).
