# 2026 congressional election prediction

Frozen forecast date: **November 2, 2026**. The entry covers all 435 House seats and 35 Senate elections. The objective is to select the most probable named winner, without optimizing against the Kalshi benchmark.

## Reproduce

Use Python 3.12:

```bash
python -m pip install -r requirements.txt
python reproduce.py
```

This produces only `predictions/full_congress_FINAL.csv` and `predictions/senate_FINAL.csv`, matching the supplied templates. All calculations use the included frozen inputs and fitted models; no internet connection is required after installing dependencies. Intermediate files are temporary and removed automatically. To reproduce the same historical training fit as well, run `python reproduce.py --refit`. To select an output directory, add `--output-dir /path/to/folder`.

## Model

House and Senate use separate, fixed procedures across election years. House uses a regularized neural network with hidden layers (32, 16), L2 penalty 100 and 30 predictors. Senate averages the predicted logits/residuals of two networks with hidden layers (16, 8), L2 penalty 30 and 9 or 7 predictors. The target is the Democratic two-party vote-share logit minus the prior presidential Democratic vote-share logit. The preceding-24-year training rule gives 2002-2024 outcomes for this forecast. Training-set median imputation, missingness indicators and standardization precede fitting. Earlier chronological out-of-fold errors calibrate marginal Student-t uncertainty with five degrees of freedom. Exact feature lists and settings are in `model_spec.json`.

House inputs include historical presidential, House, Senate and governor voting, incumbency, presidential alignment, district history and shrunk candidate overperformance. The original midterm-effect feature group is excluded. Senate uses compact historical voting and incumbency inputs; one component includes midterm terms and the other omits them. Current polling, sentiment, economic series, expert ratings and market odds are not structural-model inputs.

The final entry retains 439 reviewed structural predictions, including 14 candidate-history identity corrections and six applications outside the evaluated contest domain. These two-party probabilities approximate named-candidate probabilities by assuming the leading pair exhausts winning outcomes.

The remaining 31 predictions follow `inputs/completion_policy.json`: 12 use completed primary or special elections with a shrunk margin and judgmental change uncertainty; 17 use explicit ballot/incumbency/partisan-baseline judgments; and two open Louisiana seats use a party prior with equal-weight and square-root-fundraising allocation across ballot candidates. These completion probabilities are not historically calibrated. Independent party labels are preserved, same-party races select a person, and Louisiana forecasts concern the eventual seat winner after any required runoff. Fundraising is used only in the two Louisiana completion cases.

## Data

- `inputs/training_features.csv`, `history_panel.csv`, `calibration_raw.csv` and `map_change_years.json` preserve the project's historical training, lag-construction and calibration snapshots. These are prepared model inputs from the existing project, including its corroborated corrections and misdated NC-9 quarantine; they are not claimed to be a fresh reconstruction of every original election return.
- `inputs/downballot_pres2024_maps2026.csv` contains [The Downballot's](https://www.the-downballot.com/p/the-downballots-calculations-of-presidential) 2024 presidential totals on 2026 congressional maps. Unverified district-aligned history is left missing and handled by the frozen model pipeline.
- `inputs/electindex_races.csv` is the [ElectIndex roster snapshot](https://github.com/ElectIndex/26_us_forecast_data/blob/main/races.csv). Its candidate/office information is used, not its forecast probabilities.
- `inputs/ballot_candidates.csv` and `ballot_name_crosswalk.csv` contain the saved candidate registry and identity links. Each candidate record retains its source URL and evidence level; registry inclusion does not imply that every ballot was exhaustively certified.
- `inputs/official/ca_primary_house_2026.xlsx` is the [California June 2 primary Statement of Vote](https://elections.cdn.sos.ca.gov/sov/2026-primary/sov/76-us-rep.xlsx). `ca14_special.html` records the [August 18 CA-14 special election](https://www.sos.ca.gov/elections/prior-elections/special-elections/2026-cd14/official-canvass). `ak_primary_summary.pdf` records the [official August 18 Alaska primary](https://www.elections.alaska.gov/enr/26PRIM/results/ElectionSummaryReportRPT.pdf).
- `inputs/deepsyte_candidates_2026.csv` is the saved [DeepSyte](https://deepsyte.app/data) transcription of FEC fundraising. Only ballot-matched LA-05/06 candidates are used. Selected-candidate totals were corroborated against FEC records for [Echols](https://www.fec.gov/data/candidate/H6LA05242/?cycle=2026) and [Miguez](https://www.fec.gov/data/candidate/H6LA05234/?cycle=2026). Report-period end dates precede this forecast.
- The two `*_template.csv` files are unchanged organizer templates. Retrieval occurred November 2; these inputs are not represented as a November 1 snapshot.

## Evidence and limitations

On the selected 2014-2024 retrospective development cohort, the structural models achieved House accuracy 96.06% (2,195/2,285; Brier 0.0310) and Senate accuracy 93.22% (165/177; Brier 0.0521). Worst-year accuracy was in 2018: House 93.26% and Senate 83.33%. These repeatedly inspected cycles are not untouched tests. Their scores do not validate the full 2026 hybrid entry, its completion rules or its current missing-input pattern.

Incomplete district histories, source vintages and candidate linkage, unmodeled late changes and minor-party outcomes remain risks. No joint seat/control distribution is claimed. Following [Ebanks, Katz and King's warning about predictive overconfidence](https://gking.harvard.edu/files/gking/files/10k.pdf), high winner accuracy and a Student-t layer are not treated as proof of full distributional adequacy.

The submission probability is for the named pick, not a Kalshi price. The organizer benchmark and actual outcomes are needed to calculate competition points later; the code does not require them to generate the entry.
