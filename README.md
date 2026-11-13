# Tabular Foundation Models for Credit Default at 700k Scale

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](#local-development)
[![License: MIT](https://img.shields.io/badge/code%20license-MIT-yellow.svg)](LICENSE)

`tfm-credit-benchmark` is a reproducible academic benchmark of TabPFN-3.5,
tuned LightGBM, and RealMLP on a real binary credit-default task. The central
experiment tests whether a tabular foundation model remains competitive with a
strong tuned gradient-boosting baseline when every model receives the same
training rows and the TabPFN context grows to 700,000 examples.

The workflow is designed for Google Colab and Google Drive. It starts with a
low-cost pilot, estimates the likely runtime and memory requirements of the
full TabPFN run, and blocks that run automatically unless the pilot passes.

> Project status: implementation is complete, but benchmark results have not
> yet been generated. This README will be extended with final tables and plots
> after the signed Colab runs finish.

## Research question

At a fixed training scale of 700,000 completed Lending Club loans, how does
TabPFN-3.5 compare with:

- an Optuna-tuned LightGBM model using native categorical features;
- default LightGBM with early stopping as a reference;
- RealMLP with its tuned defaults as a neural baseline?

The primary test metrics are ROC-AUC, PR-AUC, log loss, and Brier score.
Runtime, throughput, peak host RAM, and peak GPU memory are reported separately
so predictive quality is not hidden inside a composite efficiency score.

## Scientific contract

The benchmark enforces the following invariants in code and tests:

- The cohort contains exactly 900,000 rows and never exceeds 1,000,000 rows.
- Train, validation, and test contain 700,000, 100,000, and 100,000 rows.
- Every model at a given size receives exactly the same signed training rows.
- Pilot subsets are nested: 50k is inside 100k, which is inside 200k.
- Validation is used only for early stopping and hyperparameter selection.
- Each completed model-size run scores the locked test set once.
- TabPFN training context is never silently reduced after an OOM.
- The 700k TabPFN run requires a passing pilot tied to the same split manifest.
- Every result carries dataset, split, feature, code, library, and hardware provenance.

## Dataset and target

The code downloads
[`codesignal/lending-club-loan-accepted`](https://huggingface.co/datasets/codesignal/lending-club-loan-accepted)
directly with `huggingface_hub`. The mirror is distributed as CC0 and is based
on the Lending Club accepted-loan release.

The first run resolves the current dataset commit and then downloads that exact
revision. Its SHA256 is stored in the split manifest. Later runs reuse the
pinned revision and verify the checksum.

Only completed loans are retained: `Fully Paid`, `Charged Off`, and both
completed `Does not meet the credit policy` variants. The target is one for
charged-off loans and zero for fully paid loans.

## Origination-time features

Only information available at or before loan origination is used:

```text
loan_amnt, term, int_rate, installment, grade, sub_grade, emp_length,
home_ownership, annual_inc, verification_status, purpose, addr_state, dti,
delinq_2yrs, fico_range_low, fico_range_high, inq_last_6mths, open_acc,
pub_rec, revol_bal, revol_util, total_acc, mort_acc,
pub_rec_bankruptcies, application_type, credit_history_months
```

`credit_history_months` is derived from `issue_d` and `earliest_cr_line`.
Outcome, repayment, recovery, hardship, settlement, post-issuance credit, free
text, URL, ZIP, and identifier fields are explicitly excluded.

Before any model runs, the preparation notebook calculates a univariate
ROC-AUC for every feature. A value above 0.80 stops the workflow as a possible
leakage signal. The complete sorted table is saved to Drive.

## Models

### Tuned LightGBM

Optuna TPE minimizes validation log loss. Every trial uses validation early
stopping with up to 5,000 boosting rounds and patience 200. Studies are stored
in SQLite on Drive, so a disconnected Colab session can resume. The selected
parameters are refitted once before the locked test prediction.

### RealMLP

`pytabkit`'s `RealMLP_TD_Classifier` runs once at 700k on GPU. The signed
100,000-row validation set is supplied explicitly for early stopping and
selection. No test metric influences fitting.

### TabPFN-3.5

The benchmark pins `tabpfn==9.2.0` and uses the full V3.5 model with eight
estimators, cached fitting, memory-saving mode, and the complete training
context. Test queries begin in chunks of 512. A CUDA OOM halves only the query
chunk down to 128 and never changes the fitted context.

TabPFN-3.5 weights are governed by the
[TabPFN-3.5 Non-Commercial License](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE).
They are intended for non-commercial research, testing, and evaluation. Accept
the license through Prior Labs and provide your token only through a Colab
secret named `TABPFN_TOKEN`.

## Staged workflow

Run the notebooks in this order:

| Step | Notebook | Recommended runtime | Purpose |
|---:|---|---|---|
| 0 | `00_prepare_data.ipynb` | CPU High-RAM | Download, clean, cache, split, and check leakage |
| 1 | `01_pilot.ipynb` | A100 40 GB with High-RAM | Run 50k, 100k, and 200k pilots and project 700k cost |
| 2 | `02_lightgbm_700k.ipynb` | CPU High-RAM | Fit default and tuned LightGBM at 700k |
| 3 | `03_realmlp_700k.ipynb` | A100 with High-RAM | Fit RealMLP at 700k |
| 4 | `04_tabpfn35_700k.ipynb` | A100 40 GB or better with High-RAM | Run the pilot-gated full TabPFN experiment |
| 5 | `05_compare.ipynb` | CPU runtime | Verify artifacts, bootstrap differences, and create the review bundle |

Steps 2 and 3 can run after step 0. Step 4 requires a PASS from step 1 unless
`IGNORE_PILOT_GATE=True` is set explicitly. Step 5 can be rerun at any time and
will report missing, partial, incompatible, or completed runs.

## Colab setup

1. Open the desired notebook from the `notebooks/` directory in Colab.
2. Select the recommended runtime before executing any cells.
3. For notebooks 1 and 4, add `TABPFN_TOKEN` under Colab Secrets and enable
   notebook access. The token is never printed or written to Drive.
4. Edit only the first code cell marked `EDIT THIS CELL ONLY`.
5. Run all cells from top to bottom.
6. Disconnect the runtime when the notebook finishes. Idle Colab runtimes can
   continue consuming compute units.

The default Drive root is:

```text
/content/drive/MyDrive/tfm_credit_benchmark/lending_club_900k/
```

Change `DRIVE_OUTPUT_DIR` or `EXPERIMENT_NAME` in every notebook if a different
location is desired. Keep those values identical across all six notebooks.

## Pilot report and cost gate

The pilot fits a power law to TabPFN fit and prediction times and a linear model
to peak host and GPU memory. The 700k projection includes projected fit time,
prediction time, peak memory, total hours, and Colab compute units.

`COMPUTE_UNITS_PER_HOUR` defaults to 5.4 for an A100 40 GB runtime. Edit it in
`01_pilot.ipynb` if the current Colab rate differs.

The report passes only when all required runs finish, metrics are finite,
ROC-AUC values fall inside the sanity range, TabPFN does not regress sharply,
memory stays below 90 percent of detected capacity, and projected duration is
below `MAX_SESSION_HOURS`, which defaults to 20.

Read `pilot/pilot_report.json` and `pilot/pilot_report.md` before starting the
full TabPFN run.

## Google Drive result layout

```text
lending_club_900k/
  shared/                    cohort, split, feature schema, leakage report
  studies/                   resumable Optuna SQLite studies
  pilot/                     pilot report and projections
  lgbm_default/n<size>/      default LightGBM run package
  lgbm_tuned/n<size>/        tuned LightGBM run package
  realmlp/n700000/           RealMLP run package
  tabpfn35/n<size>/          TabPFN run package
  performance_review/        tables, figures, checks, and review bundle
```

Every completed run folder includes:

```text
test_predictions.parquet
metrics.json
metrics.csv
resources.json
resources.csv
run_details.json
run_manifest.json
artifact_checksums.json
result_card.md
```

LightGBM tuned runs also contain `tuning_trials.csv` and the resumable study is
kept under `studies/`. `metrics.json` is written last and acts as the atomic
completion marker. A partial folder is never treated as finished.

The comparison notebook creates a run inventory, long and wide metrics tables,
resource summaries, paired bootstrap confidence intervals, ROC,
precision-recall, calibration, learning, fit-time, and prediction-time figures,
`comparison.md`, `comparison.json`, and a small `review_bundle.zip`.

Raw data, model weights, SQLite studies, and per-row predictions are not
committed to GitHub.

## Resume and rerun behavior

- Preparation reuses the signed cohort and split when checksums match.
- A missing or corrupt cohort cache is rebuilt from the pinned dataset revision.
- Optuna studies resume from their Drive SQLite database.
- Completed method-size folders are skipped by default.
- A partial run stops with an actionable message rather than being mistaken for a completed experiment.
- `FORCE_RERUN=True` replaces only the selected method-size folder.
- `FORCE_REBUILD_SPLIT=True` creates a new contract and invalidates old pilot reports.

## Local development

The fast test suite is CPU-only and performs no network access:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
pytest -q
```

Run a package entry point outside Colab with:

```bash
python -m tfmbench.run \
  --method lgbm \
  --train-size 700000 \
  --output-root /path/to/output
```

The tests cover cleaning, target mapping, date parsing, deterministic splits,
nested subsets, manifest tamper detection, leakage checks, metrics, bootstrap
intervals, pilot projections, gate failures, OOM retries, notebook syntax, row
limits, and forbidden characters.

## Reproducibility and limitations

- The cohort represents completed accepted loans, not all applicants or current production decisions.
- Random cohort sampling is stratified with seed 42.
- The saved dataset revision and SHA256 identify the exact source bytes.
- The split manifest hashes every row-ID array and the feature schema.
- Test predictions allow comparison tables to be recomputed without rerunning inference.
- GPU timings and memory describe the recorded Colab environment, not universal latency.
- This is an academic benchmark, not a credit decision system.

## License

Repository code is released under the [MIT License](LICENSE). Dataset and model
assets retain their own licenses. TabPFN-3.5 weights and outputs must be used
consistently with the Prior Labs non-commercial terms linked above.
