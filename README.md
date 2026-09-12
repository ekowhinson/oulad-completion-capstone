# Predicting Successful Completion of Online Programs Using Machine Learning

Data Analytics Capstone (QM640, Walsh College). Author: Francis Ekow Hinson.

This project builds and evaluates a supervised machine-learning model that predicts whether a student will successfully complete an online module (Pass or Distinction versus Fail or Withdrawn), and identifies the behavioral, academic, and demographic factors that most influence that outcome.

## Data

The study uses the Open University Learning Analytics Dataset (OULAD): 32,593 students across 22 module presentations, with 10.6 million virtual learning environment (VLE) interaction records, assessment scores, registrations, and demographics (Kuzilek, Hlosta, & Zdrahal, 2017, https://doi.org/10.1038/sdata.2017.171). The dataset is released under a CC BY 4.0 license.

The raw data is NOT committed to this repository (the clickstream table alone is about 433 MB, above GitHub file limits). Instead, run the download script below; it fetches the official archive and extracts the seven CSV tables into `data/raw/`.

Sources:
- Open University portal: https://research.stem.open.ac.uk/ouanalyse/dataset/
- UCI Machine Learning Repository (mirror, id 349): https://archive.ics.uci.edu/dataset/349

## Research questions

1. Is VLE engagement associated with successful completion?
2. Is early academic performance (passing the first assessment, score >= 40) associated with completion?
3. Are demographic and socioeconomic characteristics (IMD band, prior education) associated with completion?
4. Can a machine-learning model (eleven classifiers from seven families) predict completion significantly better than a majority-class baseline using only early information, and which features matter most?

## Getting started

```bash
pip install -r requirements.txt
python src/download_data.py
jupyter notebook notebooks/oulad_completion_analysis.ipynb
```

## Repository structure

```
notebooks/oulad_completion_analysis.ipynb   the full analysis (executed, with outputs):
                                            integration -> sample-size verification -> EDA ->
                                            RQ1-RQ3 hypothesis tests -> RQ4 models + SHAP
src/download_data.py                        downloads and extracts OULAD into data/raw/
data/                                       local data (git-ignored)
requirements.txt                            Python dependencies
```

## Analysis overview (notebooks/oulad_completion_analysis.ipynb)

The notebook follows the capstone structure end to end, with an Observations summary closing
every stage:

1. **Problem statement** — the completion gap, the timing problem, and the four research
   questions with hypotheses.
2. **Data source and description** — OULAD provenance, license, the seven tables, unit of
   analysis, target definition.
3. **Data integration and treatment** — joins to one row per registration; missing-value and
   outlier decisions, each justified in place.
4. **Exploratory data analysis** — completion by module, engagement distributions,
   early-performance splits, the IMD gradient, correlations.
5. **Sample-size verification** — recomputes the synopsis minimums (786 / 320 / 1,068 / 530), adds
   the chi-square power requirement for RQ3 (1,565) and the events-per-variable floor for the 38
   encoded features (806), and confirms the data exceeds the binding 1,565 about 21-fold.
6. **Statistical analysis** — RQ1 Welch t-test (+ Mann-Whitney, point-biserial, univariate
   logistic), RQ2 two-proportion z-test, RQ3 chi-square + Cramer's V with Wilson-CI subgroup
   precision estimates.
7. **Feature engineering** — two feature sets: day 13 (activity logged through day 13 plus
   enrolment facts, 35 encoded features) and the first-assessment checkpoint (adds the
   assessment result, 38); whole-module aggregates excluded as leakage; encoding decisions.
8. **Data splitting** — stratified 60/20/20 train/validation/test with balance checks.
9. **Model building** — majority baseline plus eleven classifiers from seven families, 5-fold
   cross-validation, champion chosen on validation AUC.
10. **Model evaluation** — champion refit on train + validation and scored once on test: AUC with
    bootstrap CI (the RQ4 hypothesis test), accuracy/precision/recall/F1, ROC curves, confusion
    matrices, McNemar paired tests; the two prediction stages; a retrospective day-13-eligible
    evaluation that excludes registrations not yet on file or already withdrawn by day 13.
11. **Important-feature selection** — permutation importance + SHAP, and a top-10-features
    refit (features ranked on the validation partition) demonstrating a lean early-warning variant.
12. **Recommended model** — the champion, with the case for it and a deployment note.
13. **Final-stage analyses** — hyperparameter tuning, probability calibration, workload-capped
    operating threshold, subgroup fairness audit, early-window sensitivity, student-grouped split,
    an unseen-presentation holdout, an operational checkpoint snapshot (only submissions received
    by the deadline plus one week, registrations still active then), the day-13 and snapshot lists
    scored on the most recent presentation after training on earlier ones, and student-clustered
    (cluster-robust) re-tests of RQ1–RQ3.
14. **Final conclusions and recommendations** — verdicts per research question, five
    operational recommendations for providers, limitations, and future work.

## Status

Final report and presentation completed (September 2026). The notebook was last re-executed end
to end on 2026-09-12; every figure quoted in the report comes from that run.
