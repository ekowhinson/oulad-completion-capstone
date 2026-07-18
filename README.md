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
4. Can a machine-learning model (logistic regression, random forest, XGBoost) predict completion significantly better than a majority-class baseline, and which features matter most?

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
5. **Sample-size verification** — recomputes the synopsis minimums (786 / 320 / 1,068 / 530)
   and confirms the data exceeds the binding 1,068 roughly thirtyfold.
6. **Statistical analysis** — RQ1 Welch t-test (+ Mann-Whitney, point-biserial, univariate
   logistic), RQ2 two-proportion z-test, RQ3 chi-square + Cramer's V with Wilson-CI subgroup
   precision estimates.
7. **Feature engineering** — early-window (first two weeks) features only; whole-module
   aggregates excluded as leakage; encoding decisions.
8. **Data splitting** — stratified 60/20/20 train/validation/test with balance checks.
9. **Model building** — majority baseline, logistic regression, random forest, XGBoost, with
   5-fold cross-validation.
10. **Model evaluation** — champion selected on validation, scored once on test: AUC with
    bootstrap CI (the RQ4 hypothesis test), accuracy/precision/recall/F1, ROC curves,
    confusion matrix.
11. **Important-feature selection** — permutation importance + SHAP, and a top-10-features
    refit demonstrating a lean early-warning variant.
12. **Recommended model** — the champion, with the case for it and a deployment note.
13. **Final conclusions and recommendations** — verdicts per research question, five
    operational recommendations for providers, limitations, and future work.

## Status

Analysis stage: the synopsis-planned analyses are implemented and executed in the notebook
above. Next: the interim analysis report write-up.
