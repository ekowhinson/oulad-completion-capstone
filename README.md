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
```

## Repository structure

```
src/download_data.py   downloads and extracts OULAD into data/raw/
data/                  local data (git-ignored)
requirements.txt       Python dependencies
```

## Status

Synopsis stage. Data integration, feature engineering, the four research-question analyses, and the modeling pipeline will be added for the interim analysis report.
