# ML Model Selection Pipeline (leakage-safe, tested, CI-ready)

A small, reproducible machine learning project that compares four classifiers on the
Breast Cancer Wisconsin (Diagnostic) dataset that ships with scikit-learn, and
shows the engineering habits that make results trustworthy.

> Educational project. Not a medical tool.

## What it demonstrates
- **No data leakage:** scaling sits inside each `Pipeline`, so it is fitted on training folds only.
- **Fair model selection:** 5-fold stratified cross-validation on the training set; the test set is used once, at the end.
- **Threshold tuning without peeking:** the decision threshold is picked from out-of-fold training predictions so that at least 95% of malignant cases are caught.
- **Reproducibility:** fixed seed, pinned split, saved metrics and plots.
- **Software practice:** unit tests (`pytest`) and a GitHub Actions workflow that runs tests and training on every push.

## Setup and run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q
python -m src.train
```
Outputs go to `reports/metrics.json` and `reports/results.png`; the model is saved to `models/best_model.joblib`.

## Results from one run (seed 42, 114 held-out test samples)
5-fold CV ROC-AUC on the training set:

| Model | CV ROC-AUC (mean +/- std) |
| --- | --- |
| Logistic regression | 0.9959 +/- 0.0050 |
| SVM (RBF, calibrated) | 0.9956 +/- 0.0048 |
| Hist. gradient boosting | 0.9942 +/- 0.0030 |
| Random forest | 0.9904 +/- 0.0072 |

Selected model: logistic regression. Held-out test: accuracy 0.982, precision 0.986,
recall 0.986, ROC-AUC 0.995 (same at threshold 0.50 and the tuned threshold 0.45).

## Honest notes
- The differences between the top models are smaller than the CV standard deviation, so this run does not prove one model is better than another.
- The dataset is small (569 rows) and easy; scores near the ceiling say little about harder real-world data.
- Next steps: repeated CV, calibration curves, SHAP/feature importance, and a harder dataset.

## Structure
```
src/data.py      stratified split
src/models.py    candidate pipelines
src/evaluate.py  metrics and threshold selection
src/train.py     CV, selection, final report, plots
tests/           pytest unit tests
.github/workflows/ci.yml
```
