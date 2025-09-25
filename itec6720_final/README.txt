ITEC 6720 — Final Project (README)

WHAT THIS IS
- Lesson 1: Mini-course snippets (small standalone scripts).
- Lesson 2: Step-by-Step Iris project (full ML workflow: load → EDA → CV → tune SVM → test → save model).

FOLDERS
- scripts/mini_course/ – MC-01 … MC-14 scripts
- scripts/step_by_step/ – 01_env_versions.py … 06_predictions.py, plus
  - iris_end_to_end_SVM_model.py (main)
  - iris_end_to_end_NB_model.py (optional)
- data/ (optional local iris.csv)
- figures/, models/, reports/

ENV / INSTALL
Python 3.x
pip install numpy scipy pandas matplotlib scikit-learn joblib

HOW TO RUN (Step-by-Step)
python scripts/step_by_step/iris_end_to_end_SVM_model.py
This script: loads data, makes EDA plots, runs 10-fold CV spot-check, grid-searches SVM, evaluates on test, saves model and summary.

(Optional) Auto-select variant:
python scripts/step_by_step/iris_end_to_end_NB_model.py

KEY OUTPUTS (generated)
- Figures: figures/eda_boxplots.png, eda_hist.png, eda_scatter_matrix.png,
  algo_boxplot.png, cv_vs_test.png, confusion_matrix.png
- Reports: reports/spotcheck_cv.csv, step_by_step_run_summary.txt
- Model: models/iris_best_model.joblib
- Console shows: 10-fold CV scores, best SVM params, TEST accuracy (~0.92), confusion matrix, classification report.

SUBMIT CHECKLIST
[ ] Scripts (mini_course + step_by_step)
[ ] Screenshots doc with captions
[ ] 500-word report (a–g)
[ ] Generated figures/reports/model
[ ] This README.txt
