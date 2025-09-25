# scripts/step_by_step/iris_end_to_end.py
"""
Iris End-to-End Project (for Final Project submission)

Outputs created:
- figures/eda_boxplots.png
- figures/eda_hist.png
- figures/eda_scatter_matrix.png
- figures/algo_boxplot.png
- figures/cv_vs_test.png
- figures/confusion_matrix.png
- reports/spotcheck_cv.csv
- reports/step_by_step_run_summary.txt
- models/iris_best_model.joblib
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from pandas import read_csv

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from joblib import dump, load



def main():
    RNG = 42
    FIGDIR = Path("figures");   FIGDIR.mkdir(parents=True, exist_ok=True)
    MODELDIR = Path("models");  MODELDIR.mkdir(parents=True, exist_ok=True)
    REPORTDIR = Path("reports"); REPORTDIR.mkdir(parents=True, exist_ok=True)
    DATADIR = Path("data");     DATADIR.mkdir(parents=True, exist_ok=True)

    # ---------- 1) Load & EDA (match blog) ----------
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    df = read_csv(url, names=names)
    (DATADIR / "iris.csv").write_text(df.to_csv(index=False))  # save local copy

    feature_names = names[:-1]
    class_col = names[-1]
    X = df[feature_names].values
    y = df[class_col].values
    target_names = np.unique(y)

    print("Shape:", df.shape)
    print("Classes:", {k: int(v) for k, v in df[class_col].value_counts().sort_index().items()})
    print(df.describe().T[['mean', 'std', 'min', 'max']])

    # EDA plots
    df[feature_names].plot(kind="box", subplots=True, layout=(2, 2),
                           sharex=False, sharey=False, figsize=(8, 6),
                           title="Univariate Boxplots")
    plt.tight_layout(); plt.savefig(FIGDIR / "eda_boxplots.png", dpi=150); plt.close()

    df[feature_names].hist(figsize=(8, 6))
    plt.suptitle("Feature Histograms"); plt.tight_layout()
    plt.savefig(FIGDIR / "eda_hist.png", dpi=150); plt.close()

    scatter_matrix(df[feature_names], figsize=(8, 8))
    plt.suptitle("Scatter Matrix")
    plt.savefig(FIGDIR / "eda_scatter_matrix.png", dpi=150); plt.close()

    # ---------- 2) Split (stratified) ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RNG
    )

    # ---------- 3) CV harness + spot-check ----------
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RNG)
    models = [
        ("LR",  Pipeline([("scaler", StandardScaler()),
                          ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", multi_class="auto"))])),
        ("LDA", LinearDiscriminantAnalysis()),
        ("KNN", Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())])),
        ("CART", DecisionTreeClassifier(random_state=RNG)),
        ("NB",  GaussianNB()),
        ("SVM", Pipeline([("scaler", StandardScaler()), ("clf", SVC())])),
    ]

    print("\nSpot-check (10-fold CV accuracy):")
    model_names, results = [], []
    for name, model in models:
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        model_names.append(name); results.append(scores)
        print(f"{name:>4}: {scores.mean():.3f} (+/- {scores.std():.3f})")

    pd.DataFrame({n: r for n, r in zip(model_names, results)}).to_csv(REPORTDIR / "spotcheck_cv.csv", index=False)

    plt.figure()
    plt.boxplot(results, tick_labels=model_names)
    plt.title("Algorithm Comparison (CV accuracy)")
    plt.savefig(FIGDIR / "algo_boxplot.png", dpi=150); plt.close()

    # ---------- 4) Hyperparameter tuning (SVM pipeline) ----------
    svm_pipe = Pipeline([("scaler", StandardScaler()), ("clf", SVC())])
    param_grid = {
        "clf__kernel": ["rbf", "linear"],
        "clf__C": [0.1, 1, 10, 100],
        "clf__gamma": ["scale", "auto", 0.1, 0.01, 0.001],
    }
    grid = GridSearchCV(svm_pipe, param_grid, scoring="accuracy", cv=cv, n_jobs=-1, return_train_score=True)
    grid.fit(X_train, y_train)
    print("\nBest SVM params:", grid.best_params_)
    print("Best CV accuracy:", round(grid.best_score_, 3))

    # ---------- 5) Ensembles (optional comparison) ----------
    rf = RandomForestClassifier(n_estimators=200, random_state=RNG, n_jobs=-1)
    gb = GradientBoostingClassifier(random_state=RNG)
    for name, mdl in [("RF", rf), ("GB", gb)]:
        scores = cross_val_score(mdl, X_train, y_train, cv=cv, scoring="accuracy")
        print(f"{name} CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

    # ---------- 6) Finalize & evaluate on test ----------
    best_model = grid.best_estimator_
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    test_acc = accuracy_score(y_test, y_pred)
    print("\nTEST accuracy:", test_acc)
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred, target_names=target_names))
 


    # ---------- 7) Persist & reload ----------
    model_path = MODELDIR / "iris_best_model.joblib"
    dump(best_model, model_path)
    reloaded = load(model_path)
    print("Loaded model test accuracy (sanity check):", accuracy_score(y_test, reloaded.predict(X_test)))

    # ---------- 8) Result figures ----------
    plt.figure()
    plt.title("CV Mean vs Test Accuracy (Best Model)")
    plt.bar(["CV mean", "Test"], [grid.best_score_, test_acc])
    plt.ylim(0.8, 1.0)
    plt.savefig(FIGDIR / "cv_vs_test.png", dpi=150); plt.close()

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=target_names, cmap="Blues", values_format="d")
    plt.title("Confusion Matrix — Test Set")
    plt.savefig(FIGDIR / "confusion_matrix.png", dpi=150); plt.close()

    with open(REPORTDIR / "step_by_step_run_summary.txt", "w", encoding="utf-8") as f:
        f.write("Iris Step-by-Step Project — Run Summary\n")
        f.write(f"Best CV accuracy: {grid.best_score_:.3f}\n")
        f.write(f"Best params: {grid.best_params_}\n")
        f.write(f"Test accuracy: {test_acc:.3f}\n")


if __name__ == "__main__":
    main()
