# 14_iris_end_to_end.py
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from joblib import dump, load

# -----------------------------
# 1) Load & EDA (minimal)
# -----------------------------
iris = load_iris(as_frame=True)
df = iris.frame  # features + target
df['target'] = iris.target

print("Shape:", df.shape)
print("Classes:", df['target'].value_counts().sort_index().to_dict())
print(df.describe().T[['mean','std','min','max']])

# -----------------------------
# 2) Train/Test split
# -----------------------------
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=7
)

print("Train/Test shapes:", X_train.shape, X_test.shape)
print("Train class counts:", np.bincount(y_train))
print("Test class counts:",  np.bincount(y_test))


# -----------------------------
# 3) Spot-check algorithms
#    (scale where it matters via Pipeline)
# -----------------------------
candidates = [
   
    ("LR", Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(solver='lbfgs', max_iter=2000))])),  # <- no multi_class
    


    ("LDA", LDA := LinearDiscriminantAnalysis()),
    ("KNN", Pipeline([('scaler', StandardScaler()),
                      ('clf', KNeighborsClassifier())])),
    ("CART", DecisionTreeClassifier(random_state=7)),
    ("NB",  GaussianNB()),
    ("SVM", Pipeline([('scaler', StandardScaler()),
                      ('clf', SVC(kernel='rbf'))]))
]

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
print("\nSpot-check (10-fold CV accuracy):")
scores_table = []
for name, model in candidates:
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    scores_table.append((name, scores.mean(), scores.std()))
    print(f"{name:>4}: {scores.mean():.3f} (+/- {scores.std():.3f})")

# -----------------------------
# 4) Tune the top model(s)
#    Example: SVM (often strong on Iris)
# -----------------------------
svm_pipe = Pipeline([('scaler', StandardScaler()),
                     ('clf', SVC())])

param_grid = {
    'clf__kernel': ['rbf', 'linear'],
    'clf__C': [0.1, 1, 10, 100],
    'clf__gamma': ['scale', 0.1, 0.01, 0.001]  # gamma used when rbf
}
grid = GridSearchCV(svm_pipe, param_grid, scoring='accuracy', cv=cv, n_jobs=-1, return_train_score=True)
grid.fit(X_train, y_train)
print("\nBest SVM params:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)

# -----------------------------
# 5) Try an ensemble (comparison)
# -----------------------------
rf = RandomForestClassifier(n_estimators=200, random_state=7)
gb = GradientBoostingClassifier(random_state=7)
for name, model in [("RF", rf), ("GB", gb)]:
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"{name} CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

# -----------------------------
# 6) Finalize, evaluate on test, save/load
#    (Pick the best: grid.best_estimator_ here)
# -----------------------------
best_model = grid.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("\nTEST accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

dump(best_model, "iris_best_model.joblib")
loaded = load("iris_best_model.joblib")
print("Loaded model test accuracy (sanity check):",
      accuracy_score(y_test, loaded.predict(X_test)))
