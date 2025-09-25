# MC-08: Algorithm evaluation metrics (accuracy + log loss)
from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
import numpy as np

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

df = read_csv(url, names=names)
X = df.iloc[:, 0:8].values
y = df.iloc[:, 8].values

# FIX: enable shuffling if you want to use random_state
kfold = KFold(n_splits=10, shuffle=True, random_state=7)

model = LogisticRegression(solver='liblinear', max_iter=1000)

# Accuracy (higher is better)
acc = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
print("Accuracy: %.3f%% (%.3f%%)" % (acc.mean() * 100.0, acc.std() * 100.0))

# Log loss (lower is better) -> cross_val_score returns *negative* log loss
neg_ll = cross_val_score(model, X, y, cv=kfold, scoring="neg_log_loss")
print("LogLoss:  %.3f (±%.3f)" % (-neg_ll.mean(), neg_ll.std()))
