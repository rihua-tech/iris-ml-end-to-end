# MC-07 (basic): Evaluate using 10-fold CV (matches the mini-course)
from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X, y = array[:, 0:8], array[:, 8]

kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = LogisticRegression(solver="liblinear", max_iter=1000, random_state=42)
results = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")

print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
