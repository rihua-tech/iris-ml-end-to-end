from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
df = read_csv(url, sep=r"\s+", names=names)

X, y = df.iloc[:, :13].values, df.iloc[:, 13].values
cv = KFold(n_splits=10, shuffle=True, random_state=7)
model = KNeighborsRegressor()

scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
rmse = np.sqrt(-scores)
print(f"RMSE: {rmse.mean():.3f} ({rmse.std():.3f})")
