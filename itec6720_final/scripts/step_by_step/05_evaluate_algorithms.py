from pandas import read_csv
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length','sepal-width','petal-length','petal-width','class']
dataset = read_csv(url, names=names)

# Split-out validation dataset
array = dataset.values
X, y = array[:,0:4], array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, y, test_size=0.20, random_state=1, shuffle=True
)

# Spot Check Algorithms (updated)
models = [
    ('LR',  make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),  # lbfgs by default; no deprecation
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', make_pipeline(StandardScaler(), KNeighborsClassifier())),
    ('CART', DecisionTreeClassifier(random_state=1)),
    ('NB',  GaussianNB()),
    ('SVM', make_pipeline(StandardScaler(), SVC()))  # default gamma='scale'
]

# evaluate each model in turn
results, names = [], []
cv = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
for name, model in models:
    cv_results = cross_val_score(model, X_train, Y_train, cv=cv, scoring='accuracy')
    results.append(cv_results); names.append(name)
    print(f'{name}: {cv_results.mean():.3f} ({cv_results.std():.3f})')

plt.boxplot(results, tick_labels=names)
plt.title('Algorithm Comparison')
plt.show()
