from joblib import load
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

model = load("models/iris_best_model.joblib")
print("Loaded object:", model)
print("Pipeline steps:", getattr(model, "named_steps", None))

X, y = load_iris(as_frame=True, return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
print("Reloaded model test acc:", round(accuracy_score(y_te, model.predict(X_te)), 3))
