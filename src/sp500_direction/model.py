from sklearn.linear_model import LogisticRegression
import joblib


def train_logistic_model(X_train, y_train, C=1.0, max_iter=1000, random_state=42):
    model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state, solver="lbfgs")
    model.fit(X_train, y_train)
    return model


def save_model(model, path: str):
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)
