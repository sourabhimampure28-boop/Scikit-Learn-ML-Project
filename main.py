"""Train and evaluate regression and classification models with Scikit-learn."""

from math import sqrt
import pandas as pd

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def run_regression():
    """Fit regression models and return a metrics DataFrame."""
    dataset = load_diabetes()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, random_state=42
        ),
        "KNN Regressor (k=5)": make_pipeline(
            StandardScaler(), KNeighborsRegressor(n_neighbors=5)
        ),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(y_test, predictions),
                "RMSE": sqrt(mean_squared_error(y_test, predictions)),
                "R2": r2_score(y_test, predictions),
            }
        )

    result_df = pd.DataFrame(results).sort_values("RMSE")
    result_df.to_csv("regression_results.csv", index=False)
    return result_df


def run_classification():
    """Fit classification models and return a metrics DataFrame."""
    dataset = load_breast_cancer()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, random_state=42),
        ),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "KNN Classifier (k=5)": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=5)
        ),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "Recall": recall_score(y_test, predictions, zero_division=0),
                "F1-score": f1_score(y_test, predictions, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, probabilities),
            }
        )

    result_df = pd.DataFrame(results).sort_values(
        "F1-score", ascending=False
    )
    result_df.to_csv("classification_results.csv", index=False)
    return result_df


def main():
    print("\\nREGRESSION RESULTS — Diabetes dataset")
    regression_results = run_regression()
    print(regression_results.round(4).to_string(index=False))

    print("\\nCLASSIFICATION RESULTS — Breast Cancer dataset")
    classification_results = run_classification()
    print(classification_results.round(4).to_string(index=False))

    print("\\nSaved regression_results.csv and classification_results.csv")


if __name__ == "__main__":
    main()
