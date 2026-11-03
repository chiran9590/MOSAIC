
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier


DATA_DIR = Path("data/processed")


def load_data(filename):
    data = np.load(DATA_DIR / filename)
    return data["X"], data["y"]


def evaluate_model(
    name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    encode_labels=False,
):
    print(f"\n{'=' * 60}")
    print(name)

    # XGBoost requires labels 0, 1, 2.
    y_fit = y_train - 1 if encode_labels else y_train

    model.fit(X_train, y_fit)

    predictions = model.predict(X_test)

    # Convert predictions back to original FI-2010 labels.
    if encode_labels:
        predictions = predictions + 1

    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[1, 2, 3],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[1, 2, 3],
        )
    )


def main():
    print("Loading fold 1...")

    X_train, y_train = load_data("train_cf_1.npz")
    X_test, y_test = load_data("test_cf_1.npz")

    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=42,
        ),
    )

    evaluate_model(
        "LOGISTIC REGRESSION",
        logistic,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=4,
        random_state=42,
    )

    evaluate_model(
        "XGBOOST",
        xgb,
        X_train,
        y_train,
        X_test,
        y_test,
        encode_labels=True,
    )


if __name__ == "__main__":
    main()
