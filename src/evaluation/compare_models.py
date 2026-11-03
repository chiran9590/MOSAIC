
from pathlib import Path

import numpy as np
import torch

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.models.transformer import MOSAICTransformer
from src.preprocessing.sequence_dataset import LOBSequenceDataset
from torch.utils.data import DataLoader


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEQUENCE_LENGTH = 100
STRIDE = 10
BATCH_SIZE = 64
RANDOM_STATE = 42

DATA_DIR = Path("data/processed")
CHECKPOINT = Path("checkpoints/mosaic_transformer_best.pth")
OUTPUT_DIR = Path("reports")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Device selection
# --------------------------------------------------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# --------------------------------------------------
# Evaluation function
# --------------------------------------------------

def report_results(name, y_true, predictions):

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
    )

    print("\n" + "=" * 60)
    print(name)

    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))

    print("\nClassification report:")

    print(
        classification_report(
            y_true,
            predictions,
            labels=[0, 1, 2],
            target_names=[
                "Class 1",
                "Class 2",
                "Class 3",
            ],
            zero_division=0,
        )
    )

    print("Confusion matrix:")

    print(
        confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1, 2],
        )
    )

    return accuracy, macro_f1


# --------------------------------------------------
# Main experiment
# --------------------------------------------------

def main():

    device = get_device()

    print("Using device:", device, flush=True)

    # ==============================================
    # 1. Load training data
    # ==============================================

    print("\nLoading training data...", flush=True)

    train_data = np.load(
        DATA_DIR / "train_cf_1.npz"
    )

    X_train_all = train_data["X"]
    y_train_all = train_data["y"] - 1

    # Chronological 80% training partition.
    split_index = int(
        len(X_train_all) * 0.8
    )

    train_indices = np.arange(
        SEQUENCE_LENGTH - 1,
        split_index,
        STRIDE,
    )

    X_train = X_train_all[train_indices]
    y_train = y_train_all[train_indices]

    # ==============================================
    # 2. Load test data
    # ==============================================

    print("Loading test data...", flush=True)

    test_dataset = LOBSequenceDataset(
        filename="test_cf_1.npz",
        sequence_length=SEQUENCE_LENGTH,
        stride=STRIDE,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_data = np.load(
        DATA_DIR / "test_cf_1.npz"
    )

    X_test_all = test_data["X"]
    y_test_all = test_data["y"] - 1

    test_indices = np.array(
        test_dataset.indices
    )

    X_test = X_test_all[test_indices]
    y_test = y_test_all[test_indices]

    print(
        "Training observations:",
        len(X_train),
        flush=True,
    )

    print(
        "Common test observations:",
        len(X_test),
        flush=True,
    )

    results = {}

    # ==============================================
    # 3. Logistic Regression
    # ==============================================

    print(
        "\nTraining Logistic Regression...",
        flush=True,
    )

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    )

    logistic.fit(
        X_train,
        y_train,
    )

    print(
        "Generating Logistic Regression predictions...",
        flush=True,
    )

    logistic_predictions = logistic.predict(
        X_test
    )

    results["Logistic Regression"] = report_results(
        "LOGISTIC REGRESSION",
        y_test,
        logistic_predictions,
    )

    # ==============================================
    # 4. HistGradientBoosting
    # ==============================================

    print(
        "\nPreparing HistGradientBoosting...",
        flush=True,
    )

    tree_model = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )

    print(
        "Fitting HistGradientBoosting...",
        flush=True,
    )

    tree_model.fit(
        X_train,
        y_train,
    )

    print(
        "Generating tree model predictions...",
        flush=True,
    )

    tree_predictions = tree_model.predict(
        X_test
    )

    results["HistGradientBoosting"] = report_results(
        "HIST GRADIENT BOOSTING",
        y_test,
        tree_predictions,
    )

    # ==============================================
    # 5. MOSAIC Transformer
    # ==============================================

    print(
        "\nLoading MOSAIC Transformer...",
        flush=True,
    )

    model = MOSAICTransformer().to(device)

    model.load_state_dict(
        torch.load(
            CHECKPOINT,
            map_location=device,
            weights_only=True,
        )
    )

    model.eval()

    transformer_predictions = []

    print(
        "Generating Transformer predictions...",
        flush=True,
    )

    with torch.no_grad():

        for X, y in test_loader:

            X = X.to(device)

            outputs = model(X)

            predictions = outputs.argmax(
                dim=1
            )

            transformer_predictions.extend(
                predictions.cpu().numpy()
            )

    transformer_predictions = np.array(
        transformer_predictions
    )

    results["Transformer"] = report_results(
        "MOSAIC TRANSFORMER",
        y_test,
        transformer_predictions,
    )

    # ==============================================
    # 6. Save predictions
    # ==============================================

    print(
        "\nSaving model predictions...",
        flush=True,
    )

    np.savez_compressed(
        OUTPUT_DIR / "model_predictions_cf1.npz",
        y_true=y_test,
        logistic=logistic_predictions,
        hist_gradient_boosting=tree_predictions,
        transformer=transformer_predictions,
        test_indices=test_indices,
    )

    # ==============================================
    # 7. Final comparison
    # ==============================================

    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("-" * 60)

    print(
        f"{'Model':25s} "
        f"{'Accuracy':12s} "
        f"{'Macro F1':12s}"
    )

    print("-" * 60)

    for name, (accuracy, macro_f1) in results.items():

        print(
            f"{name:25s} "
            f"{accuracy:<12.4f} "
            f"{macro_f1:<12.4f}"
        )

    print("\nExperiment completed successfully.")

    print(
        "Predictions saved to:",
        OUTPUT_DIR / "model_predictions_cf1.npz",
    )


if __name__ == "__main__":
    main()
