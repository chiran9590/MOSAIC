
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

from src.preprocessing.sequence_dataset import LOBSequenceDataset
from src.preprocessing.sequence_features import extract_sequence_features


SEQUENCE_LENGTH = 100
STRIDE = 10


def load_sequences(dataset):
    """Load all sequences and labels from a sequence dataset."""
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
    )

    sequences = []
    labels = []

    for x, y in loader:
        sequences.append(x.numpy())
        labels.append(y.numpy())

    return (
        np.concatenate(sequences, axis=0),
        np.concatenate(labels, axis=0),
    )


def evaluate_fold(fold):
    """Train on a complete training fold and evaluate its official test fold."""

    train_filename = f"train_cf_{fold}.npz"
    test_filename = f"test_cf_{fold}.npz"

    print(f"\nLoading fold {fold}...")

    X_train_seq, y_train = load_sequences(
        LOBSequenceDataset(
            train_filename,
            sequence_length=SEQUENCE_LENGTH,
            stride=STRIDE,
        )
    )

    X_test_seq, y_test = load_sequences(
        LOBSequenceDataset(
            test_filename,
            sequence_length=SEQUENCE_LENGTH,
            stride=STRIDE,
        )
    )

    print("Extracting sequence features...")

    X_train = extract_sequence_features(X_train_seq)
    X_test = extract_sequence_features(X_test_seq)

    print(f"Training sequences: {len(X_train)}")
    print(f"Test sequences: {len(X_test)}")
    print(f"Feature dimensions: {X_train.shape[1]}")

    model = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )

    print("Training model...")
    model.fit(X_train, y_train)

    print("Evaluating on official test fold...")
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    class_f1 = f1_score(
        y_test,
        predictions,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )

    print(f"\nFold {fold} results")
    print("-" * 35)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1, 2],
            target_names=["DOWN", "STATIONARY", "UP"],
            zero_division=0,
        )
    )

    return {
        "fold": fold,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "class_1_f1": class_f1[0],
        "class_2_f1": class_f1[1],
        "class_3_f1": class_f1[2],
        "train_sequences": len(X_train),
        "test_sequences": len(X_test),
        "y_true": y_test,
        "predictions": predictions,
    }


def main():
    Path("reports").mkdir(exist_ok=True)

    results = []
    prediction_data = {}

    for fold in range(1, 10):
        print(f"\n{'=' * 50}")
        print(f"PROCESSING OFFICIAL TEST FOLD {fold}/9")
        print("=" * 50)

        result = evaluate_fold(fold)

        prediction_data[f"y_true_fold_{fold}"] = result.pop("y_true")
        prediction_data[f"predictions_fold_{fold}"] = result.pop(
            "predictions"
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    metric_columns = [
        "accuracy",
        "macro_f1",
        "class_1_f1",
        "class_2_f1",
        "class_3_f1",
    ]

    print("\n\nFINAL OFFICIAL TEST RESULTS")
    print("=" * 60)
    print(results_df.to_string(index=False))

    print("\nMEAN PERFORMANCE ACROSS NINE FOLDS")
    print("=" * 60)
    print(results_df[metric_columns].mean().to_string())

    print("\nSTANDARD DEVIATION ACROSS NINE FOLDS")
    print("=" * 60)
    print(results_df[metric_columns].std().to_string())

    results_df.to_csv(
        "reports/final_test_results.csv",
        index=False,
    )

    np.savez_compressed(
        "reports/final_test_predictions.npz",
        **prediction_data,
    )

    print("\nSaved:")
    print("- reports/final_test_results.csv")
    print("- reports/final_test_predictions.npz")


if __name__ == "__main__":
    main()
