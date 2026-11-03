
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.preprocessing.sequence_dataset import LOBSequenceDataset
from src.preprocessing.sequence_features_v2 import (
    extract_sequence_features_v2,
)


SEQUENCE_LENGTH = 100
STRIDE = 10

MODEL_PARAMS = {
    "max_iter": 150,
    "learning_rate": 0.08,
    "max_leaf_nodes": 31,
    "l2_regularization": 1,
    "random_state": 42,
}


def build_features(dataset):
    sequences = []
    labels = []
    indices = []

    for i in range(len(dataset)):
        sequence, target = dataset[i]

        sequences.append(sequence.numpy())
        labels.append(int(target))
        indices.append(dataset.indices[i])

    sequences = np.asarray(sequences, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    features = extract_sequence_features_v2(sequences)

    return features, labels, np.asarray(indices)


def main():
    all_results = []
    all_predictions = {}

    for fold in range(1, 10):
        print(f"\n{'=' * 65}")
        print(f"OFFICIAL TEST EVALUATION — FOLD {fold}/9")
        print("=" * 65)

        train_filename = f"train_cf_{fold}.npz"
        test_filename = f"test_cf_{fold}.npz"

        # Build training sequences from the complete training fold.
        train_dataset = LOBSequenceDataset(
            train_filename,
            sequence_length=SEQUENCE_LENGTH,
            stride=STRIDE,
        )

        # Build test sequences independently.
        test_dataset = LOBSequenceDataset(
            test_filename,
            sequence_length=SEQUENCE_LENGTH,
            stride=STRIDE,
        )

        print("Extracting training features...", flush=True)
        X_train, y_train, _ = build_features(train_dataset)

        print("Extracting official test features...", flush=True)
        X_test, y_test, test_indices = build_features(test_dataset)

        print("Training model...", flush=True)

        model = HistGradientBoostingClassifier(**MODEL_PARAMS)
        model.fit(X_train, y_train)

        print("Predicting...", flush=True)
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
            average=None,
            labels=[0, 1, 2],
        )

        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1, 2],
        )

        result = {
            "fold": fold,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "class_1_f1": class_f1[0],
            "class_2_f1": class_f1[1],
            "class_3_f1": class_f1[2],
            "train_sequences": len(X_train),
            "test_sequences": len(X_test),
            "feature_count": X_train.shape[1],
        }

        all_results.append(result)

        all_predictions[f"y_true_fold_{fold}"] = y_test
        all_predictions[f"y_pred_fold_{fold}"] = predictions
        all_predictions[f"test_indices_fold_{fold}"] = test_indices
        all_predictions[f"confusion_matrix_fold_{fold}"] = matrix

        print(f"Features: {X_train.shape[1]}")
        print(f"Training sequences: {len(X_train)}")
        print(f"Test sequences: {len(X_test)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")

        print("\nClassification report:")
        print(
            classification_report(
                y_test,
                predictions,
                labels=[0, 1, 2],
                target_names=[
                    "Down",
                    "Stationary",
                    "Up",
                ],
                zero_division=0,
            )
        )

    # Save results.
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        output_dir / "final_test_results_v2.csv",
        index=False,
    )

    np.savez_compressed(
        output_dir / "final_test_predictions_v2.npz",
        **all_predictions,
    )

    # Summary statistics.
    print("\n" + "=" * 65)
    print("FINAL OFFICIAL TEST RESULTS — EXPANDED FEATURES")
    print("=" * 65)

    print(results_df.to_string(index=False))

    print("\nMEAN PERFORMANCE ACROSS NINE TEST FOLDS")

    for metric in [
        "accuracy",
        "macro_f1",
        "class_1_f1",
        "class_2_f1",
        "class_3_f1",
    ]:
        mean_value = results_df[metric].mean()
        std_value = results_df[metric].std()

        print(
            f"{metric:15s}: "
            f"{mean_value:.4f} "
            f"+/- {std_value:.4f}"
        )

    print("\nSaved:")
    print("reports/final_test_results_v2.csv")
    print("reports/final_test_predictions_v2.npz")


if __name__ == "__main__":
    main()
