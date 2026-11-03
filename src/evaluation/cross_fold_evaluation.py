
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
BATCH_SIZE = 256


def load_sequences(dataset):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
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
    filename = f"train_cf_{fold}.npz"

    data = np.load(Path("data/processed") / filename)
    split_index = int(len(data["X"]) * 0.8)

    train_dataset = LOBSequenceDataset(
        filename,
        sequence_length=SEQUENCE_LENGTH,
        stride=STRIDE,
        start_index=0,
        end_index=split_index,
    )

    validation_dataset = LOBSequenceDataset(
        filename,
        sequence_length=SEQUENCE_LENGTH,
        stride=STRIDE,
        start_index=split_index,
        end_index=None,
    )

    X_train_seq, y_train = load_sequences(train_dataset)
    X_val_seq, y_val = load_sequences(validation_dataset)

    X_train = extract_sequence_features(X_train_seq)
    X_val = extract_sequence_features(X_val_seq)

    model = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
    )

    class_f1 = f1_score(
        y_val,
        predictions,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )

    print(f"\nFold {fold}")
    print("-" * 30)
    print(f"Training sequences: {len(X_train)}")
    print(f"Validation sequences: {len(X_val)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    return {
        "fold": fold,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "class_1_f1": class_f1[0],
        "class_2_f1": class_f1[1],
        "class_3_f1": class_f1[2],
        "train_sequences": len(X_train),
        "validation_sequences": len(X_val),
    }


def main():
    Path("reports").mkdir(exist_ok=True)

    results = []

    for fold in range(1, 10):
        print(f"\nProcessing fold {fold}/9...")

        result = evaluate_fold(fold)
        results.append(result)

    results_df = pd.DataFrame(results)

    print("\nCROSS-FOLD RESULTS")
    print("==================")
    print(results_df.to_string(index=False))

    metric_columns = [
        "accuracy",
        "macro_f1",
        "class_1_f1",
        "class_2_f1",
        "class_3_f1",
    ]

    print("\nMEAN PERFORMANCE")
    print("================")
    print(results_df[metric_columns].mean())

    print("\nSTANDARD DEVIATION")
    print("==================")
    print(results_df[metric_columns].std())

    results_df.to_csv(
        "reports/cross_fold_results.csv",
        index=False,
    )

    print("\nSaved reports/cross_fold_results.csv")


if __name__ == "__main__":
    main()
