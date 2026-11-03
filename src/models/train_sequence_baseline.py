
from pathlib import Path

import numpy as np
import torch

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

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

    all_sequences = []
    all_labels = []

    for sequences, labels in loader:
        all_sequences.append(sequences.numpy())
        all_labels.append(labels.numpy())

    return (
        np.concatenate(all_sequences, axis=0),
        np.concatenate(all_labels, axis=0),
    )


def main():
    print("Loading training sequences...")

    train_dataset = LOBSequenceDataset(
        "train_cf_1.npz",
        sequence_length=SEQUENCE_LENGTH,
        stride=STRIDE,
        start_index=0,
        end_index=None,
    )

    # Use the same chronological 80/20 split as the Transformer.
    data = np.load("data/processed/train_cf_1.npz")
    split_index = int(len(data["X"]) * 0.8)

    train_dataset = LOBSequenceDataset(
        "train_cf_1.npz",
        sequence_length=SEQUENCE_LENGTH,
        stride=STRIDE,
        start_index=0,
        end_index=split_index,
    )

    validation_dataset = LOBSequenceDataset(
        "train_cf_1.npz",
        sequence_length=SEQUENCE_LENGTH,
        stride=STRIDE,
        start_index=split_index,
        end_index=None,
    )

    X_train_seq, y_train = load_sequences(train_dataset)
    X_val_seq, y_val = load_sequences(validation_dataset)

    print("Training sequence shape:", X_train_seq.shape)
    print("Validation sequence shape:", X_val_seq.shape)

    print("Extracting summary features...")

    X_train = extract_sequence_features(X_train_seq)
    X_val = extract_sequence_features(X_val_seq)

    print("Training feature shape:", X_train.shape)
    print("Validation feature shape:", X_val.shape)

    print("Training HistGradientBoosting...")

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

    print("\nSequence-aware baseline validation results")
    print("-------------------------------------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_val,
            predictions,
            labels=[0, 1, 2],
            target_names=["Class 1", "Class 2", "Class 3"],
            zero_division=0,
        )
    )

    Path("checkpoints").mkdir(exist_ok=True)

    np.savez_compressed(
        "reports/sequence_baseline_validation.npz",
        y_true=y_val,
        predictions=predictions,
    )

    print("Saved validation predictions.")


if __name__ == "__main__":
    main()
