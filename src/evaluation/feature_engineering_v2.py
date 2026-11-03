
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score

from src.preprocessing.sequence_dataset import LOBSequenceDataset
from src.preprocessing.sequence_features import extract_sequence_features
from src.preprocessing.sequence_features_v2 import (
    extract_sequence_features_v2,
)


SEQUENCE_LENGTH = 100
STRIDE = 10
FOLD = 1


def load_sequences(dataset):
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
        np.concatenate(sequences),
        np.concatenate(labels),
    )


def evaluate_features(name, X_train, y_train, X_val, y_val):

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

    print(
        f"{name:25s} | "
        f"Features: {X_train.shape[1]:4d} | "
        f"Accuracy: {accuracy:.4f} | "
        f"Macro F1: {macro_f1:.4f}"
    )

    return {
        "experiment": name,
        "feature_count": X_train.shape[1],
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def main():

    Path("reports").mkdir(exist_ok=True)

    filename = f"train_cf_{FOLD}.npz"

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

    print("Loading sequences...")

    X_train_seq, y_train = load_sequences(train_dataset)
    X_val_seq, y_val = load_sequences(validation_dataset)

    print("Extracting original features...")

    original_train = extract_sequence_features(X_train_seq)
    original_val = extract_sequence_features(X_val_seq)

    print("Extracting V2 features...")

    expanded_train = extract_sequence_features_v2(X_train_seq)
    expanded_val = extract_sequence_features_v2(X_val_seq)

    # Feature groups are arranged in blocks of 144 columns.
    groups = {
        "latest": 0,
        "mean": 1,
        "std": 2,
        "trend": 3,
        "minimum": 4,
        "maximum": 5,
        "median": 6,
        "iqr": 7,
        "recent_mean": 8,
        "recent_std": 9,
    }

    results = []

    print("\nFEATURE ENGINEERING V2 RESULTS")
    print("=" * 95)

    results.append(
        evaluate_features(
            "Original 576",
            original_train,
            y_train,
            original_val,
            y_val,
        )
    )

    results.append(
        evaluate_features(
            "Expanded 1440",
            expanded_train,
            y_train,
            expanded_val,
            y_val,
        )
    )

    # Test cumulative feature groups.
    for group_count in [5, 6, 7, 8, 9, 10]:

        selected_groups = list(groups.values())[:group_count]

        columns = np.concatenate([
            np.arange(group * 144, (group + 1) * 144)
            for group in selected_groups
        ])

        results.append(
            evaluate_features(
                f"First {group_count} groups",
                expanded_train[:, columns],
                y_train,
                expanded_val[:, columns],
                y_val,
            )
        )

    # Select features using training data only.
    print("\nCalculating mutual information rankings...")

    mi = mutual_info_classif(
        expanded_train,
        y_train,
        random_state=42,
    )

    ranking = np.argsort(mi)[::-1]

    for count in [144, 288, 432, 576, 720]:

        selected = ranking[:count]

        results.append(
            evaluate_features(
                f"Top {count} selected",
                expanded_train[:, selected],
                y_train,
                expanded_val[:, selected],
                y_val,
            )
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "macro_f1",
        ascending=False,
    )

    results_df.to_csv(
        "reports/feature_engineering_v2.csv",
        index=False,
    )

    print("\nRANKED RESULTS")
    print("=" * 95)
    print(results_df.to_string(index=False))

    print("\nSaved reports/feature_engineering_v2.csv")


if __name__ == "__main__":
    main()
