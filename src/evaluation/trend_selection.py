
import numpy as np
import pandas as pd
import torch

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from src.preprocessing.sequence_dataset import LOBSequenceDataset


def load_sequences(dataset):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
    )

    sequences = []
    labels = []

    for x, y in loader:
        sequences.append(x.numpy())
        labels.append(y.numpy())

    return np.concatenate(sequences), np.concatenate(labels)


def extract_groups(sequences):
    last = sequences[:, -1, :]
    mean = sequences.mean(axis=1)
    std = sequences.std(axis=1)

    time = np.arange(100, dtype=np.float32)
    time -= time.mean()

    trend = np.einsum(
        "t,ntf->nf",
        time,
        sequences,
    ) / np.sum(time ** 2)

    return last, mean, std, trend


def main():
    data = np.load("data/processed/train_cf_1.npz")
    split_index = int(len(data["X"]) * 0.8)

    train_dataset = LOBSequenceDataset(
        "train_cf_1.npz",
        sequence_length=100,
        stride=10,
        start_index=0,
        end_index=split_index,
    )

    val_dataset = LOBSequenceDataset(
        "train_cf_1.npz",
        sequence_length=100,
        stride=10,
        start_index=split_index,
        end_index=None,
    )

    X_train_seq, y_train = load_sequences(train_dataset)
    X_val_seq, y_val = load_sequences(val_dataset)

    train_groups = extract_groups(X_train_seq)
    val_groups = extract_groups(X_val_seq)

    last_train, mean_train, std_train, trend_train = train_groups
    last_val, mean_val, std_val, trend_val = val_groups

    print("Calculating trend feature rankings...")

    scaler = StandardScaler()
    trend_scaled = scaler.fit_transform(trend_train)

    scores = mutual_info_classif(
        trend_scaled,
        y_train,
        random_state=42,
    )

    ranked_features = np.argsort(scores)[::-1]

    feature_counts = [20, 40, 60, 144]
    results = {}

    for count in feature_counts:
        selected = ranked_features[:count]

        X_train = np.concatenate(
            [
                last_train,
                mean_train,
                std_train,
                trend_train[:, selected],
            ],
            axis=1,
        )

        X_val = np.concatenate(
            [
                last_val,
                mean_val,
                std_val,
                trend_val[:, selected],
            ],
            axis=1,
        )

        model = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        )

        print(f"\nTraining with top {count} trend features...")

        model.fit(X_train, y_train)
        predictions = model.predict(X_val)

        accuracy = accuracy_score(y_val, predictions)
        macro_f1 = f1_score(
            y_val,
            predictions,
            average="macro",
        )

        results[f"top_{count}"] = [accuracy, macro_f1]

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")

    results_df = pd.DataFrame(
        results,
        index=["accuracy", "macro_f1"],
    ).T

    print("\nTREND FEATURE SELECTION RESULTS")
    print(results_df.to_string())

    results_df.to_csv(
        "reports/trend_selection_results.csv"
    )

    print("\nSaved reports/trend_selection_results.csv")


if __name__ == "__main__":
    main()
