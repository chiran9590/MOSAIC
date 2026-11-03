
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.preprocessing.sequence_dataset import LOBSequenceDataset


SEQUENCE_LENGTH = 100
STRIDE = 10


def load_sequences(dataset):
    import torch

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

    time = np.arange(
        sequences.shape[1],
        dtype=np.float32,
    )
    time -= time.mean()

    trend = np.einsum(
        "t,ntf->nf",
        time,
        sequences,
    ) / np.sum(time ** 2)

    return {
        "latest": [last],
        "latest_mean": [last, mean],
        "latest_mean_std": [last, mean, std],
        "all_features": [last, mean, std, trend],
    }


def main():
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

    train_groups = extract_groups(X_train_seq)
    val_groups = extract_groups(X_val_seq)

    results = {}

    for name in train_groups:
        X_train = np.concatenate(
            train_groups[name],
            axis=1,
        )

        X_val = np.concatenate(
            val_groups[name],
            axis=1,
        )

        print(f"\nExperiment: {name}")
        print("Feature count:", X_train.shape[1])

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

        results[name] = (accuracy, macro_f1)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")

    print("\nFINAL ABLATION RESULTS")
    print("----------------------")

    for name, (accuracy, macro_f1) in results.items():
        print(
            f"{name:20s} "
            f"Accuracy={accuracy:.4f} "
            f"Macro F1={macro_f1:.4f}"
        )

    np.savez_compressed(
        "reports/sequence_ablation.npz",
        **{
            name: np.array(values)
            for name, values in results.items()
        },
    )


if __name__ == "__main__":
    main()
