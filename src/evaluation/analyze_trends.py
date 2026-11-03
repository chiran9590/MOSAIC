
import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

from src.preprocessing.sequence_dataset import LOBSequenceDataset


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


def main():
    data = np.load("data/processed/train_cf_1.npz")
    split_index = int(len(data["X"]) * 0.8)

    dataset = LOBSequenceDataset(
        "train_cf_1.npz",
        sequence_length=100,
        stride=10,
        start_index=0,
        end_index=split_index,
    )

    X_seq, y = load_sequences(dataset)

    time = np.arange(100, dtype=np.float32)
    time -= time.mean()

    trends = np.einsum(
        "t,ntf->nf",
        time,
        X_seq,
    ) / np.sum(time ** 2)

    print("Trend feature matrix:", trends.shape)

    # Standardise for easier interpretation.
    scaler = StandardScaler()
    trends_scaled = scaler.fit_transform(trends)

    print("Calculating mutual information...")

    scores = mutual_info_classif(
        trends_scaled,
        y,
        random_state=42,
    )

    results = pd.DataFrame({
        "feature": np.arange(1, 145),
        "mutual_information": scores,
        "mean_trend": trends.mean(axis=0),
        "trend_std": trends.std(axis=0),
    })

    results = results.sort_values(
        "mutual_information",
        ascending=False,
    )

    print("\nTop 20 trend features")
    print(results.head(20).to_string(index=False))

    results.to_csv(
        "reports/trend_feature_importance.csv",
        index=False,
    )

    print("\nSaved reports/trend_feature_importance.csv")


if __name__ == "__main__":
    main()
