
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.preprocessing.sequence_dataset import LOBSequenceDataset
from src.preprocessing.sequence_features import extract_sequence_features
from src.preprocessing.sequence_features_v2 import extract_sequence_features_v2


SEQUENCE_LENGTH = 100
STRIDE = 10
TRAIN_RATIO = 0.80

MODEL_PARAMS = {
    "max_iter": 150,
    "learning_rate": 0.08,
    "max_leaf_nodes": 31,
    "l2_regularization": 1,
    "random_state": 42,
}


def build_features(dataset, feature_version):
    sequences = []
    labels = []

    for i in range(len(dataset)):
        sequence, target = dataset[i]
        sequences.append(sequence.numpy())
        labels.append(int(target))

    sequences = np.asarray(sequences, dtype=np.float32)
    labels = np.asarray(labels)

    if feature_version == "original":
        features = extract_sequence_features(sequences)
    else:
        features = extract_sequence_features_v2(sequences)

    return features, labels


def evaluate_fold(fold, feature_version):
    filename = f"train_cf_{fold}.npz"

    data = np.load(Path("data/processed") / filename)
    total_observations = len(data["X"])

    split_index = int(total_observations * TRAIN_RATIO)

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
        end_index=total_observations,
    )

    X_train, y_train = build_features(
        train_dataset, feature_version
    )

    X_val, y_val = build_features(
        validation_dataset, feature_version
    )

    model = HistGradientBoostingClassifier(**MODEL_PARAMS)

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    return {
        "fold": fold,
        "feature_version": feature_version,
        "feature_count": X_train.shape[1],
        "accuracy": accuracy_score(y_val, predictions),
        "macro_f1": f1_score(
            y_val, predictions, average="macro"
        ),
    }


def main():
    results = []

    for fold in range(1, 10):
        for version in ["original", "expanded"]:
            print(
                f"Evaluating fold {fold}/9 | "
                f"Features: {version}",
                flush=True,
            )

            result = evaluate_fold(fold, version)
            results.append(result)

            print(
                f"Accuracy: {result['accuracy']:.4f} | "
                f"Macro F1: {result['macro_f1']:.4f}",
                flush=True,
            )

    df = pd.DataFrame(results)

    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)

    df.to_csv(
        output_dir / "cross_fold_feature_v2.csv",
        index=False,
    )

    print("\nCROSS-FOLD FEATURE COMPARISON")
    print("=" * 75)

    summary = df.groupby("feature_version")[
        ["accuracy", "macro_f1"]
    ].agg(["mean", "std"])

    print(summary.round(4))

    print("\nPER-FOLD RESULTS")
    print(
        df.pivot(
            index="fold",
            columns="feature_version",
            values="macro_f1",
        ).round(4)
    )

    print(
        "\nSaved: reports/cross_fold_feature_v2.csv"
    )


if __name__ == "__main__":
    main()
