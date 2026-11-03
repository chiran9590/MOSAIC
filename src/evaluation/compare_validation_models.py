
from pathlib import Path

import numpy as np
import torch

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

from src.preprocessing.sequence_dataset import LOBSequenceDataset
from src.preprocessing.sequence_features import extract_sequence_features
from src.models.transformer import MOSAICTransformer


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


def evaluate_model(name, y_true, predictions):
    accuracy = accuracy_score(y_true, predictions)
    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
    )

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print(
        classification_report(
            y_true,
            predictions,
            labels=[0, 1, 2],
            target_names=["Class 1", "Class 2", "Class 3"],
            zero_division=0,
        )
    )

    return accuracy, macro_f1


def main():
    Path("reports").mkdir(exist_ok=True)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Using device:", device)

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

    print("Loading sequences...")

    X_train_seq, y_train = load_sequences(train_dataset)
    X_val_seq, y_val = load_sequences(validation_dataset)

    print("Training sequences:", X_train_seq.shape)
    print("Validation sequences:", X_val_seq.shape)

    # Model 1 and 2: use only the latest observation.
    X_train_last = X_train_seq[:, -1, :]
    X_val_last = X_val_seq[:, -1, :]

    # Model 3: use summary statistics from the full sequence.
    X_train_summary = extract_sequence_features(X_train_seq)
    X_val_summary = extract_sequence_features(X_val_seq)

    predictions = {}

    # Model 1: Logistic Regression
    print("\nTraining Logistic Regression...")

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
    )

    logistic.fit(X_train_last, y_train)
    predictions["logistic"] = logistic.predict(X_val_last)

    # Model 2: HistGradientBoosting using latest observation
    print("\nTraining HistGradientBoosting (latest observation)...")

    hist_latest = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )

    hist_latest.fit(X_train_last, y_train)
    predictions["hist_latest"] = hist_latest.predict(X_val_last)

    # Model 3: HistGradientBoosting using sequence summaries
    print("\nTraining HistGradientBoosting (sequence summaries)...")

    hist_sequence = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )

    hist_sequence.fit(X_train_summary, y_train)
    predictions["hist_sequence"] = hist_sequence.predict(
        X_val_summary
    )

    # Model 4: MOSAIC Transformer
    print("\nLoading MOSAIC Transformer...")

    transformer = MOSAICTransformer().to(device)

    checkpoint = torch.load(
        "checkpoints/mosaic_transformer_best.pth",
        map_location=device,
        weights_only=True,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    transformer.load_state_dict(checkpoint)
    transformer.eval()

    transformer_predictions = []

    with torch.no_grad():
        for start in range(0, len(X_val_seq), BATCH_SIZE):
            batch = torch.tensor(
                X_val_seq[start:start + BATCH_SIZE],
                dtype=torch.float32,
                device=device,
            )

            output = transformer(batch)
            batch_predictions = output.argmax(dim=1)

            transformer_predictions.extend(
                batch_predictions.cpu().numpy()
            )

    predictions["transformer"] = np.asarray(
        transformer_predictions
    )

    # Evaluate all models on the same validation observations.
    results = {}

    for name, pred in predictions.items():
        results[name] = evaluate_model(
            name,
            y_val,
            pred,
        )

    # Save results.
    np.savez_compressed(
        "reports/four_model_validation.npz",
        y_true=y_val,
        logistic=predictions["logistic"],
        hist_latest=predictions["hist_latest"],
        hist_sequence=predictions["hist_sequence"],
        transformer=predictions["transformer"],
    )

    with open("reports/four_model_validation.txt", "w") as file:
        file.write("FOUR-MODEL VALIDATION COMPARISON\n\n")

        for name, (accuracy, macro_f1) in results.items():
            file.write(
                f"{name}: Accuracy={accuracy:.4f}, "
                f"Macro F1={macro_f1:.4f}\n"
            )

    print("\nSaved:")
    print("reports/four_model_validation.npz")
    print("reports/four_model_validation.txt")


if __name__ == "__main__":
    main()
