
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.models.transformer import MOSAICTransformer
from src.preprocessing.sequence_dataset import LOBSequenceDataset


# Configuration
SEQUENCE_LENGTH = 100
STRIDE = 10
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.0005

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)


def evaluate(model, loader, device):
    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)

            outputs = model(X)
            predicted = outputs.argmax(dim=1)

            predictions.extend(predicted.cpu().numpy())
            targets.extend(y.numpy())

    accuracy = accuracy_score(targets, predictions)

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
    )

    return accuracy, macro_f1, targets, predictions


def main():
    print("Using device:", DEVICE)

    # Original training fold
    data = np.load("data/processed/train_cf_1.npz")
    total_observations = len(data["X"])

    split_index = int(total_observations * 0.8)

    print("Total training observations:", total_observations)
    print("Training ends at:", split_index)
    print("Validation begins at:", split_index)

    # Keep sequences entirely within their respective partitions.
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
        end_index=total_observations,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("Training sequences:", len(train_dataset))
    print("Validation sequences:", len(validation_dataset))

    # Class weights from training sequences only.
    train_labels = train_dataset.y[
        train_dataset.indices
    ].numpy()

    class_counts = np.bincount(
        train_labels,
        minlength=3,
    )

    class_weights = len(train_labels) / (
        3 * np.maximum(class_counts, 1)
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=DEVICE,
    )

    print("Training class counts:", class_counts)

    model = MOSAICTransformer().to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001,
    )

    best_validation_f1 = -1.0

    # Training loop
    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0

        for X, y in train_loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(X)
            loss = criterion(outputs, y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        # Validation only — no test data used here.
        accuracy, macro_f1, _, _ = evaluate(
            model,
            validation_loader,
            DEVICE,
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Validation accuracy: {accuracy:.4f} | "
            f"Validation macro F1: {macro_f1:.4f}"
        )

        if macro_f1 > best_validation_f1:
            best_validation_f1 = macro_f1

            torch.save(
                model.state_dict(),
                CHECKPOINT_DIR / "mosaic_transformer_best.pth",
            )

            print("Saved best validation checkpoint.")

    print("\nTraining complete.")
    print("Best validation macro F1:", round(best_validation_f1, 4))


if __name__ == "__main__":
    main()
