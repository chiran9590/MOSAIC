
from pathlib import Path
import json
import joblib
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.preprocessing.sequence_features_v2 import (
    extract_sequence_features_v2,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data/processed/train_cf_9.npz"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

MODEL_PATH = CHECKPOINT_DIR / "mosaic_hgb_fold9.joblib"
METADATA_PATH = CHECKPOINT_DIR / "mosaic_hgb_fold9_metadata.json"

SEQUENCE_LENGTH = 100
STRIDE = 10
BATCH_SIZE = 64

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_training_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {DATA_PATH}"
        )

    data = np.load(DATA_PATH)

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)

    # Original FI-2010 labels: 1, 2, 3
    # Model labels: 0, 1, 2
    y = y - 1

    if not np.isin(y, [0, 1, 2]).all():
        raise ValueError("Unexpected target labels.")

    print("Training data loaded.")
    print("Observations:", len(X))
    print("Features per observation:", X.shape[1])
    print("Class distribution:", np.bincount(y))

    return X, y


# ============================================================
# BUILD SEQUENCE FEATURES IN BATCHES
# ============================================================

def build_features(X, y):

    n_observations = len(X)

    if n_observations < SEQUENCE_LENGTH:
        raise ValueError("Not enough observations for a sequence.")

    end_indices = np.arange(
        SEQUENCE_LENGTH - 1,
        n_observations,
        STRIDE,
    )

    feature_batches = []
    label_batches = []

    print("\nExtracting expanded sequence features...")

    for start in range(0, len(end_indices), BATCH_SIZE):

        batch_indices = end_indices[
            start:start + BATCH_SIZE
        ]

        sequences = np.stack([
            X[index - SEQUENCE_LENGTH + 1:index + 1]
            for index in batch_indices
        ])

        features = extract_sequence_features_v2(sequences)

        labels = y[batch_indices]

        feature_batches.append(
            np.asarray(features, dtype=np.float32)
        )

        label_batches.append(labels)

        completed = min(start + BATCH_SIZE, len(end_indices))

        print(
            f"\rProcessed {completed:,}/{len(end_indices):,} sequences",
            end="",
        )

    print()

    X_sequences = np.concatenate(feature_batches)
    y_sequences = np.concatenate(label_batches)

    if not np.isfinite(X_sequences).all():
        raise ValueError("Extracted features contain NaN or infinity.")

    print("Sequence feature shape:", X_sequences.shape)
    print("Sequence label shape:", y_sequences.shape)

    return X_sequences, y_sequences


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(X_train, y_train):

    print("\nTraining HistGradientBoosting model...")

    model = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_train)

    print("\nTraining diagnostics")
    print("--------------------")

    print(
        "Training accuracy:",
        round(accuracy_score(y_train, predictions), 4),
    )

    print(
        "Training macro F1:",
        round(
            f1_score(
                y_train,
                predictions,
                average="macro",
            ),
            4,
        ),
    )

    return model


# ============================================================
# SAVE MODEL AND METADATA
# ============================================================

def save_model(model, X_train, y_train):

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    bundle = {
        "model": model,
        "sequence_length": SEQUENCE_LENGTH,
        "stride": STRIDE,
        "input_features": 144,
        "summary_features": X_train.shape[1],
        "training_fold": 9,
        "classes": {
            0: "DOWN",
            1: "STATIONARY",
            2: "UP",
        },
    }

    joblib.dump(bundle, MODEL_PATH)

    metadata = {
        "project": "MOSAIC",
        "model": "HistGradientBoostingClassifier",
        "dataset": "FI-2010 NoAuction Z-score",
        "training_fold": 9,
        "sequence_length": SEQUENCE_LENGTH,
        "stride": STRIDE,
        "input_features": 144,
        "summary_features": int(X_train.shape[1]),
        "training_sequences": int(len(X_train)),
        "training_class_distribution": {
            "DOWN": int(np.sum(y_train == 0)),
            "STATIONARY": int(np.sum(y_train == 1)),
            "UP": int(np.sum(y_train == 2)),
        },
        "limitations": [
            "Trained on fold 9 training data only.",
            "Training metrics are not held-out test metrics.",
            "Does not establish cross-asset generalisation.",
            "Does not establish trading profitability.",
        ],
    }

    with open(METADATA_PATH, "w") as file:
        json.dump(metadata, file, indent=4)

    print("\nModel saved successfully.")
    print("Model:", MODEL_PATH)
    print("Metadata:", METADATA_PATH)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 55)
    print("MOSAIC DEPLOYMENT MODEL TRAINING")
    print("=" * 55)

    X, y = load_training_data()

    X_sequences, y_sequences = build_features(X, y)

    model = train_model(
        X_sequences,
        y_sequences,
    )

    save_model(
        model,
        X_sequences,
        y_sequences,
    )

    print("\nMOSAIC deployment model is ready.")


if __name__ == "__main__":
    main()
