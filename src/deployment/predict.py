
from pathlib import Path

import joblib
import numpy as np

from src.preprocessing.sequence_features_v2 import (
    extract_sequence_features_v2,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "checkpoints/mosaic_hgb_fold9.joblib"

CLASS_NAMES = {
    0: "DOWN",
    1: "STATIONARY",
    2: "UP",
}


class MOSAICPredictor:

    def __init__(self, model_path=MODEL_PATH):

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"MOSAIC model not found: {model_path}"
            )

        bundle = joblib.load(model_path)

        self.model = bundle["model"]
        self.sequence_length = bundle["sequence_length"]
        self.input_features = bundle["input_features"]
        self.summary_features = bundle["summary_features"]

        print("MOSAIC predictor loaded.")
        print("Sequence length:", self.sequence_length)
        print("Input features:", self.input_features)
        print("Summary features:", self.summary_features)

    def predict(self, sequence):

        sequence = np.asarray(sequence, dtype=np.float32)

        expected_shape = (
            self.sequence_length,
            self.input_features,
        )

        if sequence.shape != expected_shape:
            raise ValueError(
                f"Expected input shape {expected_shape}, "
                f"received {sequence.shape}"
            )

        if not np.isfinite(sequence).all():
            raise ValueError(
                "Input contains NaN or infinite values."
            )

        # Add batch dimension:
        # (100, 144) -> (1, 100, 144)
        sequence_batch = np.expand_dims(sequence, axis=0)

        # Extract expanded sequence features.
        features = extract_sequence_features_v2(
            sequence_batch
        )

        features = np.asarray(features, dtype=np.float32)

        if features.shape != (1, self.summary_features):
            raise ValueError(
                f"Unexpected extracted feature shape: {features.shape}"
            )

        # Predict class.
        prediction = int(self.model.predict(features)[0])

        # Predict class probabilities.
        probabilities = self.model.predict_proba(features)[0]

        probability_dict = {
            CLASS_NAMES[int(class_id)]: float(probability)
            for class_id, probability in zip(
                self.model.classes_,
                probabilities,
            )
        }

        return {
            "prediction": CLASS_NAMES[prediction],
            "class_id": prediction,
            "probabilities": probability_dict,
        }


if __name__ == "__main__":

    print("\nMOSAIC PREDICTION TEST")
    print("=" * 40)

    predictor = MOSAICPredictor()

    # Synthetic input for pipeline verification only.
    # This is NOT real market data.
    rng = np.random.default_rng(42)

    sample_sequence = rng.normal(
        size=(100, 144)
    ).astype(np.float32)

    result = predictor.predict(sample_sequence)

    print("\nPrediction result")
    print("-----------------")
    print("Direction:", result["prediction"])
    print("Class ID:", result["class_id"])

    print("\nClass probabilities:")

    for name, probability in result["probabilities"].items():
        print(f"{name}: {probability:.4f}")

    print("\nPrediction pipeline working.")
