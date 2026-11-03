
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


RESULTS_FILE = Path("reports/model_predictions_cf1.npz")
OUTPUT_DIR = Path("reports/figures")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("Loading saved predictions...")

    data = np.load(RESULTS_FILE)

    y_true = data["y_true"]

    predictions = {
        "Logistic Regression": data["logistic"],
        "HistGradientBoosting": data["hist_gradient_boosting"],
        "MOSAIC Transformer": data["transformer"],
    }

    # ------------------------------------------
    # 1. Calculate performance metrics
    # ------------------------------------------

    metrics = {}

    for name, y_pred in predictions.items():

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        macro_f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
        )

        metrics[name] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        }

        print("\n" + "=" * 60)
        print(name)

        print("Accuracy:", round(accuracy, 4))
        print("Macro F1:", round(macro_f1, 4))

        print("\nClassification Report:")

        print(
            classification_report(
                y_true,
                y_pred,
                labels=[0, 1, 2],
                target_names=[
                    "Class 1",
                    "Class 2",
                    "Class 3",
                ],
                zero_division=0,
            )
        )

    # ------------------------------------------
    # 2. Accuracy and Macro F1 comparison
    # ------------------------------------------

    names = list(metrics.keys())

    accuracies = [
        metrics[name]["accuracy"]
        for name in names
    ]

    macro_f1_scores = [
        metrics[name]["macro_f1"]
        for name in names
    ]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        x - width / 2,
        accuracies,
        width,
        label="Accuracy",
    )

    ax.bar(
        x + width / 2,
        macro_f1_scores,
        width,
        label="Macro F1",
    )

    ax.set_ylabel("Score")
    ax.set_title("MOSAIC Model Performance Comparison")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)

    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "model_comparison.png",
        dpi=300,
    )

    plt.close()

    # ------------------------------------------
    # 3. Confusion matrices
    # ------------------------------------------

    for name, y_pred in predictions.items():

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1, 2],
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[
                "Class 1",
                "Class 2",
                "Class 3",
            ],
        )

        fig, ax = plt.subplots(figsize=(7, 6))

        display.plot(
            ax=ax,
            cmap="Blues",
            values_format="d",
        )

        ax.set_title(f"{name} - Confusion Matrix")

        plt.tight_layout()

        filename = (
            name.lower()
            .replace(" ", "_")
            + "_confusion_matrix.png"
        )

        plt.savefig(
            OUTPUT_DIR / filename,
            dpi=300,
        )

        plt.close()

    # ------------------------------------------
    # 4. Save numerical metrics
    # ------------------------------------------

    with open(
        Path("reports/model_metrics.txt"),
        "w",
    ) as file:

        for name, values in metrics.items():

            file.write(f"{name}\n")

            file.write(
                f"Accuracy: {values['accuracy']:.4f}\n"
            )

            file.write(
                f"Macro F1: {values['macro_f1']:.4f}\n\n"
            )

    print("\nEvaluation analysis completed.")

    print("Figures saved in:", OUTPUT_DIR)
    print("Metrics saved in: reports/model_metrics.txt")


if __name__ == "__main__":
    main()
