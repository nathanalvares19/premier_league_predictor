# evaluate classification models
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt


def evaluate(model, X_valid_encoded, y_valid_encoded, predictions):
    predictions = model.predict(X_valid_encoded)

    print(f"Accuracy: {accuracy_score(y_valid_encoded, predictions):.5f}")

    print(
        classification_report(
            y_valid_encoded,
            predictions,
            labels=[0, 1, 2],
            target_names=["Win", "Loss", "Draw"],
            zero_division=0,
        )
    )

    ConfusionMatrixDisplay.from_predictions(
        y_valid_encoded,
        predictions,
        labels=[0, 1, 2],
        display_labels=["Win", "Loss", "Draw"],
        cmap="Blues",
    )

    plt.show()
