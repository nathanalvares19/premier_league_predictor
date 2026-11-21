# evaluate classification models

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate(y_valid_encoded, predictions):
    print(f"Accuracy: {accuracy_score(y_valid_encoded, predictions):.5f}")
    print(
        f"Precision: {precision_score(y_valid_encoded, predictions, average='macro', zero_division=0):.5f}"
    )
    print(
        f"Recall: {recall_score(y_valid_encoded, predictions, average='macro', zero_division=0):.5f}"
    )
    print(
        f"F1: {f1_score(y_valid_encoded, predictions, average='macro', zero_division=0):.5f}"
    )
