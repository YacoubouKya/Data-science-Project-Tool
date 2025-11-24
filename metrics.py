# modules/utils/metrics.py

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
import numpy as np

def classification_metrics(y_true, y_pred):
    """
    Calcule des métriques de classification.
    Gère le cas où les prédictions sont continues (ex: probabilités ou régression).
    """
    # Si les prédictions sont continues (floats), on les arrondit ou prend argmax
    if np.issubdtype(np.array(y_pred).dtype, np.floating):
        # Cas binaire : on arrondit
        if len(np.unique(y_true)) == 2:
            y_pred = (np.array(y_pred) >= 0.5).astype(int)
        else:
            # Cas multi-classes : prendre la classe max
            y_pred = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else np.round(y_pred).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted")),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted")),
    }

    # Ajout rapport complet et matrice de confusion
    #metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    #metrics["classification_report"] = classification_report(y_true, y_pred, output_dict=True)

    return metrics


def regression_metrics(y_true, y_pred):
    """
    Calcule des métriques de régression.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
    }