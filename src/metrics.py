
from __future__ import annotations
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

def classification_metrics(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "brier": brier_score_loss(y_true, y_proba),
    }
