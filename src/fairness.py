
from __future__ import annotations
import pandas as pd
from typing import Dict, Any
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import recall_score

def compute_fairness(y_true, y_pred, sensitive_series: pd.Series) -> Dict[str, Any]:
    mf = MetricFrame(
        metrics={"selection_rate": selection_rate,
                 "tpr": lambda yt, yp: recall_score(yt, yp, pos_label=1)},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_series.astype(str),
    )
    dp = demographic_parity_difference(y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_series)
    eo = equalized_odds_difference(y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_series)
    return {
        "overall": mf.overall.to_dict(),
        "by_group": mf.by_group.to_dict(),
        "demographic_parity_difference": float(dp),
        "equalized_odds_difference": float(eo),
    }

def binarize(y_proba, threshold=0.5):
    return (y_proba >= threshold).astype(int)
