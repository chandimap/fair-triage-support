
from __future__ import annotations
import numpy as np
from sklearn.neighbors import NearestNeighbors

def build_contrastive_explainer(X_embed: np.ndarray, y_scores: np.ndarray, k=50):
    """
    Very lightweight contrastive explainer:
    - Build NN index on embedded/preprocessed features
    - For a case i, find a close neighbour with opposite predicted label
    - Return the largest feature deltas as a human-readable summary
    """
    nn = NearestNeighbors(n_neighbors=min(k, len(X_embed)))
    nn.fit(X_embed)

    def explain(idx: int, feature_names: list[str], X: np.ndarray, scores: np.ndarray):
        label = int(scores[idx] >= 0.5)
        _, indices = nn.kneighbors(X[idx].reshape(1, -1), return_distance=True)
        alt = None
        for j in indices[0]:
            if int(scores[j] >= 0.5) != label:
                alt = j
                break
        if alt is None:
            return {"text": "No close contrasting case found.", "deltas": {}}
        delta = X[idx] - X[alt]
        top = np.argsort(np.abs(delta))[-5:][::-1]
        deltas = {feature_names[t]: float(delta[t]) for t in top}
        text = "Compared to a similar patient with a different predicted priority, key differences were: " +                ", ".join([f"{k} (Δ={v:.2f})" for k, v in deltas.items()])
        return {"text": text, "deltas": deltas}

    return explain
