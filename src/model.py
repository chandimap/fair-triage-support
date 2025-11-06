
from __future__ import annotations
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

def build_preprocessor(num_cols, cat_cols) -> ColumnTransformer:
    num = Pipeline(steps=[("scaler", StandardScaler())])
    cat = Pipeline(steps=[("ohe", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)], remainder="drop")

def build_model(kind: str = "logreg"):
    if kind == "logreg":
        base = LogisticRegression(max_iter=200)
    elif kind == "gb":
        base = GradientBoostingClassifier()
    else:
        raise ValueError(f"Unknown model: {kind}")
    return CalibratedClassifierCV(base, method="isotonic", cv=3)

def build_pipeline(num_cols, cat_cols, model_kind="logreg"):
    pre = build_preprocessor(num_cols, cat_cols)
    clf = build_model(model_kind)
    return Pipeline([("prep", pre), ("clf", clf)])

def fit_and_eval(pipe: Pipeline, X_train, y_train, X_val, y_val):
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_val)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(y_val, proba),
        "acc": accuracy_score(y_val, pred),
        "brier": brier_score_loss(y_val, proba),
    }

def save_pipeline(pipe: Pipeline, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_path)

def load_pipeline(path: Path) -> Pipeline:
    return joblib.load(path)
