
import argparse
from pathlib import Path
from src.config import TARGET_COL, NUM_COLS, CAT_COLS, MODELS_DIR
from src.data import load_data, split_xy
from src.model import build_pipeline, fit_and_eval, save_pipeline
from src.fairness import compute_fairness, binarize

def main(data_path: str, target: str, group_col: str, model: str):
    df = load_data(data_path)
    X, y = split_xy(df, target)
    pipe = build_pipeline(NUM_COLS, CAT_COLS, model_kind=model)

    m = int(0.8 * len(df))
    X_train, y_train = X.iloc[:m], y[:m]
    X_val, y_val = X.iloc[m:], y[m:]

    metrics = fit_and_eval(pipe, X_train, y_train, X_val, y_val)
    proba = pipe.predict_proba(X_val)[:, 1]
    y_pred = binarize(proba, threshold=0.5)
    fair = compute_fairness(y_val, y_pred, df[group_col].iloc[m:])

    out = MODELS_DIR / f"pipeline_{model}.joblib"
    save_pipeline(pipe, out)

    print("Model metrics:", metrics)
    print("Fairness summary:", {k: v for k, v in fair.items() if k not in ["by_group","overall"]})
    print("Saved model to:", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--target", type=str, default=TARGET_COL)
    ap.add_argument("--group_col", type=str, default="ethnicity")
    ap.add_argument("--model", type=str, default="logreg", choices=["logreg","gb"])
    args = ap.parse_args()
    main(args.data, args.target, args.group_col, args.model)
