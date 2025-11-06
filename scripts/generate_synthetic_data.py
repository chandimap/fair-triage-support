
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def generate(n=4000, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 90, size=n)
    sex = rng.choice(["F","M"], size=n, p=[0.55,0.45])
    ethnicity = rng.choice(["White","South Asian","Black","Other"], size=n, p=[0.70,0.15,0.08,0.07])
    dep = rng.integers(1,6,size=n)  # 1..5
    bmi = rng.normal(27, 4.5, size=n).clip(16, 50)
    hr = rng.normal(72, 10, size=n).clip(40, 130)
    tug = rng.normal(12, 3, size=n).clip(6, 40)
    walk = rng.normal(450, 80, size=n).clip(150, 700)
    pain = rng.integers(0, 11, size=n)
    mobility = rng.integers(0, 11, size=n)
    comorb = rng.integers(0, 5, size=n)

    # Base deterioration logit
    logit = (
        -3.0 + 0.02*(age-60) + 0.06*(bmi-27) + 0.02*(hr-72) + 0.07*(tug-12)
        - 0.004*(walk-450) + 0.08*(pain-5) + 0.05*(mobility-5) + 0.18*comorb
    )
    # Inject small group disparity for auditing
    eth_effect = {"White": 0.0, "South Asian": 0.18, "Black": 0.12, "Other": 0.05}
    logit += np.array([eth_effect[e] for e in ethnicity]) + 0.03*(dep-3)

    prob = 1/(1+np.exp(-logit))
    y = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        "age": age, "sex": sex, "ethnicity": ethnicity, "deprivation_quintile": dep,
        "bmi": bmi, "resting_hr": hr, "tug_sec": tug, "walk_6m": walk,
        "pain_score": pain, "mobility_score": mobility, "comorb_count": comorb,
        "deteriorated_6w": y
    })
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--out", type=str, default="data/triage_synth.csv")
    args = ap.parse_args()
    Path("data").mkdir(parents=True, exist_ok=True)
    df = generate(n=args.n)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with shape {df.shape}")
