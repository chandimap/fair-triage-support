
# Fair Triage Support (personal research project)

**Author:** Chandima Liyana Arachchige  
**Purpose:** A personal, research-oriented prototype for **fairness-aware triage decision support**.  
**Scope:** Methods, code and UI for *simulation studies*

This project implements:
- A calibrated risk model (scikit-learn: Logistic Regression / Gradient Boosting)
- A **fairness audit** (Fairlearn; optional AIF360 if available)
- A **Streamlit dashboard** with:
  - patient risk scores,
  - **contrastive explanation** (“why this patient vs a similar one”),
  - a compact **subgroup fairness widget**,
  - a short **bias-check** prompt before finalising priority.
- Analysis helpers approximating **mixed-effects** style evaluation with cluster-robust GLM (statsmodels) + learning terms.

> **Disclaimer:** This repository uses **synthetic data** and is intended **for research and education only**. Do **not** use it for real clinical decisions.

---

## 1) Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Generate synthetic triage-style data:
```bash
python scripts/generate_synthetic_data.py --n 4000 --out data/triage_synth.csv
```

Train a model and run a fairness audit:
```bash
python scripts/train_and_audit.py --data data/triage_synth.csv --target deteriorated_6w --group_col ethnicity --model logreg
```

Launch the dashboard:
```bash
python scripts/run_dashboard.py
# or:
streamlit run dashboard/app.py
```

---

## 2) Data schema (synthetic)

Columns (example):
- Demographics: `age`, `sex` (F/M), `ethnicity` (categorical), `deprivation_quintile` (1–5)
- Health & function: `bmi`, `resting_hr`, `tug_sec`, `walk_6m`, `pain_score`, `mobility_score`
- Comorbidities: `comorb_count`
- **Label:** `deteriorated_6w` (0/1) – proxy for “needs early intervention”

---

## 3) What’s inside

- `src/data.py` – load/clean/split; derive age bands  
- `src/model.py` – preprocessing pipeline, model training, **isotonic calibration**  
- `src/metrics.py` – accuracy, AUC, **Brier score**  
- `src/fairness.py` – **Fairlearn** metrics (demographic parity, equalized odds)  
- `src/explanations.py` – simple **contrastive explanation** (nearest similar case)  
- `src/analysis/mixed_effects.py` – GLM with **cluster-robust** SEs (participant clustering) + learning term  
- `dashboard/app.py` – Streamlit UI (scores, explanation, fairness, bias-check)  
- `scripts/*.py` – synthetic data gen, train+audit, run dashboard

---

## 4) Typical workflow

1. Generate or load dataset → `data/triage_synth.csv`  
2. Train & audit → view metrics and fairness gaps  
3. Explore in dashboard → inspect patient-level explanations, subgroup parity, run a **bias-check**  
4. For behavioural studies (if you run simulations): log trial-level data and analyse with `src/analysis/mixed_effects.py`.

---

## 5) Limitations & next steps

- Prototype explanations (contrastive nearest-neighbour) — can be extended to counterfactuals or SHAP.  
- Mixed-effects are approximated via GLM with **cluster-robust** errors; consider PyMC/Bambi for full hierarchical models.  
- Add **uncertainty** overlays and a small **what-if** panel for accuracy–fairness trade-offs.

---

## 6) License

MIT 
