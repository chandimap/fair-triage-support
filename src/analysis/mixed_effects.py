
from __future__ import annotations
import statsmodels.api as sm
import pandas as pd

def glm_binomial_cluster(df: pd.DataFrame, y: str, x_vars: list[str], cluster: str):
    """
    Binomial GLM with cluster-robust standard errors by participant.
    Add 'trial_index' (or similar) to x_vars to account for learning over time.
    """
    X = sm.add_constant(df[x_vars])
    model = sm.GLM(df[y], X, family=sm.families.Binomial())
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster]})
    return res

def ols_cluster(df: pd.DataFrame, y: str, x_vars: list[str], cluster: str):
    X = sm.add_constant(df[x_vars])
    model = sm.OLS(df[y], X)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster]})
    return res
