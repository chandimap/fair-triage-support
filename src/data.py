
from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split

AGE_BANDS = [(0, 39), (40, 54), (55, 69), (70, 200)]

def _age_to_band(age: float) -> str:
    for lo, hi in AGE_BANDS:
        if lo <= age <= hi:
            return f"{lo}-{hi}"
    return "unknown"

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = {"age","sex","ethnicity","deprivation_quintile","bmi","resting_hr",
              "tug_sec","walk_6m","pain_score","mobility_score","comorb_count","deteriorated_6w"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")
    df = df.dropna(subset=["age","sex","ethnicity","deteriorated_6w"])
    df["age_band"] = df["age"].apply(_age_to_band)
    df["deprivation_quintile"] = df["deprivation_quintile"].astype(int).clip(1,5).astype(str)
    df["sex"] = df["sex"].astype(str)
    df["ethnicity"] = df["ethnicity"].astype(str)
    df["deteriorated_6w"] = df["deteriorated_6w"].astype(int)
    return df

def split_xy(df: pd.DataFrame, target: str, drop_cols=None):
    drop_cols = drop_cols or []
    X = df.drop(columns=[target] + drop_cols)
    y = df[target].values
    return X, y

def train_val_test_split(df: pd.DataFrame, test_size=0.2, val_size=0.2, random_state=42):
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df["deteriorated_6w"], random_state=random_state)
    train_df, val_df = train_test_split(train_df, test_size=val_size, stratify=train_df["deteriorated_6w"], random_state=random_state)
    return train_df, val_df, test_df
