"""Preprocessing for UK-DALE house_1. Falls back to synthetic data if UKDALE_DIR is missing."""
import os, numpy as np, pandas as pd
UKDALE_DIR = os.environ.get("UKDALE_DIR", "ukdale")
# house_1 channel ids, idle threshold (W), typical ON power (W, used only for synthetic data)
APPS = {"microwave": (13, 20, 1100), "kettle": (10, 50, 2500),
        "fridge": (12, 20, 120), "washing_machine": (5, 30, 450)}

def synthetic(app, n=150000, seed=0):
    rng = np.random.default_rng(seed); on = APPS[app][2]
    p = rng.normal(2, 0.5, n).clip(0); i = 0
    while i < n:
        i += rng.integers(100, 600); d = rng.integers(10, 40)
        p[i:i+d] = rng.normal(on, on*0.03, len(p[i:i+d]))
    idx = pd.date_range("2024-01-01", periods=n, freq="6s")
    return pd.DataFrame({"power": p}, index=idx)

def load_raw(app):
    f = os.path.join(UKDALE_DIR, "house_1", f"channel_{APPS[app][0]}.dat")
    if not os.path.exists(f):
        print(f"[!] {f} not found -> using SYNTHETIC data for {app}"); return synthetic(app)
    df = pd.read_csv(f, sep=" ", names=["ts", "power"], dtype={"ts": "int64", "power": "float32"})
    df.index = pd.to_datetime(df.pop("ts"), unit="s"); return df

def clean(df):
    df = df[~df.index.duplicated()].sort_index()          # duplicates
    df = df[df.power >= 0]                                  # negative readings
    df = df.resample("6s").mean().ffill(limit=3).dropna()   # regular grid, fill short gaps
    df["power"] = df.power.clip(upper=df.power.quantile(0.9999))  # sensor glitches
    df["power_smooth"] = df.power.rolling(5, min_periods=1).mean()  # 5-pt SMA
    df["delta"] = df.power.diff().fillna(0)
    return df

def features(power, delta): return np.column_stack([power, delta])

def get_clean(app, save=True):
    df = clean(load_raw(app))
    if save: df.to_csv(f"data/{app}_clean.csv")
    return df
