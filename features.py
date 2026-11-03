import numpy as np, pandas as pd
W = 10                                           # context window = last 10 readings (about 60 s)
FEATS = ["power", "delta", "roll_mean", "roll_std"]

def roll_feats(power, delta):                    # arrays -> feature matrix (training / evaluation)
    r = pd.Series(power).rolling(W, min_periods=1)
    return np.column_stack([power, delta, r.mean().values, r.std().fillna(0).values])

def add_features(df):                            # cleaned dataframe -> adds roll_mean, roll_std
    df = df.copy(); r = df.power.rolling(W, min_periods=1)
    df["roll_mean"] = r.mean(); df["roll_std"] = r.std().fillna(0); return df

def live_features(buf, p):                       # one new reading + list of previous readings (used by the app)
    a = np.array((list(buf) + [p])[-W:], float)
    return [p, (p - buf[-1]) if len(buf) else 0.0, a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0]
