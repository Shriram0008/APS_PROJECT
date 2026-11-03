"""
Preprocessing figures for the microwave only (UK-DALE house_1, channel 13) - same steps as significance_test.py.
Run from the project root (needs ukdale/house_1/channel_13.dat):   python make_figures.py
Window default = 365 days (the report's Run B).  PowerShell, 90-day version:  $env:WINDOW_DAYS=90; python make_figures.py
Outputs: figures/fig1 ... fig9 (.png) and figures/preprocess_summary.csv (also printed)
"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
UKDALE_DIR = os.environ.get("UKDALE_DIR", "ukdale"); WINDOW_DAYS = int(os.environ.get("WINDOW_DAYS", 365))
THRESH, W, SEED = 20, 10, 42
os.makedirs("figures", exist_ok=True); sns.set_theme(); rng = np.random.default_rng(SEED)

def save(name):
    plt.tight_layout(); plt.savefig(f"figures/{name}.png", dpi=150); plt.close()

# ---- load the sample window (chunked) ----
f = os.path.join(UKDALE_DIR, "house_1", "channel_13.dat"); parts, start = [], None
for ch in pd.read_csv(f, sep=" ", names=["ts", "power"], chunksize=1_000_000):
    if start is None: start = ch.ts.iloc[0]
    end = start + WINDOW_DAYS * 86400; parts.append(ch[ch.ts < end])
    if ch.ts.iloc[-1] >= end: break
raw = pd.concat(parts); raw.index = pd.to_datetime(raw.pop("ts"), unit="s")

# ---- preprocessing, keeping intermediate columns for the graphs ----
st = {"raw_rows": len(raw), "start": str(raw.index[0]), "end": str(raw.index[-1])}
d = raw[~raw.index.duplicated()]; d = d[d.power >= 0]; st["after_dup_neg"] = len(d)
gaps = d.index.to_series().diff().dt.total_seconds().dropna()
g = d.resample("6s").mean(); st["grid_rows"] = len(g); st["missing_on_grid"] = int(g.power.isna().sum())
g = g.ffill(limit=5); st["still_missing"] = int(g.power.isna().sum()); g = g.dropna(); st["rows_after_gapfill"] = len(g)
cap = g.power.quantile(0.9999); g["power_raw"] = g.power; g["power"] = g.power.clip(upper=cap)
st["clip_W"] = round(float(cap), 1); st["clipped"] = int((g.power_raw > cap).sum())
g["smooth"] = g.power.rolling(5, min_periods=1).mean()
dt = g.index.to_series().diff(); g["delta"] = g.power.diff().where(dt <= pd.Timedelta("12s"), 0).fillna(0)
g["active"] = g.smooth > THRESH; r = g.power.rolling(W, min_periods=1); g["roll_mean"] = r.mean(); g["roll_std"] = r.std().fillna(0)
cut = int(len(g) * 0.8); steady = g.active & g.active.shift(1, fill_value=False)
st.update(active_rows=int(g.active.sum()), steady_rows=int(steady.sum()), train_rows=cut, test_rows=len(g) - cut)
summ = pd.Series(st, name="microwave"); summ.to_csv("figures/preprocess_summary.csv"); print(summ.to_string())

# ---- fig1: raw power, the busiest day ----
day = g.power_raw.resample("1D").apply(lambda s: (s > THRESH).sum()).idxmax(); dd = g.power_raw.loc[str(day.date())]
plt.figure(figsize=(9, 3.6)); plt.plot(dd.index, dd.values, lw=.8); plt.title(f"Fig 1. Raw microwave power, {day.date()} (busiest day)")
plt.ylabel("Power (W)"); save("fig1_raw_busiest_day")
# ---- fig2: raw distribution ----
plt.figure(figsize=(8, 3.6)); plt.hist(raw.power, bins=80); plt.yscale("log"); plt.xlabel("Power (W)"); plt.ylabel("Count (log)")
plt.title("Fig 2. Distribution of raw power readings"); save("fig2_distribution")
# ---- fig3: gap lengths ----
bins, labs = [12, 30, 60, 600, 3600, np.inf], ["12-30 s", "30-60 s", "1-10 min", "10 min-1 h", "> 1 h"]
gc = pd.cut(gaps, bins, labels=labs).value_counts().reindex(labs).fillna(0)
plt.figure(figsize=(8, 3.6)); plt.bar(labs, gc.values); plt.ylabel("Number of gaps"); plt.title("Fig 3. Gap lengths in the raw data"); save("fig3_gaps")
# ---- fig4: clipping before / after ----
plt.figure(figsize=(8, 3.6)); plt.hist(g.power_raw, bins=80, alpha=.5, label="before"); plt.hist(g.power, bins=80, alpha=.5, label="after")
plt.yscale("log"); plt.axvline(cap, c="r", ls="--", label=f"clip at {cap:.0f} W"); plt.legend(); plt.xlabel("Power (W)")
plt.title("Fig 4. Glitch clipping (99.99th percentile)"); save("fig4_clipping")
# ---- fig5: smoothing ----
i = int(np.argmax(g.power.values)); w = g.iloc[max(0, i - 100): i + 100]
plt.figure(figsize=(8, 3.6)); plt.plot(w.index, w.power, label="raw", lw=1); plt.plot(w.index, w.smooth, label="5-point moving average", lw=2)
plt.legend(); plt.ylabel("Power (W)"); plt.title("Fig 5. Smoothing around the highest-power event"); save("fig5_smoothing")
# ---- fig6: features ----
s = g[steady].sample(min(20000, int(steady.sum())), random_state=SEED); fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].scatter(s.power, s.delta, s=3, alpha=.3); ax[0].set_xlabel("power (W)"); ax[0].set_ylabel("delta (W)")
ax[1].scatter(s.roll_mean, s.roll_std, s=3, alpha=.3); ax[1].set_xlabel("rolling mean, last 10 readings (W)"); ax[1].set_ylabel("rolling std (W)")
fig.suptitle("Fig 6. Features of steady ON-state readings"); save("fig6_features")
# ---- fig7: ON-state ----
plt.figure(figsize=(8, 3.6)); plt.hist(g.smooth[g.active], bins=80); plt.xlabel("Smoothed power (W)")
plt.title(f"Fig 7. ON-state readings ({g.active.mean() * 100:.1f}% of rows above {THRESH} W)"); save("fig7_on_state")
# ---- fig8: chronological split ----
daily = g.power.resample("1D").mean(); plt.figure(figsize=(9, 3.6)); plt.plot(daily.index, daily.values)
plt.axvline(g.index[cut], c="r", ls="--", label="train | test (80/20)"); plt.legend(); plt.ylabel("Daily mean power (W)")
plt.title("Fig 8. Chronological train/test split"); save("fig8_split")
# ---- fig9: injected faults (illustration) ----
idx = np.where(steady.values)[0]; starts = idx[[len(idx) // 4, len(idx) // 2, 3 * len(idx) // 4]]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
for a_, s0, name, fn in zip(ax, starts, ["Spike", "Drop", "Drift (-18%)"], [lambda x: x * 2 + 150, lambda x: x * .45, lambda x: x * .82]):
    lo, hi = max(0, s0 - 20), s0 + 30; x = g.power.values[lo:hi].copy(); m = x.copy(); k = s0 - lo; m[k:k + 10] = fn(m[k:k + 10])
    a_.plot(range(len(x)), x, label="original"); a_.plot(range(len(m)), m, label="with injected fault"); a_.set_title(name); a_.set_xlabel("reading")
ax[0].set_ylabel("Power (W)"); ax[0].legend(); fig.suptitle("Fig 9. Injected sustained faults (10 readings each)"); save("fig9_injected_faults")
print("saved figures/")
