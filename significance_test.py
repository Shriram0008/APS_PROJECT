"""
Is the improved model really better, or is the difference due to test-sample variation?

Problem   : classify microwave power readings as normal (0) or fault (1)   [UK-DALE house_1, channel 13]
Baseline  : Isolation Forest      Improved: Local Outlier Factor (LOF)    (both trained WITHOUT labels, on normal data)
Test set  : ONE fixed held-out test period (last 20 % in time) with injected sustained faults; both models scored on the same rows
Statistics: block-bootstrap sampling of the test set -> 95 % CIs for F1 and for the F1 difference;
            Wilson CIs for proportions; McNemar exact test; paired block-permutation test.
Run       : python significance_test.py      (needs ukdale/house_1/channel_13.dat; set UKDALE_DIR to change the folder)
Outputs   : results/summary.json, results/summary.txt, results/bootstrap_diff.png
"""
import os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# ---------------- configuration (fixed seed -> reproducible) ----------------
SEED = 42; UKDALE_DIR = os.environ.get("UKDALE_DIR", "ukdale"); WINDOW_DAYS = int(os.environ.get("WINDOW_DAYS", 90))
CHANNEL, THRESH, W, CONTAM = 13, 20, 10, 0.02        # microwave channel, ON threshold (W), context window, contamination
SEG, FAULT_FRAC, MAX_FIT = 10, 0.10, 5000
MIN_SEG = int(os.environ.get("MIN_SEG", 8))           # fault length (readings), share of faults in test set, LOF training sample
B, BLOCK, N_PERM, ALPHA = 2000, 20, 5000, 0.05       # bootstrap resamples, block size, permutations, significance level
rng = np.random.default_rng(SEED)

# ---------------- 1. data preparation ----------------
def load_window():
    f = os.path.join(UKDALE_DIR, "house_1", f"channel_{CHANNEL}.dat"); parts, start = [], None
    for ch in pd.read_csv(f, sep=" ", names=["ts", "power"], chunksize=1_000_000):   # read in chunks, stop after the sample window
        if start is None: start = ch.ts.iloc[0]
        end = start + WINDOW_DAYS * 86400; parts.append(ch[ch.ts < end])
        if ch.ts.iloc[-1] >= end: break
    df = pd.concat(parts); df.index = pd.to_datetime(df.pop("ts"), unit="s"); return df

def clean(df):
    df = df[~df.index.duplicated()]; df = df[df.power >= 0]                     # duplicates, negatives
    g = df.resample("6s").mean().ffill(limit=5).dropna()                        # regular 6 s grid, fill gaps <= 30 s
    g["power"] = g.power.clip(upper=g.power.quantile(0.9999))                   # clip sensor glitches
    smooth = g.power.rolling(5, min_periods=1).mean()                           # 5-point moving average (used for ON/OFF only)
    dt = g.index.to_series().diff()
    g["delta"] = g.power.diff().where(dt <= pd.Timedelta("12s"), 0).fillna(0)   # change vs previous reading (0 across gaps)
    g["active"] = (smooth > THRESH).values; return g

def roll_feats(power, delta):                                                    # features: power, delta, rolling mean/std of last W readings
    r = pd.Series(power).rolling(W, min_periods=1)
    return np.column_stack([power, delta, r.mean().values, r.std().fillna(0).values])

def inject(P0, D0, te_idx, rng):                                                 # sustained faults: spike, drop, 18 % drift
    n_seg = max(8, int(FAULT_FRAC * len(te_idx) / SEG)); p = P0.copy(); lab = np.zeros(len(p), bool)
    for s in rng.choice(te_idx, n_seg, replace=False):
        e = min(len(p), s + SEG); k = rng.choice(["spike", "drop", "drift"])
        if k == "spike": p[s:e] = P0[s:e] * rng.uniform(1.5, 2.5) + 150
        elif k == "drop": p[s:e] = P0[s:e] * rng.uniform(.3, .6)
        else: p[s:e] = P0[s:e] * 0.82
        lab[s:e] = True
    sh = p - P0; d = D0 + sh - np.r_[0, sh[:-1]]
    near = pd.Series(lab).rolling(W, min_periods=1).max().astype(bool).values   # fault + recovery rows
    return p, d, lab, near

# ---------------- 2. statistics helpers ----------------
def f1(y, p):
    tp, fp, fn = int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 1)).sum()), int(((y == 1) & (p == 0)).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0

def wilson(k, n):                                                                # 95 % Wilson CI for a proportion
    c = stats.binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="wilson"); return [float(c.low), float(c.high)]

def block_bootstrap(y, p1, p2, block_id, nb):                                    # resample blocks of consecutive rows (keeps time dependence)
    blocks = [np.where(block_id == i)[0] for i in range(nb)]; out = np.empty((B, 3))
    for b in range(B):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, nb, nb)])
        a, c = f1(y[idx], p1[idx]), f1(y[idx], p2[idx]); out[b] = (a, c, c - a)
    return out

def block_permutation_p(y, p1, p2, block_id, nb, obs):                           # H0: models exchangeable -> swap predictions block by block
    cnt = 0
    for _ in range(N_PERM):
        sw = (rng.random(nb) < .5)[block_id]; a, c = np.where(sw, p2, p1), np.where(sw, p1, p2)
        cnt += abs(f1(y, c) - f1(y, a)) >= abs(obs) - 1e-12
    return (cnt + 1) / (N_PERM + 1)

# ---------------- 3. main experiment ----------------
def main():
    os.makedirs("results", exist_ok=True)
    g = clean(load_window()); P0, D0 = g.power.values.astype(float), g.delta.values.astype(float)
    act = g.active.values; steady = act & np.r_[False, act[:-1]]                # running now and in the previous reading
    is_tr = np.arange(len(g)) < int(len(g) * 0.8)                               # chronological 80/20 split
    Xtr = roll_feats(P0, D0)[is_tr & steady]; te_idx = np.where(~is_tr & steady)[0]
    p, d, lab, near = inject(P0, D0, te_idx, rng)
    keep = (~is_tr) & steady & ((~near) | lab)                                  # normal rows + fault rows (recovery rows excluded)
    X, y = roll_feats(p, d)[keep], lab[keep].astype(int)
    sc = StandardScaler().fit(Xtr); Xtr_s, X_s = sc.transform(Xtr), sc.transform(X)
    base = IsolationForest(n_estimators=200, contamination=CONTAM, random_state=SEED).fit(Xtr_s)
    sub = Xtr_s[rng.choice(len(Xtr_s), min(MAX_FIT, len(Xtr_s)), replace=False)]
    imp = LocalOutlierFactor(n_neighbors=30, contamination=CONTAM, novelty=True).fit(sub + rng.normal(0, 1e-3, sub.shape))
    p1, p2 = (base.predict(X_s) == -1).astype(int), (imp.predict(X_s) == -1).astype(int)
    n = len(y); block_id = np.arange(n) // BLOCK; nb = int(block_id.max()) + 1

    res = {"n_train": int(len(Xtr)), "n_test": int(n), "n_fault_rows": int(y.sum()), "n_blocks": nb, "models": {}}
    for name, pr in [("baseline_IF", p1), ("improved_LOF", p2)]:
        tp, fp, fn = int(((y == 1) & (pr == 1)).sum()), int(((y == 0) & (pr == 1)).sum()), int(((y == 1) & (pr == 0)).sum())
        res["models"][name] = dict(accuracy=float((pr == y).mean()), accuracy_wilson_ci=wilson((pr == y).sum(), n),
            precision=tp / (tp + fp) if tp + fp else 0.0, recall=tp / (tp + fn), recall_wilson_ci=wilson(tp, tp + fn), f1=f1(y, pr), tp=tp, fp=fp, fn=fn)
    boot = block_bootstrap(y, p1, p2, block_id, nb)
    for i, name in enumerate(["baseline_IF", "improved_LOF"]): res["models"][name]["f1_boot_ci"] = np.percentile(boot[:, i], [2.5, 97.5]).tolist()
    obs = res["models"]["improved_LOF"]["f1"] - res["models"]["baseline_IF"]["f1"]
    ci = np.percentile(boot[:, 2], [2.5, 97.5]).tolist()
    bb = int(((p1 == y) & (p2 != y)).sum()); cc = int(((p1 != y) & (p2 == y)).sum())      # McNemar discordant pairs
    p_mc = float(stats.binomtest(bb, bb + cc, 0.5).pvalue) if bb + cc else 1.0
    p_perm = float(block_permutation_p(y, p1, p2, block_id, nb, obs))
    res["difference"] = dict(f1_diff=float(obs), boot_ci95=ci, boot_se=float(boot[:, 2].std(ddof=1)), mcnemar_b_IF_right_LOF_wrong=bb,
                             mcnemar_c_LOF_right_IF_wrong=cc, mcnemar_p=p_mc, permutation_p=p_perm, alpha=ALPHA)
    reject = (p_perm < ALPHA) and (ci[0] > 0 or ci[1] < 0)
    res["decision"] = "Reject H0: the F1 difference is unlikely to be due to test-sample variation" if reject else \
                      "Fail to reject H0: the difference could be due to test-sample variation"
    json.dump(res, open("results/summary.json", "w"), indent=1)
    plt.hist(boot[:, 2], bins=40); [plt.axvline(v, c="r", ls="--") for v in ci]; plt.axvline(0, c="k")
    plt.title("Bootstrap distribution of F1 difference (LOF - IF)"); plt.xlabel("F1 difference"); plt.tight_layout(); plt.savefig("results/bootstrap_diff.png", dpi=120)
    m = res["models"]; txt = [f"Train rows {res['n_train']} | test rows {n} (faults {int(y.sum())}) | blocks {nb} | seed {SEED}"]
    for k, v in m.items():
        txt.append(f"{k}: acc {v['accuracy']:.3f} {np.round(v['accuracy_wilson_ci'],3).tolist()} | prec {v['precision']:.3f} | recall {v['recall']:.3f} "
                   f"{np.round(v['recall_wilson_ci'],3).tolist()} | F1 {v['f1']:.3f} {np.round(v['f1_boot_ci'],3).tolist()}")
    df_ = res["difference"]; txt += [f"F1 diff (LOF-IF) {df_['f1_diff']:.3f} | 95% bootstrap CI {np.round(ci,3).tolist()} | SE {df_['boot_se']:.3f}",
        f"McNemar b={bb} c={cc} p={p_mc:.4g} | block-permutation p={p_perm:.4g} | alpha={ALPHA}", "Decision: " + res["decision"]]
    open("results/summary.txt", "w").write("\n".join(txt)); print("\n".join(txt))

if __name__ == "__main__": main()
