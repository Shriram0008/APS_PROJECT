"""Train + benchmark IF / LOF / OCSVM. Usage: python train.py [appliance ...]"""
import sys, json, time, joblib, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from preprocess import APPS, get_clean, features

def inject(X, rng, n):  # fake faults: sudden spikes / drops in power
    A = X[rng.choice(len(X), n, replace=False)].copy()
    new = A[:, 0] * rng.uniform(1.5, 2.5, n) + 150
    A[:, 1] += new - A[:, 0]; A[:, 0] = new; return A

metrics = {}
for app in (sys.argv[1:] or APPS):
    rng = np.random.default_rng(42)
    df = get_clean(app); th = APPS[app][1]
    act = df[df.power_smooth > th]                      # train on ACTIVE readings only
    X = features(act.power.values, act.delta.values)
    cut = int(len(X) * 0.8); Xtr, Xte = X[:cut], X[cut:]
    sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    Xan = sc.transform(inject(Xte, rng, max(50, len(Xte)//20)))
    models = {"IF": IsolationForest(n_estimators=200, contamination=0.02, random_state=42),
              "LOF": LocalOutlierFactor(n_neighbors=30, contamination=0.02, novelty=True),
              "OCSVM": OneClassSVM(nu=0.02, gamma="scale")}
    metrics[app] = {"train_rows": len(Xtr), "models": {}}
    for name, m in models.items():
        sub = Xtr_s if name == "IF" else Xtr_s[rng.choice(len(Xtr_s), min(len(Xtr_s), 5000), replace=False)]
        t = time.time(); m.fit(sub); ft = time.time() - t
        t = time.time(); fa = (m.predict(Xte_s) == -1).mean(); det = (m.predict(Xan) == -1).mean()
        it = time.time() - t
        metrics[app]["models"][name] = {"detection_rate": round(float(det), 3), "false_alarm_rate": round(float(fa), 3),
                                        "fit_s": round(ft, 3), "predict_s": round(it, 3)}
        joblib.dump({"model": m, "scaler": sc, "threshold": th}, f"models/{app}_{name}.joblib")
    print(app, json.dumps(metrics[app]["models"], indent=1))
    plt.figure(figsize=(9, 3)); plt.plot(df.power.values[:3000]); plt.title(f"{app} - cleaned power (first 3000 samples)")
    plt.tight_layout(); plt.savefig(f"data/{app}_eda.png"); plt.close()
json.dump(metrics, open("models/metrics.json", "w"), indent=1)
