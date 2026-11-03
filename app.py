import json, time, joblib
from collections import deque
from flask import Flask, request, jsonify, send_from_directory
from features import W, live_features
app = Flask(__name__, static_folder="static"); LOG = []; CACHE = {}; BUF = deque(maxlen=W)   # BUF = last W readings
APP = "microwave"

def bundle(m):
    if m not in CACHE: CACHE[m] = joblib.load(f"models/{APP}_{m}.joblib")
    return CACHE[m]

@app.get("/")
def home(): return send_from_directory("static", "index.html")

@app.get("/api/info")      # typical running power from training, used by the demo preset buttons
def info():
    b = bundle("IF"); return jsonify(mean_power=round(b["typical"]["median"], 1), threshold=b["threshold"])

@app.get("/api/metrics")
def metrics(): return jsonify(json.load(open("models/metrics.json")))

@app.get("/api/log")
def log(): return jsonify(LOG[-50:][::-1])

@app.post("/api/reset")
def reset(): BUF.clear(); LOG.clear(); return jsonify(ok=True)

@app.post("/predict")
def predict():
    d = request.get_json(force=True)
    try:
        m = d.get("model", "IF"); v = float(d.get("voltage", 230)); p = float(d["power"])
        if p < 0 or v <= 0: raise ValueError
        b = bundle(m)
    except (KeyError, ValueError, FileNotFoundError):
        return jsonify(error="invalid input or model not found in models/"), 400
    if not BUF and d.get("prev_power") not in (None, ""): BUF.append(float(d["prev_power"]))   # first reading: use the entered previous power
    x = live_features(list(BUF), p); prev = BUF[-1] if BUF else p
    reason = "normal"
    if p < b["threshold"]:                  # appliance off -> normal
        anomaly, score = False, 0.0
    else:
        xs = b["scaler"].transform([x]); score = float(b["model"].decision_function(xs)[0])
        anomaly = bool(b["model"].predict(xs)[0] == -1); reason = "model flagged" if anomaly else "normal"
        if not anomaly and max(p, x[2]) > 1.25 * b["typical"]["p99"]:   # range guard: far above anything seen in training
            anomaly, reason = True, "range guard (above normal power range)"
    BUF.append(p)
    r = dict(model=m, power=p, prev_power=prev, delta=round(p - prev, 1), current=round(p / v, 3), roll_mean=round(x[2], 1),
             anomaly=anomaly, reason=reason, score=round(score, 4), time=time.strftime("%H:%M:%S"))
    LOG.append(r); return jsonify(r)

if __name__ == "__main__": app.run(port=5000, debug=True)
