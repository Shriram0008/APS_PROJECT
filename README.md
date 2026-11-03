# Appliance Power Anomaly Detection (UK-DALE, microwave)

Software-only anomaly detection on household appliance power data. A microwave's power readings from the UK-DALE dataset are cleaned, turned into context features, and used to train and compare three unsupervised models (Isolation Forest, Local Outlier Factor, One-Class SVM). A Flask backend and a web dashboard let you enter power readings and see whether they are flagged as anomalies.

No hardware is used. Faults are injected artificially, because UK-DALE has no fault labels.

## Features
- Data understanding and quality checks (gaps, duplicates, negatives, missing values)
- Preprocessing with before/after graphs: regular 6 s grid, gap filling, glitch clipping, smoothing, features, chronological train/test split
- Context features: power, change in power, and mean/std of the last 10 readings
- Model comparison: Isolation Forest, LOF, One-Class SVM, over 30 repeated runs
- 95% confidence intervals and hypothesis tests (Friedman, Wilcoxon with Bonferroni correction)
- Flask API and live-style dashboard with a demo sequence

## Dataset
UK-DALE (UK Domestic Appliance-Level Electricity), house_1, channel 13 (microwave), first 90 days.
Kelly, J. and Knottenbelt, W. (2015). *The UK-DALE dataset, domestic appliance-level electricity demand and whole-house demand from five UK homes.* Scientific Data 2, 150007.
The data is not included in this repository. Download it and place the unzipped folder as `ukdale/` in the project root (so that `ukdale/house_1/channel_13.dat` exists).

## Project structure
```
notebooks/   01 understanding data, 02 preprocessing, 03 training, 04 evaluation and statistics
features.py  shared feature code (used by notebooks and the app)
app.py       Flask backend
static/      dashboard (HTML + Chart.js)
models/      trained models and metrics.json
data/        graphs produced by the notebooks
```

## Setup
```
python -m venv .venv
.venv\Scripts\activate        # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## How to run
1. Place the UK-DALE folder as `ukdale/` (see above).
2. Run the notebooks in `notebooks/` in order: 01, 02, 03, 04. This creates `data/microwave_clean.csv`, the models in `models/`, and `models/metrics.json`.
3. Start the app from the project root:
   ```
   python app.py
   ```
4. Open http://localhost:5000, choose a model, press **Reset**, then **Run demo sequence** (12 normal readings, a sustained spike, then normal). You can also type your own power values and press **Detect**.

The trained models are included, so step 2 can be skipped to try the dashboard straight away.

## Method
1. Load a 90-day window of microwave readings (about 6 s sampling).
2. Remove duplicates and negatives, resample to a regular grid, fill gaps of up to 30 s, clip extreme glitches, smooth with a 5-point moving average.
3. Features: power, delta, rolling mean and rolling std over the last 10 readings.
4. Train on steady ON-state readings of the first 80% of the time period; test on the last 20%.
5. Evaluate on injected sustained faults (spike, drop, 18% drift), repeated 30 times with different seeds.

## Results (30 runs, mean with 95% CI)
| Model | Detection | False alarm | Precision | F1 |
|---|---|---|---|---|
| LOF | 0.081 | 0.016 | 0.233 | 0.119 |
| Isolation Forest | 0.305 | 0.023 | 0.432 | 0.353 |
| One-Class SVM | 0.197 | 0.031 | 0.258 | 0.219 |

The Friedman test and all pairwise Wilcoxon tests (Bonferroni corrected) are significant at alpha = 0.05. LOF has the best F1 in this setup. Graphs are in `data/`.

## Limitations
- Faults are injected, not real.
- The 30 repetitions reuse the same test period, so p-values are indicative.
- LOF needs a tiny jitter on its training data, because repeated identical rows break it.
- Isolation Forest misses sustained spikes on its own. The app adds a range guard (flags power far above the training range) on top of the models.
- Only one appliance (microwave) in one house is modelled.

## License
Add a license of your choice (for example MIT). The UK-DALE data has its own license; see the dataset page.