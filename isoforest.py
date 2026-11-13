import json

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

DATA = "machine_temperature.csv"
LABELS = "labels.json"
KEY = "realKnownCause/machine_temperature_system_failure.csv"
WINDOW = 288
GAP = pd.Timedelta(hours=1)
ALERT_COUNTS = (53, 256, 786, 2071)   # the alert volumes the baseline produced at K = 5, 4, 3, 2

df = pd.read_csv(DATA, parse_dates=["timestamp"])
df = df.groupby("timestamp", as_index=False)["value"].mean().sort_values("timestamp").reset_index(drop=True)
windows = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in json.load(open(LABELS))[KEY]]
df["label"] = False
for start, end in windows:
    df.loc[df["timestamp"].between(start, end), "label"] = True

# baseline score (same as baseline.py)
past = df["value"].shift(1)
df["z"] = (df["value"] - past.rolling(WINDOW, min_periods=WINDOW).mean()) / past.rolling(WINDOW, min_periods=WINDOW).std()

# features for Isolation Forest: only the past, never the future
v = df["value"]
df["f_value"] = v
df["f_mean_1h"] = v.rolling(12).mean()
df["f_mean_6h"] = v.rolling(72).mean()
df["f_mean_1d"] = v.rolling(288).mean()
df["f_std_1h"] = v.rolling(12).std()
df["f_std_6h"] = v.rolling(72).std()
df["f_min_6h"] = v.rolling(72).min()
df["f_change_1h"] = v - v.shift(12)
df["f_dev_1d"] = v - df["f_mean_1d"]
features = [c for c in df.columns if c.startswith("f_")]

data = df.dropna(subset=features + ["z"]).reset_index(drop=True)   # drop the first day (warm-up)
print("Readings evaluated:", len(data), "| inside windows:", f"{data['label'].mean():.1%}")

# unsupervised: the labels are NOT used to fit the model
model = IsolationForest(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(data[features])
data["iso"] = model.score_samples(data[features])   # lower = more unusual


def episodes(times):
    result, start, prev = [], None, None
    for t in times:
        if start is None:
            start = prev = t
        elif t - prev <= GAP:
            prev = t
        else:
            result.append((start, prev))
            start = prev = t
    if start is not None:
        result.append((start, prev))
    return result


def evaluate(flagged):
    eps = episodes(sorted(flagged["timestamp"]))
    hit = sum(bool(flagged["timestamp"].between(a, b).any()) for a, b in windows)
    false_eps = sum(not any(s <= b and e >= a for a, b in windows) for s, e in eps)
    return hit, len(eps), false_eps, flagged["label"].mean()


print("\nSame number of alerts for both methods (the N most unusual readings of each)")
print(f"{'alerts':>7} | {'method':<10} {'windows hit':>11} {'episodes':>9} {'false eps':>10} {'precision':>10}")
for n in ALERT_COUNTS:
    for name, col in (("baseline", "z"), ("isolation", "iso")):
        flagged = data.nsmallest(n, col)
        hit, eps, false_eps, prec = evaluate(flagged)
        print(f"{n:>7} | {name:<10} {hit:>8}/4 {eps:>9} {false_eps:>10} {prec:>9.0%}")
    print("-" * 56)

data[["timestamp", "value", "label", "z", "iso"]].to_csv("scored_iso.csv", index=False)