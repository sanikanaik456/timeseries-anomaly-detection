import json

import pandas as pd

DATA = "machine_temperature.csv"
LABELS = "labels.json"
KEY = "realKnownCause/machine_temperature_system_failure.csv"
WINDOW = 288          # 288 readings x 5 minutes = 1 day of history
GAP = pd.Timedelta(hours=1)   # alerts closer than this count as one episode

# load, and merge the 12 duplicate timestamps by averaging them
df = pd.read_csv(DATA, parse_dates=["timestamp"])
df = df.groupby("timestamp", as_index=False)["value"].mean().sort_values("timestamp").reset_index(drop=True)
windows = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in json.load(open(LABELS))[KEY]]
df["label"] = False
for start, end in windows:
    df.loc[df["timestamp"].between(start, end), "label"] = True

# baseline: how far is each reading below its own recent average?
past = df["value"].shift(1)                     # only past readings, never the current one
mean = past.rolling(WINDOW, min_periods=WINDOW).mean()
std = past.rolling(WINDOW, min_periods=WINDOW).std()
df["z"] = (df["value"] - mean) / std


def episodes(times):
    """Group alert timestamps into episodes: alerts less than GAP apart are one episode."""
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


def evaluate(alert):
    flagged = df[alert]
    eps = episodes(list(flagged["timestamp"]))
    detected = sum(bool(flagged["timestamp"].between(a, b).any()) for a, b in windows)
    false_eps = sum(not any(s <= b and e >= a for a, b in windows) for s, e in eps)
    precision = flagged["label"].mean() if len(flagged) else float("nan")
    return len(flagged), len(eps), detected, false_eps, precision


print("Baseline: alert when a reading is K standard deviations below its previous-day average")
print(f"{'K':>4} {'alerts':>8} {'episodes':>9} {'windows hit':>12} {'false episodes':>15} {'alert precision':>16}")
for k in (2, 3, 4, 5):
    n, eps, hit, false_eps, prec = evaluate(df["z"] < -k)
    print(f"{k:>4} {n:>8} {eps:>9} {hit:>9}/4 {false_eps:>15} {prec:>15.0%}")

print(f"\nBase rate: {df['label'].mean():.1%} of readings are inside a labelled window")
df.to_csv("scored.csv", index=False)