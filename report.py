import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest

DATA = "machine_temperature.csv"
LABELS = "labels.json"
KEY = "realKnownCause/machine_temperature_system_failure.csv"
WINDOW = 288
GAP = pd.Timedelta(hours=1)
N_ALERTS = 786          # alert budget used for the chart and the seed check
SEEDS = (0, 1, 2, 3, 4)

df = pd.read_csv(DATA, parse_dates=["timestamp"])
df = df.groupby("timestamp", as_index=False)["value"].mean().sort_values("timestamp").reset_index(drop=True)
windows = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in json.load(open(LABELS))[KEY]]
df["label"] = False
for start, end in windows:
    df.loc[df["timestamp"].between(start, end), "label"] = True

past = df["value"].shift(1)
df["z"] = (df["value"] - past.rolling(WINDOW, min_periods=WINDOW).mean()) / past.rolling(WINDOW, min_periods=WINDOW).std()

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
data = df.dropna(subset=features + ["z"]).reset_index(drop=True)


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


# 1. stability across random seeds (the baseline has no randomness)
print(f"Isolation Forest with {N_ALERTS} alerts, five random seeds:")
for seed in SEEDS:
    model = IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1).fit(data[features])
    data["iso"] = model.score_samples(data[features])
    hit, eps, false_eps, prec = evaluate(data.nsmallest(N_ALERTS, "iso"))
    print(f"  seed {seed}: windows hit {hit}/4, false-alarm episodes {false_eps}, precision {prec:.0%}")

# 2. final model (seed 42) and the chart
model = IsolationForest(n_estimators=300, random_state=42, n_jobs=-1).fit(data[features])
data["iso"] = model.score_samples(data[features])
base_alerts = data.nsmallest(N_ALERTS, "z")
iso_alerts = data.nsmallest(N_ALERTS, "iso")

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
for ax, alerts, title in (
    (axes[0], base_alerts, f"Baseline: {N_ALERTS} lowest rolling z-scores"),
    (axes[1], iso_alerts, f"Isolation Forest: {N_ALERTS} most unusual readings"),
):
    ax.plot(data["timestamp"], data["value"], linewidth=0.5, color="tab:blue")
    for i, (start, end) in enumerate(windows):
        ax.axvspan(start, end, color="red", alpha=0.2, label="labelled failure window" if i == 0 else None)
    ax.scatter(alerts["timestamp"], alerts["value"], s=6, color="black", label="alert", zorder=3)
    ax.set_title(title)
    ax.set_ylabel("temperature")
    ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig("plot_alerts.png", dpi=120)
print("\nSaved plot_alerts.png")