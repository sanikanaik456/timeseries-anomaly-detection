import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATA = "machine_temperature.csv"
LABELS = "labels.json"
KEY = "realKnownCause/machine_temperature_system_failure.csv"

df = pd.read_csv(DATA, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
windows = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in json.load(open(LABELS))[KEY]]

print("Rows:", len(df))
print("From", df["timestamp"].min(), "to", df["timestamp"].max())
print("Missing values:", int(df["value"].isna().sum()))
print("Duplicate timestamps:", int(df["timestamp"].duplicated().sum()))

steps = df["timestamp"].diff().dropna()
print("Most common gap between readings:", steps.mode()[0])
gaps = steps[steps > pd.Timedelta(minutes=5)]
print("Gaps longer than 5 minutes:", len(gaps))
if len(gaps):
    print(gaps.sort_values(ascending=False).head(5).to_string())

print("\nValue summary:")
print(df["value"].describe().round(2).to_string())

df["label"] = False
for start, end in windows:
    df.loc[df["timestamp"].between(start, end), "label"] = True
print("\nLabelled anomaly windows:", len(windows))
for start, end in windows:
    n = int(df["timestamp"].between(start, end).sum())
    print(f"  {start}  to  {end}   ({n} readings)")
print("Readings inside windows:", int(df["label"].sum()), f"({df['label'].mean():.1%} of all readings)")

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(df["timestamp"], df["value"], linewidth=0.5)
for i, (start, end) in enumerate(windows):
    ax.axvspan(start, end, color="red", alpha=0.25, label="labelled anomaly window" if i == 0 else None)
ax.set_title("Machine temperature with labelled anomaly windows (NAB)")
ax.set_ylabel("temperature")
ax.legend()
fig.tight_layout()
fig.savefig("plot_overview.png", dpi=120)
print("\nSaved plot_overview.png")