\# Anomaly detection on machine temperature data



Can an unsupervised method find the failure periods in a machine's temperature readings better than a simple rule? This project compares a rolling z-score baseline with an Isolation Forest, using the same number of alerts for both.



\## Data

Numenta Anomaly Benchmark (NAB), file `machine\_temperature\_system\_failure`: 22,695 readings, one every 5 minutes, from 2 Dec 2013 to 19 Feb 2014, with four labelled failure windows of about 47 hours each (10% of all readings). The data is not included here; see https://github.com/numenta/NAB for the data and its licence.



To download it:

&#x20;   Invoke-WebRequest -Uri "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/machine\_temperature\_system\_failure.csv" -OutFile machine\_temperature.csv

&#x20;   Invoke-WebRequest -Uri "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined\_windows.json" -OutFile labels.json



\## Data checks

No missing values and no gaps in the timeline. 12 duplicate timestamps, merged by averaging. The first day is dropped, because the methods need one day of history.



\## Methods

\- \*\*Baseline:\*\* alert when a reading is K standard deviations below the average of the previous day (past readings only).

\- \*\*Isolation Forest (scikit-learn):\*\* unsupervised, trained without the labels, on 9 features computed from past readings only (value, 1 hour / 6 hour / 1 day rolling means, rolling standard deviations, 6 hour minimum, 1 hour change, deviation from the 1 day mean).



\## Evaluation

Both methods get the same number of alerts (the N most unusual readings of each), so neither wins by alerting more. I report:

\- failure windows hit (at least one alert inside the window)

\- false-alarm episodes (alerts less than an hour apart count as one episode; an episode is false if it overlaps no window)

\- precision (share of alerts that fall inside a labelled window); guessing at random would give about 10%



\## Results



| Alerts | Method | Windows hit | False-alarm episodes | Precision |

|---|---|---|---|---|

| 256 | baseline | 1 of 4 | 19 | 14% |

| 256 | Isolation Forest | 2 of 4 | 2 | 79% |

| 786 | baseline | 2 of 4 | 40 | 8% |

| 786 | Isolation Forest | 4 of 4 | 20 | 70% |

| 2071 | baseline | 3 of 4 | 60 | 9% |

| 2071 | Isolation Forest | 4 of 4 | 35 | 56% |



With 786 alerts and five different random seeds, Isolation Forest hit all four windows every time, with 19 to 23 false-alarm episodes and precision of 67% to 71%.



!\[alerts](plot\_alerts.png)



\## What I found

The baseline fires on any drop below its recent average, and the data has many ordinary dips that are not labelled failures, so its alerts are no more precise than guessing at 786 alerts (8%, against a 10% base rate). Isolation Forest concentrates its alerts inside the failure windows, especially the second and the fourth.



\## Limits

\- Only four labelled failures on one machine. These results do not show that the method works in general.

\- "Window hit" needs one alert, which is lenient. At 786 alerts, windows 1 and 3 get only a few alerts each.

\- The labelled windows are about two days wide, which is generous to both methods.

\- I chose the features after looking at the data, so the comparison is not a fully blind test.

\- Some alerts outside the windows (for example near 3 February) may be real events that nobody labelled, so "false-alarm" is an upper bound.



\## Run it

&#x20;   pip install -r requirements.txt

&#x20;   python explore.py      (data checks and overview chart)

&#x20;   python baseline.py     (rolling z-score baseline)

&#x20;   python isoforest.py    (comparison at equal alert counts)

&#x20;   python report.py       (seed check and alerts chart)



Data: Numenta Anomaly Benchmark.

