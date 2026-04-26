"""
generate_sample_data.py
=======================
Generates a realistic 60-day household energy dataset.
Each row = one appliance usage session (date, time, appliance, watts, duration).
Run once to create sample_data/energy_data.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── Appliance profiles ────────────────────────────────────────────────────────
# Each entry: (typical_watts, sessions_per_day_mean, duration_hours_mean, duration_std,
#              preferred_hours, season_factor)
APPLIANCES = {
    "Air Conditioner": {
        "watts":        (1500, 200),   # (mean, std)
        "sessions":     (3,   1),
        "duration":     (2.5, 0.8),
        "peak_hours":   [14, 15, 16, 17, 18, 19, 20],
        "weekday_bias": 0.8,           # less use on weekdays vs weekends
    },
    "Refrigerator": {
        "watts":        (150,  20),
        "sessions":     (24,   2),     # runs continuously → many micro-sessions
        "duration":     (0.5, 0.1),
        "peak_hours":   list(range(24)),
        "weekday_bias": 1.0,
    },
    "Washing Machine": {
        "watts":        (500, 50),
        "sessions":     (1,  0.5),
        "duration":     (1.0, 0.2),
        "peak_hours":   [8, 9, 10, 11, 18, 19, 20],
        "weekday_bias": 0.6,           # mostly weekends
    },
    "Water Heater": {
        "watts":        (2000, 300),
        "sessions":     (2,   0.5),
        "duration":     (0.5, 0.1),
        "peak_hours":   [5, 6, 7, 18, 19],
        "weekday_bias": 1.0,
    },
    "Television": {
        "watts":        (120, 20),
        "sessions":     (2,  0.5),
        "duration":     (2.0, 0.8),
        "peak_hours":   [18, 19, 20, 21, 22],
        "weekday_bias": 0.9,
    },
    "Microwave": {
        "watts":        (1000, 100),
        "sessions":     (3,   1),
        "duration":     (0.1, 0.05),
        "peak_hours":   [7, 8, 12, 13, 19, 20],
        "weekday_bias": 1.0,
    },
    "Laptop": {
        "watts":        (60,  10),
        "sessions":     (2,  0.5),
        "duration":     (3.0, 1.0),
        "peak_hours":   [9, 10, 11, 14, 15, 20, 21],
        "weekday_bias": 1.2,
    },
    "Lights": {
        "watts":        (40,  10),
        "sessions":     (5,   1),
        "duration":     (2.0, 0.5),
        "peak_hours":   [6, 7, 18, 19, 20, 21, 22],
        "weekday_bias": 1.0,
    },
    "Electric Fan": {
        "watts":        (75,  10),
        "sessions":     (2,   1),
        "duration":     (3.0, 1.0),
        "peak_hours":   [13, 14, 15, 16, 22, 23],
        "weekday_bias": 0.9,
    },
    "Iron": {
        "watts":        (1100, 100),
        "sessions":     (1,   0.3),
        "duration":     (0.5, 0.2),
        "peak_hours":   [7, 8, 17, 18],
        "weekday_bias": 0.5,
    },
}

# ── Date range: 60 days ending today ─────────────────────────────────────────
END_DATE   = datetime(2025, 3, 31)
START_DATE = END_DATE - timedelta(days=59)
dates      = [START_DATE + timedelta(days=i) for i in range(60)]

rows = []

for current_date in dates:
    is_weekend = current_date.weekday() >= 5

    for appliance, profile in APPLIANCES.items():
        # How many sessions today?
        w_bias   = profile["weekday_bias"]
        sessions = max(0, int(np.random.normal(
            profile["sessions"][0] * (w_bias if not is_weekend else 1.0),
            profile["sessions"][1]
        )))

        for _ in range(sessions):
            # Pick a start hour (weighted towards preferred hours)
            hour = np.random.choice(profile["peak_hours"])

            # Add some minute-level noise
            minute = np.random.randint(0, 60)

            # Watts with some noise
            watts = max(10, np.random.normal(*profile["watts"]))

            # Duration
            duration = max(0.05, np.random.normal(*profile["duration"]))

            rows.append({
                "date":           current_date.strftime("%Y-%m-%d"),
                "time":           f"{hour:02d}:{minute:02d}",
                "appliance":      appliance,
                "watts":          round(watts, 1),
                "duration_hours": round(duration, 2),
            })

df = pd.DataFrame(rows)

# Sort by date + time
df["_dt"] = pd.to_datetime(df["date"] + " " + df["time"])
df = df.sort_values("_dt").drop(columns=["_dt"]).reset_index(drop=True)

# Save
os.makedirs("sample_data", exist_ok=True)
df.to_csv("sample_data/energy_data.csv", index=False)
print(f"✅  Generated {len(df):,} rows across {df['date'].nunique()} days "
      f"and {df['appliance'].nunique()} appliances.")
print(df.head())
