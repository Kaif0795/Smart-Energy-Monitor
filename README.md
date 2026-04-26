#  Smart Energy Monitoring & Management System

A complete software-based household energy monitoring system built with Python and Streamlit.
No hardware required — works entirely with simulated or uploaded CSV data.

---

##  Project Structure

```
smart_energy/
├── app.py                          ← Main Streamlit dashboard (6 tabs)
├── generate_sample_data.py         ← One-time script to create sample data
├── requirements.txt
├── sample_data/
│   └── energy_data.csv             ← 60-day, 10-appliance, 2349-row dataset
└── modules/
    ├── __init__.py
    ├── data_processor.py           ← Parsing, kWh calc, aggregations
    ├── predictor.py                ← Polynomial regression ML model
    ├── optimizer.py                ← Rule-based tips + efficiency scoring
    └── report_generator.py         ← CSV & PDF export
```

---

##  Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate sample data (already included)
```bash
python generate_sample_data.py
```

### 3. Launch the dashboard
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

##  CSV Format

Upload your own data with these columns:

| Column | Type | Example |
|--------|------|---------|
| date | YYYY-MM-DD | 2025-01-15 |
| time | HH:MM | 14:30 |
| appliance | string | Air Conditioner |
| watts | float | 1500 |
| duration_hours | float | 2.5 |

---

##  Features

### Tab 1 — Dashboard
- KPI cards: total kWh, avg daily, peak-hour share, estimated cost, top appliance
- Daily line chart with 7-day rolling average and 75th-percentile threshold line
- Avg. kWh by day-of-week bar chart
- Appliance-wise horizontal bar chart (colour-scaled)
- Energy distribution donut chart
- Hourly consumption heatmap with peak-band highlighting

### Tab 2 — Data Input
- Searchable/filterable raw data table
- Manual record entry form (date / time / appliance / watts / duration)
- Inline kWh & cost calculator

### Tab 3 — Predictions
- Polynomial-regression (degree-2) model with StandardScaler pipeline
- Model accuracy: MAE, RMSE, R², MAPE
- Adjustable forecast horizon: 3–30 days
- Forecast chart with actual, fitted, predicted lines and ±10% confidence band
- Forecast detail table with date, kWh, and cost per day

### Tab 4 — Optimisation
- 0–100 Efficiency Score
- Per-appliance efficiency grades (A+, A, B, C, D) vs benchmarks
- Peak-hour shift potential: shiftable kWh and estimated cost saving
- Expandable tips per appliance (10 appliances, 3–5 tips each)

### Tab 5 — Alerts
- Anomaly-flagged high-usage days (top 10%)
- Peak-ratio alert if >45% of energy falls in 5–9 PM
- Appliance-level Grade-D warnings
- Simulated real-time current-hour display with peak/off-peak status

### Tab 6 — Reports
- Download CSV: raw data, daily totals, appliance summary, or forecast
- Generate & download PDF report (summary + table + tips + alerts)
- On-screen Markdown summary with download option

---

##  ML Model Details

- **Algorithm**: Polynomial Regression (degree=2) via scikit-learn Pipeline
- **Features**: Integer day-index (0 … N)
- **Target**: Daily kWh total
- **Pipeline**: `PolynomialFeatures → StandardScaler → LinearRegression`
- **Clipping**: Forecasts clipped to ±3σ around training mean
- **Metrics reported**: MAE, RMSE, R², MAPE

---

##  Dependencies

| Library | Purpose |
|---------|---------|
| streamlit | Web UI framework |
| pandas | Data wrangling |
| numpy | Numerical operations |
| plotly | Interactive charts |
| scikit-learn | ML pipeline |
| fpdf2 | PDF generation |

---

##  Supported Appliances

Air Conditioner, Refrigerator, Washing Machine, Water Heater, Television,
Microwave, Laptop, Lights, Electric Fan, Iron
