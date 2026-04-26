"""
modules/data_processor.py
=========================
Handles all data ingestion, cleaning, transformation, and aggregation.

Public API
----------
DataProcessor(df)            – initialise with a raw DataFrame
  .daily_consumption()       – pd.DataFrame  [date, kwh]
  .weekly_consumption()      – pd.DataFrame  [week_label, kwh]
  .appliance_consumption()   – pd.DataFrame  [appliance, kwh, pct]
  .hourly_pattern()          – pd.DataFrame  [hour, kwh]
  .day_of_week_pattern()     – pd.DataFrame  [day_name, kwh]
  .detect_high_usage()       – pd.DataFrame  high-usage dates
  .peak_hour_usage()         – pd.DataFrame  [hour, kwh] for peak band
  .total_consumption()       – float   (kWh)
  .average_daily_consumption()  – float (kWh)
  .cost_estimate(rate)       – float   (currency)
  .summary_stats()           – dict
"""

import pandas as pd
import numpy as np
from datetime import datetime


# ── Column names expected in the raw CSV ─────────────────────────────────────
REQUIRED_COLS = {"date", "time", "appliance", "watts", "duration_hours"}

# Peak-hour band definition (17:00 – 21:00)
PEAK_HOURS = list(range(17, 22))


class DataProcessor:
    """Processes raw energy-usage records into analytical views."""

    # ── Construction / validation ─────────────────────────────────────────────
    def __init__(self, df: pd.DataFrame):
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")
        self.df = df.copy()
        self._preprocess()

    def _preprocess(self):
        """Parse types, derive helper columns, compute kWh per row."""
        df = self.df

        # ── Datetime parsing ──────────────────────────────────────────────────
        df["date"]     = pd.to_datetime(df["date"])
        df["datetime"] = pd.to_datetime(
            df["date"].dt.strftime("%Y-%m-%d") + " " + df["time"].astype(str),
            errors="coerce"
        )

        # ── Numeric coercion ──────────────────────────────────────────────────
        df["watts"]          = pd.to_numeric(df["watts"],          errors="coerce").fillna(0)
        df["duration_hours"] = pd.to_numeric(df["duration_hours"], errors="coerce").fillna(0)

        # ── Energy (kWh = Watts × hours / 1000) ──────────────────────────────
        df["kwh"] = (df["watts"] * df["duration_hours"]) / 1000.0

        # ── Temporal helpers ──────────────────────────────────────────────────
        df["hour"]          = df["datetime"].dt.hour
        df["day_of_week"]   = df["datetime"].dt.day_name()
        df["week_number"]   = df["datetime"].dt.isocalendar().week.astype(int)
        df["month"]         = df["datetime"].dt.month
        df["week_start"]    = df["datetime"] - pd.to_timedelta(
            df["datetime"].dt.dayofweek, unit="d"
        )

        # ── Is peak hour? ─────────────────────────────────────────────────────
        df["is_peak"] = df["hour"].isin(PEAK_HOURS)

        self.df = df

    # ── Aggregations ──────────────────────────────────────────────────────────

    def daily_consumption(self) -> pd.DataFrame:
        """Total kWh per calendar day, sorted chronologically."""
        result = (
            self.df.groupby("date")["kwh"]
            .sum()
            .reset_index()
            .sort_values("date")
        )
        result["kwh"] = result["kwh"].round(3)
        return result

    def weekly_consumption(self) -> pd.DataFrame:
        """Total kWh per ISO week, labelled as 'Week N (DD Mon)'."""
        result = (
            self.df.groupby(["week_number", "week_start"])["kwh"]
            .sum()
            .reset_index()
            .sort_values("week_start")
        )
        result["week_label"] = result.apply(
            lambda r: f"Wk {r['week_number']}  ({r['week_start'].strftime('%d %b')})", axis=1
        )
        result["kwh"] = result["kwh"].round(3)
        return result[["week_label", "kwh"]]

    def appliance_consumption(self) -> pd.DataFrame:
        """Total kWh and percentage share per appliance, descending."""
        result = (
            self.df.groupby("appliance")["kwh"]
            .sum()
            .reset_index()
            .sort_values("kwh", ascending=False)
        )
        total = result["kwh"].sum()
        result["pct"]  = ((result["kwh"] / total) * 100).round(1)
        result["kwh"]  = result["kwh"].round(3)
        return result.reset_index(drop=True)

    def hourly_pattern(self) -> pd.DataFrame:
        """Average kWh per hour-of-day across all days."""
        result = (
            self.df.groupby("hour")["kwh"]
            .mean()
            .reset_index()
            .rename(columns={"kwh": "avg_kwh"})
        )
        result["avg_kwh"] = result["avg_kwh"].round(4)
        return result

    def day_of_week_pattern(self) -> pd.DataFrame:
        """Average daily kWh by day-of-week."""
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        result = (
            self.df.groupby(["day_of_week", "date"])["kwh"]
            .sum()
            .reset_index()
            .groupby("day_of_week")["kwh"]
            .mean()
            .reindex(order)
            .reset_index()
            .rename(columns={"kwh": "avg_kwh"})
        )
        result["avg_kwh"] = result["avg_kwh"].round(3)
        return result

    def detect_high_usage(self, percentile: float = 75) -> pd.DataFrame:
        """Return days where total kWh exceeds the given percentile threshold."""
        daily     = self.daily_consumption()
        threshold = daily["kwh"].quantile(percentile / 100)
        return daily[daily["kwh"] > threshold].copy()

    def peak_hour_usage(self) -> pd.DataFrame:
        """Total kWh consumed during peak hours, per appliance."""
        peak_df = self.df[self.df["is_peak"]]
        result  = (
            peak_df.groupby("appliance")["kwh"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        result["kwh"] = result["kwh"].round(3)
        return result

    # ── Scalar metrics ────────────────────────────────────────────────────────

    def total_consumption(self) -> float:
        return round(self.df["kwh"].sum(), 3)

    def average_daily_consumption(self) -> float:
        return round(self.daily_consumption()["kwh"].mean(), 3)

    def cost_estimate(self, rate_per_kwh: float = 8.0) -> float:
        """Estimated electricity cost (₹ or any currency) for the loaded period."""
        return round(self.total_consumption() * rate_per_kwh, 2)

    def summary_stats(self) -> dict:
        """Return a dictionary of headline KPIs for display on the dashboard."""
        daily  = self.daily_consumption()
        return {
            "total_kwh":        self.total_consumption(),
            "avg_daily_kwh":    self.average_daily_consumption(),
            "max_daily_kwh":    round(daily["kwh"].max(), 3),
            "min_daily_kwh":    round(daily["kwh"].min(), 3),
            "num_days":         daily["date"].nunique(),
            "num_appliances":   self.df["appliance"].nunique(),
            "peak_kwh":         round(self.df[self.df["is_peak"]]["kwh"].sum(), 3),
            "cost_estimate":    self.cost_estimate(),
            "top_appliance":    self.appliance_consumption().iloc[0]["appliance"],
        }


# ── Standalone utility: parse & validate an uploaded file ────────────────────

def load_and_validate(uploaded_file) -> pd.DataFrame:
    """
    Read a CSV from a Streamlit UploadedFile (or file path) and
    return a validated DataFrame ready for DataProcessor.
    Raises ValueError with a user-friendly message on bad input.
    """
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        raise ValueError(f"Could not read file: {exc}")

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"File is missing required columns: {missing}\n"
            f"Required: {REQUIRED_COLS}"
        )
    if len(df) == 0:
        raise ValueError("The file contains no data rows.")
    return df
