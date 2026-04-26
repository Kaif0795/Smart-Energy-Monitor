"""
modules/optimizer.py
====================
Rule-based energy optimization engine.

Analyses the processed data and returns:
  • Appliance-specific suggestions
  • Peak-hour shift recommendations
  • Efficiency ratings per appliance
  • Estimated savings

Public API
----------
EnergyOptimizer(processor)
  .get_appliance_suggestions()  – list[dict]
  .get_peak_shift_savings()     – dict
  .get_efficiency_ratings()     – pd.DataFrame
  .get_alert_messages()         – list[str]
  .overall_score()              – int  (0-100 "efficiency score")
"""

import pandas as pd
import numpy as np
from typing import List, Dict


# ── Static knowledge base ─────────────────────────────────────────────────────
# Each appliance entry:
#   'benchmark_kwh_day' – reasonable daily average (kWh)
#   'peak_reducible'    – fraction of its usage that can be shifted off-peak
#   'tips'              – list of actionable suggestions
KNOWLEDGE_BASE = {
    "Air Conditioner": {
        "benchmark_kwh_day": 3.0,
        "peak_reducible":    0.60,
        "saving_pct":        25,
        "tips": [
            "🌡️  Set thermostat to 24–26 °C (every 1 °C increase saves ~6% energy).",
            "🕔  Avoid running AC during peak hours 5 PM – 9 PM; pre-cool rooms earlier.",
            "🧹  Clean/replace AC filters monthly — clogged filters waste up to 15% extra power.",
            "🪟  Use curtains & blinds to block afternoon heat, reducing AC load.",
            "💨  Use ceiling fans alongside AC to feel 3 °C cooler at the same setting.",
        ],
    },
    "Water Heater": {
        "benchmark_kwh_day": 1.0,
        "peak_reducible":    0.50,
        "saving_pct":        30,
        "tips": [
            "🚿  Use an on-demand (instant) heater instead of a storage tank.",
            "🌡️  Lower temperature to 50 °C — adequate and saves ~10% energy.",
            "🕐  Schedule heating in the morning (6–7 AM) to avoid peak-hour tariffs.",
            "🧰  Insulate the tank and hot-water pipes to retain heat longer.",
        ],
    },
    "Washing Machine": {
        "benchmark_kwh_day": 0.5,
        "peak_reducible":    0.80,
        "saving_pct":        40,
        "tips": [
            "🕙  Run washes late at night (10 PM – 6 AM) when tariffs are lowest.",
            "🧺  Always run full loads — a half-full machine uses nearly the same energy.",
            "❄️  Use cold-water cycles; 90% of the energy in a wash goes to heating water.",
            "♻️  Use the quick-wash mode for lightly soiled clothes.",
        ],
    },
    "Refrigerator": {
        "benchmark_kwh_day": 1.8,
        "peak_reducible":    0.0,
        "saving_pct":        10,
        "tips": [
            "🌡️  Set fridge to 3–5 °C and freezer to -18 °C (optimal efficiency).",
            "📍  Keep the fridge away from the oven, dishwasher, and direct sunlight.",
            "🔒  Check door seals — a leaky seal can waste up to 30% extra energy.",
            "🥦  Let hot food cool to room temperature before refrigerating.",
            "🧹  Vacuum condenser coils every 6 months for better heat dissipation.",
        ],
    },
    "Television": {
        "benchmark_kwh_day": 0.3,
        "peak_reducible":    0.20,
        "saving_pct":        20,
        "tips": [
            "🔅  Reduce screen brightness; auto-brightness saves up to 20% energy.",
            "⏻  Enable 'auto-off after idle' or sleep timer — TVs in standby still draw power.",
            "📺  Consider upgrading to an OLED or QLED TV (40–50% more efficient than CRT/plasma).",
            "🔌  Unplug the TV when on vacation — standby mode uses energy 24/7.",
        ],
    },
    "Lights": {
        "benchmark_kwh_day": 0.5,
        "peak_reducible":    0.30,
        "saving_pct":        50,
        "tips": [
            "💡  Replace all incandescent bulbs with LED (75–80% energy savings).",
            "🌞  Use natural daylight where possible; open blinds during daytime.",
            "🔦  Install occupancy/motion sensors in bathrooms and hallways.",
            "🔅  Use dimmers — dimming by 50% saves ~40% energy.",
        ],
    },
    "Microwave": {
        "benchmark_kwh_day": 0.1,
        "peak_reducible":    0.10,
        "saving_pct":        15,
        "tips": [
            "♨️  Prefer microwave over conventional oven for reheating (uses 80% less energy).",
            "🍱  Cover food while cooking — reduces cooking time by up to 25%.",
            "🔌  Unplug when not in use — the digital clock draws constant standby power.",
        ],
    },
    "Laptop": {
        "benchmark_kwh_day": 0.2,
        "peak_reducible":    0.10,
        "saving_pct":        20,
        "tips": [
            "⚡  Enable battery-saver / power-efficiency mode in OS settings.",
            "🌑  Use dark mode — OLED screens consume up to 40% less power with dark themes.",
            "💤  Configure sleep/hibernate after 5 minutes of inactivity.",
            "🔌  Unplug the charger once battery reaches 100%; trickle charging wastes energy.",
        ],
    },
    "Electric Fan": {
        "benchmark_kwh_day": 0.4,
        "peak_reducible":    0.15,
        "saving_pct":        10,
        "tips": [
            "🔄  Run fans counter-clockwise in summer to create a wind-chill effect.",
            "⏻  Turn fans off when leaving the room — fans cool people, not rooms.",
            "⭐  Look for BEE 5-star rated fans when replacing (30% more efficient).",
        ],
    },
    "Iron": {
        "benchmark_kwh_day": 0.1,
        "peak_reducible":    0.60,
        "saving_pct":        25,
        "tips": [
            "🕑  Iron large batches at once to avoid repeated heating/cooling cycles.",
            "🕒  Schedule ironing in the early morning or late evening (off-peak).",
            "♨️  Use the lowest effective temperature setting for each fabric type.",
            "💧  Use a steam generator iron — irons faster with less heat.",
        ],
    },
}

# Alert thresholds
DAILY_ALERT_THRESHOLD_KWH = 20.0   # flag if a single day exceeds this
PEAK_RATIO_ALERT          = 0.45   # flag if >45% of energy is used in peak hours


class EnergyOptimizer:
    """Generates appliance-specific tips, peak-shift savings, and efficiency scores."""

    def __init__(self, processor):
        """
        Parameters
        ----------
        processor : DataProcessor instance (already initialised with data)
        """
        self.processor = processor
        self._app_df   = processor.appliance_consumption()
        self._daily_df = processor.daily_consumption()
        self._total    = processor.total_consumption()

    # ── Per-appliance suggestions ─────────────────────────────────────────────

    def get_appliance_suggestions(self) -> List[Dict]:
        """
        Return a list of dicts, one per known appliance found in data.
        Each dict: {appliance, kwh, pct, priority, tips, estimated_saving_kwh}
        """
        suggestions = []
        for _, row in self._app_df.iterrows():
            app = row["appliance"]
            kb  = KNOWLEDGE_BASE.get(app)
            if kb is None:
                continue
            saving_kwh = round(row["kwh"] * kb["saving_pct"] / 100, 2)
            suggestions.append({
                "appliance":           app,
                "kwh":                 row["kwh"],
                "pct":                 row["pct"],
                "priority":            "🔴 High" if row["pct"] > 20 else (
                                       "🟡 Medium" if row["pct"] > 10 else "🟢 Low"),
                "tips":                kb["tips"],
                "estimated_saving_kwh": saving_kwh,
                "saving_pct":          kb["saving_pct"],
            })
        # Sort by consumption desc
        suggestions.sort(key=lambda x: x["kwh"], reverse=True)
        return suggestions

    # ── Peak-shift analysis ───────────────────────────────────────────────────

    def get_peak_shift_savings(self) -> Dict:
        """
        Estimate how much energy (and cost) can be saved by shifting
        peak-hour usage to off-peak hours.
        """
        peak_df        = self.processor.peak_hour_usage()
        total_peak_kwh = peak_df["kwh"].sum()
        peak_ratio     = total_peak_kwh / max(self._total, 1e-9)

        shiftable_kwh  = 0.0
        breakdown      = []
        for _, row in peak_df.iterrows():
            app = row["appliance"]
            kb  = KNOWLEDGE_BASE.get(app, {})
            reducible = kb.get("peak_reducible", 0.20)
            shift     = round(row["kwh"] * reducible, 3)
            shiftable_kwh += shift
            breakdown.append({
                "appliance":    app,
                "peak_kwh":    row["kwh"],
                "shiftable_kwh": shift,
            })

        # Assume 20 % cheaper off-peak tariff
        saving_cost = round(shiftable_kwh * 8.0 * 0.20, 2)

        return {
            "total_peak_kwh":  round(total_peak_kwh, 3),
            "peak_ratio_pct":  round(peak_ratio * 100, 1),
            "shiftable_kwh":   round(shiftable_kwh, 3),
            "estimated_saving": saving_cost,
            "breakdown":        breakdown,
        }

    # ── Efficiency ratings ────────────────────────────────────────────────────

    def get_efficiency_ratings(self) -> pd.DataFrame:
        """
        Compare each appliance's average daily kWh against the benchmark.
        Returns a DataFrame with an efficiency rating (A+, A, B, C, D).
        """
        n_days  = self.processor.summary_stats()["num_days"]
        ratings = []
        for _, row in self._app_df.iterrows():
            app   = row["appliance"]
            kb    = KNOWLEDGE_BASE.get(app)
            if kb is None:
                continue
            avg_day   = row["kwh"] / max(n_days, 1)
            benchmark = kb["benchmark_kwh_day"]
            ratio     = avg_day / max(benchmark, 1e-9)

            if ratio <= 0.80:
                grade = "A+"
                color = "green"
            elif ratio <= 1.00:
                grade = "A"
                color = "lightgreen"
            elif ratio <= 1.25:
                grade = "B"
                color = "orange"
            elif ratio <= 1.60:
                grade = "C"
                color = "darkorange"
            else:
                grade = "D"
                color = "red"

            ratings.append({
                "Appliance":   app,
                "Daily kWh":   round(avg_day, 3),
                "Benchmark":   benchmark,
                "Ratio":       round(ratio, 2),
                "Grade":       grade,
            })

        return pd.DataFrame(ratings).sort_values("Ratio", ascending=False)

    # ── Alert messages ────────────────────────────────────────────────────────

    def get_alert_messages(self) -> List[str]:
        """Return a list of ⚠️ alert strings for abnormal usage patterns."""
        alerts = []

        # High single-day usage
        high_days = self.processor.detect_high_usage(percentile=90)
        for _, row in high_days.iterrows():
            alerts.append(
                f"⚠️  **High usage on {row['date'].strftime('%d %b %Y')}**: "
                f"{row['kwh']:.2f} kWh (top 10% of recorded days)."
            )

        # High peak-hour ratio
        peak_info = self.get_peak_shift_savings()
        if peak_info["peak_ratio_pct"] > PEAK_RATIO_ALERT * 100:
            alerts.append(
                f"⚠️  **{peak_info['peak_ratio_pct']:.1f}% of your energy** is consumed "
                f"during peak hours (5–9 PM). Consider shifting loads to off-peak times."
            )

        # Appliances far over benchmark
        ratings = self.get_efficiency_ratings()
        for _, row in ratings[ratings["Grade"] == "D"].iterrows():
            alerts.append(
                f"⚠️  **{row['Appliance']}** is consuming {row['Ratio']:.1f}× "
                f"its benchmark. Immediate attention recommended."
            )

        return alerts

    # ── Overall efficiency score ──────────────────────────────────────────────

    def overall_score(self) -> int:
        """
        Compute a 0–100 efficiency score.
        100 = every appliance at or below benchmark.
        0   = all appliances severely over benchmark.
        """
        ratings = self.get_efficiency_ratings()
        if ratings.empty:
            return 50
        # Score per appliance: max(0, 100 - (ratio-1)*100)
        scores = ratings["Ratio"].apply(lambda r: max(0, min(100, 100 - (r - 1) * 80)))
        return int(scores.mean())
