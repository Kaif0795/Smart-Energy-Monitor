"""
modules/report_generator.py
============================
Generates downloadable reports from analysis results.

  export_csv(df)             – return bytes of a CSV
  export_summary_pdf(stats, appliance_df, suggestions, alerts)
                             – return bytes of a PDF report
"""

import io
import pandas as pd
from datetime import datetime

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False


# ── CSV export ────────────────────────────────────────────────────────────────

def export_csv(df: pd.DataFrame) -> bytes:
    """Return the DataFrame as UTF-8 encoded CSV bytes."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


# ── PDF export ────────────────────────────────────────────────────────────────

def _s(text: str) -> str:
    """Sanitise text to latin-1 so basic FPDF fonts don't choke on emojis/unicode."""
    return str(text).encode("latin-1", errors="ignore").decode("latin-1")


class _EnergyReport(FPDF if HAS_FPDF else object):
    """Custom FPDF subclass with header/footer branding."""

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_fill_color(30, 90, 180)
        self.set_text_color(255, 255, 255)
        self.cell(0, 12, "  Smart Energy Monitoring System -- Report", ln=True, fill=True)
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Page {self.page_no()}", align="C")


def export_summary_pdf(
    stats: dict,
    appliance_df: pd.DataFrame,
    suggestions: list,
    alerts: list,
) -> bytes:
    """
    Build and return a multi-section PDF summary report.

    Parameters
    ----------
    stats        : dict from DataProcessor.summary_stats()
    appliance_df : pd.DataFrame from DataProcessor.appliance_consumption()
    suggestions  : list[dict] from EnergyOptimizer.get_appliance_suggestions()
    alerts       : list[str]  from EnergyOptimizer.get_alert_messages()

    Returns
    -------
    bytes — PDF file content
    """
    if not HAS_FPDF:
        # Fallback: plain-text "PDF" as bytes
        lines = ["Smart Energy Report\n", "=" * 40 + "\n"]
        lines.append(f"Total kWh    : {stats.get('total_kwh', 'N/A')}\n")
        lines.append(f"Avg Daily    : {stats.get('avg_daily_kwh', 'N/A')}\n")
        return "".join(lines).encode("utf-8")

    pdf = _EnergyReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "", 11)

    # ── Section helper ────────────────────────────────────────────────────────
    def section_title(title: str):
        # Strip non-latin chars (emojis) for basic PDF font compatibility
        clean = title.encode("latin-1", errors="ignore").decode("latin-1")
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_fill_color(220, 235, 255)
        pdf.cell(0, 9, f"  {clean}", ln=True, fill=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.ln(2)

    def kv_row(label: str, value: str, indent: int = 5):
        pdf.set_x(indent + 10)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(65, 7, _s(label))
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, _s(str(value)), ln=True)

    # ── 1. Summary KPIs ───────────────────────────────────────────────────────
    section_title("1.  Energy Summary")
    kv_row("Total Consumption:",   f"{stats.get('total_kwh', 0):.2f} kWh")
    kv_row("Average Daily Usage:", f"{stats.get('avg_daily_kwh', 0):.2f} kWh/day")
    kv_row("Peak Day Usage:",      f"{stats.get('max_daily_kwh', 0):.2f} kWh")
    kv_row("Number of Days:",      str(stats.get("num_days", 0)))
    kv_row("Appliances Tracked:",  str(stats.get("num_appliances", 0)))
    kv_row("Estimated Cost:",      f"Rs. {stats.get('cost_estimate', 0):,.2f}")
    kv_row("Top Appliance:",       str(stats.get("top_appliance", "N/A")))
    pdf.ln(4)

    # ── 2. Appliance Consumption Table ────────────────────────────────────────
    section_title("2.  Appliance-wise Consumption")
    col_w = [70, 35, 30]
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(180, 210, 255)
    for header, w in zip(["Appliance", "Total kWh", "Share (%)"], col_w):
        pdf.cell(w, 7, header, border=1, fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    for i, row in appliance_df.iterrows():
        fill = (i % 2 == 0)
        pdf.set_fill_color(245, 248, 255) if fill else pdf.set_fill_color(255, 255, 255)
        pdf.cell(col_w[0], 6, _s(str(row["appliance"])), border=1, fill=fill)
        pdf.cell(col_w[1], 6, f"{row['kwh']:.2f}", border=1, fill=fill, align="R")
        pdf.cell(col_w[2], 6, f"{row['pct']:.1f}%", border=1, fill=fill, align="R")
        pdf.ln()
    pdf.ln(4)

    # ── 3. Optimisation Suggestions ───────────────────────────────────────────
    section_title("3.  Top Optimisation Suggestions")
    pdf.set_font("Helvetica", "", 9)
    for s in suggestions[:5]:          # top 5 appliances
        pdf.set_font("Helvetica", "B", 10)
        # Strip emoji for PDF compatibility
        app_clean = s["appliance"]
        pdf.cell(0, 7, _s(f"{app_clean}  ({s['kwh']:.2f} kWh  |  save ~{s['saving_pct']}%)"), ln=True)
        pdf.set_font("Helvetica", "", 9)
        for tip in s["tips"][:2]:
            clean_tip = tip.encode("ascii", errors="ignore").decode()
            pdf.set_x(15)
            pdf.multi_cell(0, 5, _s(f"- {clean_tip.strip()}"))
        pdf.ln(2)

    # ── 4. Alerts ─────────────────────────────────────────────────────────────
    if alerts:
        section_title("4.  Alerts & Warnings")
        pdf.set_font("Helvetica", "", 9)
        for alert in alerts[:8]:
            clean = alert.replace("**", "").encode("ascii", errors="ignore").decode()
            pdf.multi_cell(0, 5, _s(clean.strip()))
            pdf.ln(1)

    # ── Output ────────────────────────────────────────────────────────────────
    return bytes(pdf.output())
