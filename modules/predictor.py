"""
modules/predictor.py
====================
ML-based energy consumption forecasting.

Uses a Polynomial Regression (degree-2) model trained on daily kWh totals.
Polynomial features capture mild seasonal / trend curves better than a
plain linear fit on short 60-day windows.

Public API
----------
EnergyPredictor()
  .train(daily_df)          – fit model; returns in-sample predictions (np.array)
  .predict(n_days)          – forecast the next n_days (np.array)
  .metrics                  – dict  {mae, rmse, r2, mape}
  .last_n_days              – int   number of training days
"""

import numpy as np
import pandas as pd
from sklearn.linear_model   import LinearRegression
from sklearn.preprocessing  import PolynomialFeatures, StandardScaler
from sklearn.pipeline       import Pipeline
from sklearn.metrics        import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")


class EnergyPredictor:
    """Polynomial-regression energy forecaster."""

    def __init__(self, degree: int = 2):
        """
        Parameters
        ----------
        degree : polynomial degree (default 2 — quadratic).
        """
        self.degree   = degree
        self.pipeline = Pipeline([
            ("poly",   PolynomialFeatures(degree=degree, include_bias=True)),
            ("scaler", StandardScaler()),
            ("lr",     LinearRegression()),
        ])
        self.trained      = False
        self.last_n_days  = 0
        self.metrics: dict = {}
        self._y_mean      = 0.0   # used to clip unreasonable forecasts

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, daily_df: pd.DataFrame) -> np.ndarray:
        """
        Fit the model on a daily-consumption DataFrame.

        Parameters
        ----------
        daily_df : pd.DataFrame with columns ['date', 'kwh']

        Returns
        -------
        y_pred_train : np.ndarray of in-sample fitted values
        """
        df = daily_df.sort_values("date").reset_index(drop=True)

        # ── Feature: integer day index (0, 1, 2, …) ──────────────────────────
        X = np.arange(len(df)).reshape(-1, 1)
        y = df["kwh"].values.astype(float)

        self.pipeline.fit(X, y)
        self.trained     = True
        self.last_n_days = len(df)
        self._y_mean     = float(y.mean())
        self._y_std      = float(y.std())

        # ── In-sample predictions ─────────────────────────────────────────────
        y_pred = self.pipeline.predict(X)
        y_pred = np.clip(y_pred, 0, None)  # energy can't be negative

        # ── Evaluation metrics ────────────────────────────────────────────────
        mae  = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2   = r2_score(y, y_pred)
        # MAPE — guard against zero actual values
        with np.errstate(divide="ignore", invalid="ignore"):
            mape = np.mean(np.abs((y - y_pred) / np.where(y == 0, np.nan, y))) * 100
            mape = float(np.nanmean(mape)) if np.isscalar(mape) else float(mape)

        self.metrics = {
            "MAE":  round(mae,  4),
            "RMSE": round(rmse, 4),
            "R²":   round(r2,   4),
            "MAPE": round(mape, 2),
        }
        return y_pred

    # ── Forecasting ───────────────────────────────────────────────────────────

    def predict(self, n_days: int = 7) -> np.ndarray:
        """
        Predict energy consumption for the next n_days beyond the training window.

        Returns
        -------
        np.ndarray of length n_days (kWh per day)
        """
        if not self.trained:
            raise RuntimeError("Call .train() before .predict()")

        future_X   = np.arange(
            self.last_n_days,
            self.last_n_days + n_days
        ).reshape(-1, 1)

        forecast   = self.pipeline.predict(future_X)

        # ── Clip to a reasonable range (±3σ around training mean) ─────────────
        lower = max(0.0, self._y_mean - 3 * self._y_std)
        upper = self._y_mean + 3 * self._y_std
        forecast = np.clip(forecast, lower, upper)

        return forecast

    # ── Convenience helpers ───────────────────────────────────────────────────

    def get_trend(self) -> str:
        """
        Return a human-readable trend description based on the model slope
        over the last vs first quarter of training data.
        """
        if not self.trained:
            return "Unknown (model not trained)"

        n       = self.last_n_days
        X_start = np.array([[0]])
        X_end   = np.array([[n - 1]])
        v_start = float(self.pipeline.predict(X_start)[0])
        v_end   = float(self.pipeline.predict(X_end)[0])
        delta   = v_end - v_start
        pct     = (delta / max(v_start, 1e-9)) * 100

        if abs(pct) < 3:
            return f"Stable (±{abs(pct):.1f}% over period)"
        elif pct > 0:
            return f"⬆  Increasing (+{pct:.1f}% over period)"
        else:
            return f"⬇  Decreasing ({pct:.1f}% over period)"
