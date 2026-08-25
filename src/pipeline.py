"""Load India CO₂ per-capita data, fit a linear trend, forecast, and apply policy."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "India_Emissions.filtered"
    / "co-emissions-per-capita.csv"
)
VALUE_COL = "CO₂ emissions per capita"
# Pre-1990 levels are near zero and not a useful trend for near-term policy.
MODEL_START_YEAR = 1990
# Hold out the most recent years so evaluation is out-of-sample in time.
TEST_START_YEAR = 2016
DEFAULT_HORIZON_YEAR = 2050


def load_clean(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    clean = df[["Year", VALUE_COL]].copy()
    clean = clean.dropna().drop_duplicates(subset=["Year"]).sort_values("Year")
    clean["Year"] = clean["Year"].astype(int)
    return clean.reset_index(drop=True)


def modern_period(df: pd.DataFrame, start_year: int = MODEL_START_YEAR) -> pd.DataFrame:
    return df.loc[df["Year"] >= start_year].copy().reset_index(drop=True)


def time_split(
    df: pd.DataFrame, test_start: int = TEST_START_YEAR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df.loc[df["Year"] < test_start].copy()
    test = df.loc[df["Year"] >= test_start].copy()
    if train.empty or test.empty:
        raise ValueError("Time split produced an empty train or test set.")
    return train, test


def fit_linear(train: pd.DataFrame) -> LinearRegression:
    model = LinearRegression()
    model.fit(train[["Year"]], train[VALUE_COL])
    return model


def predict_years(model: LinearRegression, years) -> np.ndarray:
    frame = pd.DataFrame({"Year": np.asarray(list(years), dtype=int)})
    return model.predict(frame)


def evaluation_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_holdout(modern: pd.DataFrame) -> dict:
    """Fit on years before TEST_START_YEAR; score later years."""
    train, test = time_split(modern)
    model = fit_linear(train)
    train_pred = predict_years(model, train["Year"])
    test_pred = predict_years(model, test["Year"])
    return {
        "model": model,
        "train": train,
        "test": test,
        "train_pred": train_pred,
        "test_pred": test_pred,
        "train_metrics": evaluation_metrics(train[VALUE_COL], train_pred),
        "test_metrics": evaluation_metrics(test[VALUE_COL], test_pred),
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
    }


def fit_full_trend(modern: pd.DataFrame) -> LinearRegression:
    """Refit on all modern years for the displayed forecast."""
    return fit_linear(modern)


def historical_fit(model: LinearRegression, modern: pd.DataFrame) -> pd.DataFrame:
    out = modern.copy()
    out["fitted"] = predict_years(model, out["Year"])
    return out


def bau_forecast(
    model: LinearRegression,
    last_year: int,
    horizon_year: int,
) -> pd.DataFrame:
    if horizon_year <= last_year:
        return pd.DataFrame(columns=["Year", "bau"])
    years = np.arange(last_year + 1, horizon_year + 1)
    return pd.DataFrame({"Year": years, "bau": predict_years(model, years)})


def apply_policy(
    bau: pd.DataFrame,
    last_hist_year: int,
    annual_reduction: float,
) -> pd.DataFrame:
    """Scale the BAU path by (1 - r) each year after the last observed year.

    r = 0 is the model's trend. r > 0 is a what-if: extra annual cuts vs that trend.
    This is a scenario, not a prediction of real policy.
    """
    out = bau.copy()
    if out.empty:
        out["policy"] = pd.Series(dtype=float)
        return out
    years_since = out["Year"] - last_hist_year
    factor = (1.0 - annual_reduction) ** years_since
    out["policy"] = np.maximum(0.0, out["bau"] * factor)
    return out


def build_projection(
    modern: pd.DataFrame,
    annual_reduction: float = 0.0,
    horizon_year: int = DEFAULT_HORIZON_YEAR,
) -> dict:
    holdout = evaluate_holdout(modern)
    full_model = fit_full_trend(modern)
    last_year = int(modern["Year"].max())
    fitted = historical_fit(full_model, modern)
    bau = bau_forecast(full_model, last_year, horizon_year)
    scenario = apply_policy(bau, last_year, annual_reduction)
    return {
        "holdout": holdout,
        "full_model": full_model,
        "fitted": fitted,
        "scenario": scenario,
        "last_year": last_year,
        "slope": float(full_model.coef_[0]),
        "intercept": float(full_model.intercept_),
        "last_observed": float(modern.loc[modern["Year"] == last_year, VALUE_COL].iloc[0]),
    }
