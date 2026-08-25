import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from pipeline import (
    VALUE_COL,
    apply_policy,
    bau_forecast,
    evaluate_holdout,
    fit_full_trend,
    historical_fit,
    load_clean,
    modern_period,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(ROOT / "data/raw/India_Emissions.filtered/co-emissions-per-capita.csv")
print(df.head())
print(df.tail())
print(df.info())
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:")
print(df.duplicated().sum())

clean_df = load_clean()
print(clean_df.head())
print(clean_df.tail())

plt.figure(figsize=(10, 5))
plt.plot(clean_df["Year"], clean_df[VALUE_COL])
plt.title("India CO₂ Emissions Per Capita")
plt.xlabel("Year")
plt.ylabel("CO₂ emissions per capita (t/person)")
plt.grid(True)
plt.tight_layout()
eda_path = OUTPUTS / "eda_emissions.png"
plt.savefig(eda_path, dpi=150)
plt.close()
print(f"\nSaved EDA chart: {eda_path}")

# Regression — modern period only (1990–2024).
modern = modern_period(clean_df)
x = modern[["Year"]]
y = modern[VALUE_COL]
print("\nRegression features (first 5 years):")
print(x.head())
print(y.head())

holdout = evaluate_holdout(modern)
print("\nHoldout evaluation (train < 2016, test 2016–2024)")
print("Train:", holdout["train_metrics"])
print("Test:", holdout["test_metrics"])
print(f"Holdout slope: {holdout['slope']:.4f} t/person per year")

full_model = fit_full_trend(modern)
fitted = historical_fit(full_model, modern)
last_year = int(modern["Year"].max())
horizon = 2050
bau = bau_forecast(full_model, last_year, horizon)
example_cut = 0.02
scenario = apply_policy(bau, last_year, example_cut)

print(f"\nFull-sample slope (1990–{last_year}): {full_model.coef_[0]:.4f} t/person per year")
print(f"Business-as-usual {horizon}: {float(bau.loc[bau['Year'] == horizon, 'bau'].iloc[0]):.3f} t/person")
print(
    f"Policy {example_cut:.0%} annual cut vs BAU, {horizon}: "
    f"{float(scenario.loc[scenario['Year'] == horizon, 'policy'].iloc[0]):.3f} t/person"
)
print("\nForecast sample:")
print(scenario.head())
print(scenario.tail())

plt.figure(figsize=(10, 5))
plt.plot(clean_df["Year"], clean_df[VALUE_COL], label="Observed")
plt.plot(fitted["Year"], fitted["fitted"], label="Linear fit (1990–2024)")
plt.plot(scenario["Year"], scenario["bau"], label="BAU forecast")
plt.plot(scenario["Year"], scenario["policy"], label="Policy: 2% extra annual cut vs BAU")
plt.title("India CO₂ per capita: trend, forecast, and example policy")
plt.xlabel("Year")
plt.ylabel("CO₂ emissions per capita (t/person)")
plt.legend()
plt.grid(True)
plt.tight_layout()
forecast_path = OUTPUTS / "forecast_policy.png"
plt.savefig(forecast_path, dpi=150)
plt.close()
print(f"\nSaved forecast chart: {forecast_path}")
