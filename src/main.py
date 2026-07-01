import pandas as pd
import os
print(os.path.exists("data/raw/India_Emissions.filtered/co-emissions-per-capita.csv"))

df = pd.read_csv(
    "data/raw/India_Emissions.filtered/co-emissions-per-capita.csv"
)
#Loading Data Frame
print(df.head())

print(df.tail())

print(df.head(10))

print(df.info())

print(df.columns)

#Cleaning Data Frame
print("\nMissing values:")

print(df.isnull().sum())

print("\nDuplicate rows:")

print(df.duplicated().sum())

clean_df = df[["Year", "CO₂ emissions per capita"]]

print(clean_df.head())
print(clean_df.tail())






