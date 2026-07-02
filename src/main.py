import pandas as pd
import matplotlib.pyplot as plt
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
#Exploratory Data Analysis

plt.figure(figsize=(10,5))

plt.plot(

clean_df["Year"],
clean_df["CO₂ emissions per capita"]






)





plt.title("India CO₂ Emissions Per Capita")
plt.xlabel("Year")
plt.ylabel("CO₂ Emissions Per Capita")

plt.grid(True)

plt.show()







