import pandas as pd


data = {
   

    "Year": [1990, 1991, 1992, 1993],
    "Emissions": [578, 601, 620, 650]




}

df = pd.DataFrame(data)
print(df)

df["Reduced_20"] = df["Emissions"] * 0.8
print(df)
