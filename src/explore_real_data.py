import pandas as pd
import os


print(os.path.exists("data/raw/India_Emissions.filtered/co-emissions-per-capita.csv"))


df = pd.read_csv(

    "data/raw/India_Emissions.filtered/co-emissions-per-capita.csv"


)

print(df.head())





