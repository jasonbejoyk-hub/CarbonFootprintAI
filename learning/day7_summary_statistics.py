import pandas as pd

data = {


    "Year": [1990, 1991, 1992, 1993],
    "Emissions": [578, 601, 620, 650]

    




}
df = pd.DataFrame(data)

print("Average:", df["Emissions"].mean())
#Mean is equal to 612.25
print("Maximum:", df["Emissions"].max())
#Maximum is equal to 650
print("Minimum:", df["Emissions"].min())
#Minimum is equal to 578



