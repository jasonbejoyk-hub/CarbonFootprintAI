import pandas as pd


data = {

    "Year" : [1990, 1991, 1992, 1993],
    "Emissions": [578, 601, 620, 650]
}
df = pd.DataFrame(data)

high_emissions = df["Emissions"] > 620
#Filter the Data Frame to only include rows with emission  650




print(high_emissions)






