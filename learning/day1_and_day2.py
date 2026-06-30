import pandas as pd

data = {

"Year" : [2010, 2011, 2012, 2013],
"Emissions": [100, 200, 300, 450]




}

df = pd.DataFrame(data)
print(df["Year"])
print(df[["Year", "Emissions"]])




