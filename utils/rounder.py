import pandas as pd

files = [
    "load_p.csv.bz2",
    "load_p_forecasted.csv.bz2",
    "load_q.csv.bz2",
    "load_q_forecasted.csv.bz2",
    "prices.csv.bz2",
    "prod_p.csv.bz2",
    "prod_p_forecasted.csv.bz2",
    "prod_v.csv.bz2"
]

for f in files:
    df = pd.read_csv(f, sep=";")
    df = df.round(decimals=1)
    print (df)
    df.to_csv(f, sep=";", index=False)
