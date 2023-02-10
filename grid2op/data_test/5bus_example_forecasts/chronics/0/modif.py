import pandas as pd
import numpy as np

for el in ["load_p", "load_q", "prod_p"]:
    dt = pd.read_csv(f"{el}.csv.bz2", sep=";")
    
    arr = np.ones((12, dt.shape[1]))
    add = np.vstack([np.cumsum(arr, 0) for _ in range(dt.shape[0])])
    dt = dt.loc[dt.index.repeat(12)]
    dt += add
    dt.to_csv(f"{el}_forecasted.csv.bz2", sep=";", index=False, header=True)