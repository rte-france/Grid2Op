import pandas as pd
import numpy as np
import grid2op
import os

for chron_nm in ["2035-01-15_0", "2035-08-20_0"]:
    for fn in ["load_p", "load_q", "prod_p"]:
        df_ = pd.read_csv(os.path.join(".", chron_nm, f"{fn}.csv.bz2"), sep=";")
        df_ = df_.iloc[:(288*2)]
        df_.to_csv(os.path.join(".", chron_nm, f"{fn}.csv.bz2"), sep=";", header=True, index=False)
        
        df_ = pd.read_csv(os.path.join(".", chron_nm, f"{fn}_forecasted.csv.bz2"), sep=";")
        df_ = df_.iloc[:(288*2*12)]
        df_.to_csv(os.path.join(".", chron_nm, f"{fn}_forecasted.csv.bz2"), sep=";", header=True, index=False)
        