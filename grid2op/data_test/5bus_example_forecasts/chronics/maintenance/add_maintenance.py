import pandas as pd
import numpy as np

line_name = ['0_1_0', '0_2_1', '0_3_2', '0_4_3', '1_2_4', '2_3_5', '2_3_6',
             '3_4_7']
line_maint_id = 5
indx_maint_start = 6
indx_maint_stop = 10

load = pd.read_csv("load_p.csv.bz2", sep=";")
n_row = load.shape[0]

maintenance = np.zeros((n_row, len(line_name)))
maintenance[indx_maint_start:indx_maint_stop,line_maint_id] = 1.
maint = pd.DataFrame(maintenance, columns=line_name)
maint.to_csv("maintenance.csv.bz2", sep=";", index=False, header=True)
