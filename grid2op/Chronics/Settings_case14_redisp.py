"""
This file contains the settings (path to the case file, chronics converter etc.) that allows to make a simple
environment with a powergrid of only 5 buses, 3 laods, 2 generators and 8 powerlines.
"""
import os
import pkg_resources
import numpy as np
from pathlib import Path

file_dir = Path(__file__).parent.absolute()
grid2op_root = file_dir.parent.absolute()
dat_dir = os.path.abspath(os.path.join(grid2op_root, "data"))
case_dir = "case14_redisp"
grid_file = "case14_redisp.json"

case14_redisp_CASEFILE = os.path.join(dat_dir, case_dir, grid_file)
case14_redisp_CHRONICSPATH = os.path.join(dat_dir, case_dir, "chronics")

case14_redisp_TH_LIM = np.array([3.84900179e+02, 3.84900179e+02, 2.28997102e+05, 2.28997102e+05,
                                   2.28997102e+05, 1.52664735e+04, 2.28997102e+05, 3.84900179e+02,
                                   3.84900179e+02, 1.83285800e+02, 3.84900179e+02, 3.84900179e+02,
                                   2.28997102e+05, 2.28997102e+05, 6.93930612e+04, 3.84900179e+02,
                                   3.84900179e+02, 2.40562612e+02, 3.40224266e+03, 3.40224266e+03])
