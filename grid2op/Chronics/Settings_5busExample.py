"""
This file contains the settings (path to the case file, chronics converter etc.) that allows to make a simple
environment with a powergrid of only 5 buses, 3 laods, 2 generators and 8 powerlines.
"""
import os
from pathlib import Path

file_dir = Path(__file__).parent.absolute()
grid2op_root = file_dir.parent.absolute()
grid2op_root = str(grid2op_root)
dat_dir = os.path.abspath(os.path.join(grid2op_root, "data"))
case_dir = "5bus_example"
grid_file = "5bus_example.json"

EXAMPLE_CASEFILE = os.path.join(dat_dir, case_dir, grid_file)
EXAMPLE_CHRONICSPATH = os.path.join(dat_dir, case_dir, "chronics")

CASE_5_GRAPH_LAYOUT = [(0, 0), (0, 400), (200, 400), (400, 400), (400, 0)]
