"""
This file contains the settings (path to the case file, chronics converter etc.) that allows to make a simple
environment with a powergrid of only 5 buses, 3 laods, 2 generators and 8 powerlines.
"""
import os
import pkg_resources
import copy
import warnings

# the reference powergrid was different than the default case14 of the litterature.
EXAMPLE_CASEFILE = os.path.abspath(os.path.join(pkg_resources.resource_filename(__name__, "data"),
                                                "5bus_example", "5bus_example.json"))
EXAMPLE_CHRONICSPATH = os.path.join(pkg_resources.resource_filename(__name__, "data"), "5bus_example", "chronics")

CASE_5_GRAPH_LAYOUT = [(0, 0), (0, 400), (200, 400), (400, 400), (400, 0)]