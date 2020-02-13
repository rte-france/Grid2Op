"""
This file contains the settings (path to the case file, chronics converter etc.) that allows to make a simple
environment with a powergrid of only 5 buses, 3 laods, 2 generators and 8 powerlines.
"""
import os
import pkg_resources
import numpy as np

# the reference powergrid was different than the default case14 of the litterature.
case14_test_CASEFILE = os.path.abspath(os.path.join(pkg_resources.resource_filename(__name__, "data"),
                                                "case14_test", "case14_test.json"))

case14_test_CHRONICSPATH = os.path.join(pkg_resources.resource_filename(__name__, "data"),
                                          "case14_test",
                                          "chronics")

case14_test_TH_LIM = np.array([   352.8251645 ,    352.8251645 , 183197.68156979, 183197.68156979,
                                   183197.68156979,  12213.17877132, 183197.68156979,    352.8251645,
                                      352.8251645 ,    352.8251645 ,    352.8251645 ,    352.8251645,
                                   183197.68156979, 183197.68156979, 183197.68156979,    352.8251645,
                                      352.8251645 ,    352.8251645 ,   2721.79412618,   2721.79412618])