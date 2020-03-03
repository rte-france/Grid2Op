"""
This file contains the settings (path to the case file, chronics converter etc.) that allows to make a simple
environment with a powergrid of only 5 buses, 3 laods, 2 generators and 8 powerlines.
"""
import os
import pkg_resources
import numpy as np

# the reference powergrid was different than the default case14 of the litterature with adjusted voltages
# for lower levels
case14_real_CASEFILE = os.path.abspath(os.path.join(pkg_resources.resource_filename(__name__, "data"),
                                                    "case14_realistic", "case14_realistic.json"))

case14_real_CHRONICSPATH = os.path.join(pkg_resources.resource_filename(__name__, "data"),
                                        "case14_realistic", "chronics")

case14_real_TH_LIM = np.array([ 384.900179  ,  384.900179  ,  380.        ,  380.        ,
                                157.        ,  380.        ,  380.        , 1077.7205012 ,
                                461.8802148 ,  769.80036   ,  269.4301253 ,  384.900179  ,
                                760.        ,  380.        ,  760.        ,  384.900179  ,
                                230.9401074 ,  170.79945452, 3402.24266   , 3402.24266   ])