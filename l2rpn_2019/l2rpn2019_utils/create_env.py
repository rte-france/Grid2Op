import os
import sys
import grid2op
import grid2op.main
import argparse

import pdb

from grid2op.ChronicsHandler import GridStateFromFileWithForecasts
from grid2op.Runner import Runner
from grid2op.Reward import L2RPNReward
from grid2op.Settings_L2RPN2019 import L2RPN2019_DICT_NAMES, L2RPN2019_CASEFILE
from grid2op.Settings_L2RPN2019 import ReadPypowNetData

from datetime import timedelta
import numpy as np
import pandas as pd
import copy


PATH_DATA_DEFAULT = os.path.abspath(os.path.join("data", "data_l2rpn_2019"))
PATH_DATA = PATH_DATA_DEFAULT
if not os.path.exists(os.path.join("l2rpn2019_utils", "data_location.py")):
    # the script to download the data has been used, so i use that to retrieve where it has been installed
    try:
        from l2rpn2019_utils.data_location import L2RPN_TRAINING_SET as PATH_DATA
    except Exception as e:
        # impossible to load the data
        pass

# todo add confirmation to download data


def make_l2rpn2109_env(path_data=PATH_DATA):
    env = grid2op.make("l2rpn_2019", chronics_path=path_data)
    return env


def get_submitted_controller(submission_dir):
    sys.path.append(submission_dir)
    try:
        import submission
    except ImportError:
        raise ImportError('The submission folder provided (\"{}\") should contain a file submission.py containing your '
                          'controler named as the class Submission.'.format(submission_dir))

    try:
        submitted_controler = submission.Submission
    except:
        raise Exception('Did not find a class named Submission within submission.py; your submission controler should'
                        ' be a class named Submission in submission.py file directly within the ZIP submission file.')
    return submitted_controler


def main(path_save=None,
         submission_dir=".",
         nb_episode=1,
         nb_process=1,
         path_chronics=PATH_DATA,
         path_parameters=None,
         max_timestep=None):

    if path_save is not None:
        path_save = os.path.abspath(path_save)
    else:
        path_save = None

    submitted_controler = get_submitted_controller(submission_dir)

    gridStateclass_kwargs = {"gridvalueClass": ReadPypowNetData}
    if max_timestep is not None and max_timestep > 0:
        gridStateclass_kwargs["max_iter"] = int(max_timestep)

    res = grid2op.main.main(nb_episode=nb_episode,
                            agent_class=submitted_controler,
                            path_casefile=L2RPN2019_CASEFILE,
                            path_chronics=path_chronics,
                            names_chronics_to_backend=L2RPN2019_DICT_NAMES,
                            gridStateclass_kwargs=gridStateclass_kwargs,
                            reward_class=L2RPNReward,
                            path_save=path_save,
                            nb_process=nb_process,
                            path_parameters=path_parameters)
    if path_save is not None:
        print("Done and data saved in : \"{}\"".format(path_save))
    return res


def get_runner(path_chronics=PATH_DATA,
               submission_dir="."):
    submitted_controler = get_submitted_controller(submission_dir)
    runner = Runner(init_grid_path=L2RPN2019_CASEFILE,
                    path_chron=path_chronics,
                    names_chronics_to_backend=L2RPN2019_DICT_NAMES,
                    gridStateclass_kwargs={"gridvalueClass": ReadPypowNetData},
                    rewardClass=L2RPNReward,
                    agentClass=submitted_controler)
    return runner


