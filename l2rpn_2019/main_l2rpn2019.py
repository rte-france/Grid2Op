# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

"""
This file is here to emulate the behavior of the pypownet `main` function.

With this file,
"""
import argparse
import os

from l2rpn2019_utils.create_env import main, PATH_DATA

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch the evaluation of the Grid2Op ("Grid To Operate") code.')
    parser.add_argument('--path_save', default=None,
                        help='The path where the log of the experience will be stored '
                             '(default: None -> nothing stored)')
    parser.add_argument('--nb_process', type=int, default=1,
                        help='The number of process used for each evaluation (note that if nb_process > '
                             'nb_episode then nb_episode is used.')
    parser.add_argument('--nb_episode', type=int, default=1,
                        help='The number of episode to play (default 1)')
    parser.add_argument('--path_chronics', type=str, default=PATH_DATA,
                        help='Path where the chronics (temporal variation of loads and production usually) are located')
    parser.add_argument('--path_parameters', default=None,
                        help='Path where the parameters of the game are stored')
    parser.add_argument('--submission_dir', default=".",
                        help='Path where the agent that need to be checked is located.')

    args = parser.parse_args()

    if args.path_save is not None:
        path_save = str(args.path_save)
    else:
        path_save = None

    if args.path_parameters is not None:
        path_parameter = str(args.path_parameters)
    else:
        path_parameter = None

    if not os.path.exists(args.path_chronics):
        raise RuntimeError("Unable to find L2RPN 2019 chronics at \"{}\".\nYou can download the training"
                           "set with:\n \t\"python l2rpn2019_utils\\download_training_data.py --help\" "
                           "or\nif the data are located on your computer use the \"path_chronics\" argument "
                           "of this script: :\n \t"
                           "\"python main_l2rpn2019.py --path_chronics=PAHT/WHERE/L2RPN2019DATA/ARE/LOCATED\""
                           "".format(args.path_chronics))

    if path_save is not None:
        root_dir = os.path.split(path_save)[0]
        repo_exp = os.path.split(path_save)[-1]
        if not os.path.exists(root_dir):
            print("Creating the directory \"{}\" in which the experiments will be saved as "
                  "\"{}\"".format(root_dir, repo_exp))
            os.mkdir(root_dir)

    submission_dir = os.path.abspath(args.submission_dir)
    if not os.path.exists(os.path.join(submission_dir, "submission.py")):
        raise RuntimeError("Impossible to find the file \"submission.py\" at \"{}\". This means you didn't provide any "
                           "agent for which the performance need to be checked. This sc".format(submission_dir))

    res = main(path_save=path_save,
               submission_dir=submission_dir,
               nb_process=args.nb_process,
               nb_episode=args.nb_episode,
               path_chronics=args.path_chronics,
               path_parameters=path_parameter)
