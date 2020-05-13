# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

"""
TODO documentation of this function!

"""
import os

import argparse

from grid2op.Observation import CompleteObservation
from grid2op.Chronics import Multifolder
from grid2op.Reward import FlatReward
from grid2op.Agent import DoNothingAgent
from grid2op.Backend import PandaPowerBackend
from grid2op.Rules import AlwaysLegal
from grid2op.Runner import Runner


def main_run(path_casefile=None,
             path_chronics=None,
             path_parameters=None,
             chronics_class=Multifolder,
             backend_class=PandaPowerBackend,
             agent_class=DoNothingAgent,
             reward_class=FlatReward,
             observation_class=CompleteObservation,
             legalAct_class=AlwaysLegal,
             nb_episode=3,
             nb_process=1,
             path_save=None,
             names_chronics_to_backend=None,
             gridStateclass_kwargs={}):
    init_grid_path = os.path.abspath(path_casefile)

    path_chron = os.path.abspath(path_chronics)

    parameters_path = path_parameters

    runner = Runner(init_grid_path=init_grid_path,
                    path_chron=path_chron,
                    parameters_path=parameters_path,
                    names_chronics_to_backend=names_chronics_to_backend,
                    gridStateclass=chronics_class,
                    gridStateclass_kwargs=gridStateclass_kwargs,
                    backendClass=backend_class,
                    rewardClass=reward_class,
                    agentClass=agent_class,
                    observationClass=observation_class,
                    legalActClass=legalAct_class)

    res = runner.run(nb_episode=nb_episode, nb_process=nb_process, path_save=path_save)
    return res


def cli_main():
    parser = argparse.ArgumentParser(description='Launch the evaluation of the Grid2Op ("Grid To Operate") code.')
    parser.add_argument('--path_save', default=None,
                        help='The path where the log of the experience will be stored (default: None -> nothing stored)')
    parser.add_argument('--nb_process', type=int, default=1,
                        help='The number of process used for each evaluation (note that if nb_process > nb_episode then nb_episode is used.')
    parser.add_argument('--nb_episode', type=int, default=3,
                        help='The number of episode to play (default 3)')
    parser.add_argument('--path_casefile', type=str, required=True,
                        help='Path where the case file is located (casefile is the file describing the powergrid)')
    parser.add_argument('--path_chronics', type=str, required=True,
                        help='Path where the chronics (temporal variation of loads and production usually are located)')
    parser.add_argument('--path_parameters', default=None,
                        help='Path where the _parameters of the game are stored')

    args = parser.parse_args()
    return args


def main_cli(args=None):
    if args is None:
        args = cli_main()

    if args.path_save is not None:
        path_save = str(args.path_save)
    else:
        path_save = None
    if args.path_parameters is not None:
        path_parameter = str(args.path_parameters)
    else:
        path_parameter = None

    names_chronics_to_backend = None

    # actually performing the run
    msg_ = "Running Grid2Op:\n\t- on case file at \"{case_file}\"\n\t- with data located at \"{data}\""
    msg_ += "\n\t- using {process} process(es)\n\t- for {nb_episode} episodes"
    if args.path_save is None:
        msg_ += "\n\t- results will not be saved"
    else:
        msg_ += "\n\t- results will be saved in \"{}\"".format(args.path_save)
    print(msg_.format(case_file=args.path_casefile, data=args.path_chronics, process=args.nb_process,
                      nb_episode=args.nb_episode))
    res = main_run(path_save=path_save,
                   nb_process=args.nb_process,
                   nb_episode=args.nb_episode,
                   path_casefile=args.path_casefile,
                   path_chronics=args.path_chronics,
                   path_parameters=path_parameter,
                   names_chronics_to_backend=names_chronics_to_backend)
    print("The results are:")
    for chron_name, _, cum_reward, nb_time_step, max_ts in res:
        msg_tmp = "\tFor chronics located at {}\n".format(chron_name)
        msg_tmp += "\t\t - cumulative reward: {:.2f}\n".format(cum_reward)
        msg_tmp += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(nb_time_step, max_ts)
        print(msg_tmp)


if __name__ == "__main__":
    args = cli_main()
    main_cli(args)
