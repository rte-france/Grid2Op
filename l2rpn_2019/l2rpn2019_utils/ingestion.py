# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import sys
import grid2op
import grid2op.main

from grid2op.Reward import L2RPNReward
from grid2op.Settings_L2RPN2019 import L2RPN2019_DICT_NAMES as names_chronics_to_backend
from grid2op.Settings_L2RPN2019 import ReadPypowNetData
from grid2op.Settings_L2RPN2019 import L2RPN2019_CASEFILE


def run_episode_as_codalab():
    # read arguments
    input_dir = os.path.abspath(sys.argv[1])
    output_dir = os.path.abspath(sys.argv[2])
    program_dir = os.path.abspath(sys.argv[3])
    submission_dir = os.path.abspath(sys.argv[4])

    # create output dir if not existing
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # add proper directories to path
    sys.path.append(program_dir)
    sys.path.append(submission_dir)

    try:
        import submission
    except ImportError:
        raise ImportError('The submission folder should contain a file submission.py containing your controler named '
                          'as the class Submission.')

    multi_episode_path = os.path.join(input_dir)

    try:
        submitted_controler = submission.Submission
    except:
        raise Exception('Did not find a class named Submission within submission.py; your submission controler should'
                        ' be a class named Submission in submission.py file directly within the ZIP submission file.')

    res = grid2op.main.main(nb_episode=2,
                            agent_class=submitted_controler,
                            path_casefile=L2RPN2019_CASEFILE,
                            path_chronics=multi_episode_path,
                            names_chronics_to_backend=names_chronics_to_backend,
                            gridStateclass_kwargs={"gridvalueClass": ReadPypowNetData, "max_iter": 100},
                            reward_class=L2RPNReward,
                            path_save=os.path.abspath(os.path.join(output_dir, "saved_experiment"))
                            )
    print("Done and data saved in : \"{}\"".format(os.path.join(output_dir, "saved_experiment")))


if __name__ == "__main__":
    run_episode_as_codalab()
