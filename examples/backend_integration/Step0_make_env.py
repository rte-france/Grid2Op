# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

"""
In this script we will create a function to make a proper grid2Op environment that will use
to test / check our backend.

Grid2op supports lots of functions. Here we mainly spend our time to deactivate them in 
order to have something has clean and predictable as possible.

The most notable thing in this script is the use of `BackendConverter`. This class allows to
manipulate (agent, time series etc.) data as if they come for the `source_backend_class` but
internally, grid2op uses the `target_backend_class` (in this case the default PandaPowerBackend)
to carry out the computation.

This is especially usefull when you want to write a new "backend" because:

1) as the "source_backend_class" is never really used, you are not forced
   to implement everything at once before being able to make some tests
2) it provides a default "mapping" from elements in the grid you load that
   might have different names (*eg* "load_1_2" in PandaPowerBackend and 
   "load_2_1" in your backend) and you are still able to read the time
   series provided in the grid2op package.

We recommend not to spend much time looking at this code but keep in mind that 
when backends are developed, BackendConverter might be a usefull tools.
"""

import warnings

import grid2op
from grid2op.Action import CompleteAction
from grid2op.Converter import BackendConverter
from grid2op.Backend import PandaPowerBackend
from grid2op.Reward import ConstantReward
from grid2op.Opponent import BaseOpponent


def make_env_for_backend(env_name, backend_class):
    # env_name: one of:
    # - rte_case5_example: the grid in the documentation (completely fake grid)
    # - l2rpn_case14_sandbox: inspired from IEEE 14
    # - l2rpn_neurips_2020_track1: inspired from IEEE 118 (only a third of it)
    # - l2rpn_wcci_2022_dev: inspired from IEEE 118 (entire grid)
    if env_name == "rte_case5_example":
        pass
    elif env_name == "l2rpn_case14_sandbox":
        pass
    elif env_name == "l2rpn_neurips_2020_track1":
        pass
    elif env_name == "l2rpn_wcci_2022_dev":
        pass
    else:
        raise RuntimeError(f"Unknown grid2op environment name {env_name} used when developping a new backend.")
        
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        env = grid2op.make(env_name,
                           test=True,
                           action_class=CompleteAction,  # we tell grid2op we will manipulate all type of actions
                           reward_class=ConstantReward,  # we don't have yet redispatching data, that might be use by the reward
                           opponent_class=BaseOpponent,  # we deactivate the opponents
                           backend=BackendConverter(source_backend_class=backend_class,
                                                    target_backend_class=PandaPowerBackend,
                                                    use_target_backend_name=True)
                          )
    obs = env.reset()
    return env, obs
