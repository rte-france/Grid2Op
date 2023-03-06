# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

"""
This script provides, given the implementation of a (at least minimal) backend
some standard usage of said backend with grid2op, the way people "normally"
interacts with it.

"""
import copy
import pandas as pd

from Step5_modify_topology import CustomBackend_Minimal


if __name__ == "__main__":
    import grid2op
    from grid2op.Action import CompleteAction
    import os
    import warnings
    
    path_grid2op = grid2op.__file__
    path_data_test = os.path.join(os.path.split(path_grid2op)[0], "data")
    
    env_name = "rte_case5_example"
    # one of:
    # - rte_case5_example: the grid in the documentation (completely fake grid)
    # - l2rpn_case14_sandbox: inspired from IEEE 14
    # - l2rpn_neurips_2020_track1: inspired from IEEE 118 (only a third of it)
    # - l2rpn_wcci_2022_dev: inspired from IEEE 118 (entire grid)
    
    # change the load (to be more consistent with standard grid2op usage, we do it
    # using an environment that uses a real backend, and not our "second step toward a backend")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        env = grid2op.make(env_name, test=True, action_class=CompleteAction, backend=CustomBackend_Minimal())
    obs = env.reset()
    
    ########### First "test" perform nothing and see what it gives
    done = False
    nb_step = 0
    while True:
        obs, reward, done, info = env.step(env.action_space())
        if done:
            break
        nb_step += 1
    print(f"{nb_step} have been made with your backend")
    