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
from tqdm import tqdm
from Step5_modify_topology import CustomBackend_Minimal


if __name__ == "__main__":
    import grid2op
    from grid2op.Action import CompleteAction
    import os
    import warnings
    from Step0_make_env import make_env_for_backend
    from grid2op.Agent import RecoPowerlineAgent
    from grid2op.Reward import L2RPNReward
    
    path_grid2op = grid2op.__file__
    path_data_test = os.path.join(os.path.split(path_grid2op)[0], "data")
    
    env_name = "rte_case5_example"
    # one of:
    # - rte_case5_example: the grid in the documentation (completely fake grid)
    # - l2rpn_case14_sandbox: inspired from IEEE 14
    # - l2rpn_neurips_2020_track1: inspired from IEEE 118 (only a third of it)
    # - l2rpn_wcci_2022_dev: inspired from IEEE 118 (entire grid)
    converter_env, _ = make_env_for_backend(env_name, CustomBackend_Minimal)
    
    # "real" usecase that corresponds to a realistic use of a
    # backend for grid2op. (note that users are totally not aware of what's
    # going on behing the scene)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        env = grid2op.make(env_name,
                           test=True,
                           action_class=CompleteAction,
                           backend=CustomBackend_Minimal(),
                           reward_class=L2RPNReward,  # we use this mainly for the "greedy agent" (see below)
                           
                           # this is because the load / gen / line names might be different than the one read by pandapower.
                           # so we tell grid2op to use the names it founds in the time series (for loads and generators)
                           # and map them to name found on the grid (as defined in the backend)
                           # but as we don't want to come-up with this dictionnary by hands (which would be
                           # for a real usecase THE ONLY way to go) we simply rely on the grid2op automatic
                           # conversion offered by the converter
                           names_chronics_to_grid=converter_env.backend.names_target_to_source  
                           )
    obs = env.reset()
    
    ########### First "test" perform nothing and see what it gives
    done = False
    nb_step = 0
    with tqdm() as pbar:
        while True:
            obs, reward, done, info = env.step(env.action_space())
            if done:
                break
            nb_step += 1
            pbar.update()
    print(f"{nb_step} steps have been made with your backend with do nothing")
    
    ########## Second "test" perform random actions every now and then
    env.seed(0)
    obs = env.reset()
    done = False
    nb_step = 0
    with tqdm() as pbar:
        while True:
            if nb_step % 10 == 9:
                # do a randome action sometime
                act = env.action_space.sample()
            else:
                # do nothing most of the time
                act = env.action_space()
            obs, reward, done, info = env.step(act)
            if done:
                break
            nb_step += 1
            pbar.update()
    print(f"{nb_step} steps have been made with your backend with some random actions")
    
    ########### Third "test" using an "agent" that "does smart actions" (greedy agent)
    done = False
    nb_step = 0
    obs = env.reset()
    reward = 0.
    agent = RecoPowerlineAgent(env.action_space)
    with tqdm() as pbar:
        while True:
            act = agent.act(obs, reward)
            obs, reward, done, info = env.step(act)
            if done:
                break
            nb_step += 1
            pbar.update()
    print(f"{nb_step} steps have been made with the greedy agent")
    