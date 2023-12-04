# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.


"""
This file aims at profiling a case where the "simulate" function is heavily used.
"""

import grid2op
from grid2op.gym_compat import GymEnv

import warnings
try:
    from lightsim2grid import LightSimBackend
    bk_cls = LightSimBackend
    nm_bk_used = "LightSimBackend"
    print("LightSimBackend used")
except ImportError:
    from grid2op.Backend import PandaPowerBackend
    bk_cls = PandaPowerBackend
    nm_bk_used = "PandaPowerBackend"
    print("PandaPowerBackend used")

import os
import cProfile
import pdb


NB_SIMULATE = 10
ENV_NAME = "l2rpn_icaps_2021_small"
ENV_NAME = "l2rpn_idf_2023"


def make_env(env_name=ENV_NAME):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        fake_env = grid2op.make(env_name, test=True)
    param = fake_env.parameters
    param.NO_OVERFLOW_DISCONNECTION = True
    env = grid2op.make(env_name, backend=bk_cls(), param=param)
    env.seed(0)
    env.reset()
    gym_env = GymEnv(env)
    return gym_env, env


def run_env(gym_env, cp_gym_env, env, cp_env):
    done = False
    while not done:
        act = {}
        cp_gym_env.enable()
        obs, reward, done, truncated, info = gym_env.step(act)
        cp_gym_env.disable()
        
    done = False 
    while not done:
        act = env.action_space()
        cp_env.enable()
        obs, reward, done, info = env.step(act)
        cp_env.disable()


if __name__ == "__main__":
    gym_env, env = make_env()
    cp_gym = cProfile.Profile()
    cp_env = cProfile.Profile()
    run_env(gym_env, cp_gym, env, cp_env)
    nm_f, ext = os.path.splitext(__file__)
    nm_out_gym = f"gym_{nm_f}_{nm_bk_used}_{ENV_NAME}_gymenv.prof"
    nm_out_env = f"gym_{nm_f}_{nm_bk_used}_{ENV_NAME}_env.prof"
    cp_gym.dump_stats(nm_out_gym)
    cp_env.dump_stats(nm_out_env)
    print("You can view profiling grid2op raw results with:\n\tsnakeviz {}".format(nm_out_env))
    print("You can view profiling gym results with:\n\tsnakeviz {}".format(nm_out_gym))
# base: 66.7 s
# sans copy dans simulate: 65.2
