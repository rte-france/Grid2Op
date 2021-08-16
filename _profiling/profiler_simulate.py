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


def make_env():
    env_name = "l2rpn_icaps_2021"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        fake_env = grid2op.make(env_name, test=True)
    param = fake_env.parameters
    param.NO_OVERFLOW_DISCONNECTION = True
    env = grid2op.make(env_name+"_small", backend=LightSimBackend(), param=param)
    return env


def run_env(env):
    done = False
    while not done:
        act = env.action_space()
        obs, reward, done, info = env.step(act)
        if not done:
            simulate(obs, env.action_space())


def simulate(obs, act):
    simobs, rim_r, sim_d, sim_info = obs.simulate(act)


if __name__ == "__main__":
    env = make_env()
    cp = cProfile.Profile()
    cp.enable()
    run_env(env)
    cp.disable()
    nm_f, ext = os.path.splitext(__file__)
    nm_out = f"{nm_f}_{nm_bk_used}.prof"
    cp.dump_stats(nm_out)
    print("You can view profiling results with:\n\tsnakeviz {}".format(nm_out))
