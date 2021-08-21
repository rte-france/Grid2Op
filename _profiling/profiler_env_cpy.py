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

from tqdm import tqdm
import os
import cProfile
import pdb

from profiler_simulate import make_env, bk_cls, nm_bk_used


def run_env(env, max_step=100):
    done = False
    step_cnt = 0
    with tqdm(total=max_step) as pbar:
        while not done:
            act = env.action_space()
            obs, reward, done, info = env.step(act)
            if not done:
                copy_env(env)
            step_cnt += 1
            if step_cnt > max_step:
                break
            pbar.update(1)


def copy_env(env):
    res = env.copy()


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
