# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

"""
This file should be used to assess the performance of grid2op in "runner" mode: the loading time are not studied,
neither are the import times.

Data are loaded only once, when the environment is "done" the programm stops.

This corresponds to the situation: you have a trained agent, you want to assess its performance using the runner
"""

import numpy as np
import os

from grid2op import make
from grid2op.Parameters import Parameters
from grid2op.Converter import IdToAct
from grid2op.Rules import AlwaysLegal
from grid2op.Backend import PandaPowerBackend
import cProfile

from utils_benchmark import run_env, str2bool, ProfileAgent

try:
    from lightsim2grid import LightSimBackend
    light_sim_avail = True
except ImportError:
    light_sim_avail = False

ENV_NAME = "rte_case5_example"
ENV_NAME = "rte_case14_realistic"
MAX_TS = 1000


class TestAgent(ProfileAgent):
    def __init__(self, action_space, env_name, action_space_converter=IdToAct, **kwargs_converter):
        ProfileAgent.__init__(self, action_space, env_name=env_name,
                              action_space_converter=action_space_converter, **kwargs_converter)

        self.nb_act_done = 0
        self.act_this = True

    def my_act(self, transformed_obs, reward, done=False):
        if self.act_this:
            res = self.nb_act_done
            self.nb_act_done += 1
            self.nb_act_done %= len(self.action_space.all_actions)
            self.act_this = False
        else:
            res = -1
            self.act_this = True
        return res


def main(max_ts, name, use_lightsim=False, test_env=True):
    param = Parameters()
    if use_lightsim:
        if light_sim_avail:
            backend = LightSimBackend()
        else:
            raise RuntimeError("LightSimBackend not available")
    else:
        backend = PandaPowerBackend()

    param.init_from_dict({"NO_OVERFLOW_DISCONNECTION": True})

    env_klu = make(name, backend=backend, param=param, gamerules_class=AlwaysLegal, test=test_env)
    agent = TestAgent(action_space=env_klu.action_space, env_name=name)

    cp = cProfile.Profile()
    cp.enable()
    nb_ts_klu, time_klu, aor_klu, gen_p_klu, gen_q_klu = run_env(env_klu, max_ts, agent)
    cp.disable()
    nm_f, ext = os.path.splitext(__file__)
    nm_out = "{}_{}_{}.prof".format(nm_f, "lightsim" if use_ls else "pp", name)
    cp.dump_stats(nm_out)
    print("You can view profiling results with:\n\tsnakeviz {}".format(nm_out))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark pyKLU and Pandapower Backend for an agent that takes every '
                                                 'topological action possible')
    parser.add_argument('--name', default=ENV_NAME, type=str,
                        help='Environment name to be used for the benchmark.')
    parser.add_argument('--number', type=int, default=MAX_TS,
                        help='Maximum number of time steps for which the benchamark will be run.')
    parser.add_argument("--use_ls", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Use the LightSim2Grid Backend.")
    parser.add_argument("--no_test", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Do not use a test environment for the profiling (default to False: meaning you use a test env)")

    args = parser.parse_args()

    max_ts = int(args.number)
    name = str(args.name)
    use_ls = args.use_ls
    test_env = not args.no_test
    main(max_ts, name, use_lightsim=use_ls, test_env=test_env)
