# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

"""
This file should be used to assess the performance of grid2op in "early training" mode: a random agent is used
each time its game over the environment is reset
"""

import numpy as np
import os

from grid2op import make
from grid2op.Chronics import GridStateFromFile
from grid2op.Parameters import Parameters
from grid2op.Converter import IdToAct
from grid2op.Rules import AlwaysLegal
from grid2op.Backend import PandaPowerBackend
import cProfile

from utils_benchmark import run_env_with_reset, str2bool, ProfileAgent

try:
    from lightsim2grid.LightSimBackend import LightSimBackend
    light_sim_avail = True
except ImportError:
    light_sim_avail = False

ENV_NAME = "rte_case5_example"
ENV_NAME = "rte_case14_realistic"
MAX_TS = 1000


class TestAgent(ProfileAgent):
    def __init__(self,
                 action_space,
                 env_name,
                 action_space_converter=IdToAct,
                 nb_quiet=0,
                 **kwargs_converter):
        ProfileAgent.__init__(self, action_space, env_name=env_name,
                              action_space_converter=action_space_converter, **kwargs_converter)
        self.nb_act_done = 0
        self.act_this = 0
        self.nb_quiet = nb_quiet
        self._nb_quiet_1 = self.nb_quiet - 1
        self.nb_act = len(self.action_space.all_actions)

    def my_act(self, transformed_obs, reward, done=False):
        if self.act_this % self.nb_quiet == self._nb_quiet_1:
            # do an action
            res = self.space_prng.randint(self.nb_act)
        else:
            # do nothing
            res = 0
        self.act_this += 1
        return res


def main(max_ts, name, use_lightsim=False):
    param = Parameters()
    if use_lightsim:
        if light_sim_avail:
            backend = LightSimBackend()
        else:
            raise RuntimeError("LightSimBackend not available")
    else:
        backend = PandaPowerBackend()

    # param.init_from_dict({"NO_OVERFLOW_DISCONNECTION": True})

    env_klu = make(name, backend=backend, param=param, gamerules_class=AlwaysLegal, test=True,
                   data_feeding_kwargs={"chunk_size": 128, "max_iter": max_ts, "gridvalueClass": GridStateFromFile}
                   )
    agent = TestAgent(action_space=env_klu.action_space, env_name=name, nb_quiet=2)
    agent.seed(42)
    # nb_quiet = 2 : do a random action once every 2 timesteps
    agent.seed(42)
    cp = cProfile.Profile()
    cp.enable()
    nb_ts_klu, time_klu, aor_klu, gen_p_klu, gen_q_klu, reset_count = run_env_with_reset(env_klu, max_ts, agent,
                                                                                         seed=69)
    cp.disable()
    nm_f, ext = os.path.splitext(__file__)
    nm_out = "{}_{}_{}.prof".format(nm_f, "lightsim" if use_ls else "pp", name)
    cp.dump_stats(nm_out)
    print("You can view profiling results with:\n\tsnakeviz {}".format(nm_out))

    print("There were {} resets".format(reset_count))


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
                        help="Activate nice mode.")

    args = parser.parse_args()

    max_ts = int(args.number)
    name = str(args.name)
    use_ls = args.use_ls
    main(max_ts, name, use_lightsim=use_ls)
