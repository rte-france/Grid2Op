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
from grid2op.Agent import AgentWithConverter
from grid2op.Parameters import Parameters
from grid2op.Converter import IdToAct
from grid2op.Rules import AlwaysLegal
from grid2op.Backend import PandaPowerBackend
import cProfile

from utils_benchmark import run_env, str2bool

try:
    from lightsim2grid.LightSimBackend import LightSimBackend
    light_sim_avail = True
except ImportError:
    light_sim_avail = False

ENV_NAME = "rte_case5_example"
ENV_NAME = "rte_case14_realistic"
MAX_TS = 1000


class TestAgent(AgentWithConverter):
    def __init__(self, action_space, env_name, action_space_converter=IdToAct, **kwargs_converter):
        AgentWithConverter.__init__(self, action_space, action_space_converter=action_space_converter, **kwargs_converter)
        self.action_space.all_actions = []

        # do nothing
        all_actions_tmp = [action_space()]

        # powerline switch: disconnection
        for i in range(action_space.n_line):
            if env_name == "rte_case14_realistic":
                if i == 18:
                    continue
            elif env_name == "rte_case5_example":
                pass
            elif env_name == "rte_case118_example":
                if i == 6:
                    continue
                if i == 26:
                    continue
                if i == 72:
                    continue
                if i == 73:
                    continue
                if i == 80:
                    continue
                if i == 129:
                    continue
                if i == 140:
                    continue
                if i == 176:
                    continue
                if i == 177:
                    continue
            all_actions_tmp.append(action_space.disconnect_powerline(line_id=i))

        # other type of actions
        all_actions_tmp += action_space.get_all_unitary_topologies_set(action_space)
        # self.action_space.all_actions += action_space.get_all_unitary_redispatch(action_space)

        if env_name == "rte_case14_realistic":
            # remove action that makes the powerflow diverge
            breaking_acts = [action_space({"set_bus": {"lines_or_id": [(7,2), (8,1), (9,1)],
                                                       "lines_ex_id": [(17,2)],
                                                       "generators_id": [(2,2)],
                                                       "loads_id": [(4,1)]}}),
                             action_space({"set_bus": {"lines_or_id": [(10, 2), (11, 1), (19,2)],
                                                       "lines_ex_id": [(16, 2)],
                                                       "loads_id": [(5, 1)]}}),
                             action_space({"set_bus": {"lines_or_id": [(5, 1)],
                                                       "lines_ex_id": [(2, 2)],
                                                       "generators_id": [(1, 2)],
                                                       "loads_id": [(1, 1)]}}),
                             action_space({"set_bus": {"lines_or_id": [(6, 2), (15, 2), (16, 1)],
                                                       "lines_ex_id": [(3, 2), (5, 2)],
                                                       "loads_id": [(2, 1)]}}),
                            action_space({"set_bus": {"lines_or_id": [(18, 1)],
                                                      "lines_ex_id": [(15, 2), (19, 2)],
                                                      }})
            ]
        elif env_name == "rte_case118_example":
            breaking_acts = [action_space({"set_bus": {"lines_or_id": [(100, 2), (129, 1), (173, 2)],
                                                       # "lines_ex_id": [(17,2)],
                                                       "generators_id": [(2, 2)],
                                                       "loads_id": [(6, 1)]
                                                       }}),
                             action_space({"set_bus": {"lines_or_id": [(100, 2), (129, 1), (173, 2)],
                                                       # "lines_ex_id": [(17,2)],
                                                       "generators_id": [(2, 2)],
                                                       "loads_id": [(6, 2)]
                                                       }}),
                             action_space({"set_bus": {"lines_or_id": [(100, 2), (129, 1), (173, 2)],
                                                       # "lines_ex_id": [(17,2)],
                                                       "generators_id": [(2, 1)],
                                                       "loads_id": [(6, 1)]
                                                       }}),
                             action_space({"set_bus": {"lines_or_id": [(140, 1)],
                                                       "lines_ex_id": [(129, 2)],
                                                       # "generators_id": [(2, 1)],
                                                       # "loads_id": [(6, 1)]
                                                       }}),
                             action_space({"set_bus": {"lines_or_id": [(57, 2), (80, 1), (83, 2)],
                                                       "lines_ex_id": [(2, 2), (13, 2), (24, 2), (35, 2)],
                                                       "generators_id": [(6, 2)],
                                                       "loads_id": [(8, 2)]
                                                       }}),
                             action_space({"set_bus": {"lines_or_id": [(57, 2), (80, 1), (83, 2)],
                                                       "lines_ex_id": [(2, 2), (13, 2), (24, 2), (35, 2)],
                                                       "generators_id": [(6, 2)],
                                                       "loads_id": [(8, 1)]
                                                       }}),
                             action_space({"set_bus": {"lines_or_id": [(57, 2), (80, 1), (83, 2)],
                                                       "lines_ex_id": [(2, 2), (13, 2), (24, 2), (35, 2)],
                                                       "generators_id": [(6, 1)],
                                                       "loads_id": [(8, 2)]
                                                       }}),
                             action_space({"set_bus": {"lines_or_id": [(57, 2), (80, 1), (83, 2)],
                                                       "lines_ex_id": [(2, 2), (13, 2), (24, 2), (35, 2)],
                                                       "generators_id": [(6, 1)],
                                                       "loads_id": [(8, 1)]
                                                       }}),
            ]
        else:
            breaking_acts = [action_space({"set_bus": {"lines_or_id": [(0,2), (1,2), (2,2), (3,1)],
                                                       "generators_id": [(0,1)],
                                                       "loads_id": [(0,1)]}}),
                             ]

        # filter out actions that break everything
        all_actions = []
        for el in all_actions_tmp:
            if not el in breaking_acts:
                all_actions.append(el)

        # set the action to the action space
        self.action_space.all_actions = all_actions

        # add the action "reset everything to 1 bus"
        self.action_space.all_actions.append(action_space({"set_bus": np.ones(action_space.dim_topo, dtype=np.int),
                                                           "set_line_status": np.ones(action_space.n_line, dtype=np.int)}))
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


def main(max_ts, name, use_lightsim=False):
    param = Parameters()
    if use_lightsim:
        if light_sim_avail:
            backend = LightSimBackend()
        else:
            raise RuntimeError("LightSimBackend not available")
    else:
        backend = PandaPowerBackend()

    param.init_from_dict({"NO_OVERFLOW_DISCONNECTION": True})

    env_klu = make(name, backend=backend, param=param, gamerules_class=AlwaysLegal, test=True)
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
                        help="Activate nice mode.")

    args = parser.parse_args()

    max_ts = int(args.number)
    name = str(args.name)
    use_ls = args.use_ls
    main(max_ts, name, use_lightsim=use_ls)
