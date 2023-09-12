# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LightSim2grid, LightSim2grid a implements a c++ backend targeting the Grid2Op platform.

import time
import numpy as np
from tqdm import tqdm
import argparse
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct


class ProfileAgent(AgentWithConverter):
    def __init__(self,
                 action_space,
                 env_name,
                 action_space_converter=IdToAct,
                 **kwargs_converter
                 ):
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
            elif env_name == "rte_case118_example" or env_name.startswith("l2rpn_neurips_2020_track2"):
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
            elif env_name == "l2rpn_wcci_2020":
                if i == 2:
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
        elif env_name == "rte_case118_example" or env_name.startswith("l2rpn_neurips_2020_track2"):
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
                             action_space({"set_bus": {"lines_or_id": [(100, 2), (129, 1), (173, 2)],
                                                       # "lines_ex_id": [(2, 2), (13, 2), (24, 2), (35, 2)],
                                                       "generators_id": [(2, 1)],
                                                       "loads_id": [(6, 2)]
                                                       }}),
            ]
        elif env_name == "l2rpn_wcci_2020":
            breaking_acts = [action_space({"set_bus": {"lines_or_id": [(5, 2), (6, 2)],
                                                       "lines_ex_id": [(1, 2), (2, 1), (4, 2), (55, 2)],
                                                       # "generators_id": [(2, 2)],
                                                       # "loads_id": [(6, 1)]
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

        # add the action "reset everything to bus 1"
        self.action_space.all_actions.append(action_space({"set_bus": np.ones(action_space.dim_topo, dtype=int),
                                                           "set_line_status": np.ones(action_space.n_line,
                                                                                      dtype=int)}))


def print_res(env_klu, env_pp,
              nb_ts_klu, nb_ts_pp,
              time_klu, time_pp,
              aor_klu, aor_pp,
              gen_p_klu, gen_p_pp,
              gen_q_klu, gen_q_pp):
    print("Overall speed-up of KLU vs pandapower (for grid2opbackend) {:.2f}\n".format(time_pp / time_klu))
    print("PyKLU Backend {} time steps in {}s ({:.2f} it/s)".format(nb_ts_klu, time_klu, nb_ts_klu/time_klu))
    print("\tTime apply act: {:.2f}ms".format(1000. * env_klu._time_apply_act / nb_ts_klu))
    print("\tTime powerflow: {:.2f}ms".format(1000. * env_klu._time_powerflow / nb_ts_klu))
    print("\tTime extract observation: {:.2f}ms".format(1000. * env_klu._time_extract_obs / nb_ts_klu))

    print("Pandapower Backend {} time steps in {}s ({:.2f} it/s)".format(nb_ts_pp, time_pp, nb_ts_pp/time_pp))
    print("\tTime apply act: {:.2f}ms".format(1000. * env_pp._time_apply_act / nb_ts_pp))
    print("\tTime powerflow: {:.2f}ms".format(1000. * env_pp._time_powerflow / nb_ts_pp))
    print("\tTime extract observation: {:.2f}ms".format(1000. * env_pp._time_extract_obs / nb_ts_pp))

    print("Absolute value of the difference for aor: {}".format(np.max(np.abs(aor_klu - aor_pp))))
    print("Absolute value of the difference for gen_p: {}".format(np.max(np.abs(gen_p_klu - gen_p_pp))))
    print("Absolute value of the difference for gen_q: {}".format(np.max(np.abs(gen_q_klu - gen_q_pp))))


def run_env(env, max_ts, agent):
    nb_rows = min(env.chronics_handler.max_timestep(), max_ts)
    aor = np.zeros((nb_rows, env.n_line))
    gen_p = np.zeros((nb_rows, env.n_gen))
    gen_q = np.zeros((nb_rows, env.n_gen))
    obs = env.get_obs()
    done = False
    reward = env.reward_range[0]
    nb_ts = 0
    prev_act = None
    beg_ = time.perf_counter()
    with tqdm(total=nb_rows) as pbar:
        while not done:
            act = agent.act(obs, reward, done)
            obs, reward, done, info = env.step(act)
            aor[nb_ts, :] = obs.a_or
            gen_p[nb_ts, :] = obs.prod_p
            gen_q[nb_ts, :] = obs.prod_q
            nb_ts += 1
            pbar.update(1)
            if nb_ts >= max_ts:
                break
            # if np.sum(obs.line_status) < obs.n_line - 1 * (nb_ts % 2 == 1):
            #     print("There is a bug following action; {}".format(act))
            prev_act = act
            # if done:
            #     print(act)
    end_ = time.perf_counter()
    total_time = end_ - beg_
    return nb_ts, total_time, aor, gen_p, gen_q


def run_env_with_reset(env, max_ts, agent, seed=None):
    nb_rows = min(env.chronics_handler.max_timestep(), max_ts)
    aor = np.zeros((nb_rows, env.n_line))
    gen_p = np.zeros((nb_rows, env.n_gen))
    gen_q = np.zeros((nb_rows, env.n_gen))
    if seed is not None:
        env.seed(seed)
    obs = env.reset()
    done = False
    reward = env.reward_range[0]
    nb_ts = 0
    beg_ = time.perf_counter()
    reset_count = 0
    with tqdm(total=nb_rows) as pbar:
        while not done:
            act = agent.act(obs, reward, done)
            obs, reward, done, info = env.step(act)
            aor[nb_ts, :] = obs.a_or
            gen_p[nb_ts, :] = obs.prod_p
            gen_q[nb_ts, :] = obs.prod_q
            nb_ts += 1
            pbar.update(1)
            if nb_ts >= max_ts:
                break

            if done:
                # I reset
                reward = env.reward_range[0]
                obs = env.reset()
                reset_count += 1
                done = False
    end_ = time.perf_counter()
    total_time = end_ - beg_
    return nb_ts, total_time, aor, gen_p, gen_q, reset_count


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')