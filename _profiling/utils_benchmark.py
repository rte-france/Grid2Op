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
    beg_ = time.time()
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
            if done:
                print(act)
    end_ = time.time()
    total_time = end_ - beg_
    return nb_ts, total_time, aor, gen_p, gen_q


def run_env_with_reset(env, max_ts, agent):
    nb_rows = min(env.chronics_handler.max_timestep(), max_ts)
    aor = np.zeros((nb_rows, env.n_line))
    gen_p = np.zeros((nb_rows, env.n_gen))
    gen_q = np.zeros((nb_rows, env.n_gen))
    obs = env.get_obs()
    done = False
    reward = env.reward_range[0]
    nb_ts = 0
    beg_ = time.time()
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
                done = False
                reward = env.reward_range[0]
                obs = env.reset()
    end_ = time.time()
    total_time = end_ - beg_
    return nb_ts, total_time, aor, gen_p, gen_q


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')