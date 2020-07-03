# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
from multiprocessing import Process, Pipe
import numpy as np

from grid2op.dtypes import dt_int
from grid2op.Exceptions import Grid2OpException, MultiEnvException
from grid2op.Space import GridObjects
from grid2op.Environment.BaseMultiProcessEnv import BaseMultiProcessEnvironment
from grid2op.Action import BaseAction


class MultiEnvMultiProcess(BaseMultiProcessEnvironment):
    """
    This class allows to evaluate a single agent instance on multiple environments running in parrallel.

    It is a kind of :class:`BaseMultiProcessEnvironment`. For more information you can consult the
    documentation of this parent class. This class allows to interact at the same time with different copy of
    possibly different environments in parallel

    Attributes
    -----------
    envs: `list:grid2op.Environment.Environment`
        Al list of environments for which the evaluation will be made in parallel.

    nb_envs: ``list:int``
        Number of parallel underlying environment that will be handled. 
        MUST be the same length as the parameter `envs`.
        The total number of subprocesses will be the sum of this list.


    Examples
    --------
    This class can be used as:

    .. code-block:: python

        import grid2op
        from grid2op.Environment import MultiEnvMultiProcess
        env0 = grid2op.make()  # create an environment
        env1 = grid2op.make()  # create a second environment, that can be similar, or not
        # it is recommended to filter or create the environment with different parameters, otherwise this class
        # is of little interest
        envs = [env0, env1]  # list of all environments created
        nb_envs = [1, 7]  # number of "copies" of each environment that will be made.
        # in this case the first one will be copied only once, and the second one 7 times.
        # the total number of environments used in the multi env will be the sum(nb_envs), here 8.

        multi_env = MultiEnvMultiProcess(envs=envs, nb_envs=nb_envs)
        # and now you can use it like any other grid2op environment (almost)
        observations = multi_env.reset()

    """
    def __init__(self, envs, nb_envs):
        try:
            nb_envs = np.array(nb_envs)
            nb_envs = nb_envs.astype(dt_int)
        except Exception as exc_:
            raise MultiEnvException("\"nb_envs\" argument should be a list of integers. We could not "
                                    "convert it to such with error \"{}\"".format(exc_))

        if np.any(nb_envs < 0):
            raise MultiEnvException("You ask to perform \"{}\" copy of an environment. This is a negative "
                                    "integer. I cannot do that. Please make sure \"nb_envs\" argument "
                                    "is all made of strictly positive integers and not {}."
                                    "".format(np.min(nb_envs), nb_envs))
        if np.any(nb_envs == 0):
            raise MultiEnvException("You ask to perform 0 copy of an environment. This is not supported at "
                                    "the moment. Please make sure \"nb_envs\" argument "
                                    "is all made of strictly positive integers and not {}."
                                    "".format(nb_envs))

        all_envs = []
        for e, n in enumerate(nb_envs):
            all_envs += [envs[e] for _ in range(n)]
        super().__init__(all_envs)


if __name__ == "__main__":
    from tqdm import tqdm
    from grid2op import make
    from grid2op.Agent import DoNothingAgent

    nb_env = [2,2,1,1,2]  # change that to adapt to your system
    NB_STEP = 100  # number of step for each environment

    env = make()
    env.seed(42)
    envs = [env, env, env, env, env]

    agent = DoNothingAgent(env.action_space)
    multi_envs = MultiEnvMultiProcess(envs, nb_env)

    obs = multi_envs.reset()
    rews = [env.reward_range[0] for i in range(multi_envs.nb_env)]
    dones = [False for i in range(multi_envs.nb_env)]

    total_reward = 0.
    for i in tqdm(range(NB_STEP)):
        acts = [None for _ in range(multi_envs.nb_env)]
        for env_act_id in range(multi_envs.nb_env):
            acts[env_act_id] = agent.act(obs[env_act_id], rews[env_act_id], dones[env_act_id])
        obs, rews, dones, infos = multi_envs.step(acts)
        total_reward += np.sum(rews)
        len(rews)

    multi_envs.close()

    ob = env.reset()
    rew = env.reward_range[0]
    done = False
    total_reward_single = 0
    for i in tqdm(range(NB_STEP)):
        act = agent.act(ob, rew, done)
        ob, rew, done, info = env.step(act)
        if done:
            ob = env.reset()
        total_reward_single += np.sum(rew)
    env.close()
    print("total_reward mluti_env: {}".format(total_reward))
    print("total_reward single env: {}".format(total_reward_single))

