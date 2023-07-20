# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np

from grid2op.Environment.baseMultiProcessEnv import BaseMultiProcessEnvironment


class SingleEnvMultiProcess(BaseMultiProcessEnvironment):
    """
    This class allows to evaluate a single agent instance on multiple environments running in parallel.

    It is a kind of :class:`BaseMultiProcessEnvironment`. For more information you can consult the
    documentation of this parent class. It allows to interact at the same time with different copy of the
    (same) environment in parallel

    Attributes
    -----------
    env: `list::grid2op.Environment.Environment`
        Al list of environments for which the evaluation will be made in parallel.

    nb_env: ``int``
        Number of parallel underlying environment that will be handled. It is also the size of the list of actions
        that need to be provided in :func:`MultiEnvironment.step` and the return sizes of the list of this
        same function.

    Examples
    --------

    An example on how you can best leverage this class is given in the getting_started notebooks. Another simple
    example is:

    .. code-block:: python

        from grid2op.BaseAgent import DoNothingAgent
        from grid2op.MakeEnv import make
        from grid2op.Environment import SingleEnvMultiProcess

        # create a simple environment
        env = make()
        # number of parrallel environment
        nb_env = 2  # change that to adapt to your system
        NB_STEP = 100  # number of step for each environment

        # create a simple agent
        agent = DoNothingAgent(env.action_space)

        # create the multi environment class
        multi_envs = SingleEnvMultiProcess(env=env, nb_env=nb_env)

        # making is usable
        obs = multi_envs.reset()
        rews = [env.reward_range[0] for i in range(nb_env)]
        dones = [False for i in range(nb_env)]

        # performs the appropriated steps
        for i in range(NB_STEP):
            acts = [None for _ in range(nb_env)]
            for env_act_id in range(nb_env):
                acts[env_act_id] = agent.act(obs[env_act_id], rews[env_act_id], dones[env_act_id])
            obs, rews, dones, infos = multi_envs.step(acts)

            # DO SOMETHING WITH THE AGENT IF YOU WANT

        # close the environments
        multi_envs.close()
        # close the initial environment
        env.close()

    """

    def __init__(self, env, nb_env, obs_as_class=True, return_info=True, logger=None):
        envs = [env for _ in range(nb_env)]
        super().__init__(
            envs,
            obs_as_class=obs_as_class,
            return_info=return_info,
            logger=logger.getChild("SingleEnvMultiProcess")
            if logger is not None
            else None,
        )


if __name__ == "__main__":
    from tqdm import tqdm
    from grid2op import make
    from grid2op.Agent import DoNothingAgent

    nb_env = 8  # change that to adapt to your system
    NB_STEP = 100  # number of step for each environment

    env = make()
    env.seed(42)

    agent = DoNothingAgent(env.action_space)
    multi_envs = SingleEnvMultiProcess(env, nb_env)

    obs = multi_envs.reset()
    rews = [env.reward_range[0] for i in range(nb_env)]
    dones = [False for i in range(nb_env)]

    total_reward = 0.0
    for i in tqdm(range(NB_STEP)):
        acts = [None for _ in range(nb_env)]
        for env_act_id in range(nb_env):
            acts[env_act_id] = agent.act(
                obs[env_act_id], rews[env_act_id], dones[env_act_id]
            )
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
