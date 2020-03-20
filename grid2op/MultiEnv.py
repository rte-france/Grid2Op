"""
This class allows to evaluate a single agent instance on multiple environments running in parrallel.

It uses the python "multiprocessing" framework to work, and thus is suitable only on a single machine with multiple
cores (cpu / thread). We do not recommend to use this method on a cluster of different machines.

This class uses the following representation:

- an :grid2op.Agent.Agent: lives in a main process
- different environment lives into different processes
- a call to :func:`MultiEnv.step` will perform one step per environment, in parallel using a ``Pipe`` to transfer data
  to and from the main process from each individual environment process. It is a synchronous function. It means
  it will wait for every environment to finish the step before returning all the information.

There are some limitations. For example, even if forecast are available, it's not possible to use forecast of the
observations. This imply that :func:`grid2op.Observation.Observation.simulate` is not available when using
:class:`MultiEnvironment`

Compare to regular Environments, :class:`MultiEnvironment` simply stack everything. You need to send not a single
:class:`grid2op.Action.Action` but as many actions as there are underlying environments. You receive not one single
:class:`grid2op.Observation.Observation` but as many observations as the number of underlying environments.

A broader support of regular grid2op environment capabilities as well as support for
:func:`grid2op.Observation.Observation.simulate` call will be added in the future.

An example on how you can best leverage this class is given in the getting_started notebooks. Another simple example is:

.. code-block:: python

    from grid2op.Agent import DoNothingAgent
    from grid2op.MakeEnv import make

    # create a simple environment
    env = make()
    # number of parrallel environment
    nb_env = 2  # change that to adapt to your system
    NB_STEP = 100  # number of step for each environment

    # create a simple agent
    agent = DoNothingAgent(env.action_space)

    # create the multi environment class
    multi_envs = MultiEnvironment(env=env, nb_env=nb_env)

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

import copy
import os
import time
from multiprocessing import Process, Pipe
import numpy as np


try:
    from .Exceptions import *
    from .Space import GridObjects
    from .Environment import Environment
    from .Action import Action

except (ImportError, ModuleNotFoundError):
    from Exceptions import *
    from Space import GridObjects
    from Environment import Environment
    from Action import Action

    from Agent import DoNothingAgent
    from MakeEnv import make

import pdb

# TODO test this class.


class RemoteEnv(Process):
    """
    This class represent the environment that is executed on a remote process.

    Note that the environment is only created in the subprocess, and is not available in the main process. Once created
    it is not possible to access anything directly from it in the main process, where the Agent lives. Only the
    :class:`grid2op.Observation.Observation` are forwarded to the agent.

    """
    def __init__(self, env_params, remote, parent_remote, seed, name=None):
        Process.__init__(self, group=None, target=None, name=name)
        self.backend = None
        self.env = None
        self.env_params = env_params
        self.remote = remote
        self.parent_remote = parent_remote
        self.seed_used = seed
        self.space_prng = None

    def init_env(self):
        """
        Initialize the environment  that will perform all the computation of this process.
        Remember the environment only lives in this process. It cannot
        be transfer to / from the main process.

        This function also makes sure the chronics are read in different order accross all processes. This is done
        by calling the :func:`grid2op.ChronicsHandler.GridValue.shuffle` method. An example of how to use this function
        is provided in :func:`grid2op.ChronicsHandler.Multifolder.shuffle`.

        """
        # TODO documentation
        # TODO seed of the environment.

        self.space_prng = np.random.RandomState()
        self.space_prng.seed(seed=self.seed_used)
        self.backend = self.env_params["backendClass"]()
        del self.env_params["backendClass"]
        self.env = Environment(**self.env_params, backend=self.backend)
        self.env.chronics_handler.shuffle(shuffler=lambda x: x[self.space_prng.choice(len(x), size=len(x), replace=False)])

    def _clean_observation(self, obs):
        obs._forecasted_grid = []
        obs._forecasted_inj = []
        obs._obs_env = None
        obs.action_helper = None

    def get_obs_ifnotconv(self):
        # TODO dirty hack because of wrong chronics
        # need to check!!!
        conv = False
        obs = None
        while not conv:
            try:
                obs = self.env.reset()
                conv = True
            except:
                pass
        return obs

    def run(self):
        if self.env is None:
            self.init_env()

        while True:
            cmd, data = self.remote.recv()
            if cmd == 'get_spaces':
                self.remote.send((self.env.observation_space, self.env.action_space))
            elif cmd == 's':
                # perform a step
                obs, reward, done, info = self.env.step(data)
                if done:
                    # if done do a reset
                    obs = self.get_obs_ifnotconv()
                self._clean_observation(obs)
                self.remote.send((obs, reward, done, info))
            elif cmd == 'r':
                # perfom a reset
                obs = self.get_obs_ifnotconv()
                self._clean_observation(obs)
                self.remote.send(obs)
            elif cmd == 'c':
                # close everything
                self.env.close()
                self.remote.close()
                break
            elif cmd == 'z':
                # adapt the chunk size
                self.env.set_chunk_size(data)
            else:
                raise NotImplementedError


class MultiEnvironment(GridObjects):
    """
    This class allows to execute in parallel multiple environments. This can be usefull to train agents such as
    A3C for example.

    Note that it is not per se an Environment, and lack many members of such.

    It uses the multiprocessing python module as a way to handle this parallelism. As some objects are not pickable,
    all the process are completely independant. It is not possible to directly get back in the main process (the one
    where this environment has been created) any attributes of any of the underlying environment.

    Attributes
    -----------
    imported_env: `grid2op.Environment.Environment`
        The environment to duplicated and for which the evaluation will be made in parallel.

    nb_env: ``int``
        Number of parallel underlying environment that will be handled. It is also the size of the list of actions
        that need to be provided in :func:`MultiEnvironment.step` and the return sizes of the list of this
        same function.

    """
    def __init__(self, nb_env, env):
        GridObjects.__init__(self)
        # self.init_grid(env)
        self.imported_env = env
        self.nb_env = nb_env

        self._remotes, self._work_remotes = zip(*[Pipe() for _ in range(self.nb_env)])

        env_params = [env.get_kwargs() for _ in range(self.nb_env)]
        for el in env_params:
            el["backendClass"] = type(env.backend)
        self._ps = [RemoteEnv(env_params=env_,
                              remote=work_remote,
                              parent_remote=remote,
                              name="env: {}".format(i),
                              seed=np.random.randint(np.iinfo(np.uint32).max))
                    for i, (work_remote, remote, env_) in enumerate(zip(self._work_remotes, self._remotes, env_params))]

        for p in self._ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self._work_remotes:
            remote.close()

        self._waiting = True

    def _send_act(self, actions):
        for remote, action in zip(self._remotes, actions):
            remote.send(('s', action))
        self._waiting = True

    def _wait_for_obs(self):
        results = [remote.recv() for remote in self._remotes]
        self._waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def step(self, actions):
        """
        Perform a step in all the underlying environments.
        If one or more of the underlying environments encounters a game over, it is automatically restarted.

        The observation sent back to the user is the observation after the :func:`grid2op.Environment.Environment.reset`
        has been called.

        It has no impact on the other underlying environments.

        Parameters
        ----------
        actions: ``list``
            List of :attr:`MultiEnvironment.nb_env` :class:`grid2op.Action.Action`. Each action will be executed
            in the corresponding underlying environment.

        Returns
        -------
        obs: ``list``
            List all the observations returned by each underlying environment.

        rews: ``list``
            List all the rewards returned by each underlying environment.

        dones: ``list``
            List all the "done" returned by each underlying environment. If one of this value is "True" this means
            the environment encounter a game over.

        infos
        """
        if len(actions) != self.nb_env:
            raise MultiEnvException("Incorrect number of actions provided. You provided {} actions, but the "
                                    "MultiEnvironment counts {} different environment."
                                    "".format(len(actions), self.nb_env))
        for act in actions:
            if not isinstance(act, Action):
                raise MultiEnvException("All actions send to MultiEnvironment.step should be of type \"grid2op.Action\""
                                        "and not {}".format(type(act)))

        self._send_act(actions)
        obs, rews, dones, infos = self._wait_for_obs()
        return obs, rews, dones, infos

    def reset(self):
        """
        Reset all the environments, and return all the associated observation.

        Returns
        -------
        res: ``list``
            The list of all observations. This list counts :attr:`MultiEnvironment.nb_env` elements, each one being
            an :class:`grid2OP.Observation.Observations`.

        """
        for remote in self._remotes:
            remote.send(('r', None))
        res = [remote.recv() for remote in self._remotes]
        return np.stack(res)

    def close(self):
        """
        Close all the environments and all the processes.
        """
        for remote in self._remotes:
            remote.send(('c', None))

    def set_chunk_size(self, new_chunk_size):
        """
        Dynamically adapt the amount of data read from the hard drive. Usefull to set it to a low integer value (eg 10
        or 100) at the beginning of the learning process, when agent fails pretty quickly.

        This takes effect only after a reset has been performed.

        Parameters
        ----------
        new_chunk_size: ``int``
            The new chunk size (positive integer)

        """
        try:
            new_chunk_size = int(new_chunk_size)
        except Exception as e:
            raise Grid2OpException("Impossible to set the chunk size. It should be convertible a integer, and not"
                                   "{}".format(new_chunk_size))

        if new_chunk_size <= 0:
            raise Grid2OpException("Impossible to read less than 1 data at a time. Please make sure \"new_chunk_size\""
                                   "is a positive integer.")

        for remote in self._remotes:
            remote.send(('z', new_chunk_size))


if __name__ == "__main__":
    from tqdm import tqdm

    env = make()

    nb_env = 8  # change that to adapt to your system
    NB_STEP = 1000  # number of step for each environment

    agent = DoNothingAgent(env.action_space)
    multi_envs = MultiEnvironment(env=env, nb_env=nb_env)

    obs = multi_envs.reset()
    rews = [env.reward_range[0] for i in range(nb_env)]
    dones = [False for i in range(nb_env)]

    total_reward = 0.
    for i in tqdm(range(NB_STEP)):
        acts = [None for _ in range(nb_env)]
        for env_act_id in range(nb_env):
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

