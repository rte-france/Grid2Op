# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
from multiprocessing import Process, Pipe, Array
import numpy as np
import warnings
import time

from grid2op.Exceptions import EnvError
from grid2op.dtypes import dt_int
from grid2op.Exceptions import Grid2OpException, MultiEnvException
from grid2op.Space import GridObjects
from grid2op.Environment.environment import Environment
from grid2op.Action import BaseAction


class RemoteEnv(Process):
    """
    INTERNAL

     .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    This class represent the environment that is executed on a remote process.

    Note that the environment is only created in the subprocess, and is not available in the main process. Once created
    it is not possible to access anything directly from it in the main process, where the BaseAgent lives. Only the
    :class:`grid2op.Observation.BaseObservation` are forwarded to the agent.

    """

    def __init__(
        self,
        env_params,
        p_id,
        remote,
        parent_remote,
        seed,
        logger=None,
        name=None,
        return_info=True,
        _obs_to_vect=True,
    ):
        Process.__init__(self, group=None, target=None, name=name)

        if logger is None:
            import logging

            self.logger = logging.getLogger(__name__)
            self.logger.disabled = True
        else:
            self.logger = logger.getChild(f"grid2op_RemoteEnv_{p_id}")

        self.p_id = p_id
        self.backend = None
        self.env = None
        self.env_params = env_params
        self.remote = remote
        self.parent_remote = parent_remote
        self.seed_used = seed
        self.space_prng = None
        self.fast_forward = 0
        self.all_seeds = []

        # internal do not modify  # Do not work (in the sens that is it less efficient)
        self.return_info = return_info
        self._obs_to_vect = _obs_to_vect
        self._comp_time = 0.0

    def init_env(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Initialize the environment  that will perform all the computation of this process.
        Remember the environment only lives in this process. It cannot
        be transfer to / from the main process.

        This function also makes sure the chronics are read in different order accross all processes. This is done
        by calling the :func:`grid2op.Chronics.GridValue.shuffle` method. An example of how to use this function
        is provided in :func:`grid2op.Chronics.Multifolder.shuffle`.

        """
        self.space_prng = np.random.RandomState()
        self.space_prng.seed(seed=self.seed_used)
        self.backend = self.env_params["_raw_backend_class"]()
        with warnings.catch_warnings():
            # warnings have bee already sent in the main process, no need to resend them
            warnings.filterwarnings("ignore")
            if "logger" in self.env_params:
                # disable the logger of the environment, to force the use of this one
                del self.env_params["logger"]
            self.env = Environment(
                **self.env_params, backend=self.backend, logger=self.logger
            )

        env_seed = self.space_prng.randint(np.iinfo(dt_int).max)
        self.all_seeds = self.env.seed(env_seed)
        self.env.chronics_handler.shuffle(
            shuffler=lambda x: x[
                self.space_prng.choice(len(x), size=len(x), replace=False)
            ]
        )

    def _clean_observation(self, obs):
        obs._forecasted_grid = []
        obs._forecasted_inj = []
        obs._obs_env = None
        obs.action_helper = None
        return obs

    def get_obs_ifnotconv(self):
        # warnings.warn(f"get_obs_ifnotconv is used")
        # TODO dirty hack because of wrong chronics
        # need to check!!!
        conv = False
        obs_v = None
        obs = None
        while not conv:
            try:
                self.env.reset()
                if self.fast_forward > 0:
                    self.env.fast_forward_chronics(
                        self.space_prng.randint(0, self.fast_forward)
                    )
                obs = self.env.get_obs()
                obs_v = obs.to_vect()
                if np.all(np.isfinite(obs_v)):
                    # i make sure that everything is not Nan
                    # other i consider it's "divergence" so "game over"
                    conv = True
            except Exception as exc_:
                pass
        if self._obs_to_vect:
            res = obs_v
        else:
            res = obs
        return res

    def run(self):
        if self.env is None:
            self.init_env()

        while True:
            cmd, data = self.remote.recv()
            if cmd == "get_spaces":
                self.remote.send((self.env.observation_space, self.env.action_space))
            elif cmd == "s":
                # perform a step
                beg_ = time.perf_counter()
                if data is None:
                    data = self.env.action_space()
                else:
                    data = self.env.action_space.from_vect(data)
                obs, reward, done, info = self.env.step(data)
                obs_v = obs.to_vect()
                if done or (~np.isfinite(obs_v)).any():
                    # if done do a reset
                    res_obs = self.get_obs_ifnotconv()
                elif self._obs_to_vect:
                    res_obs = obs.to_vect()
                else:
                    res_obs = self._clean_observation(obs)

                if not self.return_info:
                    info = None
                end_ = time.perf_counter()
                self._comp_time += end_ - beg_
                self.remote.send((res_obs, reward, done, info))
            elif cmd == "r":
                # perfom a reset
                obs_v = self.get_obs_ifnotconv()
                self.remote.send(obs_v)
            elif cmd == "c":
                # close everything
                self.env.close()
                self.remote.close()
                break
            elif cmd == "z":
                # adapt the chunk size
                self.env.set_chunk_size(data)
            elif cmd == "o":
                # get_obs
                tmp = self.env.get_obs()
                if self._obs_to_vect:
                    res_obs = tmp.to_vect()
                else:
                    res_obs = self._clean_observation(tmp)
                self.remote.send(res_obs)
            elif cmd == "f":
                # fast forward the chronics when restart
                self.fast_forward = int(data)
            elif cmd == "seed":
                self.remote.send((self.seed_used, self.all_seeds))
            elif cmd == "params":
                self.remote.send(self.env.parameters)
            elif cmd == "comp_time":
                self.remote.send(self._comp_time)
            elif cmd == "powerflow_time":
                self.remote.send(self.env.backend.comp_time)
            elif cmd == "step_time":
                self.remote.send(self.env._time_step)
            elif cmd == "set_filter":
                self.env.chronics_handler.set_filter(data)
                self.remote.send(None)
            elif cmd == "set_id":
                self.env.set_id(data)
                self.remote.send(None)
            elif cmd == "sim":
                action = self.env.action_space.from_vect(data)
                obs = self.env.get_obs()
                sim_obs, sim_reward, sim_done, sim_info = obs.simulate(action)
                sim_obs_v = sim_obs.to_vect()
                self.remote.send((sim_obs_v, sim_reward, sim_done, sim_info))
            elif hasattr(self.env, cmd):
                tmp = getattr(self.env, cmd)
                self.remote.send(tmp)
            else:
                raise NotImplementedError


class BaseMultiProcessEnvironment(GridObjects):
    """
    This class allows to evaluate a single agent instance on multiple environments running in parrallel.

    It uses the python "multiprocessing" framework to work, and thus is suitable only on a single machine with multiple
    cores (cpu / thread). We do not recommend to use this method on a cluster of different machines.

    This class uses the following representation:

    - an :class:`grid2op.BaseAgent.BaseAgent`: lives in a main process
    - different environments lives into different processes
    - a call to :func:`MultiEnv.step` will perform one step per environment, in parallel using a ``Pipe`` to transfer data
      to and from the main process from each individual environment process. It is a synchronous function. It means
      it will wait for every environment to finish the step before returning all the information.

    There are some limitations. For example, even if forecast are available, it's not possible to use forecast of the
    observations. This imply that :func:`grid2op.Observation.BaseObservation.simulate` is not available when using
    :class:`MultiEnvironment`

    Compare to regular Environments, :class:`MultiEnvironment` simply stack everything. You need to send not a single
    :class:`grid2op.Action.BaseAction` but as many actions as there are underlying environments. You receive not one single
    :class:`grid2op.Observation.BaseObservation` but as many observations as the number of underlying environments.

    A broader support of regular grid2op environment capabilities as well as support for
    :func:`grid2op.Observation.BaseObservation.simulate` call might be added in the future.

    **NB** As opposed to :func:`Environment.step` a call to :func:`BaseMultiProcessEnvironment.step` or any of
    its derived class (:class:`SingleEnvMultiProcess` or :class:`MultiEnvMultiProcess`) if a sub environment
    is "done" then it is automatically reset. This means entails that you can call
    :func:`BaseMultiProcessEnvironment.step` without worrying about having to reset.

    Attributes
    -----------
    envs: `list::grid2op.Environment.Environment`
        Al list of environments for which the evaluation will be made in parallel.

    nb_env: ``int``
        Number of parallel underlying environment that will be handled. It is also the size of the list of actions
        that need to be provided in :func:`MultiEnvironment.step` and the return sizes of the list of this
        same function.

    obs_as_class: ``bool``
        Whether to convert the observations back to :class:`grid2op.Observation` object to to leave them as
        numpy array. Default (`obs_as_class=True`) to send them as observation object, but it's slower.

    return_info: ``bool``
        Whether to return the information dictionary or not (might speed up computation)

    """

    def __init__(self, envs, obs_as_class=True, return_info=True, logger=None):
        GridObjects.__init__(self)
        self.__closed = False
        for env in envs:
            if not isinstance(env, Environment):
                raise MultiEnvException(
                    'You provided environment of type "{}" which is not supported.'
                    "Please only provide a grid2op.Environment.Environment class."
                    "".format(type(env))
                )

        self.nb_env = len(envs)
        max_int = np.iinfo(dt_int).max
        _remotes, _work_remotes = zip(*[Pipe() for _ in range(self.nb_env)])

        env_params = [sub_env.get_kwargs(with_backend=False) for sub_env in envs]
        self._ps = [
            RemoteEnv(
                env_params=env_,
                p_id=i,
                remote=work_remote,
                parent_remote=remote,
                name="{}_{}".format(envs[i].name, i),
                return_info=return_info,
                seed=envs[i].space_prng.randint(max_int),
                logger=logger.getChild("BaseMultiProcessEnvironment")
                if logger is not None
                else None,
            )
            for i, (work_remote, remote, env_) in enumerate(
                zip(_work_remotes, _remotes, env_params)
            )
        ]

        # on windows, this has to be created after
        self.envs = envs
        self._remotes = _remotes
        self._work_remotes = _work_remotes

        for p in self._ps:
            p.daemon = (
                True  # if the main process crashes, we should not cause things to hang
            )
            p.start()
        for remote in self._work_remotes:
            remote.close()
        self.obs_as_class = obs_as_class
        # self.__return_info = return_info
        self._waiting = True

    def _send_act(self, actions):
        for remote, action in zip(self._remotes, actions):
            vect = action.to_vect()
            # vect = None  # TODO
            remote.send(("s", vect))
        self._waiting = True

    def _wait_for_obs(self):
        results = [remote.recv() for remote in self._remotes]
        self._waiting = False
        obs, rews, dones, infos = zip(*results)
        if self.obs_as_class:
            obs = [
                self.envs[e].observation_space.from_vect(ob) for e, ob in enumerate(obs)
            ]
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def copy(self):
        raise NotImplementedError(
            "It is not possible to copy multiprocessing environments at the moment."
        )

    def step(self, actions):
        """
        Perform a step in all the underlying environments.
        If one or more of the underlying environments encounters a game over, it is automatically restarted.

        The observation sent back to the user is the observation after the :func:`grid2op.Environment.Environment.reset`
        has been called.

        As opposed to :class:`Environment.step` a call to this function will automatically reset
        any of the underlying environments in case one of them is "done". This is performed the following way.
        In the case one underlying environment is over (due to game over or due to end of the chronics), then:

        - the corresponding "done" is returned as ``True``
        - the corresponding observation returned is not the observation of the last time step (corresponding to the
          underlying environment that is game over) but is the first observation after reset.

        At the next call to step, the flag done will be (if not game over arise) set to ``False`` and the
        corresponding observation is the next observation of this underlying environment: every thing works
        as usual in this case.

        We did that because restarting the game over environment added un necessary complexity.

        Parameters
        ----------
        actions: ``list``
            List of :attr:`MultiEnvironment.nb_env` :class:`grid2op.Action.BaseAction`. Each action will be executed
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

        infos: ``list``
            List of dictionaries corresponding

        Examples
        ---------
        You can use this class as followed:

        .. code-block:: python

            import grid2op
            from grid2op.Environment import BaseMultiProcessEnv
            env1 = grid2op.make()  # create an environment of your choosing
            env2 = grid2op.make()  # create another environment of your choosing

            multi_env = BaseMultiProcessEnv([env1, env2])
            obss = multi_env.reset()
            obs1, obs2 = obss  # here i extract the observation of the first environment and of the second one
            # note that you cannot do obs1.simulate().
            # this is equivalent to a call to
            # obs1 = env1.reset(); obs2 = env2.reset()

            # then you can do regular steps
            action_env1 = env1.action_space()
            action_env2 = env2.action_space()
            obss, rewards, dones, infos = env.step([action_env1, action_env2])
            # if you define
            # obs1, obs2 = obss
            # r1, r2 = rewards
            # done1, done2 = dones
            # info1, info2 = infos
            # in this case, it is equivalent to calling
            # obs1, r1, done1, info1 = env1.step(action_env1)
            # obs2, r2, done2, info2 = env2.step(action_env2)

        Let us now focus on the "automatic" reset part.

        .. code-block:: python

            # see above for the creation of a multi_env and the proper imports
            multi_env = BaseMultiProcessEnv([env1, env2])
            action_env1 = env1.action_space()
            action_env2 = env2.action_space()
            obss, rewards, dones, infos = env.step([action_env1, action_env2])

            # say dones[0] is ``True``
            # in this case if you define
            # obs1 = obss[0]
            # r1=rewards[0]
            # done1=done[0]
            # info1=info[0]
            # in that case it is equivalent to the "single processed" code
            # obs1_tmp, r1_tmp, done1_tmp, info1_tmp = env1.step(action_env1)
            # done1 = done1_tmp
            # r1 = r1_tmp
            # info1 = info1_tmp
            # obs1_aux = env1.reset()
            # obs1 = obs1_aux
            # CAREFULLL in this case, obs1 is NOT obs1_tmp but is really

        """
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        if len(actions) != self.nb_env:
            raise MultiEnvException(
                "Incorrect number of actions provided. You provided {} actions, but the "
                "MultiEnvironment counts {} different environment."
                "".format(len(actions), self.nb_env)
            )
        for act in actions:
            if not isinstance(act, BaseAction):
                raise MultiEnvException(
                    'All actions send to MultiEnvironment.step should be of type "grid2op.BaseAction"'
                    "and not {}".format(type(act))
                )

        self._send_act(actions)
        obs, rews, dones, infos = self._wait_for_obs()
        return obs, rews, dones, infos

    def reset(self):
        """
        Reset all the environments, and return all the associated observation.

        **NB** Except in some specific occasion, there is no need to call this function reset. Indeed, when
        a sub environment is "done" then it is automatically restarted in the
        :func:BaseMultiEnvMultiProcess.step` function.

        Returns
        -------
        res: ``list``
            The list of all observations. This list counts :attr:`MultiEnvironment.nb_env` elements, each one being
            an :class:`grid2op.Observation.BaseObservation`.

        """
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        for remote in self._remotes:
            remote.send(("r", None))
        res = [remote.recv() for e, remote in enumerate(self._remotes)]
        if self.obs_as_class:
            res = [
                self.envs[e].observation_space.from_vect(el) for e, el in enumerate(res)
            ]
        return np.stack(res)

    def close(self):
        """
        Close all the environments and all the processes.
        """
        if self.__closed:
            return

        for remote in self._remotes:
            remote.send(("c", None))
        self.__closed = True

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
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        try:
            new_chunk_size = int(new_chunk_size)
        except Exception as e:
            raise Grid2OpException(
                "Impossible to set the chunk size. It should be convertible a integer, and not"
                "{}".format(new_chunk_size)
            )

        if new_chunk_size <= 0:
            raise Grid2OpException(
                'Impossible to read less than 1 data at a time. Please make sure "new_chunk_size"'
                "is a positive integer."
            )

        for remote in self._remotes:
            remote.send(("z", new_chunk_size))

    def set_ff(self, ff_max=7 * 24 * 60 / 5):
        """
        This method is primarily used for training.

        The problem this method aims at solving is the following: most of grid2op environments starts a Monday at
        00:00. This method will "fast forward" an environment for a random number of timestep between 0 and ``ff_max``
        """
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        try:
            ff_max = int(ff_max)
        except Exception as exc_:
            raise RuntimeError(
                "ff_max parameters should be convertible to an integer."
            ) from exc_

        for remote in self._remotes:
            remote.send(("f", ff_max))

    def get_seeds(self):
        """
        Get the seeds used to initialize each sub environments.
        """
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        for remote in self._remotes:
            remote.send(("seed", None))
        res = [remote.recv() for remote in self._remotes]
        return res

    def get_parameters(self):
        """
        Get the parameters of each sub environments
        """
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        for remote in self._remotes:
            remote.send(("params", None))
        res = [remote.recv() for remote in self._remotes]
        return res

    def get_obs(self):
        """implement the get_obs function that is "broken" if you use the __getattr__"""
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        for remote in self._remotes:
            remote.send(("o", None))
        res = [
            self.envs[e].observation_space.from_vect(remote.recv())
            for e, remote in enumerate(self._remotes)
        ]
        return res

    def _send_sim(self, actions):
        for remote, action in zip(self._remotes, actions):
            remote.send(("sim", action.to_vect()))
        self._waiting = True

    def simulate(self, actions):
        """
        Perform the equivalent of `obs.simulate` in all the underlying environment

        Parameters
        ----------
        actions: ``list``
            List of all action to simulate

        Returns
        ---------
        sim_obs:
            The observation resulting from the simulation
        sim_rews:
            The reward resulting from the simulation
        sim_dones:
            For each simulation, whether or not this the simulated action lead to a game over
        sim_infos:
            Additional information for each simulated actions.

        Examples
        --------

        You can use this feature like:

        .. code-block:: python

            import grid2op
            from grid2op.Environment import BaseMultiProcessEnvironment

            env_name = "l2rpn_case14_sandbox"  # or any other name
            env1 = grid2op.make(env_name)
            env2 = grid2op.make(env_name)

            multi_env = BaseMultiProcessEnvironment([env1, env2])
            obss = multi_env.reset()

            # simulate
            actions = [env1.action_space(), env2.action_space()]
            sim_obss, sim_rs, sim_ds, sim_is = multi_env.simulate(actions)

        """
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        if len(actions) != self.nb_env:
            raise MultiEnvException(
                "Incorrect number of actions provided. You provided {} actions, but the "
                "MultiEnvironment counts {} different environment."
                "".format(len(actions), self.nb_env)
            )
        for act in actions:
            if not isinstance(act, BaseAction):
                raise MultiEnvException(
                    "All actions send to MultiEnvironment.step should be of type "
                    '"grid2op.BaseAction" and not {}'.format(type(act))
                )

        self._send_sim(actions)
        sim_obs, sim_rews, sim_dones, sim_infos = self._wait_for_obs()
        return sim_obs, sim_rews, sim_dones, sim_infos

    def __getattr__(self, name):
        """
        This function is used to get the attribute of the underlying sub environments.

        Note that setting attributes or information to the sub_env this way will not work. This method only allows
        to get the value of some attributes, NOT to modify them.

        /!\ **DANGER** /!\ is you use this function, you are entering the danger zone. This might not work and
        make your all python session dies without any notice. You've been warned.

        Parameters
        ----------
        name: ``str``
            Name of the attribute you want to get the value, for each sub_env

        Returns
        -------
        res: ``list``
            The value of the given attribute for each sub env. Again, use with care.

        """
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        res = True
        for sub_env in self.envs:
            if not hasattr(sub_env, name):
                res = False
        if not res:
            raise RuntimeError(
                'At least one of the sub_env has not the attribute "{}". This will not be '
                "executed.".format(name)
            )
        for remote in self._remotes:
            remote.send((name, None))
        res = [remote.recv() for remote in self._remotes]
        return res

    def get_comp_time(self):
        """
        Get the computation time (only of the step part, corresponds to sub_env.comp_time) of each sub environments
        """
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        for remote in self._remotes:
            remote.send(("comp_time", None))
        res = [remote.recv() for remote in self._remotes]
        return res

    def get_powerflow_time(self):
        """
        Get the computation time (corresponding to sub_env.backend.comp_time) of each sub environments
        """
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        for remote in self._remotes:
            remote.send(("powerflow_time", None))
        res = [remote.recv() for remote in self._remotes]
        return res

    def get_step_time(self):
        """
        Get the computation time (corresponding to sub_env._time_step) of each sub environments
        """
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        for remote in self._remotes:
            remote.send(("step_time", None))
        res = [remote.recv() for remote in self._remotes]
        return res

    def set_filter(self, filter_funs):
        """
        Set a `filter_fun` for each of the underlying environment.

        See :func:`grid2op.Chronis.MultiFolder.set_filter` for more information

        Examples
        --------
        TODO usage example
        """
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        if callable(filter_funs):
            filter_funs = [filter_funs for _ in range(self.nb_env)]
        if len(filter_funs) != self.nb_env:
            raise RuntimeError(
                "filter_funs should be either a single function that will be applied "
                "identically to each sub_env or a list of callable functions."
            )
        for el in filter_funs:
            if not callable(el):
                raise RuntimeError(
                    "filter_funs should be composed of callable elements, such as functions "
                    "that can be use with `env.chronics_handler.set_filter`"
                )
        for sub_env_id, remote in enumerate(self._remotes):
            remote.send(("set_filter", filter_funs[sub_env_id]))
        res = [remote.recv() for remote in self._remotes]
        return res

    def set_id(self, id_):
        """
        Set a chronics id for each of the underlying environment to be used for each of the sub_env.

        See :func:`grid2op.Environment.Environment.set_id` for more information

        Examples
        --------
        TODO usage example
        """
        if self.__closed:
            raise EnvError("This environment is closed, you cannot use it.")
        if isinstance(id_, int):
            id_ = [id_ for _ in range(self.nb_env)]
        if len(id_) != self.nb_env:
            raise RuntimeError(
                "id_ should be either a single integer or an integer that represents "
                "the chronics to use."
            )
        for el in id_:
            if not isinstance(el, int):
                raise RuntimeError("id_ should be composed of integers.")
        for sub_env_id, remote in enumerate(self._remotes):
            remote.send(("set_id", id_[sub_env_id]))
        res = [remote.recv() for remote in self._remotes]
        return res

    def __del__(self):
        """when the environment is garbage collected, free all the memory, including cross reference to itself in the observation space."""
        if not self.__closed:
            self.close()


if __name__ == "__main__":
    from tqdm import tqdm
    from grid2op import make
    from grid2op.Agent import DoNothingAgent

    nb_env = 8  # change that to adapt to your system
    NB_STEP = 100  # number of step for each environment

    env = make()
    env.seed(42)
    envs = [env for _ in range(nb_env)]

    agent = DoNothingAgent(env.action_space)
    multi_envs = BaseMultiProcessEnvironment(envs)

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
