"""
This module is here to facilitate the evaluation of agent.
It can handles all types of :class:`grid2op.Agent`.

"""
import time
import warnings

import numpy as np
import copy

import os
import sys

from multiprocessing import Pool

import json

import pdb

try:
    from .Action import HelperAction, Action, TopologyAction
    from .Exceptions import *
    from .Observation import CompleteObservation, ObservationHelper, Observation
    from .Reward import FlatReward, RewardHelper, Reward
    from .GameRules import GameRules, AllwaysLegal, LegalAction
    from .Environment import Environment
    from .ChronicsHandler import ChronicsHandler, GridStateFromFile, GridValue
    from .Backend import Backend
    from .BackendPandaPower import PandaPowerBackend
    from .Parameters import Parameters
    from .Agent import DoNothingAgent, Agent
    from .EpisodeData import EpisodeData
    from ._utils import _FakePbar
    from .VoltageControler import ControlVoltageFromFile

except (ModuleNotFoundError, ImportError):
    from Action import HelperAction, Action, TopologyAction
    from Exceptions import *
    from Observation import CompleteObservation, ObservationHelper, Observation
    from Reward import FlatReward, RewardHelper, Reward
    from GameRules import GameRules, AllwaysLegal, LegalAction
    from Environment import Environment
    from ChronicsHandler import ChronicsHandler, GridStateFromFile, GridValue
    from Backend import Backend
    from BackendPandaPower import PandaPowerBackend
    from Parameters import Parameters
    from Agent import DoNothingAgent, Agent
    from EpisodeData import EpisodeData
    from _utils import _FakePbar
    from VoltageControler import ControlVoltageFromFile


# TODO have a vectorized implementation of everything in case the agent is able to act on multiple environment
# at the same time. This might require a lot of work, but would be totally worth it! (especially for Neural Net based agents)

# TODO add a more suitable logging strategy

# TODO use gym logger if specified by the user.


class DoNothingLog:
    """
    A class to emulate the behaviour of a logger, but that does absolutely nothing.
    """
    INFO = 2
    WARNING = 1
    ERROR = 0

    def __init__(self, max_level=2):
        self.max_level = max_level

    def warn(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


class ConsoleLog(DoNothingLog):
    """
    A class to emulate the behaviour of a logger, but that prints on the console
    """

    def __init__(self, max_level=2):
        DoNothingLog.__init__(self, max_level)

    def warn(self, *args, **kwargs):
        if self.max_level >= self.WARNING:
            if args:
                print("WARNING: \"{}\"".format(", ".join(args)))
            if kwargs:
                print("WARNING: {}".format(kwargs))

    def info(self, *args, **kwargs):
        if self.max_level >= self.INFO:
            if args:
                print("INFO: \"{}\"".format(", ".join(args)))
            if kwargs:
                print("INFO: {}".format(kwargs))

    def error(self, *args, **kwargs):
        if self.max_level >= self.ERROR:
            if args:
                print("ERROR: \"{}\"".format(", ".join(args)))
            if kwargs:
                print("ERROR: {}".format(kwargs))

    def warning(self, *args, **kwargs):
        if self.max_level >= self.WARNING:
            if args:
                print("WARNING: \"{}\"".format(", ".join(args)))
            if kwargs:
                print("WARNING: {}".format(kwargs))

#TODO i think runner.env are not close, like, never closed :eyes:


class Runner(object):
    """
    A runner is a utilitary tool that allows to create environment, and run simulations more easily.
    This specific class as for main purpose to evaluate the performance of a trained :class:`grid2op.Agent`, rather
    than to train it. Of course, it is possible to adapt it for a specific training mechanisms. Examples of such
    will be made available in the future.

    Attributes
    ----------
    envClass: ``type``
        The type of the environment used for the game. The class should be given, and **not** an instance (object) of
        this class. The default is the :class:`grid2op.Environment`. If modified, it should derived from this class.

    actionClass: ``type``
        The type of action that can be performed by the agent / bot / controler. The class should be given, and
        **not** an instance of this class. This type
        should derived from :class:`grid2op.Action`. The default is :class:`grid2op.TopologyAction`.

    observationClass: ``type``
        This type represents the class that will be used to build the :class:`grid2op.Observation` visible by the
        :class:`grid2op.Agent`. As :attr:`Runner.actionClass`, this should be a type, and **not** and instance (object)
        of this type. This type should derived from :class:`grid2op.Observation`. The default is
        :class:`grid2op.CompleteObservation`.

    rewardClass: ``type``
        Representes the type used to build the rewards that are given to the :class:`Agent`. As
        :attr:`Runner.actionClass`, this should be a type, and **not** and instance (object) of this type.
        This type should derived from :class:`grid2op.Reward`. The default is :class:`grid2op.ConstantReward` that
        **should not** be used to train or evaluate an agent, but rather as debugging purpose.

    gridStateclass: ``type``
        This types control the mechanisms to read chronics and assign data to the powergrid. Like every "\.*Class"
        attributes the type should be pass and not an intance (object) of this type. Its default is
        :class:`grid2op.GridStateFromFile` and it must be a subclass of :class:`grid2op.GridValue`.

    legalActClass: ``type``
        This types control the mechanisms to assess if an :class:`grid2op.Action` is legal.
        Like every "\.*Class" attributes the type should be pass and not an intance (object) of this type.
        Its default is :class:`grid2op.AllwaysLegal` and it must be a subclass of :class:`grid2op.LegalAction`.

    backendClass: ``type``
        This types control the backend, *eg.* the software that computes the powerflows.
        Like every "\.*Class" attributes the type should be pass and not an intance (object) of this type.
        Its default is :class:`grid2op.PandaPowerBackend` and it must be a subclass of :class:`grid2op.Backend`.

    agentClass: ``type``
        This types control the type of Agent, *eg.* the bot / controler that will take :class:`grid2op.Action` and
        avoid cascading failures.
        Like every "\.*Class" attributes the type should be pass and not an intance (object) of this type.
        Its default is :class:`grid2op.DoNothingAgent` and it must be a subclass of :class:`grid2op.Agent`.

    logger:
        A object than can be used to log information, either in a text file, or by printing them to the command prompt.

    init_grid_path: ``str``
        This attributes store the path where the powergrid data are located. If a relative path is given, it will be
        extended as an absolute path.

    names_chronics_to_backend: ``dict``
        See description of :func:`grid2op.ChronicsHelper.initialize` for more information about this dictionnary

    parameters_path: ``str``, optional
        Where to look for the :class:`grid2op.Environment` :class:`grid2op.Parameters`. It defaults to ``None`` which
        corresponds to using default values.

    parameters: :class:`grid2op.Parameters`
        Type of _parameters used. This is an instance (object) of type :class:`grid2op.Parameters` initialized from
        :attr:`Runner.parameters_path`

    path_chron: ``str``
        Path indicatng where to look for temporal data.

    chronics_handler: :class:`grid2op.ChronicsHandler`
        Initialized from :attr:`Runner.gridStateclass` and :attr:`Runner.path_chron` it represents the input data used
        to generate grid state by the :attr:`Runner.env`

    backend: :class:`grid2op.Backend`
        Used to compute the powerflow. This object has the type given by :attr:`Runner.backendClass`

    env: :class:`grid2op.Environment`
        Represents the environment which the agent / bot / control must control through action. It is initialized from
        the :attr:`Runner.envClass`

    agent: :class:`grid2op.Agent`
        Represents the agent / bot / controler that takes action performed on a environment (the powergrid) to maximize
        a certain reward.

    verbose: ``bool``
        If ``True`` then detailed output of each steps are written.

    gridStateclass_kwargs: ``dict``
        Additional keyword arguments used to build the :attr:`Runner.chronics_handler`

    thermal_limit_a: ``numpy.ndarray``
        The thermal limit for the environment (if any).
    """

    def __init__(self,
                 # full path where grid state is located, eg "./data/test_Pandapower/case14.json"
                 init_grid_path: str,
                 path_chron,  # path where chronics of injections are stored
                 parameters_path=None,
                 names_chronics_to_backend=None,
                 actionClass=TopologyAction,
                 observationClass=CompleteObservation,
                 rewardClass=FlatReward,
                 legalActClass=AllwaysLegal,
                 envClass=Environment,
                 gridStateclass=GridStateFromFile,
                 # type of chronics to use. For example GridStateFromFile if forecasts are not used, or GridStateFromFileWithForecasts otherwise
                 backendClass=PandaPowerBackend,
                 agentClass=DoNothingAgent,  # class used to build the agent
                 agentInstance=None,
                 verbose=False,
                 gridStateclass_kwargs={},
                 voltageControlerClass=ControlVoltageFromFile,
                 thermal_limit_a=None,
                 max_iter=-1
                 ):
        """
        Initialize the Runner.

        Parameters
        ----------
        init_grid_path: ``str``
            Madantory, used to initialize :attr:`Runner.init_grid_path`.

        path_chron: ``str``
            Madantory where to look for chronics data, used to initialize :attr:`Runner.path_chron`.

        parameters_path: ``str`` or ``dict``, optional
            Used to initialize :attr:`Runner.parameters_path`. If it's a string, this will suppose parameters are
            located at this path, if it's a dictionary, this will use the parameters converted from this dictionary.

        names_chronics_to_backend: ``dict``, optional
            Used to initialize :attr:`Runner.names_chronics_to_backend`.

        actionClass: ``type``, optional
            Used to initialize :attr:`Runner.actionClass`.

        observationClass: ``type``, optional
            Used to initialize :attr:`Runner.observationClass`.

        rewardClass: ``type``, optional
            Used to initialize :attr:`Runner.rewardClass`. Default to :class:`grid2op.ConstantReward` that
            *should not** be used to train or evaluate an agent, but rather as debugging purpose.

        legalActClass: ``type``, optional
            Used to initialize :attr:`Runner.legalActClass`.

        envClass: ``type``, optional
            Used to initialize :attr:`Runner.envClass`.

        gridStateclass: ``type``, optional
            Used to initialize :attr:`Runner.gridStateclass`.

        backendClass: ``type``, optional
            Used to initialize :attr:`Runner.backendClass`.

        agentClass: ``type``, optional
            Used to initialize :attr:`Runner.agentClass`.

        agentInstance: :class:`grid2op.Agent.Agent`
            Used to initialize the agent. Note that either :attr:`agentClass` or :attr:`agentInstance` is used
            at the same time. If both ot them are ``None`` or both of them are "not ``None``" it throw an error.

        verbose: ``bool``, optional
            Used to initialize :attr:`Runner.verbose`.

        thermal_limit_a: ``numpy.ndarray``
            The thermal limit for the environment (if any).

        voltagecontrolerClass: :class:`grid2op.VoltageControler.ControlVoltageFromFile`, optional
            The controler that will change the voltage setpoints of the generators.

        """

        if not isinstance(envClass, type):
            raise Grid2OpException(
                "Parameter \"envClass\" used to build the Runner should be a type (a class) and not an object "
                "(an instance of a class). It is currently \"{}\"".format(
                    type(envClass)))
        if not issubclass(envClass, Environment):
            raise RuntimeError("Impossible to create a runner without an evnrionment derived from grid2op.Environement"
                               " class. Please modify \"envClass\" parameter.")
        self.envClass = envClass

        if not isinstance(actionClass, type):
            raise Grid2OpException(
                "Parameter \"actionClass\" used to build the Runner should be a type (a class) and not an object "
                "(an instance of a class). It is currently \"{}\"".format(
                    type(actionClass)))
        if not issubclass(actionClass, Action):
            raise RuntimeError("Impossible to create a runner without an action class derived from grid2op.Action. "
                               "Please modify \"actionClass\" parameter.")
        self.actionClass = actionClass

        if not isinstance(observationClass, type):
            raise Grid2OpException(
                "Parameter \"observationClass\" used to build the Runner should be a type (a class) and not an object "
                "(an instance of a class). It is currently \"{}\"".format(
                    type(observationClass)))
        if not issubclass(observationClass, Observation):
            raise RuntimeError("Impossible to create a runner without an observation class derived from "
                               "grid2op.Observation. Please modify \"observationClass\" parameter.")
        self.observationClass = observationClass

        if not isinstance(rewardClass, type):
            raise Grid2OpException(
                "Parameter \"rewardClass\" used to build the Runner should be a type (a class) and not an object "
                "(an instance of a class). It is currently \"{}\"".format(
                    type(rewardClass)))
    
        if not issubclass(rewardClass, Reward):
            raise RuntimeError("Impossible to create a runner without an observation class derived from "
                               "grid2op.Reward. Please modify \"rewardClass\" parameter.")
        self.rewardClass = rewardClass

        if not isinstance(gridStateclass, type):
            raise Grid2OpException(
                "Parameter \"gridStateclass\" used to build the Runner should be a type (a class) and not an object "
                "(an instance of a class). It is currently \"{}\"".format(
                    type(gridStateclass)))
        if not issubclass(gridStateclass, GridValue):
            raise RuntimeError("Impossible to create a runner without an chronics class derived from "
                               "grid2op.GridValue. Please modify \"gridStateclass\" parameter.")
        self.gridStateclass = gridStateclass

        if not isinstance(legalActClass, type):
            raise Grid2OpException(
                "Parameter \"legalActClass\" used to build the Runner should be a type (a class) and not an object "
                "(an instance of a class). It is currently \"{}\"".format(
                    type(legalActClass)))
        if not issubclass(legalActClass, LegalAction):

            raise RuntimeError("Impossible to create a runner without a class defining legal actions derived "
                               "from grid2op.LegalAction. Please modify \"legalActClass\" parameter.")
        self.legalActClass = legalActClass

        if not isinstance(backendClass, type):
            raise Grid2OpException(
                "Parameter \"legalActClass\" used to build the Runner should be a type (a class) and not an object "
                "(an instance of a class). It is currently \"{}\"".format(
                    type(backendClass)))
        if not issubclass(backendClass, Backend):
            raise RuntimeError("Impossible to create a runner without a backend class derived from grid2op.GridValue. "
                               "Please modify \"backendClass\" parameter.")
        self.backendClass = backendClass

        if agentClass is not None:
            if agentInstance is not None:
                raise RuntimeError("Impossible to build the backend. Only one of AgentClass or agentInstance can be "
                                   "used (both are not None).")
            if not isinstance(agentClass, type):
                raise Grid2OpException(
                    "Parameter \"agentClass\" used to build the Runner should be a type (a class) and not an object "
                    "(an instance of a class). It is currently \"{}\"".format(
                        type(agentClass)))
            if not issubclass(agentClass, Agent):
                raise RuntimeError("Impossible to create a runner without an agent class derived from grid2op.Agent. "
                                   "Please modify \"agentClass\" parameter.")
            self.agentClass = agentClass
            self._useclass = True
            self.agent = None
        elif agentInstance is not None:
            if not isinstance(agentInstance, Agent):
                raise RuntimeError("Impossible to create a runner without an agent class derived from grid2op.Agent. "
                                   "Please modify \"agentInstance\" parameter.")
            self.agentClass = None
            self._useclass = False
            self.agent = agentInstance
        else:
            raise RuntimeError("Impossible to build the backend. Either AgentClass or agentInstance must be provided "
                               "and both are None.")

        self.logger = ConsoleLog(
            DoNothingLog.INFO if verbose else DoNothingLog.ERROR)

        # store _parameters
        self.init_grid_path = init_grid_path
        self.names_chronics_to_backend = names_chronics_to_backend

        # game _parameters
        if isinstance(parameters_path, str):
            self.parameters_path = parameters_path
            self.parameters = Parameters(parameters_path)
        elif isinstance(parameters_path, dict):
            self.parameters = Parameters()
            self.parameters.init_from_dict(parameters_path)
        elif parameters_path is None:
            self.parameters_path = parameters_path
            self.parameters = Parameters()
        else:
            raise RuntimeError("Impossible to build the parameters. The argument \"parameters_path\" should either"
                               "be a string or a dictionary.")

        # chronics of grid state
        self.path_chron = path_chron
        self.gridStateclass_kwargs = gridStateclass_kwargs
        self.max_iter = max_iter
        if max_iter > 0:
            self.gridStateclass["max_iter"] = max_iter
        self.chronics_handler = ChronicsHandler(chronicsClass=self.gridStateclass,
                                                path=self.path_chron,
                                                **self.gridStateclass_kwargs)

        # the backend, used to compute powerflows
        self.backend = self.backendClass()

        # build the environment
        self.env = None

        self.verbose = verbose

        self.thermal_limit_a = thermal_limit_a

        # controler for voltage
        if not issubclass(voltageControlerClass, ControlVoltageFromFile):
            raise Grid2OpException("Parameter \"voltagecontrolClass\" should derive from \"ControlVoltageFromFile\".")
        self.voltageControlerClass = voltageControlerClass

    def _new_env(self, chronics_handler, backend, parameters):
        res = self.envClass(init_grid_path=self.init_grid_path,
                            chronics_handler=chronics_handler,
                            backend=backend,
                            parameters=parameters,
                            names_chronics_to_backend=self.names_chronics_to_backend,
                            actionClass=self.actionClass,
                            observationClass=self.observationClass,
                            rewardClass=self.rewardClass,
                            legalActClass=self.legalActClass,
                            voltagecontrolerClass=self.voltageControlerClass)

        if self.thermal_limit_a is not None:
            res.set_thermal_limit(self.thermal_limit_a)

        if self._useclass:
            agent = self.agentClass(res.helper_action_player)
        else:
            agent = self.agent
        return res, agent

    def init_env(self):
        """
        Function used to initialized the environment and the agent.
        It is called by :func:`Runner.reset`.

        Returns
        -------
        ``None``

        """
        self.env, self.agent = self._new_env(self.chronics_handler, self.backend, self.parameters)

    def reset(self):
        """
        Used to reset an environment. This method is called at the beginning of each new episode.
        If the environment is not initialized, then it initializes it with :func:`Runner.make_env`.

        Returns
        -------
        ``None``

        """
        if self.env is None:
            self.init_env()
        else:
            self.env.reset()

    def run_one_episode(self, indx=0, path_save=None, pbar=False):
        """
        Function used to run one episode of the :attr:`Runner.agent` and see how it performs in the :attr:`Runner.env`.

        Parameters
        ----------
        indx: ``int``
            The number of episode previously run

        path_save: ``str``, optional
            Path where to save the data. See the description of :mod:`grid2op.Runner` for the structure of the saved
            file.

        Returns
        -------
        cum_reward: ``float``
            The cumulative reward obtained by the agent during this episode

        time_step: ``int``
            The number of timesteps that have been played before the end of the episode (because of a "game over" or
            because there were no more data)

        """
        self.reset()
        res = self._run_one_episode(self.env, self.agent, self.logger, indx, path_save, pbar=pbar)
        return res

    @staticmethod
    def _run_one_episode(env, agent, logger, indx, path_save=None, pbar=False):
        done = False
        time_step = int(0)
        dict_ = {}
        time_act = 0.
        cum_reward = 0.

        # reset the environment
        env.chronics_handler.tell_id(indx-1)
        # the "-1" above is because the environment will be reset. So it will increase id of 1.
        obs = env.reset()

        # compute the size and everything if it needs to be stored
        nb_timestep_max = env.chronics_handler.max_timestep()
        efficient_storing = nb_timestep_max > 0
        nb_timestep_max = max(nb_timestep_max, 0)

        if path_save is None:
            # i don't store anything on drive, so i don't need to store anything on memory
            nb_timestep_max = 0

        times = np.full(nb_timestep_max, fill_value=np.NaN, dtype=np.float)
        rewards = np.full(nb_timestep_max, fill_value=np.NaN, dtype=np.float)
        actions = np.full((nb_timestep_max, env.action_space.n),
                          fill_value=np.NaN, dtype=np.float)
        env_actions = np.full(
            (nb_timestep_max, env.helper_action_env.n), fill_value=np.NaN, dtype=np.float)
        observations = np.full(
            (nb_timestep_max+1, env.observation_space.n), fill_value=np.NaN, dtype=np.float)
        disc_lines = np.full(
            (nb_timestep_max, env.backend.n_line), fill_value=np.NaN, dtype=np.bool)
        disc_lines_templ = np.full(
            (1, env.backend.n_line), fill_value=False, dtype=np.bool)

        if path_save is not None:
            # store observation at timestep 0
            if efficient_storing:
                observations[time_step, :] = obs.to_vect()
            else:
                observations = np.concatenate((observations, obs.to_vect()))

        episode = EpisodeData(actions=actions, env_actions=env_actions,
                              observations=observations,
                              rewards=rewards, disc_lines=disc_lines, times=times,
                              observation_space=env.observation_space,
                              action_space=env.action_space,
                              helper_action_env=env.helper_action_env,
                              path_save=path_save, disc_lines_templ=disc_lines_templ,
                              logger=logger, name=env.chronics_handler.get_name())

        episode.set_parameters(env)

        beg_ = time.time()

        reward = env.reward_range[0]
        done = False

        next_pbar = [False]
        with Runner._make_progress_bar(pbar, nb_timestep_max, next_pbar) as pbar_:
            while not done:
                beg__ = time.time()
                act = agent.act(obs, reward, done)
                end__ = time.time()
                time_act += end__ - beg__

                obs, reward, done, info = env.step(act)  # should load the first time stamp
                cum_reward += reward
                time_step += 1
                pbar_.update(1)

                episode.incr_store(efficient_storing, time_step, end__ - beg__,
                                   reward, env.env_modification, act, obs, info)
            end_ = time.time()

        episode.set_meta(env, time_step, cum_reward)

        li_text = ["Env: {:.2f}s", "\t - apply act {:.2f}s", "\t - run pf: {:.2f}s",
                   "\t - env update + observation: {:.2f}s", "Agent: {:.2f}s", "Total time: {:.2f}s",
                   "Cumulative reward: {:1f}"]
        msg_ = "\n".join(li_text)
        logger.info(msg_.format(
            env._time_apply_act + env._time_powerflow + env._time_extract_obs,
            env._time_apply_act, env._time_powerflow, env._time_extract_obs,
            time_act, end_ - beg_, cum_reward))

        episode.set_episode_times(env, time_act, beg_, end_)

        episode.to_disk()

        name_chron = env.chronics_handler.get_name()

        return name_chron, cum_reward, int(time_step)

    @staticmethod
    def _make_progress_bar(pbar, total, next_pbar):
        """
        Parameters
        ----------
        pbar: ``bool`` or ``type`` or ``object``
            How to display the progress bar, understood as follow:

            - if pbar is ``None`` nothing is done.
            - if pbar is a boolean, tqdm pbar are used, if tqdm package is available and installed on the system
              [if ``true``]. If it's false it's equivalent to pbar being ``None``
            - if pbar is a ``type`` ( a class), it is used to build a progress bar at the highest level (episode) and
              and the lower levels (step during the episode). If it's a type it muyst accept the argument "total"
              and "desc" when being built, and the closing is ensured by this method.
            - if pbar is an object (an instance of a class) it is used to make a progress bar at this highest level
              (episode) but not at lower levels (setp during the episode)
        """
        pbar_ = _FakePbar()
        next_pbar[0] = False

        if isinstance(pbar, bool):
            if pbar:
                try:
                    from tqdm import tqdm
                    pbar_ = tqdm(total=total, desc="episode")
                    next_pbar[0] = True
                except (ImportError, ModuleNotFoundError):
                    pass
        elif isinstance(pbar, type):
            pbar_ = pbar(total=total, desc="episode")
            next_pbar[0] = pbar
        elif isinstance(pbar, object):
            pbar_ = pbar
        return pbar_

    def run_sequential(self, nb_episode, path_save=None, pbar=False):
        """
        This method is called to see how well an agent performed on a sequence of episode.

        Parameters
        ----------
        nb_episode: ``int``
            Number of episode to play.

        path_save: ``str``, optional
            If not None, it specifies where to store the data. See the description of this module :mod:`Runner` for
            more information

        pbar: ``bool`` or ``type`` or ``object``
            How to display the progress bar, understood as follow:

            - if pbar is ``None`` nothing is done.
            - if pbar is a boolean, tqdm pbar are used, if tqdm package is available and installed on the system
              [if ``true``]. If it's false it's equivalent to pbar being ``None``
            - if pbar is a ``type`` ( a class), it is used to build a progress bar at the highest level (episode) and
              and the lower levels (step during the episode). If it's a type it muyst accept the argument "total"
              and "desc" when being built, and the closing is ensured by this method.
            - if pbar is an object (an instance of a class) it is used to make a progress bar at this highest level
              (episode) but not at lower levels (setp during the episode)

        Returns
        -------
        res: ``list``
            List of tuple. Each tuple having 5 elements:

              - "id_chron" unique identifier of the episode
              - "name_chron" name of chronics
              - "cum_reward" the cumulative reward obtained by the :attr:`Runner.Agent` on this episode i
              - "nb_time_step": the number of time steps played in this episode.
              - "max_ts" : the maximum number of time steps of the chronics

        """
        res = [(None, None, None, None, None) for _ in range(nb_episode)]

        next_pbar = [False]
        with self._make_progress_bar(pbar, nb_episode, next_pbar) as pbar_:
            for i in range(nb_episode):
                name_chron, cum_reward, nb_time_step = self.run_one_episode(path_save=path_save, indx=i, pbar=next_pbar[0])
                id_chron = self.chronics_handler.get_id()
                max_ts = self.chronics_handler.max_timestep()
                res[i] = (id_chron, name_chron, cum_reward, nb_time_step, max_ts)
                pbar_.update(1)
        return res

    @staticmethod
    def _one_process_parrallel(runner, episode_this_process, process_id, path_save=None):
        chronics_handler = ChronicsHandler(chronicsClass=runner.gridStateclass,
                                           path=runner.path_chron,
                                           **runner.gridStateclass_kwargs)
        parameters = copy.deepcopy(runner.parameters)
        backend = runner.backendClass()
        nb_episode_this_process = len(episode_this_process)
        res = [(None, None, None) for _ in range(nb_episode_this_process)]
        for i, p_id in enumerate(episode_this_process):
            env, agent = runner._new_env(chronics_handler=chronics_handler,
                                         backend=backend,
                                         parameters=parameters)
            name_chron, cum_reward, nb_time_step = Runner._run_one_episode(
                env, agent, runner.logger, p_id, path_save)
            id_chron = chronics_handler.get_id()
            max_ts = chronics_handler.max_timestep()
            res[i] = (id_chron, name_chron, cum_reward, nb_time_step, max_ts)
        return res

    def run_parrallel(self, nb_episode, nb_process=1, path_save=None):
        """
        This method will run in parrallel, independantly the nb_episode over nb_process.

        Note that it restarts completely the :attr:`Runner.backend` and :attr:`Runner.env` if the computation
        is actually performed with more than 1 cores (nb_process > 1)

        It uses the python multiprocess, and especially the :class:`multiprocess.Pool` to perform the computations.
        This implies that all runs are completely independant (they happen in different process) and that the
        memory consumption can be big. Tests may be recommended if the amount of RAM is low.

        It has the same return type as the :func:`Runner.run_sequential`.

        Parameters
        ----------
        nb_episode: ``int``
            Number of episode to simulate

        nb_process: ``int``, optional
            Number of process used to play the nb_episode. Default to 1.

        path_save: ``str``, optional
            If not None, it specifies where to store the data. See the description of this module :mod:`Runner` for
            more information

        Returns
        -------
        res: ``list``
            List of tuple. Each tuple having 3 elements:

              - "i" unique identifier of the episode (compared to :func:`Runner.run_sequential`, the elements of the
                returned list are not necessarily sorted by this value)
              - "cum_reward" the cumulative reward obtained by the :attr:`Runner.Agent` on this episode i
              - "nb_time_step": the number of time steps played in this episode.
              - "max_ts" : the maximum number of time steps of the chronics

        """
        if nb_process <= 0:
            raise RuntimeError(
                "Runner: you need at least 1 process to run episodes")
        if nb_process == 1:
            warnings.warn(
                "Runner.run_parrallel: number of process set to 1. Failing back into sequential mod.")
            return [self.run_sequential(nb_episode, path_save=path_save)]
        else:
            if self.env is not None:
                self.env.close()
                self.env = None
            self.backend = self.backendClass()

            nb_process = int(nb_process)
            process_ids = [[] for i in range(nb_process)]
            for i in range(nb_episode):
                process_ids[i % nb_process].append(i)

            res = []
            with Pool(nb_process) as p:
                tmp = p.starmap(Runner._one_process_parrallel,
                                [(self, pn, i, path_save) for i, pn in enumerate(process_ids)])
            for el in tmp:
                res += el
        return res

    def run(self, nb_episode, nb_process=1, path_save=None, max_iter=None, pbar=False):
        """
        Main method of the :class:`Runner` class. It will either call :func:`Runner.run_sequential` if "nb_process" is
        1 or :func:`Runner.run_parrallel` if nb_process >= 2.

        Parameters
        ----------
        nb_episode: ``int``
            Number of episode to simulate

        nb_process: ``int``, optional
            Number of process used to play the nb_episode. Default to 1.

        path_save: ``str``, optional
            If not None, it specifies where to store the data. See the description of this module :mod:`Runner` for
            more information

        Returns
        -------
        res: ``list``
            List of tuple. Each tuple having 3 elements:

              - "i" unique identifier of the episode (compared to :func:`Runner.run_sequential`, the elements of the
                returned list are not necessarily sorted by this value)
              - "cum_reward" the cumulative reward obtained by the :attr:`Runner.Agent` on this episode i
              - "nb_time_step": the number of time steps played in this episode.

        """
        if nb_episode < 0:
            raise RuntimeError(
                "Impossible to run a negative number of scenarios.")
        if nb_episode == 0:
            res = []
        else:
            if nb_process <= 0:
                raise RuntimeError(
                    "Impossible to run using less than 1 process.")
            if max_iter is not None:
                self.chronics_handler.set_max_iter(max_iter)
            if nb_process == 1:
                self.logger.info("Sequential runner used.")
                res = self.run_sequential(nb_episode, path_save=path_save, pbar=pbar)
            else:
                self.logger.info("Parrallel runner used.")
                res = self.run_parrallel(
                    nb_episode, nb_process=nb_process, path_save=path_save)
        return res
