# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import warnings
import copy
import numpy as np
from multiprocessing import get_start_method, get_context, Pool
from typing import Tuple, List, Union

from grid2op.Environment import BaseEnv
from grid2op.Action import BaseAction, TopologyAction, DontAct
from grid2op.Exceptions import Grid2OpException, EnvError
from grid2op.Observation import CompleteObservation, BaseObservation
from grid2op.Opponent.opponentSpace import OpponentSpace
from grid2op.Reward import FlatReward, BaseReward
from grid2op.Rules import AlwaysLegal
from grid2op.Environment import Environment
from grid2op.Chronics import ChronicsHandler, GridStateFromFile, GridValue
from grid2op.Backend import Backend, PandaPowerBackend
from grid2op.Parameters import Parameters
from grid2op.Agent import DoNothingAgent, BaseAgent
from grid2op.VoltageControler import ControlVoltageFromFile
from grid2op.dtypes import dt_float
from grid2op.Opponent import BaseOpponent, NeverAttackBudget
from grid2op.operator_attention import LinearAttentionBudget
from grid2op.Space import DEFAULT_N_BUSBAR_PER_SUB
from grid2op.Episode import EpisodeData
# on windows if i start using sequential, i need to continue using sequential
# if i start using parallel i need to continue using parallel
# so i force the usage of the "starmap" stuff even if there is one process on windows
from grid2op._glop_platform_info import _IS_WINDOWS, _IS_LINUX, _IS_MACOS

from grid2op.Runner.aux_fun import (
    _aux_run_one_episode,
    _aux_make_progress_bar,
    _aux_one_process_parrallel,
)
from grid2op.Runner.basic_logger import DoNothingLog, ConsoleLog


runner_returned_type = Union[Tuple[str, str, float, int, int],
                             Tuple[str, str, float, int, int, EpisodeData],
                             Tuple[str, str, float, int, int, EpisodeData, int]]

# TODO have a vectorized implementation of everything in case the agent is able to act on multiple environment
# at the same time. This might require a lot of work, but would be totally worth it!
# (especially for Neural Net based agents)

# TODO add a more suitable logging strategy

# TODO use gym logger if specified by the user.
# TODO: if chronics are "loop through" multiple times, only last results are saved. :-/

KEY_TIME_SERIE_ID = "time serie id"

class Runner(object):
    """
    A runner is a utility tool that allows to run simulations more easily.

    It is a more convenient way to execute the
    following loops:

    .. code-block:: python

        import grid2op
        from grid2op.Agent import RandomAgent # for example...
        from grid2op.Runner import Runner

        env = grid2op.make("l2rpn_case14_sandbox")

        ###############
        # the gym loops
        nb_episode = 5
        for i in range(nb_episode):
            obs = env.reset()
            done = False
            reward = env.reward_range[0]
            while not done:
                act = agent.act(obs, reward, done)
                obs, reward, done, info = env.step(act)

        ###############
        # equivalent with use of a Runner
        runner = Runner(**env.get_params_for_runner(), agentClass=RandomAgent)
        res = runner.run(nb_episode=nn_episode)


    This specific class as for main purpose to evaluate the performance of a trained
    :class:`grid2op.Agent.BaseAgent` rather than to train it.

    It has also the good property to be able to save the results of a experiment in a standardized
    manner described in the :class:`grid2op.Episode.EpisodeData`.

    **NB** we do not recommend to create a runner from scratch by providing all the arguments. We strongly
    encourage you to use the :func:`grid2op.Environment.Environment.get_params_for_runner` for
    creating a runner.

    Attributes
    ----------
    envClass: ``type``
        The type of the environment used for the game. The class should be given, and **not** an instance (object) of
        this class. The default is the :class:`grid2op.Environment`. If modified, it should derived from this class.

    other_env_kwargs: ``dict``
        Other kwargs used to build the environment (None for "nothing")
        
    actionClass: ``type``
        The type of action that can be performed by the agent / bot / controler. The class should be given, and
        **not** an instance of this class. This type
        should derived from :class:`grid2op.BaseAction`. The default is :class:`grid2op.TopologyAction`.

    observationClass: ``type``
        This type represents the class that will be used to build the :class:`grid2op.BaseObservation` visible by the
        :class:`grid2op.BaseAgent`. As :attr:`Runner.actionClass`, this should be a type, and **not** and instance
        (object)
        of this type. This type should derived from :class:`grid2op.BaseObservation`. The default is
        :class:`grid2op.CompleteObservation`.

    rewardClass: ``type``
        Representes the type used to build the rewards that are given to the :class:`BaseAgent`. As
        :attr:`Runner.actionClass`, this should be a type, and **not** and instance (object) of this type.
        This type should derived from :class:`grid2op.BaseReward`. The default is :class:`grid2op.ConstantReward` that
        **should not** be used to train or evaluate an agent, but rather as debugging purpose.

    gridStateclass: ``type``
        This types control the mechanisms to read chronics and assign data to the powergrid. Like every "\\.*Class"
        attributes the type should be pass and not an intance (object) of this type. Its default is
        :class:`grid2op.GridStateFromFile` and it must be a subclass of :class:`grid2op.GridValue`.

    legalActClass: ``type``
        This types control the mechanisms to assess if an :class:`grid2op.BaseAction` is legal.
        Like every "\\.*Class" attributes the type should be pass and not an intance (object) of this type.
        Its default is :class:`grid2op.AlwaysLegal` and it must be a subclass of :class:`grid2op.BaseRules`.

    backendClass: ``type``
        This types control the backend, *eg.* the software that computes the powerflows.
        Like every "\\.*Class" attributes the type should be pass and not an intance (object) of this type.
        Its default is :class:`grid2op.PandaPowerBackend` and it must be a subclass of :class:`grid2op.Backend`.

    backend_kwargs: ``dict``
        Optional arguments used to build the backend. These arguments will not 
        be copied to create the backend used by the runner. They might
        required to be pickeable on some plateform when using multi processing.
        
    agentClass: ``type``
        This types control the type of BaseAgent, *eg.* the bot / controler that will take :class:`grid2op.BaseAction`
        and
        avoid cascading failures.
        Like every "\\.*Class" attributes the type should be pass and not an intance (object) of this type.
        Its default is :class:`grid2op.DoNothingAgent` and it must be a subclass of :class:`grid2op.BaseAgent`.

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

    opponent_action_class: ``type``, optional
        The action class used for the opponent. The opponent will not be able to use action that are invalid with
        the given action class provided. It defaults to :class:`grid2op.Action.DontAct` which forbid any type
        of action possible.

    opponent_class: ``type``, optional
        The opponent class to use. The default class is :class:`grid2op.Opponent.BaseOpponent` which is a type
        of opponents that does nothing.

    opponent_init_budget: ``float``, optional
        The initial budget of the opponent. It defaults to 0.0 which means the opponent cannot perform any action
        if this is not modified.

    opponent_budget_per_ts: ``float``, optional
        The budget increase of the opponent per time step

    opponent_budget_class: ``type``, optional
        The class used to compute the attack cost.

    grid_layout: ``dict``, optional
        The layout of the grid (position of each substation) usefull if you need to plot some things for example.

    TODO
    _attention_budget_cls=LinearAttentionBudget,
    _kwargs_attention_budget=None,
    _has_attention_budget=False

    Examples
    --------
    Different examples are showed in the description of the main method :func:`Runner.run`

    Notes
    -----

    Runner does not necessarily behave normally when "nb_process" is not 1 on some platform (windows and some
    version of macos). Please read the documentation, and especially the :ref:`runner-multi-proc-warning`
    for more information and possible way to disable this feature.

    """

    FORCE_SEQUENTIAL = "GRID2OP_RUNNER_FORCE_SEQUENTIAL"

    def __init__(
        self,
        init_env_path: str,
        init_grid_path: str,
        path_chron,  # path where chronics of injections are stored
        n_busbar=DEFAULT_N_BUSBAR_PER_SUB,
        name_env="unknown",
        parameters_path=None,
        names_chronics_to_backend=None,
        actionClass=TopologyAction,
        observationClass=CompleteObservation,
        rewardClass=FlatReward,
        legalActClass=AlwaysLegal,
        envClass=Environment,
        other_env_kwargs=None,
        gridStateclass=GridStateFromFile,
        # type of chronics to use. For example GridStateFromFile if forecasts are not used,
        # or GridStateFromFileWithForecasts otherwise
        backendClass=PandaPowerBackend,
        backend_kwargs=None,
        agentClass=DoNothingAgent,  # class used to build the agent
        agentInstance=None,
        verbose=False,
        gridStateclass_kwargs={},
        voltageControlerClass=ControlVoltageFromFile,
        thermal_limit_a=None,
        max_iter=-1,
        other_rewards={},
        opponent_space_type=OpponentSpace,
        opponent_action_class=DontAct,
        opponent_class=BaseOpponent,
        opponent_init_budget=0.0,
        opponent_budget_per_ts=0.0,
        opponent_budget_class=NeverAttackBudget,
        opponent_attack_duration=0,
        opponent_attack_cooldown=99999,
        opponent_kwargs={},
        grid_layout=None,
        with_forecast=True,
        attention_budget_cls=LinearAttentionBudget,
        kwargs_attention_budget=None,
        has_attention_budget=False,
        logger=None,
        use_compact_episode_data=False,
        kwargs_observation=None,
        observation_bk_class=None,
        observation_bk_kwargs=None,
        
        # experimental: whether to read from local dir or generate the classes on the fly:
        _read_from_local_dir=False,
        _is_test=False,  # TODO not implemented !!
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

        use_compact_episode_data:  ``bool``, optional
            Whether to use :class:`grid2op.Episode.CompactEpisodeData` instead of :class:`grid2op.Episode.EpisodeData` to store 
            Episode to disk (allows it to be replayed later). Defaults to False.

        # TODO documentation on the opponent
        # TOOD doc for the attention budget
        """
        self._n_busbar = n_busbar
        self.with_forecast = with_forecast
        self.name_env = name_env
        if not isinstance(envClass, type):
            raise Grid2OpException(
                'Parameter "envClass" used to build the Runner should be a type (a class) and not an object '
                '(an instance of a class). It is currently "{}"'.format(type(envClass))
            )
        if not issubclass(envClass, Environment):
            raise RuntimeError(
                "Impossible to create a runner without an evnrionment derived from grid2op.Environement"
                ' class. Please modify "envClass" parameter.'
            )
        self.envClass = envClass
        if other_env_kwargs is not None:
            self.other_env_kwargs = other_env_kwargs
        else:
            self.other_env_kwargs = {}

        if not isinstance(actionClass, type):
            raise Grid2OpException(
                'Parameter "actionClass" used to build the Runner should be a type (a class) and not an object '
                '(an instance of a class). It is currently "{}"'.format(
                    type(actionClass)
                )
            )
        if not issubclass(actionClass, BaseAction):
            raise RuntimeError(
                "Impossible to create a runner without an action class derived from grid2op.BaseAction. "
                'Please modify "actionClass" parameter.'
            )
        self.actionClass = actionClass

        if not isinstance(observationClass, type):
            raise Grid2OpException(
                'Parameter "observationClass" used to build the Runner should be a type (a class) and not an object '
                '(an instance of a class). It is currently "{}"'.format(
                    type(observationClass)
                )
            )
        if not issubclass(observationClass, BaseObservation):
            raise RuntimeError(
                "Impossible to create a runner without an observation class derived from "
                'grid2op.BaseObservation. Please modify "observationClass" parameter.'
            )
        self.observationClass = observationClass
        if isinstance(rewardClass, type):
            if not issubclass(rewardClass, BaseReward):
                raise RuntimeError(
                    "Impossible to create a runner without an observation class derived from "
                    'grid2op.BaseReward. Please modify "rewardClass" parameter.'
                )
        else:
            if not isinstance(rewardClass, BaseReward):
                raise RuntimeError(
                    "Impossible to create a runner without an observation class derived from "
                    'grid2op.BaseReward. Please modify "rewardClass" parameter.'
                )

        self.rewardClass = rewardClass

        if not isinstance(gridStateclass, type):
            raise Grid2OpException(
                'Parameter "gridStateclass" used to build the Runner should be a type (a class) and not an object '
                '(an instance of a class). It is currently "{}"'.format(
                    type(gridStateclass)
                )
            )
        if not issubclass(gridStateclass, GridValue):
            raise RuntimeError(
                "Impossible to create a runner without an chronics class derived from "
                'grid2op.GridValue. Please modify "gridStateclass" parameter.'
            )
        self.gridStateclass = gridStateclass

        self.envClass._check_rules_correct(legalActClass)
        self.legalActClass = legalActClass

        if not isinstance(backendClass, type):
            raise Grid2OpException(
                'Parameter "legalActClass" used to build the Runner should be a type (a class) and not an object '
                '(an instance of a class). It is currently "{}"'.format(
                    type(backendClass)
                )
            )
        if not issubclass(backendClass, Backend):
            raise RuntimeError(
                "Impossible to create a runner without a backend class derived from grid2op.GridValue. "
                'Please modify "backendClass" parameter.'
            )
        self.backendClass = backendClass
        if backend_kwargs is not None:
            self._backend_kwargs = backend_kwargs
        else:
            self._backend_kwargs = {}
        
        self.__can_copy_agent = True
        if agentClass is not None:
            if agentInstance is not None:
                raise RuntimeError(
                    "Impossible to build the Runner. Only one of agentClass or agentInstance can be "
                    "used (both are set / both are not None)."
                )
            if not isinstance(agentClass, type):
                raise Grid2OpException(
                    'Parameter "agentClass" used to build the Runner should be a type (a class) and not an object '
                    '(an instance of a class). It is currently "{}"'.format(
                        type(agentClass)
                    )
                )
            if not issubclass(agentClass, BaseAgent):
                raise RuntimeError(
                    "Impossible to create a runner without an agent class derived from "
                    "grid2op.BaseAgent. "
                    'Please modify "agentClass" parameter.'
                )
            self.agentClass = agentClass
            self._useclass = True
            self.agent = None
        elif agentInstance is not None:
            if not isinstance(agentInstance, BaseAgent):
                raise RuntimeError(
                    "Impossible to create a runner without an agent class derived from "
                    "grid2op.BaseAgent. "
                    'Please modify "agentInstance" parameter.'
                )
            self.agentClass = None
            self._useclass = False
            self.agent = agentInstance
            # Test if we can copy the agent for parallel runs
            try:
                copy.copy(self.agent)
            except Exception as exc_:
                self.__can_copy_agent = False
        else:
            raise RuntimeError(
                "Impossible to build the backend. Either AgentClass or agentInstance must be provided "
                "and both are None."
            )
        self.agentInstance = agentInstance

        self._read_from_local_dir = _read_from_local_dir
        self._observation_bk_class = observation_bk_class
        self._observation_bk_kwargs = observation_bk_kwargs

        self.logger = ConsoleLog(DoNothingLog.INFO if verbose else DoNothingLog.ERROR)
        if logger is None:
            import logging

            self.logger = logging.getLogger(__name__)
            if verbose:
                self.logger.setLevel("debug")
            else:
                self.logger.disabled = True
        else:
            self.logger = logger.getChild("grid2op_Runner")

        self.use_compact_episode_data = use_compact_episode_data

        # store _parameters
        self.init_env_path = init_env_path
        self.init_grid_path = init_grid_path
        self.names_chronics_to_backend = names_chronics_to_backend

        # game _parameters
        self.parameters_path = parameters_path
        if isinstance(parameters_path, str):
            self.parameters = Parameters(parameters_path)
        elif isinstance(parameters_path, dict):
            self.parameters = Parameters()
            self.parameters.init_from_dict(parameters_path)
        elif parameters_path is None:
            self.parameters = Parameters()
        else:
            raise RuntimeError(
                'Impossible to build the parameters. The argument "parameters_path" should either '
                "be a string or a dictionary."
            )

        # chronics of grid state
        self.path_chron = path_chron
        self.gridStateclass_kwargs = gridStateclass_kwargs
        self.max_iter = max_iter
        if max_iter > 0:
            self.gridStateclass_kwargs["max_iter"] = max_iter

        self.verbose = verbose
        self.thermal_limit_a = thermal_limit_a

        # controler for voltage
        if not issubclass(voltageControlerClass, ControlVoltageFromFile):
            raise Grid2OpException(
                'Parameter "voltagecontrolClass" should derive from "ControlVoltageFromFile".'
            )
        self.voltageControlerClass = voltageControlerClass
        self._other_rewards = other_rewards

        # for opponent (should be defined here) after the initialization of BaseEnv
        self._opponent_space_type = opponent_space_type
        if not issubclass(opponent_action_class, BaseAction):
            raise EnvError(
                "Impossible to make an environment with an opponent action class not "
                "derived from BaseAction"
            )
        try:
            self.opponent_init_budget = dt_float(opponent_init_budget)
        except Exception as e:
            raise EnvError(
                'Impossible to convert "opponent_init_budget" to a float with error {}'.format(
                    e
                )
            )
        if self.opponent_init_budget < 0.0:
            raise EnvError(
                "If you want to deactive the opponent, please don't set its budget to a negative number."
                'Prefer the use of the DontAct action type ("opponent_action_class=DontAct" '
                "and / or set its budget to 0."
            )
        if not issubclass(opponent_class, BaseOpponent):
            raise EnvError(
                "Impossible to make an opponent with a type that does not inherit from BaseOpponent."
            )
        self.opponent_action_class = opponent_action_class
        self.opponent_class = opponent_class
        self.opponent_init_budget = opponent_init_budget
        self.opponent_budget_per_ts = opponent_budget_per_ts
        self.opponent_budget_class = opponent_budget_class
        self.opponent_attack_duration = opponent_attack_duration
        self.opponent_attack_cooldown = opponent_attack_cooldown
        self.opponent_kwargs = opponent_kwargs
        self.grid_layout = grid_layout

        # attention budget
        self._attention_budget_cls = attention_budget_cls
        self._kwargs_attention_budget = copy.deepcopy(kwargs_attention_budget)
        self._has_attention_budget = has_attention_budget

        # custom observation building
        if kwargs_observation is None:
            kwargs_observation = {}
        self._kwargs_observation = copy.deepcopy(kwargs_observation)

        # otherwise on windows / macos it sometimes fail in the runner in multi process
        # on linux like OS i prefer to generate all the proper classes accordingly
        if _IS_LINUX:
            pass
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                with self.init_env() as env:
                    bk_class = type(env.backend)
                    pass

        # not implemented !
        self._is_test = _is_test

        self.__used = False

    def _make_new_backend(self):
        try:
            res = self.backendClass(**self._backend_kwargs)
        except TypeError:
            # for backward compatibility, some backend might not
            # handle full kwargs (that might be added later)
            import inspect
            possible_params = inspect.signature(self.backendClass.__init__).parameters
            this_kwargs = {}
            for el in self._backend_kwargs:
                if el in possible_params:
                    this_kwargs[el] = self._backend_kwargs[el]
                else:
                    warnings.warn("Runner: your backend does not support the kwargs "
                                  f"`{el}={self._backend_kwargs[el]}`. This usually "
                                  "means it is outdated. Please upgrade it.")
            res = self.backendClass(**this_kwargs)
        return res
    
    def _new_env(self, parameters) -> Tuple[BaseEnv, BaseAgent]:
        chronics_handler = ChronicsHandler(
            chronicsClass=self.gridStateclass,
            path=self.path_chron,
            **self.gridStateclass_kwargs
        )
        backend = self._make_new_backend()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = self.envClass.init_obj_from_kwargs(
                n_busbar=self._n_busbar,
                other_env_kwargs=self.other_env_kwargs,
                init_env_path=self.init_env_path,
                init_grid_path=self.init_grid_path,
                chronics_handler=chronics_handler,
                backend=backend,
                parameters=parameters,
                name=self.name_env,
                names_chronics_to_backend=self.names_chronics_to_backend,
                actionClass=self.actionClass,
                observationClass=self.observationClass,
                rewardClass=self.rewardClass,
                legalActClass=self.legalActClass,
                voltagecontrolerClass=self.voltageControlerClass,
                other_rewards=self._other_rewards,
                opponent_space_type=self._opponent_space_type,
                opponent_action_class=self.opponent_action_class,
                opponent_class=self.opponent_class,
                opponent_init_budget=self.opponent_init_budget,
                opponent_budget_per_ts=self.opponent_budget_per_ts,
                opponent_budget_class=self.opponent_budget_class,
                opponent_attack_duration=self.opponent_attack_duration,
                opponent_attack_cooldown=self.opponent_attack_cooldown,
                kwargs_opponent=self.opponent_kwargs,
                with_forecast=self.with_forecast,
                attention_budget_cls=self._attention_budget_cls,
                kwargs_attention_budget=self._kwargs_attention_budget,
                has_attention_budget=self._has_attention_budget,
                logger=self.logger,
                kwargs_observation=self._kwargs_observation,
                observation_bk_class=self._observation_bk_class,
                observation_bk_kwargs=self._observation_bk_kwargs,
                _raw_backend_class=self.backendClass,
                _read_from_local_dir=self._read_from_local_dir,
            )

        if self.thermal_limit_a is not None:
            res.set_thermal_limit(self.thermal_limit_a)

        if self.grid_layout is not None:
            res.attach_layout(self.grid_layout)

        if self._useclass:
            agent = self.agentClass(res.action_space)
        else:
            if self.__can_copy_agent:
                agent = copy.copy(self.agent)
            else:
                agent = self.agent
        return res, agent

    def init_env(self) -> BaseEnv:
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Function used to initialized the environment and the agent.
        It is called by :func:`Runner.reset`.
        """
        env, self.agent = self._new_env(self.parameters)
        return env

    def reset(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Used to reset an environment. This method is called at the beginning of each new episode.
        If the environment is not initialized, then it initializes it with :func:`Runner.make_env`.
        """
        pass

    def run_one_episode(
        self,
        indx=0,
        path_save=None,
        pbar=False,
        env_seed=None,
        max_iter=None,
        agent_seed=None,
        episode_id=None,
        detailed_output=False,
        add_nb_highres_sim=False,
        init_state=None,
        reset_options=None,
    ) -> runner_returned_type:
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Function used to run one episode of the :attr:`Runner.agent` and see how it performs in the :attr:`Runner.env`.

        Parameters
        ----------
        indx: ``int``
            The index of the episode to run (ignored if `episode_id` is not None)

        path_save: ``str``, optional
            Path where to save the data. See the description of :mod:`grid2op.Runner` for the structure of the saved
            file.
            
        detailed_output: 
            See descr. of :func:`Runner.run` method
        
        add_nb_highres_sim: 
            See descr. of :func:`Runner.run` method

        Returns
        -------
        TODO DEPRECATED DOC
        
        cum_reward: ``np.float32``
            The cumulative reward obtained by the agent during this episode

        time_step: ``int``
            The number of timesteps that have been played before the end of the episode (because of a "game over" or
            because there were no more data)

        """
        self.reset()
        with self.init_env() as env:  
            # small piece of code to detect the 
            # episode id
            if episode_id is None:
                # user did not provide any episode id, I check in the reset_options
                if reset_options is not None:
                    if KEY_TIME_SERIE_ID in reset_options:
                        indx = int(reset_options[KEY_TIME_SERIE_ID])
                        del reset_options[KEY_TIME_SERIE_ID]
            else:
                # user specified an episode id, I use it.
                indx = episode_id                      
            res = _aux_run_one_episode(
                env,
                self.agent,
                self.logger,
                indx,
                path_save,
                pbar=pbar,
                env_seed=env_seed,
                max_iter=max_iter,
                agent_seed=agent_seed,
                detailed_output=detailed_output,
                use_compact_episode_data = self.use_compact_episode_data,
                init_state=init_state,
                reset_option=reset_options,
            )
            if max_iter is not None:
                env.chronics_handler._set_max_iter(-1)
            
            id_chron = env.chronics_handler.get_id()
        # `res` here necessarily contains detailed_output and nb_highres_call  
        if not add_nb_highres_sim:
            res = res[:-1]
        if not detailed_output:
            res = res[:-1]
        
        # new in 1.10.2: id_chron is computed from here
        res = (id_chron, *res)
        return res

    def _run_sequential(
        self,
        nb_episode,
        path_save=None,
        pbar=False,
        env_seeds=None,
        agent_seeds=None,
        max_iter=None,
        episode_id=None,
        add_detailed_output=False,
        add_nb_highres_sim=False,
        init_states=None,
        reset_options=None,
    ) -> List[runner_returned_type]:
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

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

        env_seeds: ``list``
            An iterable of the seed used for the experiments. By default ``None``, no seeds are set. If provided,
            its size should match ``nb_episode``.

        episode_id: ``list``
            For each of the nb_episdeo you want to compute, it specifies the id of the chronix that will be used.
            By default ``None``, no seeds are set. If provided,
            its size should match ``nb_episode``.

        add_detailed_output: 
            see :func:`Runner.run` method
        
        init_states: 
            see :func:`Runner.run` method

        Returns
        -------
        res: ``list``
            List of tuple. Each tuple having 5 elements:

              - "id_chron" unique identifier of the episode
              - "name_chron" name of chronics
              - "cum_reward" the cumulative reward obtained by the :attr:`Runner.BaseAgent` on this episode i
              - "nb_time_step": the number of time steps played in this episode.
              - "max_ts" : the maximum number of time steps of the chronics
              - "episode_data" : The :class:`EpisodeData` corresponding to this episode run

        """
        res = [(None, None, None, None, None, None) 
               for _ in range(nb_episode)]

        next_pbar = [False]
        with _aux_make_progress_bar(pbar, nb_episode, next_pbar) as pbar_:
            for i in range(nb_episode):
                env_seed = None
                if env_seeds is not None:
                    env_seed = env_seeds[i]
                agt_seed = None
                if agent_seeds is not None:
                    agt_seed = agent_seeds[i]
                init_state = None
                if init_states is not None:
                    init_state = init_states[i]
                reset_opt = None
                if reset_options is not None:
                    # we copy it because we might remove the "time serie id"
                    # from it
                    reset_opt = reset_options[i].copy()
                # if no "episode_id" is provided i used the i th one   
                ep_id = i 
                if episode_id is not None:
                    # if episode_id is provided, I use this one
                    ep_id = episode_id[i]  # otherwise i use the provided one
                else:
                    # if it's not provided, I check if one is used in the `reset_options`
                    if reset_opt is not None:
                        if KEY_TIME_SERIE_ID in reset_opt:
                            ep_id = int(reset_opt[KEY_TIME_SERIE_ID])
                            del reset_opt[KEY_TIME_SERIE_ID]
                (
                    id_chron,
                    name_chron,
                    cum_reward,
                    nb_time_step,
                    max_ts,
                    episode_data,
                    nb_call_highres_sim,
                ) = self.run_one_episode(
                    path_save=path_save,
                    indx=ep_id,
                    episode_id=ep_id,
                    pbar=next_pbar[0],
                    env_seed=env_seed,
                    agent_seed=agt_seed,
                    max_iter=max_iter,
                    detailed_output=True,
                    add_nb_highres_sim=True,
                    init_state=init_state,
                    reset_options=reset_opt
                )
                res[i] = (id_chron,
                          name_chron,
                          float(cum_reward),
                          nb_time_step,
                          max_ts
                          )
                if add_detailed_output:
                    res[i] = (*res[i], episode_data)
                if add_nb_highres_sim:
                    res[i] = (*res[i], nb_call_highres_sim)
                pbar_.update(1)
        return res

    def _run_parrallel(
        self,
        nb_episode,
        nb_process=1,
        path_save=None,
        env_seeds=None,
        agent_seeds=None,
        max_iter=None,
        episode_id=None,
        add_detailed_output=False,
        add_nb_highres_sim=False,
        init_states=None,
        reset_options=None,
    ) -> List[runner_returned_type]:
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        This method will run in parallel, independently the nb_episode over nb_process.

        In case the agent cannot be cloned using `copy.copy`: nb_process is set to 1

        Note that it restarts completely the :attr:`Runner.backend` and :attr:`Runner.env` if the computation
        is actually performed with more than 1 cores (nb_process > 1)

        It uses the python multiprocess, and especially the :class:`multiprocess.Pool` to perform the computations.
        This implies that all runs are completely independent (they happen in different process) and that the
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

        env_seeds: ``list``
            An iterable of the seed used for the experiments. By default ``None``, no seeds are set. If provided,
            its size should match ``nb_episode``.

        agent_seeds: ``list``
            An iterable that contains the seed used for the environment. By default ``None`` means no seeds are set.
            If provided, its size should match the ``nb_episode``. The agent will be seeded at the beginning of each
            scenario BEFORE calling `agent.reset()`.

        add_detailed_output: 
            See :func:`Runner.run` method
        
        init_states:
            See :func:`Runner.run` method
            
        Returns
        -------
        res: ``list``
            List of tuple. Each tuple having 3 elements:

              - "i" unique identifier of the episode (compared to :func:`Runner.run_sequential`, the elements of the
                returned list are not necessarily sorted by this value)
              - "cum_reward" the cumulative reward obtained by the :attr:`Runner.BaseAgent` on this episode i
              - "nb_time_step": the number of time steps played in this episode.
              - "max_ts" : the maximum number of time steps of the chronics
              - "episode_data" : The :class:`EpisodeData` corresponding to this episode run

        """
        if nb_process <= 0:
            raise RuntimeError("Runner: you need at least 1 process to run episodes")
        force_sequential = False
        tmp = os.getenv(Runner.FORCE_SEQUENTIAL)
        if tmp is not None:
            force_sequential = int(tmp) > 0
        if nb_process == 1 or (not self.__can_copy_agent) or force_sequential:
            # on windows if i start using sequential, i need to continue using sequential
            # if i start using parallel i need to continue using parallel
            # so i force the usage of the sequential mode
            self.logger.warn(
                "Runner.run_parrallel: number of process set to 1. Failing back into sequential mode."
            )
            return self._run_sequential(
                nb_episode,
                path_save=path_save,
                env_seeds=env_seeds,
                max_iter=max_iter,
                agent_seeds=agent_seeds,
                episode_id=episode_id,
                add_detailed_output=add_detailed_output,
                add_nb_highres_sim=add_nb_highres_sim,
                init_states=init_states,
                reset_options=reset_options
            )
        else:
            self._clean_up()

            nb_process = int(nb_process)
            process_ids = [[] for i in range(nb_process)]
            for i in range(nb_episode):
                if episode_id is None:
                    # user does not provide episode_id
                    if reset_options is not None:
                        # we copy them, because we might delete some things from them
                        reset_options = [el.copy() for el in reset_options]  
                        
                        # we check if the reset_options contains the "time serie id"
                        if KEY_TIME_SERIE_ID in reset_options[i]:
                            this_ep_id = int(reset_options[i][KEY_TIME_SERIE_ID])
                            del reset_options[i][KEY_TIME_SERIE_ID]
                        else:
                            this_ep_id = i
                    else:
                        this_ep_id = i
                    process_ids[i % nb_process].append(this_ep_id)
                else:
                    # user provided episode_id, we use this one
                    process_ids[i % nb_process].append(episode_id[i])

            if env_seeds is None:
                seeds_env_res = [None for _ in range(nb_process)]
            else:
                # split the seeds according to the process
                seeds_env_res = [[] for _ in range(nb_process)]
                for i in range(nb_episode):
                    seeds_env_res[i % nb_process].append(env_seeds[i])

            if agent_seeds is None:
                seeds_agt_res = [None for _ in range(nb_process)]
            else:
                # split the seeds according to the process
                seeds_agt_res = [[] for _ in range(nb_process)]
                for i in range(nb_episode):
                    seeds_agt_res[i % nb_process].append(agent_seeds[i])
                    
            if init_states is None:
                init_states_res = [None for _ in range(nb_process)]
            else:
                # split the init states according to the process
                init_states_res = [[] for _ in range(nb_process)]
                for i in range(nb_episode):
                    init_states_res[i % nb_process].append(init_states[i])

            if reset_options is None:
                reset_options_res = [None for _ in range(nb_process)]
            else:
                # split the reset options according to the process
                reset_options_res = [[] for _ in range(nb_process)]
                for i in range(nb_episode):
                    reset_options_res[i % nb_process].append(reset_options[i])
                
            res = []
            if _IS_LINUX:
                lists = [(self,) for _ in enumerate(process_ids)]
            else:
                lists = [(Runner(**self._get_params()),) for _ in enumerate(process_ids)]
            
            for i, pn in enumerate(process_ids):
                lists[i] = (*lists[i],
                            pn,
                            i,
                            path_save,
                            seeds_env_res[i],
                            seeds_agt_res[i],
                            max_iter,
                            add_detailed_output,
                            add_nb_highres_sim,
                            init_states_res[i],
                            reset_options_res[i])
                
            if get_start_method() == 'spawn':
                # https://github.com/rte-france/Grid2Op/issues/600
                with get_context("spawn").Pool(nb_process) as p:
                    tmp = p.starmap(_aux_one_process_parrallel, lists)
            else:            
                with Pool(nb_process) as p:
                    tmp = p.starmap(_aux_one_process_parrallel, lists)
            for el in tmp:
                res += el
        return res

    def _get_params(self):
        res = {
            "init_grid_path": self.init_grid_path,
            "init_env_path": self.init_env_path,
            "path_chron": self.path_chron,  # path where chronics of injections are stored
            "name_env": self.name_env,
            "parameters_path": self.parameters_path,
            "names_chronics_to_backend": self.names_chronics_to_backend,
            "actionClass": self.actionClass,
            "observationClass": self.observationClass,
            "rewardClass": self.rewardClass,
            "legalActClass": self.legalActClass,
            "envClass": self.envClass,
            "gridStateclass": self.gridStateclass,
            "backendClass": self.backendClass,
            "backend_kwargs": self._backend_kwargs,
            "agentClass": self.agentClass,
            "agentInstance": self.agentInstance,
            "verbose": self.verbose,
            "gridStateclass_kwargs": copy.deepcopy(self.gridStateclass_kwargs),
            "voltageControlerClass": self.voltageControlerClass,
            "thermal_limit_a": self.thermal_limit_a,
            "max_iter": self.max_iter,
            "other_rewards": copy.deepcopy(self._other_rewards),
            "opponent_space_type": self._opponent_space_type,
            "opponent_action_class": self.opponent_action_class,
            "opponent_class": self.opponent_class,
            "opponent_init_budget": self.opponent_init_budget,
            "opponent_budget_per_ts": self.opponent_budget_per_ts,
            "opponent_budget_class": self.opponent_budget_class,
            "opponent_attack_duration": self.opponent_attack_duration,
            "opponent_attack_cooldown": self.opponent_attack_cooldown,
            "opponent_kwargs": copy.deepcopy(self.opponent_kwargs),
            "grid_layout": copy.deepcopy(self.grid_layout),
            "with_forecast": self.with_forecast,
            "attention_budget_cls": self._attention_budget_cls,
            "kwargs_attention_budget": self._kwargs_attention_budget,
            "has_attention_budget": self._has_attention_budget,
            "logger": self.logger,
            "use_compact_episode_data": self.use_compact_episode_data,
            "kwargs_observation": self._kwargs_observation,
            "_read_from_local_dir": self._read_from_local_dir,
            "_is_test": self._is_test,
        }
        return res

    def _clean_up(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        close the environment if it has been created

        """
        pass

    def run(
        self,
        nb_episode,
        *,  # force kwargs
        nb_process=1,
        path_save=None,
        max_iter=None,
        pbar=False,
        env_seeds=None,
        agent_seeds=None,
        episode_id=None,
        add_detailed_output=False,
        add_nb_highres_sim=False,
        init_states=None,
        reset_options=None,
    ) -> List[runner_returned_type]:
        """
        Main method of the :class:`Runner` class. It will either call :func:`Runner._run_sequential` if "nb_process" is
        1 or :func:`Runner._run_parrallel` if nb_process >= 2.

        Parameters
        ----------
        nb_episode: ``int``
            Number of episode to simulate

        nb_process: ``int``, optional
            Number of process used to play the nb_episode. Default to 1. **NB** Multitoprocessing is deactivated
            on windows based platform (it was not fully supported so we decided to remove it)

        path_save: ``str``, optional
            If not None, it specifies where to store the data. See the description of this module :mod:`Runner` for
            more information

        max_iter: ``int``
            Maximum number of iteration you want the runner to perform.
            
            .. warning::
                (only for grid2op >= 1.10.3) If set in this parameters, it will
                erase all values that may be present in the `reset_options` kwargs (key `"max step"`) 
                
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

        env_seeds: ``list``
            An iterable of the seed used for the environment. By default ``None``, no seeds are set. If provided,
            its size should match ``nb_episode``.

        agent_seeds: ``list``
            An iterable that contains the seed used for the environment. By default ``None`` means no seeds are set.
            If provided, its size should match the ``nb_episode``. The agent will be seeded at the beginning of each
            scenario BEFORE calling `agent.reset()`.

        episode_id: ``list``
            For each of the nb_episdeo you want to compute, it specifies the id of the chronix that will be used.
            By default ``None``, no seeds are set. If provided,
            its size should match ``nb_episode``.
            
            .. warning::
                (only for grid2op >= 1.10.3)  If set in this parameters, it will
                erase all values that may be present in the `reset_options` kwargs (key `"time serie id"`).
                
            .. danger::
                As of now, it's not properly handled to compute twice the same `episode_id` more than once using the runner
                (more specifically, the computation will happen but file might not be saved correctly on the 
                hard drive: attempt to save all the results in the same location. We do not advise to do it)

        add_detailed_output: ``bool``
            A flag to add an :class:`EpisodeData` object to the results, containing a lot of information about the run

        add_nb_highres_sim: ``bool``
            Whether to add an estimated number of "high resolution simulator" called performed by the agent (either by
            obs.simulate, or by obs.get_forecast_env or by obs.get_simulator)
        
        init_states:
            (added in grid2op 1.10.2) Possibility to set the initial state of the powergrid (when calling `env.reset`). 
            It should either be:
            
            - a dictionary representing an action (see doc of :func:`grid2op.Environment.Environment.reset`)
            - a grid2op action (see doc of :func:`grid2op.Environment.Environment.reset`)
            - a list / tuple of one of the above with the same size as the number of episode you want.
            
            If you provide a dictionary or a grid2op action, then this element will be used for all scenarios you
            want to run.
            
            .. warning::
                (only for grid2op >= 1.10.3)  If set in this parameters, it will
                erase all values that may be present in the `reset_options` kwargs (key `"init state"`).

        reset_options:
            (added in grid2op 1.10.3) Possibility to customize the call to `env.reset` made internally by
            the Runner. More specifically, it will pass a custom `options` when the runner calls 
            `env.reset(..., options=XXX)`.
            
            It should either be:
            
            - a dictionary that can be used directly by :func:`grid2op.Environment.Environment.reset`. 
              In this case the same dictionary will be used for all the episodes computed by the runner.
            - a list / tuple of one of the above with the same size as the number of episode you want to
              compute which allow a full customization for each episode.
              
            .. warning::
                If the kwargs `max_iter` is present when calling `runner.run` function, then the key `max step`
                will be ignored in all the `reset_options` dictionary.
              
            .. warning::
                If the kwargs `episode_id` is present when calling `runner.run` function, then the key `time serie id`
                will be ignored in all the `reset_options` dictionary.
              
            .. warning::
                If the kwargs `init_states` is present when calling `runner.run` function, then the key `init state`
                will be ignored in all the `reset_options` dictionary.
                
            .. danger::
                If you provide the key "time serie id" in one of the `reset_options` dictionary, we recommend
                you do it for all `reset options` otherwise you might not end up computing the correct episodes.
                
            .. danger::
                As of now, it's not properly handled to compute twice the same `time serie` more than once using the runner
                (more specifically, the computation will happen but file might not be saved correctly on the 
                hard drive: attempt to save all the results in the same location. We do not advise to do it)
                        
        Returns
        -------
        res: ``list``
            List of tuple. Each tuple having 3[4] elements:

              - "id_chron" unique identifier of the episode
              - "name_chron" name of the time series (usually it is the path where it is stored)
              - "cum_reward" the cumulative reward obtained by the :attr:`Runner.Agent` on this episode i
              - "nb_time_step": the number of time steps played in this episode.
              - "total_step": the total number of time steps possible in this episode.
              - "episode_data" : [Optional] The :class:`EpisodeData` corresponding to this episode run only
                if `add_detailed_output=True`
              - "add_nb_highres_sim": [Optional] The estimated number of calls to high resolution simulator made
                by the agent. Only preset if `add_nb_highres_sim=True` in the kwargs

        Examples
        --------

        You can use the runner this way:

        .. code-block:: python

            import grid2op
            from gri2op.Runner import Runner
            from grid2op.Agent import RandomAgent

            env = grid2op.make("l2rpn_case14_sandbox")
            runner = Runner(**env.get_params_for_runner(), agentClass=RandomAgent)
            res = runner.run(nb_episode=1)

        If you would rather to provide an agent instance (and not a class) you can do it this way:

        .. code-block:: python

            import grid2op
            from gri2op.Runner import Runner
            from grid2op.Agent import RandomAgent

            env = grid2op.make("l2rpn_case14_sandbox")
            my_agent = RandomAgent(env.action_space)
            runner = Runner(**env.get_params_for_runner(), agentClass=None, agentInstance=my_agent)
            res = runner.run(nb_episode=1)

        Finally, in the presence of stochastic environments or stochastic agent you might want to set the seeds for
        ensuring reproducible experiments you might want to seed both the environment and your agent. You can do that
        by passing `env_seeds` and `agent_seeds` parameters (on the example bellow, the agent will be seeded with 42
        and the environment with 0.

        .. code-block:: python

            import grid2op
            from gri2op.Runner import Runner
            from grid2op.Agent import RandomAgent

            env = grid2op.make("l2rpn_case14_sandbox")
            my_agent = RandomAgent(env.action_space)
            runner = Runner(**env.get_params_for_runner(), agentClass=None, agentInstance=my_agent)
            res = runner.run(nb_episode=1, agent_seeds=[42], env_seeds=[0])

        Since grid2op 1.10.2 you can also set the initial state of the grid when
        calling the runner. You can do that with the kwargs `init_states`, for example like this:
        
        .. code-block:: python

            import grid2op
            from gri2op.Runner import Runner
            from grid2op.Agent import RandomAgent

            env = grid2op.make("l2rpn_case14_sandbox")
            my_agent = RandomAgent(env.action_space)
            runner = Runner(**env.get_params_for_runner(), agentClass=None, agentInstance=my_agent)
            res = runner.run(nb_episode=1,
                             agent_seeds=[42],
                             env_seeds=[0],
                             init_states=[{"set_line_status": [(0, -1)]}]
                             )
        
        .. note::
            We recommend that you provide `init_states` as a list having a length of
            `nb_episode`. Each episode will be initialized with the provided
            element of the list. However, if you provide only one element, then
            all episodes you want to compute will be initialized with this same
            action.
            
        .. note::
            At the beginning of each episode, if an `init_state` is set, 
            the environment is reset with a call like: `env.reset(options={"init state": init_state})`
            
            This is why we recommend you to use dictionary to set the initial state so 
            that you can control what exactly is done (set the `"method"`) more 
            information about this on the doc of the :func:`grid2op.Environment.Environment.reset`
            function.
        
        Since grid2op 1.10.3 you can also customize the way the runner will "reset" the
        environment with the kwargs `reset_options`. 
        
        Concretely, if you specify `runner.run(..., reset_options=XXX)` then the environment
        will be reset with a call to `env.reset(options=reset_options)`.
        
        As for the init states kwargs, reset_options can be either a dictionnary, in this 
        case the same dict will be used for running all the episode or a list / tuple
        of dictionnaries with the same size as the `nb_episode` kwargs.
        
        .. code-block:: python
        
            import grid2op
            from gri2op.Runner import Runner
            from grid2op.Agent import RandomAgent

            env = grid2op.make("l2rpn_case14_sandbox")
            my_agent = RandomAgent(env.action_space)
            runner = Runner(**env.get_params_for_runner(), agentClass=None, agentInstance=my_agent)
            res = runner.run(nb_episode=2,
                             agent_seeds=[42, 43],
                             env_seeds=[0, 1],
                             reset_options={"init state": {"set_line_status": [(0, -1)]}}
                             )
            # same initial state will be used for the two epusode
        
            res2 = runner.run(nb_episode=2,
                              agent_seeds=[42, 43],
                              env_seeds=[0, 1],
                              reset_options=[{"init state": {"set_line_status": [(0, -1)]}},
                                             {"init state": {"set_line_status": [(1, -1)]}}]
                              ) 
            # two different initial states will be used: the first one for the 
            # first episode and the second one for the second
            
        .. note::
            In case of conflicting inputs, for example when you specify:
            
            .. code-block:: python

                runner.run(...,
                           init_states=XXX,
                           reset_options={"init state"=YYY}
                           )
            
            or 
            
            .. code-block:: python

                runner.run(...,
                           max_iter=XXX,
                           reset_options={"max step"=YYY}
                           )
            
            or 
            
            .. code-block:: python

                runner.run(...,
                           episode_id=XXX,
                           reset_options={"time serie id"=YYY}
                           )
                           
            Then: 1) a warning is issued to inform you that you might have
            done something wrong and 2) the value in `XXX` above (*ie* the
            value provided in the `runner.run` kwargs) is always used
            instead of the value `YYY` (*ie* the value present in the
            reset_options).
            
            In other words, the arguments of the `runner.run` have the
            priority over the arguments passed to the `reset_options`.
        
        .. danger::
            If you provide the key "time serie id" in one of the `reset_options` 
            dictionary, we recommend
            you do it for all `reset_options` otherwise you might not end up 
            computing the correct episodes.
                
        """
        if nb_episode < 0:
            raise RuntimeError("Impossible to run a negative number of scenarios.")

        if env_seeds is not None:
            if len(env_seeds) != nb_episode:
                raise RuntimeError(
                    'You want to compute "{}" run(s) but provide only "{}" different seeds '
                    "(environment)."
                    "".format(nb_episode, len(env_seeds))
                )

        if agent_seeds is not None:
            if len(agent_seeds) != nb_episode:
                raise RuntimeError(
                    'You want to compute "{}" run(s) but provide only "{}" different seeds (agent).'
                    "".format(nb_episode, len(agent_seeds))
                )

        if episode_id is not None:
            if len(episode_id) != nb_episode:
                raise RuntimeError(
                    'You want to compute "{}" run(s) but provide only "{}" different ids.'
                    "".format(nb_episode, len(episode_id))
                )

        if init_states is not None:
            if isinstance(init_states, (dict, BaseAction)):
                # user provided one initial state, I copy it to all 
                # evaluation
                init_states = [init_states.copy() for _ in range(nb_episode)]
            elif isinstance(init_states, (list, tuple, np.ndarray)):
                # user provided a list of initial states, it should match the
                # number of scenarios
                if len(init_states) != nb_episode:
                    raise RuntimeError(
                        'You want to compute "{}" run(s) but provide only "{}" different initial state.'
                        "".format(nb_episode, len(init_states))
                    )
                for i, el in enumerate(init_states):
                    if not isinstance(el, (dict, BaseAction)):
                        raise RuntimeError("When specifying `init_states` kwargs with a list (or a tuple) "
                                           "it should be a list (or a tuple) of dictionary or BaseAction. "
                                           f"You provided {type(el)} at position {i}.")
            else:
                raise RuntimeError("When using `init_state` in the runner, you should make sure to use "
                                   "either use dictionnary, grid2op actions or list / tuple of actions.")
        
        if reset_options is not None:
            if isinstance(reset_options, dict):
                for k in reset_options:
                    if not k in self.envClass.KEYS_RESET_OPTIONS:
                        raise RuntimeError("Wehn specifying `reset options` all keys of the dictionary should "
                                            "be compatible with the available reset options of your environment "
                                            f"class. You provided the key \"{k}\" for the provided dictionary but"
                                            f"possible keys are limited to {self.envClass.KEYS_RESET_OPTIONS}.")
                # user provided one initial state, I copy it to all 
                # evaluation
                reset_options = [reset_options.copy() for _ in range(nb_episode)]
            elif isinstance(reset_options, (list, tuple, np.ndarray)):
                # user provided a list ofreset_options, it should match the
                # number of scenarios
                if len(reset_options) != nb_episode:
                    raise RuntimeError(
                        'You want to compute "{}" run(s) but provide only "{}" different reset options.'
                        "".format(nb_episode, len(reset_options))
                    )
                for i, el in enumerate(reset_options):
                    if not isinstance(el, dict):
                        raise RuntimeError("When specifying `reset_options` kwargs with a list (or a tuple) "
                                           "it should be a list (or a tuple) of dictionary or BaseAction. "
                                           f"You provided {type(el)} at position {i}.")
                for i, el in enumerate(reset_options):
                    for k in el:
                        if not k in self.envClass.KEYS_RESET_OPTIONS:
                            raise RuntimeError("Wehn specifying `reset options` all keys of the dictionary should "
                                               "be compatible with the available reset options of your environment "
                                               f"class. You provided the key \"{k}\" for the {i}th dictionary but"
                                               f"possible keys are limited to {self.envClass.KEYS_RESET_OPTIONS}.")
            else:
                raise RuntimeError("When using `reset_options` in the runner, you should make sure to use "
                                   "either use dictionnary, grid2op actions or list / tuple of actions.")
            
        if max_iter is not None:
            max_iter = int(max_iter)

        if nb_episode == 0:
            res = []
        else:
            try:
                if nb_process <= 0:
                    raise RuntimeError("Impossible to run using less than 1 process.")
                self.__used = True
                if nb_process == 1:
                    self.logger.info("Sequential runner used.")
                    res = self._run_sequential(
                        nb_episode,
                        path_save=path_save,
                        pbar=pbar,
                        env_seeds=env_seeds,
                        max_iter=max_iter,
                        agent_seeds=agent_seeds,
                        episode_id=episode_id,
                        add_detailed_output=add_detailed_output,
                        add_nb_highres_sim=add_nb_highres_sim,
                        init_states=init_states,
                        reset_options=reset_options
                    )
                else:
                    if add_detailed_output and (_IS_WINDOWS or _IS_MACOS):
                        self.logger.warn(
                            "Parallel run are not fully supported on windows or macos when "
                            '"add_detailed_output" is True. So we decided '
                            "to fully deactivate them."
                        )
                        res = self._run_sequential(
                            nb_episode,
                            path_save=path_save,
                            pbar=pbar,
                            env_seeds=env_seeds,
                            max_iter=max_iter,
                            agent_seeds=agent_seeds,
                            episode_id=episode_id,
                            add_detailed_output=add_detailed_output,
                            add_nb_highres_sim=add_nb_highres_sim,
                            init_states=init_states,
                            reset_options=reset_options
                        )
                    else:
                        self.logger.info("Parallel runner used.")
                        res = self._run_parrallel(
                            nb_episode,
                            nb_process=nb_process,
                            path_save=path_save,
                            env_seeds=env_seeds,
                            max_iter=max_iter,
                            agent_seeds=agent_seeds,
                            episode_id=episode_id,
                            add_detailed_output=add_detailed_output,
                            add_nb_highres_sim=add_nb_highres_sim,
                            init_states=init_states,
                            reset_options=reset_options
                        )
            finally:
                self._clean_up()
        return res
