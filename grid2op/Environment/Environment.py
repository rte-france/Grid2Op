# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import os
import copy
import warnings
import numpy as np

from grid2op.dtypes import dt_float, dt_bool
from grid2op.Action import ActionSpace, BaseAction, TopologyAction, DontAct, CompleteAction
from grid2op.Exceptions import *
from grid2op.Observation import CompleteObservation, ObservationSpace, BaseObservation
from grid2op.Reward import FlatReward, RewardHelper, BaseReward
from grid2op.Rules import RulesChecker, AlwaysLegal, BaseRules
from grid2op.Backend import Backend
from grid2op.Chronics import ChronicsHandler
from grid2op.VoltageControler import ControlVoltageFromFile, BaseVoltageController
from grid2op.Environment.BaseEnv import BaseEnv
from grid2op.Opponent import BaseOpponent, NeverAttackBudget


class Environment(BaseEnv):
    """
    This class is the grid2op implementation of the "Environment" entity in the RL framework.

    Attributes
    ----------

    name: ``str``
        The name of the environment

    action_space: :class:`grid2op.Action.ActionSpace`
        Another name for :attr:`Environment.helper_action_player` for gym compatibility.

    observation_space:  :class:`grid2op.Observation.ObservationSpace`
        Another name for :attr:`Environment.helper_observation` for gym compatibility.

    reward_range: ``(float, float)``
        The range of the reward function

    metadata: ``dict``
        For gym compatibility, do not use

    spec: ``None``
        For Gym compatibility, do not use

    viewer: ``object``
        Used to display the powergrid. Currently not supported.

    """
    def __init__(self,
                 init_grid_path: str,
                 chronics_handler,
                 backend,
                 parameters,
                 name="unknown",
                 names_chronics_to_backend=None,
                 actionClass=TopologyAction,
                 observationClass=CompleteObservation,
                 rewardClass=FlatReward,
                 legalActClass=AlwaysLegal,
                 voltagecontrolerClass=ControlVoltageFromFile,
                 other_rewards={},
                 thermal_limit_a=None,
                 with_forecast=True,
                 epsilon_poly=1e-4,  # precision of the redispatching algorithm we don't recommend to go above 1e-4
                 tol_poly=1e-2,  # i need to compute a redispatching if the actual values are "more than tol_poly" the values they should be
                 opponent_action_class=DontAct,
                 opponent_class=BaseOpponent,
                 opponent_init_budget=0.,
                 opponent_budget_per_ts=0.,
                 opponent_budget_class=NeverAttackBudget,
                 opponent_attack_duration=0,
                 opponent_attack_cooldown=99999,
                 kwargs_opponent={},
                 _raw_backend_class=None
                 ):
        BaseEnv.__init__(self,
                         parameters=parameters,
                         thermal_limit_a=thermal_limit_a,
                         epsilon_poly=epsilon_poly,
                         tol_poly=tol_poly,
                         other_rewards=other_rewards,
                         with_forecast=with_forecast,
                         voltagecontrolerClass=voltagecontrolerClass,
                         opponent_action_class=opponent_action_class,
                         opponent_class=opponent_class,
                         opponent_budget_class=opponent_budget_class,
                         opponent_init_budget=opponent_init_budget,
                         opponent_budget_per_ts=opponent_budget_per_ts,
                         opponent_attack_duration=opponent_attack_duration,
                         opponent_attack_cooldown=opponent_attack_cooldown,
                         kwargs_opponent=kwargs_opponent)
        if name == "unknown":
            warnings.warn("It is NOT recommended to create an environment without \"make\" and EVEN LESS "
                          "to use an environment without a name")
        self.name = name

        # for gym compatibility (initialized below)
        self.action_space = None
        self.observation_space = None
        self.reward_range = None
        self.viewer = None
        self.metadata = None
        self.spec = None

        if _raw_backend_class is None:
            self._raw_backend_class = type(backend)
        else:
            self._raw_backend_class = _raw_backend_class

        # for plotting
        self._init_backend(init_grid_path, chronics_handler, backend,
                           names_chronics_to_backend, actionClass, observationClass,
                           rewardClass, legalActClass)

    def _init_backend(self,
                      init_grid_path, chronics_handler, backend,
                      names_chronics_to_backend, actionClass, observationClass,
                      rewardClass, legalActClass):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Create a proper and valid environment.
        """

        if not isinstance(rewardClass, type):
            raise Grid2OpException("Parameter \"rewardClass\" used to build the Environment should be a type (a class) "
                                   "and not an object (an instance of a class). "
                                   "It is currently \"{}\"".format(type(rewardClass)))
        if not issubclass(rewardClass, BaseReward):
            raise Grid2OpException("Parameter \"rewardClass\" used to build the Environment should derived form "
                                   "the grid2op.BaseReward class, type provided is \"{}\"".format(type(rewardClass)))
        self._rewardClass = rewardClass
        self._actionClass = actionClass
        self._observationClass = observationClass

        # backend
        self._init_grid_path = os.path.abspath(init_grid_path)

        if not isinstance(backend, Backend):
            raise Grid2OpException( "Parameter \"backend\" used to build the Environment should derived form the "
                                    "grid2op.Backend class, type provided is \"{}\"".format(type(backend)))
        self.backend = backend
        # all the above should be done in this exact order, otherwise some weird behaviour might occur
        # this is due to the class attribute
        self.backend.set_env_name(self.name)
        self.backend.load_grid(self._init_grid_path)  # the real powergrid of the environment
        self.backend.load_redispacthing_data(os.path.split(self._init_grid_path)[0])
        self.backend.load_grid_layout(os.path.split(self._init_grid_path)[0])
        self.backend.assert_grid_correct()
        self._has_been_initialized()  # really important to include this piece of code! and just here after the
        # backend has loaded everything
        self._line_status = np.ones(shape=self.n_line, dtype=dt_bool)

        if self._thermal_limit_a is None:
            self._thermal_limit_a = self.backend.thermal_limit_a.astype(dt_float)
        else:
            self.backend.set_thermal_limit(self._thermal_limit_a.astype(dt_float))

        *_, tmp = self.backend.generators_info()

        # rules of the game
        if not isinstance(legalActClass, type):
            raise Grid2OpException("Parameter \"legalActClass\" used to build the Environment should be a type "
                                   "(a class) and not an object (an instance of a class). "
                                   "It is currently \"{}\"".format(type(legalActClass)))
        if not issubclass(legalActClass, BaseRules):
            raise Grid2OpException(
                "Parameter \"legalActClass\" used to build the Environment should derived form the "
                "grid2op.BaseRules class, type provided is \"{}\"".format(
                    type(legalActClass)))
        self._game_rules = RulesChecker(legalActClass=legalActClass)
        self._legalActClass = legalActClass

        # action helper
        if not isinstance(actionClass, type):
            raise Grid2OpException("Parameter \"actionClass\" used to build the Environment should be a type (a class) "
                                   "and not an object (an instance of a class). "
                                   "It is currently \"{}\"".format(type(legalActClass)))
        if not issubclass(actionClass, BaseAction):
            raise Grid2OpException(
                "Parameter \"actionClass\" used to build the Environment should derived form the "
                "grid2op.BaseAction class, type provided is \"{}\"".format(
                    type(actionClass)))

        if not isinstance(observationClass, type):
            raise Grid2OpException("Parameter \"actionClass\" used to build the Environment should be a type (a class) "
                                   "and not an object (an instance of a class). "
                                   "It is currently \"{}\"".format(type(legalActClass)))
        if not issubclass(observationClass, BaseObservation):
            raise Grid2OpException(
                "Parameter \"observationClass\" used to build the Environment should derived form the "
                "grid2op.BaseObservation class, type provided is \"{}\"".format(
                    type(observationClass)))

        # action affecting the grid that will be made by the agent
        self._helper_action_class = ActionSpace.init_grid(gridobj=self.backend)
        self._helper_action_player = self._helper_action_class(gridobj=self.backend,
                                                               actionClass=actionClass,
                                                               legal_action=self._game_rules.legal_action)

        # action that affect the grid made by the environment.
        self._helper_action_env = self._helper_action_class(gridobj=self.backend,
                                                            actionClass=CompleteAction,
                                                            legal_action=self._game_rules.legal_action)
        self._helper_observation_class = ObservationSpace.init_grid(gridobj=self.backend)
        self._helper_observation = self._helper_observation_class(gridobj=self.backend,
                                                                  observationClass=observationClass,
                                                                  rewardClass=rewardClass,
                                                                  env=self)
        # handles input data
        if not isinstance(chronics_handler, ChronicsHandler):
            raise Grid2OpException(
                "Parameter \"chronics_handler\" used to build the Environment should derived form the "
                "grid2op.ChronicsHandler class, type provided is \"{}\"".format(
                    type(chronics_handler)))
        self.chronics_handler = chronics_handler
        self.chronics_handler.initialize(self.name_load, self.name_gen,
                                         self.name_line, self.name_sub,
                                         names_chronics_to_backend=names_chronics_to_backend)
        self.names_chronics_to_backend = names_chronics_to_backend

        # test to make sure the backend is consistent with the chronics generator
        self.chronics_handler.check_validity(self.backend)

        # reward function
        self._reward_helper = RewardHelper(self._rewardClass)
        self._reward_helper.initialize(self)
        for k, v in self.other_rewards.items():
            v.initialize(self)

        # controler for voltage
        if not issubclass(self._voltagecontrolerClass, BaseVoltageController):
            raise Grid2OpException("Parameter \"voltagecontrolClass\" should derive from \"ControlVoltageFromFile\".")

        self._voltage_controler = self._voltagecontrolerClass(gridobj=self.backend,
                                                              controler_backend=self.backend)

        # create the opponent
        # At least the 3 following attributes should be set before calling _create_opponent
        self._create_opponent()

        # performs one step to load the environment properly (first action need to be taken at first time step after
        # first injections given)
        self._reset_maintenance()
        self._reset_redispatching()
        do_nothing = self._helper_action_env({})
        *_, fail_to_start, info = self.step(do_nothing)
        if fail_to_start:
            raise Grid2OpException("Impossible to initialize the powergrid, the powerflow diverge at iteration 0. "
                                   "Available information are: {}".format(info))

        # test the backend returns object of the proper size
        self.backend.assert_grid_correct_after_powerflow()

        # for gym compatibility
        self.action_space = self._helper_action_player  # this should be an action !!!
        self.observation_space = self._helper_observation  # this return an observation.
        self.reward_range = self._reward_helper.range()
        self.viewer = None
        self.viewer_fig = None

        self.metadata = {'render.modes': []}
        self.spec = None

        self.current_reward = self.reward_range[0]
        self.done = False

        # reset everything to be consistent
        self._reset_vectors_and_timings()

    def _voltage_control(self, agent_action, prod_v_chronics):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Update the environment action "action_env" given a possibly new voltage setpoint for the generators. This
        function can be overide for a more complex handling of the voltages.

        It must update (if needed) the voltages of the environment action :attr:`BaseEnv.env_modification`

        Parameters
        ----------
        agent_action: :class:`grid2op.Action.Action`
            The action performed by the player (or do nothing is player action were not legal or ambiguous)

        prod_v_chronics: ``numpy.ndarray`` or ``None``
            The voltages that has been specified in the chronics

        """
        volt_control_act = self._voltage_controler.fix_voltage(self.current_obs,
                                                               agent_action,
                                                               self._env_modification,
                                                               prod_v_chronics)
        return volt_control_act

    def set_chunk_size(self, new_chunk_size):
        """
        For an efficient data pipeline, it can be usefull to not read all part of the input data
        (for example for load_p, prod_p, load_q, prod_v). Grid2Op support the reading of large chronics by "chunk"
        of given size.

        Reading data in chunk can also reduce the memory footprint, useful in case of multiprocessing environment while
        large chronics.

        It is critical to set a small chunk_size in case of training machine learning algorithm (reinforcement
        learning agent) at the beginning when the agent performs poorly, the software might spend most of its time
        loading the data.

        **NB** this has no effect if the chronics does not support this feature.

        **NB** The environment need to be **reset** for this to take effect (it won't affect the chronics already
        loaded)

        Parameters
        ----------
        new_chunk_size: ``int`` or ``None``
            The new chunk size (positive integer)

        Examples
        ---------
        Here is an example on how to use this function

        .. code-block:: python

            import grid2op

            # I create an environment
            env = grid2op.make("rte_case5_example", test=True)
            env.set_chunk_size(100)
            # and now data will be read from the hard drive 100 time steps per 100 time steps
            # instead of the whole episode at once.

        """
        if new_chunk_size is None:
            self.chronics_handler.set_chunk_size(new_chunk_size)
            return

        try:
            new_chunk_size = int(new_chunk_size)
        except Exception as e:
            raise Grid2OpException("Impossible to set the chunk size. It should be convertible a integer, and not"
                                   "{}".format(new_chunk_size))

        if new_chunk_size <= 0:
            raise Grid2OpException("Impossible to read less than 1 data at a time. Please make sure \"new_chunk_size\""
                                   "is a positive integer.")

        self.chronics_handler.set_chunk_size(new_chunk_size)

    def set_id(self, id_):
        """
        Set the id that will be used at the next call to :func:`Environment.reset`.

        **NB** this has no effect if the chronics does not support this feature.

        **NB** The environment need to be **reset** for this to take effect.

        Parameters
        ----------
        id_: ``int``
            the id of the chronics used.

        Examples
        --------
        Here an example that will loop 10 times through the same chronics (always using the same injection then):

        .. code-block:: python

            import grid2op
            from grid2op import make
            from grid2op.BaseAgent import DoNothingAgent

            env = make("rte_case14_realistic")  # create an environment
            agent = DoNothingAgent(env.action_space)  # create an BaseAgent

            for i in range(10):
                env.set_id(0)  # tell the environment you simply want to use the chronics with ID 0
                obs = env.reset()  # it is necessary to perform a reset
                reward = env.reward_range[0]
                done = False
                while not done:
                    act = agent.act(obs, reward, done)
                    obs, reward, done, info = env.step(act)

        And here you have an example on how you can loop through the scenarios in a given order:

        .. code-block:: python

            import grid2op
            from grid2op import make
            from grid2op.BaseAgent import DoNothingAgent

            env = make("rte_case14_realistic")  # create an environment
            agent = DoNothingAgent(env.action_space)  # create an BaseAgent
            scenario_order = [1,2,3,4,5,10,8,6,5,7,78, 8]
            for id_ in scenario_order:
                env.set_id(id_)  # tell the environment you simply want to use the chronics with ID 0
                obs = env.reset()  # it is necessary to perform a reset
                reward = env.reward_range[0]
                done = False
                while not done:
                    act = agent.act(obs, reward, done)
                    obs, reward, done, info = env.step(act)

        """
        try:
            id_ = int(id_)
        except:
            raise EnvError("the \"id_\" parameters should be convertible to integer and not be of type {}"
                           "".format(type(id_)))

        self.chronics_handler.tell_id(id_-1)

    def attach_renderer(self, graph_layout=None):
        """
        This function will attach a renderer, necessary to use for plotting capabilities.

        Parameters
        ----------
        graph_layout: ``dict``
            Here for backward compatibility. Currently not used.

            If you want to set a specific layout call :func:`BaseEnv.attach_layout`

            If ``None`` this class will use the default substations layout provided when the environment was created.
            Otherwise it will use the data provided.

        Examples
        ---------
        Here is how to use the function

        .. code-block:: python

            import grid2op

            # create the environment
            env = grid2op.make()

            if False:
                # if you want to change the default layout of the powergrid
                # assign coordinates (0., 0.) to all substations (this is a dummy thing to do here!)
                layout = {sub_name: (0., 0.) for sub_name in env.name_sub}
                env.attach_layout(layout)
                # NB again, this code will make everything look super ugly !!!! Don't change the
                # default layout unless you have a reason to.

            # and if you want to use the renderer
            env.attach_renderer()

            # and now you can "render" (plot) the state of the grid
            obs = env.reset()
            done = False
            reward = env.reward_range[0]
            while not done:
                env.render()
                action = agent.act(obs, reward, done)
                obs, reward, done, info = env.step(action)

        """
        # Viewer already exists: skip
        if self.viewer is not None:
            return

        # Do we have the dependency
        try:
            from grid2op.PlotGrid import PlotMatplot
        except ImportError:
            err_msg = "Cannot attach renderer: missing dependency\n" \
                      "Please install matplotlib or run pip install grid2op[optional]"
            raise Grid2OpException(err_msg) from None

        self.viewer = PlotMatplot(self._helper_observation)
        self.viewer_fig = None
        # Set renderer modes
        self.metadata = {'render.modes': ["human", "silent"]}

    def __str__(self):
        return '<{} instance named {}>'.format(type(self).__name__, self.name)
        # TODO be closer to original gym implementation

    def reset_grid(self):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            This is automatically called when using `env.reset`

        Reset the backend to a clean state by reloading the powergrid from the hard drive.
        This might takes some time.

        If the thermal has been modified, it also modify them into the new backend.

        """
        self.backend.reset(self._init_grid_path)  # the real powergrid of the environment
        self.backend.assert_grid_correct()

        if self._thermal_limit_a is not None:
            self.backend.set_thermal_limit(self._thermal_limit_a.astype(dt_float))

        self._backend_action = self._backend_action_class()
        do_nothing = self._helper_action_env({})
        *_, fail_to_start, info = self.step(do_nothing)
        if fail_to_start:
            raise Grid2OpException("Impossible to initialize the powergrid, the powerflow diverge at iteration 0. "
                                   "Available information are: {}".format(info))

    def add_text_logger(self, logger=None):
        """
        Add a text logger to this  :class:`Environment`

        Logging is for now an incomplete feature, really incomplete (not used)

        Parameters
        ----------
        logger:
           The logger to use

        """
        self.logger = logger
        return self

    def reset(self):
        """
        Reset the environment to a clean state.
        It will reload the next chronics if any. And reset the grid to a clean state.

        This triggers a full reloading of both the chronics (if they are stored as files) and of the powergrid,
        to ensure the episode is fully over.

        This method should be called only at the end of an episode.

        Examples
        --------
        The standard "gym loop" can be done with the following code:

        .. code-block:: python

            import grid2op

            # create the environment
            env = grid2op.make()

            # and now you can "render" (plot) the state of the grid
            obs = env.reset()
            done = False
            reward = env.reward_range[0]
            while not done:
                action = agent.act(obs, reward, done)
                obs, reward, done, info = env.step(action)
        """
        super().reset()
        self.chronics_handler.next_chronics()
        self.chronics_handler.initialize(self.backend.name_load, self.backend.name_gen,
                                         self.backend.name_line, self.backend.name_sub,
                                         names_chronics_to_backend=self.names_chronics_to_backend)
        self.current_obs = None
        self._env_modification = None
        self._reset_maintenance()
        self._reset_redispatching()
        self._reset_vectors_and_timings()  # it need to be done BEFORE to prevent cascading failure when there has been
        self.reset_grid()
        if self.viewer_fig is not None:
            del self.viewer_fig
            self.viewer_fig = None
        # if True, then it will not disconnect lines above their thermal limits
        self._reset_vectors_and_timings()  # and it needs to be done AFTER to have proper timings at tbe beginning

        # reset the opponent
        self._oppSpace.reset()
        return self.get_obs()

    def render(self, mode='human'):
        """
        Render the state of the environment on the screen, using matplotlib
        Also returns the Matplotlib figure

        Examples
        --------
        Rendering need first to define a "renderer" which can be done with the following code:

        .. code-block:: python

            import grid2op

            # create the environment
            env = grid2op.make()

            # if you want to use the renderer
            env.attach_renderer()

            # and now you can "render" (plot) the state of the grid
            obs = env.reset()
            done = False
            reward = env.reward_range[0]
            while not done:
                env.render()  # this piece of code plot the grid
                action = agent.act(obs, reward, done)
                obs, reward, done, info = env.step(action)
        """
        # Try to create a plotter instance
        # Does nothing if viewer exists
        # Raises if matplot is not installed
        self.attach_renderer()
        
        # Check mode is correct
        if mode not in self.metadata["render.modes"]:
            err_msg = "Renderer mode \"{}\" not supported. Available modes are {}."
            raise Grid2OpException(err_msg.format(mode, self.metadata["render.modes"]))

        # Render the current observation
        fig = self.viewer.plot_obs(self.current_obs, figure=self.viewer_fig, redraw=True)

        # First time show for human mode
        if self.viewer_fig is None and mode == "human":
            fig.show()
        else: # Update the figure content
            fig.canvas.draw()

        # Store to re-use the figure
        self.viewer_fig = fig
        # Return the figure in case it needs to be saved/used
        return self.viewer_fig

    def copy(self):
        """
        Performs a deep copy of the environment

        Unless you have a reason to, it is not advised to make copy of an Environment.

        Examples
        --------
        It should be used as follow:

        .. code-block:: python

            import grid2op
            env = grid2op.make()
            cpy_of_env = env.copy()


        """
        tmp_backend = self.backend
        self.backend = None

        tmp_obs_space = self._helper_observation
        self.observation_space = None
        self._helper_observation = None

        obs_tmp = self.current_obs
        self.current_obs = None

        volt_cont = self._voltage_controler
        self._voltage_controler = None

        res = copy.deepcopy(self)
        res.backend = tmp_backend.copy()
        res._helper_observation = tmp_obs_space.copy()
        res.observation_space = res._helper_observation
        res.current_obs = obs_tmp.copy()
        res._voltage_controler = volt_cont.copy()

        if self._thermal_limit_a is not None:
            res.backend.set_thermal_limit(self._thermal_limit_a)
        self.backend = tmp_backend
        self.observation_space = tmp_obs_space
        self._helper_observation = tmp_obs_space
        self.current_obs = obs_tmp
        self._voltage_controler = volt_cont
        return res

    def get_kwargs(self, with_backend=True):
        """
        This function allows to make another Environment with the same parameters as the one that have been used
        to make this one.

        This is useful especially in cases where Environment is not pickable (for example if some non pickable c++
        code are used) but you still want to make parallel processing using "MultiProcessing" module. In that case,
        you can send this dictionary to each child process, and have each child process make a copy of ``self``

        **NB** This function should not be used to make a copy of an environment. Prefer using :func:`Environment.copy`
        for such purpose.


        Returns
        -------
        res: ``dict``
            A dictionary that helps build an environment like ``self`` (which is NOT a copy of self) but rather
            an instance of an environment with the same properties.

        Examples
        --------
        It should be used as follow:

        .. code-block:: python

            import grid2op
            from grid2op.Environment import Environment
            env = grid2op.make()  # create the environment of your choice
            copy_of_env = Environment(**env.get_kwargs())
            # And you can use this one as you would any other environment.
            # NB this is not a "proper" copy. for example it will not be at the same step, it will be possible
            # seeded with a different seed.
            # use `env.copy()` to make a proper copy of an environment.

        """
        res = {}
        res["init_grid_path"] = self._init_grid_path
        res["chronics_handler"] = copy.deepcopy(self.chronics_handler)
        if with_backend:
            res["backend"] = self.backend.copy()
        res["parameters"] = copy.deepcopy(self.parameters)
        res["names_chronics_to_backend"] = copy.deepcopy(self.names_chronics_to_backend)
        res["actionClass"] = self._actionClass
        res["observationClass"] = self._observationClass
        res["rewardClass"] = self._rewardClass
        res["legalActClass"] = self._legalActClass
        res["epsilon_poly"] = self._epsilon_poly
        res["tol_poly"] = self._tol_poly
        res["thermal_limit_a"] = self._thermal_limit_a
        res["voltagecontrolerClass"] = self._voltagecontrolerClass
        res["other_rewards"] = {k: v.rewardClass for k, v in self.other_rewards.items()}
        res["name"] = self.name
        res["_raw_backend_class"] = self._raw_backend_class
        res["with_forecast"] = self.with_forecast

        res["opponent_action_class"] = self._opponent_action_class
        res["opponent_class"] = self._opponent_class
        res["opponent_init_budget"] = self._opponent_init_budget
        res["opponent_budget_per_ts"] = self._opponent_budget_per_ts
        res["opponent_budget_class"] = self._opponent_budget_class
        res["opponent_attack_duration"] = self._opponent_attack_duration
        res["opponent_attack_cooldown"] = self._opponent_attack_cooldown
        res["kwargs_opponent"] = self._kwargs_opponent
        return res

    def get_params_for_runner(self):
        """
        This method is used to initialize a proper :class:`grid2op.Runner.Runner` to use this specific environment.

        Examples
        --------
        It should be used as followed:

        .. code-block:: python

            import grid2op
            from grid2op.Runner import Runner
            from grid2op.Agent import DoNothingAgent  # for example
            env = grid2op.make()  # create the environment of your choice

            # create the proper runner
            runner = Runner(**env.get_params_for_runner(), agentClass=DoNothingAgent)

            # now you can run
            runner.run(nb_episode=1)  # run for 1 episode

        """
        res = {}
        res["init_grid_path"] = self._init_grid_path
        res["path_chron"] = self.chronics_handler.path
        res["parameters_path"] = self.parameters.to_dict()
        res["names_chronics_to_backend"] = self.names_chronics_to_backend
        res["actionClass"] = self._actionClass
        res["observationClass"] = self._observationClass
        res["rewardClass"] = self._rewardClass
        res["legalActClass"] = self._legalActClass
        res["envClass"] = Environment
        res["gridStateclass"] = self.chronics_handler.chronicsClass
        res["backendClass"] = self._raw_backend_class
        res["verbose"] = False
        dict_ = copy.deepcopy(self.chronics_handler.kwargs)
        if 'path' in dict_:
            # path is handled elsewhere
            del dict_["path"]
        res["gridStateclass_kwargs"] = dict_
        res["thermal_limit_a"] = self._thermal_limit_a
        res["voltageControlerClass"] = self._voltagecontrolerClass
        res["other_rewards"] = {k: v.rewardClass for k, v in self.other_rewards.items()}
        res["grid_layout"] = self.grid_layout
        res["name_env"] = self.name

        res["opponent_action_class"] = self._opponent_action_class
        res["opponent_class"] = self._opponent_class
        res["opponent_init_budget"] = self._opponent_init_budget
        res["opponent_budget_per_ts"] = self._opponent_budget_per_ts
        res["opponent_budget_class"] = self._opponent_budget_class
        res["opponent_attack_duration"] = self._opponent_attack_duration
        res["opponent_attack_cooldown"] = self._opponent_attack_cooldown
        res["opponent_kwargs"] = self._kwargs_opponent
        return res
