"""
This module defines the :class:`Environment` the higher level representation of the world with which an
:class:`grid2op.Agent` will interact.

The environment receive an :class:`grid2op.Action.Action` from the :class:`grid2op.Agent.Agent` in the
:func:`Environment.step`
and returns an
:class:`grid2op.Observation.Observation` that the :class:`grid2op.Agent.Agent` will use to perform the next action.

An environment is better used inside a :class:`grid2op.Runner.Runner`, mainly because runners abstract the interaction
between environment and agent, and ensure the environment are properly reset after each episode.

It is however totally possible to use as any gym Environment.

Example (adapted from gym documentation available at
`gym random_agent.py <https://github.com/openai/gym/blob/master/examples/agents/random_agent.py>`_ ):

.. code-block:: python

    import grid2op
    from grid2op.Agent import DoNothingAgent
    env = grid2op.make()
    agent = DoNothingAgent(env.action_space)
    env.seed(0)
    episode_count = 100
    reward = 0
    done = False
    total_reward = 0
    for i in range(episode_count):
        ob = env.reset()
        while True:
           action = agent.act(ob, reward, done)
           ob, reward, done, _ = env.step(action)
           total_reward += reward
           if done:
               # in this case the episode is over
               break

    # Close the env and write monitor result info to disk
    env.close()
    print("The total reward was {:.2f}".format(total_reward))

"""

import numpy as np
import os
import copy

try:
    from .Space import GridObjects
    from .BasicEnv import _BasicEnv
    from .Action import HelperAction, Action, TopologyAction
    from .Exceptions import *
    from .Observation import CompleteObservation, ObservationHelper, Observation
    from .Reward import FlatReward, RewardHelper, Reward
    from .GameRules import GameRules, AllwaysLegal, LegalAction
    from .Parameters import Parameters
    from .Backend import Backend
    from .ChronicsHandler import ChronicsHandler
    from .PlotPyGame import Renderer
    from .VoltageControler import ControlVoltageFromFile
except (ModuleNotFoundError, ImportError):
    from Space import GridObjects
    from BasicEnv import _BasicEnv
    from Action import HelperAction, Action, TopologyAction
    from Exceptions import *
    from Observation import CompleteObservation, ObservationHelper, Observation
    from Reward import FlatReward, RewardHelper, Reward
    from GameRules import GameRules, AllwaysLegal, LegalAction
    from Parameters import Parameters
    from Backend import Backend
    from ChronicsHandler import ChronicsHandler
    from PlotPyGame import Renderer
    from VoltageControler import ControlVoltageFromFile

import pdb


# TODO code "start from a given time step" -> link to the "skip" method of GridValue

# TODO have a viewer / renderer now

class Environment(_BasicEnv):
    """

    Attributes
    ----------
    logger: ``logger``
        Use to store some information (currently in beta status)

    time_stamp: ``datetime.time``
        Current time of the chronics

    nb_time_step: ``int``
        Number of time steps played this episode

    parameters: :class:`grid2op.Parameters.Parameters`
        Parameters used for the game

    rewardClass: ``type``
        Type of reward used. Should be a subclass of :class:`grid2op.Reward.Reward`

    init_grid_path: ``str``
        The path where the description of the powergrid is located.

    backend: :class:`grid2op.Backend.Backend`
        The backend used to compute powerflows and cascading failures.

    game_rules: :class:`grid2op.GameRules.GameRules`
        The rules of the game (define which actions are legal and which are not)

    helper_action_player: :class:`grid2op.Action.HelperAction`
        Helper used to manipulate more easily the actions given to / provided by the :class:`grid2op.Agent` (player)

    helper_action_env: :class:`grid2op.Action.HelperAction`
        Helper used to manipulate more easily the actions given to / provided by the environment to the backend.

    helper_observation: :class:`grid2op.Observation.ObservationHelper`
        Helper used to generate the observation that will be given to the :class:`grid2op.Agent`

    current_obs: :class:`grid2op.Observation.Observation`
        The current observation (or None if it's not intialized)

    no_overflow_disconnection: ``bool``
        Whether or not cascading failures are computed or not (TRUE = the powerlines above their thermal limits will
        not be disconnected). This is initialized based on the attribute
        :attr:`grid2op.Parameters.Parameters.NO_OVERFLOW_DISCONNECTION`.

    timestep_overflow: ``numpy.ndarray``, dtype: int
        Number of consecutive timesteps each powerline has been on overflow.

    nb_timestep_overflow_allowed: ``numpy.ndarray``, dtype: int
        Number of consecutive timestep each powerline can be on overflow. It is usually read from
        :attr:`grid2op.Parameters.Parameters.NB_TIMESTEP_POWERFLOW_ALLOWED`.

    hard_overflow_threshold: ``float``
        Number of timestep before an :class:`grid2op.Agent.Agent` can reconnet a powerline that has been disconnected
        by the environment due to an overflow.

    env_dc: ``bool``
        Whether the environment computes the powerflow using the DC approximation or not. It is usually read from
        :attr:`grid2op.Parameters.Parameters.ENV_DC`.

    chronics_handler: :class:`grid2op.ChronicsHandler.ChronicsHandler`
        Helper to get the modification of each time step during the episode.

    names_chronics_to_backend: ``dict``
        Configuration file used to associated the name of the objects in the backend
        (both extremities of powerlines, load or production for
        example) with the same object in the data (:attr:`Environment.chronics_handler`). The idea is that, usually
        data generation comes from a different software that does not take into account the powergrid infrastructure.
        Hence, the same "object" can have a different name. This mapping is present to avoid the need to rename
        the "object" when providing data. A more detailed description is available at
        :func:`grid2op.ChronicsHandler.GridValue.initialize`.

    reward_helper: :class:`grid2p.Reward.RewardHelper`
        Helper that is called to compute the reward at each time step.

    action_space: :class:`grid2op.Action.HelperAction`
        Another name for :attr:`Environment.helper_action_player` for gym compatibility.

    observation_space:  :class:`grid2op.Observation.ObservationHelper`
        Another name for :attr:`Environment.helper_observation` for gym compatibility.

    reward_range: ``(float, float)``
        The range of the reward function

    metadata: ``dict``
        For gym compatibility, do not use

    spec: ``None``
        For Gym compatibility, do not use

    viewer: ``object``
        Used to display the powergrid. Currently not supported.

    env_modification: :class:`grid2op.Action.Action`
        Representation of the actions of the environment for the modification of the powergrid.

    current_reward: ``float``
        The reward of the current time step

    TODO update with maintenance, hazards etc. see below
    # store actions "cooldown"
    times_before_line_status_actionable
    max_timestep_line_status_deactivated
    times_before_topology_actionable
    max_timestep_topology_deactivated
    time_next_maintenance
    duration_next_maintenance
    hard_overflow_threshold
    time_remaining_before_reconnection

    # redispacthing
    target_dispatch
    actual_dispatch

    gen_activeprod_t:
        Should be initialized at 0. for "step" to properly recognize it's the first time step of the game

    """
    def __init__(self,
                 init_grid_path: str,
                 chronics_handler,
                 backend,
                 parameters,
                 names_chronics_to_backend=None,
                 actionClass=TopologyAction,
                 observationClass=CompleteObservation,
                 rewardClass=FlatReward,
                 legalActClass=AllwaysLegal,
                 voltagecontrolerClass=ControlVoltageFromFile,
                 thermal_limit_a=None,
                 epsilon_poly=1e-2,
                 tol_poly=1e-6
                 ):
        """
        Initialize the environment. See the descirption of :class:`grid2op.Environment.Environment` for more information.

        Parameters
        ----------
        init_grid_path: ``str``
            Used to initailize :attr:`Environment.init_grid_path`

        chronics_handler
        backend
        parameters
        names_chronics_to_backend
        actionClass
        observationClass
        rewardClass
        legalActClass
        """
        # TODO documentation!!

        _BasicEnv.__init__(self,
                          parameters=parameters,
                          thermal_limit_a=thermal_limit_a,
                          epsilon_poly=epsilon_poly,
                          tol_poly=tol_poly)
        # the voltage controler
        self.voltagecontrolerClass = voltagecontrolerClass
        self.voltage_controler = None

        # for gym compatibility (initialized below)
        self.action_space = None
        self.observation_space = None
        self.reward_range = None
        self.viewer = None
        self.metadata = None
        self.spec = None

        # for plotting
        self.graph_layout = None
        self.init_backend(init_grid_path, chronics_handler, backend,
                          names_chronics_to_backend, actionClass, observationClass,
                          rewardClass, legalActClass)

    def init_backend(self,
                     init_grid_path, chronics_handler, backend,
                     names_chronics_to_backend, actionClass, observationClass,
                     rewardClass, legalActClass):

        if not isinstance(rewardClass, type):
            raise Grid2OpException("Parameter \"rewardClass\" used to build the Environment should be a type (a class) "
                                   "and not an object (an instance of a class). "
                                   "It is currently \"{}\"".format(type(rewardClass)))
        if not issubclass(rewardClass, Reward):
            raise Grid2OpException(
                "Parameter \"rewardClass\" used to build the Environment should derived form the grid2op.Reward class, "
                "type provided is \"{}\"".format(
                    type(rewardClass)))
        self.rewardClass = rewardClass
        self.actionClass = actionClass
        self.observationClass = observationClass

        # backend
        self.init_grid_path = os.path.abspath(init_grid_path)

        if not isinstance(backend, Backend):
            raise Grid2OpException(
                "Parameter \"backend\" used to build the Environment should derived form the grid2op.Backend class, "
                "type provided is \"{}\"".format(
                    type(backend)))
        self.backend = backend
        self.backend.load_grid(self.init_grid_path)  # the real powergrid of the environment

        self.backend.load_redispacthing_data(os.path.split(self.init_grid_path)[0])
        self.backend.assert_grid_correct()
        self.init_grid(backend)
        self._has_been_initialized()
        if self._thermal_limit_a is None:
            self._thermal_limit_a = self.backend.thermal_limit_a
        else:
            self.backend.set_thermal_limit(self._thermal_limit_a)

        *_, tmp = self.backend.generators_info()

        # rules of the game
        if not isinstance(legalActClass, type):
            raise Grid2OpException("Parameter \"legalActClass\" used to build the Environment should be a type "
                                   "(a class) and not an object (an instance of a class). "
                                   "It is currently \"{}\"".format(type(legalActClass)))
        if not issubclass(legalActClass, LegalAction):
            raise Grid2OpException(
                "Parameter \"legalActClass\" used to build the Environment should derived form the "
                "grid2op.LegalAction class, type provided is \"{}\"".format(
                    type(legalActClass)))
        self.game_rules = GameRules(legalActClass=legalActClass)
        self.legalActClass = legalActClass

        # action helper
        if not isinstance(actionClass, type):
            raise Grid2OpException("Parameter \"actionClass\" used to build the Environment should be a type (a class) "
                                   "and not an object (an instance of a class). "
                                   "It is currently \"{}\"".format(type(legalActClass)))
        if not issubclass(actionClass, Action):
            raise Grid2OpException(
                "Parameter \"actionClass\" used to build the Environment should derived form the "
                "grid2op.Action class, type provided is \"{}\"".format(
                    type(actionClass)))

        if not isinstance(observationClass, type):
            raise Grid2OpException("Parameter \"actionClass\" used to build the Environment should be a type (a class) "
                                   "and not an object (an instance of a class). "
                                   "It is currently \"{}\"".format(type(legalActClass)))
        if not issubclass(observationClass, Observation):
            raise Grid2OpException(
                "Parameter \"observationClass\" used to build the Environment should derived form the "
                "grid2op.Observation class, type provided is \"{}\"".format(
                    type(observationClass)))

        # action affecting the grid that will be made by the agent
        self.helper_action_player = HelperAction(gridobj=self.backend,
                                                 actionClass=actionClass,
                                                 legal_action=self.game_rules.legal_action)

        # action that affect the grid made by the environment.
        self.helper_action_env = HelperAction(gridobj=self.backend,
                                              actionClass=Action,
                                              legal_action=self.game_rules.legal_action)

        self.helper_observation = ObservationHelper(gridobj=self.backend,
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
        self.reward_helper = RewardHelper(rewardClass=rewardClass)
        self.reward_helper.initialize(self)

        # controler for voltage
        if not issubclass(self.voltagecontrolerClass, ControlVoltageFromFile):
            raise Grid2OpException("Parameter \"voltagecontrolClass\" should derive from \"ControlVoltageFromFile\".")

        self.voltage_controler = self.voltagecontrolerClass(gridobj=self.backend,
                                                            controler_backend=self.backend)

        # performs one step to load the environment properly (first action need to be taken at first time step after
        # first injections given)
        self._reset_maintenance()
        do_nothing = self.helper_action_env({})
        *_, fail_to_start, _ = self.step(do_nothing)
        if fail_to_start:
            raise Grid2OpException("Impossible to initialize the powergrid, the powerflow diverge at iteration 0.")

        # test the backend returns object of the proper size
        self.backend.assert_grid_correct_after_powerflow()

        # for gym compatibility
        self.action_space = self.helper_action_player  # this should be an action !!!
        self.observation_space = self.helper_observation  # this return an observation.
        self.reward_range = self.reward_helper.range()
        self.viewer = None

        self.metadata = {'render.modes': ["human", "rgb_array"]}
        self.spec = None

        self.current_reward = self.reward_range[0]
        self.done = False
        self._reset_vectors_and_timings()

    def _voltage_control(self, agent_action, prod_v_chronics):
        """
        Update the environment action "action_env" given a possibly new voltage setpoint for the generators. This
        function can be overide for a more complex handling of the voltages.

        It mush update (if needed) the voltages of the environment action :attr:`BasicEnv.env_modification`

        Parameters
        ----------
        agent_action: :class:`grid2op.Action.Action`
            The action performed by the player (or do nothing is player action were not legal or ambiguous)

        prod_v_chronics: ``numpy.ndarray`` or ``None``
            The voltages that has been specified in the chronics

        """
        self.env_modification += self.voltage_controler.fix_voltage(self.current_obs,
                                                                    agent_action,
                                                                    self.env_modification,
                                                                    prod_v_chronics)

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

        **NB** this has no effect if the chronics does not support this feature. TODO see xxx for more information

        **NB** The environment need to be **reset** for this to take effect (it won't affect the chronics already
        loaded)

        Parameters
        ----------
        new_chunk_size: ``int`` or ``None``
            The new chunk size (positive integer)

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

        **NB** this has no effect if the chronics does not support this feature. TODO see xxx for more information

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
            from grid2op.Agent import DoNothingAgent

            env = make("case14_redisp")  # create an environment
            agent = DoNothingAgent(env.action_space)  # create an Agent

            for i in range(10):
                env.set_id(0)  # tell the environment you simply want to use the chronics with ID 0
                obs = env.reset()  # it is necessary to perform a reset
                reward = env.reward_range[0]
                done = False
                while not done:
                    act = agent.act(obs, reward, done)
                    obs, reward, done, info = env.step(act)

        """
        self.chronics_handler.tell_id(id_-1)

    def attach_renderer(self, graph_layout=None):
        if self.viewer is not None:
            return
        graph_layout = self.graph_layout if graph_layout is None else graph_layout
        if graph_layout is not None:
            self.viewer = Renderer(graph_layout, observation_space=self.helper_observation)
            self.viewer.reset(self)
        else:
            raise PlotError("No layout are available for the powergrid. Renderer is not possible.")

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)
        # TODO be closer to original gym implementation
        # if self.spec is None:
        #     return '<{} instance>'.format(type(self).__name__)
        # else:
        #     return '<{}<{}>>'.format(type(self).__name__, self.spec.id)

    def reset_grid(self):
        """
        Reset the backend to a clean state by reloading the powergrid from the hard drive. This might takes some time.

        If the thermal has been modified, it also modify them into the new backend.

        """
        self.backend.load_grid(self.init_grid_path)  # the real powergrid of the environment
        self.backend.assert_grid_correct()

        if self._thermal_limit_a is not None:
            self.backend.set_thermal_limit(self._thermal_limit_a)

        do_nothing = self.helper_action_env({})
        self.step(do_nothing)
        # test the backend returns object of the proper size
        self.backend.assert_grid_correct_after_powerflow()

    def add_text_logger(self, logger=None):
        """
        Add a text logger to this  :class:`Environment`

        Logging is for now an incomplete feature. It will get improved
        Parameters
        ----------
            logger:
               The logger to use

        """
        self.logger = logger
        return self

    def seed(self, seed=None):
        """
        Set the seed of this :class:`Environment` for a better control and to ease reproducible experiments.

        This is not supported yet.

        Parameters
        ----------
            seed: ``int``
               The seed to set.

        """
        try:
            seed = np.array(seed).astype('int64')
        except Exception as e:
            raise Grid2OpException("Impossible to seed with the seed provided. Make sure it can be converted to a"
                                   "numpy 64 integer.")
        # example from gym
        # self.np_random, seed = seeding.np_random(seed)
        # TODO make that more clean, see example of seeding @ https://github.com/openai/gym/tree/master/gym/utils
        self.chronics_handler.seed(seed)
        self.helper_observation.seed(seed)
        self.helper_action_player.seed(seed)
        self.helper_action_env.seed(seed)
        return [seed]

    def reset(self):
        """
        Reset the environment to a clean state.
        It will reload the next chronics if any. And reset the grid to a clean state.

        This triggers a full reloading of both the chronics (if they are stored as files) and of the powergrid,
        to ensure the episode is fully over.

        This method should be called only at the end of an episode.
        """
        self.chronics_handler.next_chronics()
        self.chronics_handler.initialize(self.backend.name_load, self.backend.name_gen,
                                         self.backend.name_line, self.backend.name_sub,
                                         names_chronics_to_backend=self.names_chronics_to_backend)
        self.current_obs = None
        self._reset_maintenance()
        self._reset_redispatching()
        self.reset_grid()
        if self.viewer is not None:
            self.viewer.reset(self)
        # if True, then it will not disconnect lines above their thermal limits
        self._reset_vectors_and_timings()
        return self.get_obs()

    def render(self, mode='human'):
        err_msg = "Impossible to use the renderer, please set it up with  \"env.init_renderer(graph_layout)\", " \
                  "graph_layout being the position of each substation of the powergrid that you must provide"
        self.attach_renderer()
        if mode == "human":
            if self.viewer is not None:
                has_quit = self.viewer.render(self.current_obs,
                                              reward=self.current_reward,
                                              timestamp=self.time_stamp,
                                              done=self.done)
                if has_quit:
                    self.close()
                    exit()
            else:
                raise Grid2OpException(err_msg)
        elif mode == "rgb_array":
            if self.viewer is not None:
                return np.array(self.viewer.get_rgb(self.current_obs,
                                                    reward=self.current_reward,
                                                    timestamp=self.time_stamp,
                                                    done=self.done))
            else:
                raise Grid2OpException(err_msg)
        else:
            raise Grid2OpException("Renderer mode \"{}\" not supported.".format(mode))

    def copy(self):
        """
        performs a deep copy of the environment

        Returns
        -------

        """
        tmp_backend = self.backend
        self.backend = None
        res = copy.deepcopy(self)
        res.backend = tmp_backend.copy()
        if self._thermal_limit_a is not None:
            res.backend.set_thermal_limit(self._thermal_limit_a)
        self.backend = tmp_backend
        return res

    def get_kwargs(self):
        """
        This function allows to make another Environment with the same parameters as the one that have been used
        to make this one.

        This is usefull especially in cases where Environment is not pickable (for example if some non pickable c++
        code are used) but you still want to make parallel processing using "MultiProcessing" module. In that case,
        you can send this dictionnary to each child process, and have each child process make a copy of ``self``

        Returns
        -------
        res: ``dict``
            A dictionnary that helps build an environment like ``self``

        Examples
        --------
        It should be used as follow:

        .. code-block:: python

            import grid2op
            from grid2op.Environment import Environment
            env = grid2op.make()  # create the environment of your choice
            copy_of_env = Environment(**env.get_kwargs())
            # And you can use this one as you would any other environment.

        """
        res = {}
        res["init_grid_path"] = self.init_grid_path
        res["chronics_handler"] = copy.deepcopy(self.chronics_handler)
        res["parameters"] = copy.deepcopy(self.parameters)
        res["names_chronics_to_backend"] = copy.deepcopy(self.names_chronics_to_backend)
        res["actionClass"] = self.actionClass
        res["observationClass"] = self.observationClass
        res["rewardClass"] = self.rewardClass
        res["legalActClass"] = self.legalActClass
        res["epsilon_poly"] = self._epsilon_poly
        res["tol_poly"] = self._tol_poly
        res["thermal_limit_a"] = self._thermal_limit_a
        res["voltagecontrolerClass"] = self.voltagecontrolerClass
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
            env = grid2op.make()  # create the environment of your choice
            agent = DoNothingAgent(env.actoin_space)

            # create the proper runner
            runner = Runner(**env.get_params_for_runner(), agentClass=DoNothingAgent)

            # now you can run
            runner.run(nb_episode=1)  # run for 1 episode

        """
        res = {}
        res["init_grid_path"] = self.init_grid_path
        res["path_chron"] = self.chronics_handler.path
        res["parameters_path"] = self.parameters.to_dict()
        res["names_chronics_to_backend"] = self.names_chronics_to_backend
        res["actionClass"] = self.actionClass
        res["observationClass"] = self.observationClass
        res["rewardClass"] = self.rewardClass
        res["legalActClass"] = self.legalActClass
        res["envClass"] = Environment
        res["gridStateclass"] = self.chronics_handler.chronicsClass
        res["backendClass"] = type(self.backend)  # TODO
        res["verbose"] = False
        dict_ = copy.deepcopy(self.chronics_handler.kwargs)
        if 'path' in dict_:
            # path is handled elsewhere
            del dict_["path"]
        res["gridStateclass_kwargs"] = dict_
        res["thermal_limit_a"] = self._thermal_limit_a
        res["voltageControlerClass"] = self.voltagecontrolerClass

        # TODO make a test for that
        return res