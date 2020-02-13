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
import time
import os
import copy

try:
    from .Space import GridObjects
    from .Action import HelperAction, Action, TopologyAction
    from .Exceptions import *
    from .Observation import CompleteObservation, ObservationHelper, Observation
    from .Reward import FlatReward, RewardHelper, Reward
    from .GameRules import GameRules, AllwaysLegal, LegalAction
    from .Parameters import Parameters
    from .Backend import Backend
    from .ChronicsHandler import ChronicsHandler
    from .PlotPyGame import Renderer
except (ModuleNotFoundError, ImportError):
    from Space import GridObjects
    from Action import HelperAction, Action, TopologyAction
    from Exceptions import *
    from Observation import CompleteObservation, ObservationHelper, Observation
    from Reward import FlatReward, RewardHelper, Reward
    from GameRules import GameRules, AllwaysLegal, LegalAction
    from Parameters import Parameters
    from Backend import Backend
    from ChronicsHandler import ChronicsHandler
    from PlotPyGame import Renderer

import pdb


# TODO code "start from a given time step" -> link to the "skip" method of GridValue

# TODO have a viewer / renderer now

class Environment(GridObjects):
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
                 epsilon_poly=1e-2,
                 tol_poly=1e-6):
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

        GridObjects.__init__(self)

        # some timers
        self._time_apply_act = 0
        self._time_powerflow = 0
        self._time_extract_obs = 0
        self._epsilon_poly = epsilon_poly
        self._tol_poly = tol_poly

        # define logger
        self.logger = None

        # and calendar data
        self.time_stamp = None
        self.nb_time_step = 0

        # specific to power system
        if not isinstance(parameters, Parameters):
            raise Grid2OpException("Parameter \"parameters\" used to build the Environment should derived form the "
                                   "grid2op.Parameters class, type provided is \"{}\"".format(type(parameters)))
        self.parameters = parameters

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
        # observation
        self.current_obs = None

        # type of power flow to play
        # if True, then it will not disconnect lines above their thermal limits
        self.no_overflow_disconnection = self.parameters.NO_OVERFLOW_DISCONNECTION
        self.timestep_overflow = np.zeros(shape=(self.n_line,), dtype=np.int)
        self.nb_timestep_overflow_allowed = np.full(shape=(self.n_line,),
                                                    fill_value=self.parameters.NB_TIMESTEP_POWERFLOW_ALLOWED)
        # store actions "cooldown"
        self.times_before_line_status_actionable = np.zeros(shape=(self.n_line,), dtype=np.int)
        self.max_timestep_line_status_deactivated = self.parameters.NB_TIMESTEP_LINE_STATUS_REMODIF

        self.times_before_topology_actionable = np.zeros(shape=(self.n_sub,), dtype=np.int)
        self.max_timestep_topology_deactivated = self.parameters.NB_TIMESTEP_TOPOLOGY_REMODIF

        # for maintenance operation
        self.time_next_maintenance = np.zeros(shape=(self.n_line,), dtype=np.int) - 1
        self.duration_next_maintenance = np.zeros(shape=(self.n_line,), dtype=np.int)

        # hazard (not used outside of this class, information is given in `time_remaining_before_line_reconnection`
        self._hazard_duration = np.zeros(shape=(self.n_line,), dtype=np.int)

        # hard overflow part
        self.hard_overflow_threshold = self.parameters.HARD_OVERFLOW_THRESHOLD
        self.time_remaining_before_line_reconnection = np.full(shape=(self.n_line,), fill_value=0, dtype=np.int)
        self.env_dc = self.parameters.ENV_DC

        # redispatching data
        self.target_dispatch = None
        self.actual_dispatch = None
        self.gen_uptime = None
        self.gen_downtime = None
        self.gen_activeprod_t = None
        self._reset_redispatching()

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

        # store environment modifications
        self._injection = None
        self._maintenance = None
        self._hazards = None
        self.env_modification = None

        # reward
        self.reward_helper = RewardHelper(rewardClass=rewardClass)
        self.reward_helper.initialize(self)

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
        self._thermal_limit_a = None

    def set_thermal_limit(self, thermal_limit):
        """
        Set the thermal limit effectively.

        Parameters
        ----------
        thermal_limit: ``numpy.ndarray``
            The new thermal limit. It must be a numpy ndarray vector (or convertible to it). For each powerline it
            gives the new thermal limit.
        """
        try:
            tmp = np.array(thermal_limit).flatten().astype(np.float)
        except Exception as e:
            raise Grid2OpException("Impossible to convert the vector as input into a 1d numpy float array.")
        if tmp.shape[0] != self.n_line:
            raise Grid2OpException("Attempt to set thermal limit on {} powerlines while there are {}"
                                   "on the grid".format(tmp.shape[0], self.n_line))
        if np.any(~np.isfinite(tmp)):
            raise Grid2OpException("Impossible to use non finite value for thermal limits.")

        self._thermal_limit_a = tmp
        self.backend.set_thermal_limit(self._thermal_limit_a)

    def set_id(self, id_):
        """
        Set the id that will be used at the next call to :func:`Environment.reset`.

        **NB** this has no effect if the chronics does not support this feature. TODO see xxx for more information

        Parameters
        ----------
        id_: ``int``
            the id of the chronics used.

        """
        self.chronics_handler.tell_id(id_-1)

    def attach_renderer(self, graph_layout):
        self.viewer = Renderer(graph_layout, observation_space=self.helper_observation)
        self.viewer.reset(self)

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)
        # TODO be closer to original gym implementation
        # if self.spec is None:
        #     return '<{} instance>'.format(type(self).__name__)
        # else:
        #     return '<{}<{}>>'.format(type(self).__name__, self.spec.id)

    def __enter__(self):
        """
        Support *with-statement* for the environment.

        Examples
        --------

        .. code-block:: python

            import grid2op
            import grid2op.Agent
            with grid2op.make() as env:
                agent = grid2op.Agent.DoNothingAgent(env.action_space)
                act = env.action_space()
                obs, r, done, info = env.step(act)
                act = agent.act(obs, r, info)
                obs, r, done, info = env.step(act)

        """
        return self

    def __exit__(self, *args):
        """
        Support *with-statement* for the environment.
        """
        self.close()
        # propagate exception
        return False

    def _reset_maintenance(self):
        self.time_next_maintenance = np.zeros(shape=(self.backend.n_line,), dtype=np.int) - 1
        self.duration_next_maintenance = np.zeros(shape=(self.backend.n_line,), dtype=np.int)
        self.time_remaining_before_reconnection = np.full(shape=(self.backend.n_line,), fill_value=0, dtype=np.int)

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

    def _update_actions(self):
        """
        Retrieve the actions to perform the update of the underlying powergrid represented by
         the :class:`grid2op.Backend`in the next time step.
        A call to this function will also read the next state of :attr:`chronics_handler`, so it must be called only
        once per time step.

        Returns
        --------
        res: :class:`grid2op.Action.Action`
            The action representing the modification of the powergrid induced by the Backend.
        """
        timestamp, tmp, maintenance_time, maintenance_duration, hazard_duration = self.chronics_handler.next_time_step()
        if "injection" in tmp:
            self._injection = tmp["injection"]
        else:
            self._injection = None
        if 'maintenance' in tmp:
            self._maintenance = tmp['maintenance']
        else:
            self._maintenance = None
        if "hazards" in tmp:
            self._hazards = tmp["hazards"]
        else:
            self._hazards = None
        self.time_stamp = timestamp
        self.duration_next_maintenance = maintenance_duration
        self.time_next_maintenance = maintenance_time
        self._hazard_duration = hazard_duration
        return self.helper_action_env({"injection": self._injection, "maintenance": self._maintenance,
                                       "hazards": self._hazards})

    def get_obs(self):
        """
        Return the observations of the current environment made by the :class:`grid2op.Agent.Agent`.

        Returns
        -------
        res: :class:`grid2op.Observation.Observation`
            The current Observation given to the :class:`grid2op.Agent.Agent` / bot / controler.
        """
        res = self.helper_observation(env=self)
        return res

    def _get_reward(self, action, has_error, is_done, is_illegal, is_ambiguous):
        return self.reward_helper(action, self, has_error, is_done, is_illegal, is_ambiguous)

    def _is_done(self, has_error, is_done):
        no_more_data = self.chronics_handler.done()
        return has_error or is_done or no_more_data

    def _update_time_reconnection_hazards_maintenance(self):
        """
        This supposes that :attr:`Environment.time_remaining_before_line_reconnection` is already updated
        with the cascading failure, soft overflow and hard overflow.

        It also supposes that :func:`Environment._update_actions` has been called, so that the vectors
        :attr:`Environment.duration_next_maintenance`, :attr:`Environment.time_next_maintenance` and
        :attr:`Environment._hazard_duration` are updated with the most recent values.

        Finally the Environment supposes that this method is called before calling :func:`Environment.get_obs`

        This function integrates the hazards and maintenance in the
        :attr:`Environment.time_remaining_before_line_reconnection` vector.
        For example, if a powerline `i` has no problem
        of overflow, but is affected by a hazard, :attr:`Environment.time_remaining_before_line_reconnection`
        should be updated with the duration of this hazard (stored in one of the three vector mentionned in the
        above paragraph)

        For this Environment, we suppose that the maximum of the 3 values are taken into account. The reality would
        be more complicated.

        Returns
        -------

        """
        self.time_remaining_before_line_reconnection = np.maximum(self.time_remaining_before_line_reconnection,
                                                                  self.duration_next_maintenance)
        self.time_remaining_before_line_reconnection = np.maximum(self.time_remaining_before_line_reconnection,
                                                                  self._hazard_duration)

    def _aux_redisp(self, redisp_act, target_p, avail_gen, previous_redisp):
        # delta_gen_min = np.maximum(-self.gen_max_ramp_down+previous_redisp, self.gen_pmin-target_p)
        # delta_gen_max = np.minimum(self.gen_max_ramp_up+previous_redisp, self.gen_pmax-target_p)
        delta_gen_min = np.maximum(-self.gen_max_ramp_down+previous_redisp,
                                   self.gen_pmin - (target_p - previous_redisp))
        delta_gen_max = np.minimum(self.gen_max_ramp_up+previous_redisp,
                                   self.gen_pmax - (target_p - previous_redisp))

        min_disp = np.sum(delta_gen_min[avail_gen])
        max_disp = np.sum(delta_gen_max[avail_gen])
        new_redisp = None
        except_ = None
        val_sum = +np.sum(redisp_act[avail_gen])-np.sum(redisp_act)
        if val_sum < min_disp:
            except_ = InvalidRedispatching("Impossible to perform this redispatching. Minimum ramp (or pmin) for "
                                           "available generators is not enough to absord "
                                           "{}MW. min possible is {}MW".format(val_sum, min_disp))
        elif val_sum > max_disp:
            except_ = InvalidRedispatching("Impossible to perform this redispatching. Maximum ramp (or pmax) for "
                                           "available generators is not enough to absord "
                                           "{}MW, max possible is {}MW".format(val_sum, max_disp))
        elif np.abs(val_sum) <= self._tol_poly:
            # i don't need to modify anything so i should be good
            new_redisp = 0.0 * redisp_act
        else:
            new_redisp, except_ = self._aux_aux_redisp(delta_gen_min,
                                                       delta_gen_max,
                                                       avail_gen,
                                                       redisp_act,
                                                       val_sum)

        return new_redisp, except_

    def _aux_aux_redisp(self, delta_gen_min, delta_gen_max, avail_gen, redisp_act, sum_value):
        except_ = None
        new_redisp = 0.*redisp_act
        if not np.sum(avail_gen):
            # there are no available generators
            except_ = NotEnoughGenerators("Sum of available generator is too low to meet the demand.")
            return None, except_

        try:
            t_zerosum = self._get_t(redisp_act[avail_gen],
                                    pmin=delta_gen_min[avail_gen],
                                    pmax=delta_gen_max[avail_gen],
                                    total_dispatch=sum_value)
        except Exception as e:
            # i can't implement redispatching due to impossibility to dispatch on the other generator
            # it's a non valid action
            except_ = e
            return None, except_

        new_redisp_tmp = self._get_poly(t=t_zerosum,
                                    pmax=delta_gen_max[avail_gen],
                                    pmin=delta_gen_min[avail_gen],
                                    tmp_p=redisp_act[avail_gen])
        new_redisp[avail_gen] = new_redisp_tmp
        # self.actual_dispatch[avail_gen] = actual_dispatch_tmp
        return new_redisp, except_

    def _get_redisp_zero_sum(self, redisp_act, new_p, redisp_this_act):
        """

        Parameters
        ----------
        action

        redisp_act:
            the redispatching part of the action

        new_p:
            the new target generation for each generator

        Returns
        -------

        """

        # make the target dispatch a 0-sum vector (using only dispatchable unit, not dispatched)
        # dispatch only the generator that are at zero
        avail_gen = self.target_dispatch == 0.  # generators with a redispatching target cannot be redispatched again
        avail_gen = avail_gen & (redisp_this_act == 0.)  # generator on which I act this time step cannot be redispatched again
        avail_gen = avail_gen & self.gen_redispatchable  # i can only redispatched dispatchable generators
        avail_gen = avail_gen & (new_p > 0.)

        # get back the previous value for the dispatchable generators
        target_disp = 1.0 * redisp_act
        # target_disp[avail_gen] = self.actual_dispatch[avail_gen]
        new_redisp, except_ = self._aux_redisp(target_disp, new_p, avail_gen, self.actual_dispatch)
        if except_ is None:
            new_redisp += redisp_act
        return new_redisp, except_

    def _compute_actual_dispatch(self, new_p):
        # this automated conrol only affect turned-on generators that are dispatchable
        except_ = None
        turned_on_gen = new_p > 0.
        gen_redispatchable = self.gen_redispatchable & turned_on_gen

        # make sure that rampmin and max are met
        new_p_if_redisp_ok = new_p + self.actual_dispatch
        gen_min = np.maximum(self.gen_pmin, self.gen_activeprod_t - self.gen_max_ramp_down)
        gen_max = np.minimum(self.gen_pmax, self.gen_activeprod_t + self.gen_max_ramp_up)

        if np.any((gen_min[gen_redispatchable] > new_p_if_redisp_ok[gen_redispatchable]) |
                   (new_p_if_redisp_ok[gen_redispatchable] > gen_max[gen_redispatchable])) and \
            np.any(self.gen_activeprod_t != 0.):

            # i am in a case where the target redispatching is not possible, due to the new values
            # i need to come up with a solution to fix that
            # note that the condition "np.any(self.gen_activeprod_t != 0.)" is added because at the first time
            # step there is no need to check all that.
            # but take into account pmin and pmax
            curtail_generation = 1. * new_p_if_redisp_ok
            mask_min = (new_p_if_redisp_ok < gen_min + self._epsilon_poly) & gen_redispatchable
            mask_max = (new_p_if_redisp_ok > gen_max - self._epsilon_poly) & gen_redispatchable

            minimum_redisp = gen_min - new_p
            maximum_redisp = gen_max - new_p
            new_dispatch = 1. * self.actual_dispatch

            if np.any(mask_min) or np.any(mask_max):
                # modify the implemented redispatching to take into account this "curtailement"
                # due to physical limitation

                curtail_generation[mask_min] = gen_min[mask_min]  # + self._epsilon_poly
                curtail_generation[mask_max] = gen_max[mask_max]  # - self._epsilon_poly

                diff_th_imp = curtail_generation - new_p_if_redisp_ok
                new_dispatch[mask_min] += diff_th_imp[mask_min] + self._epsilon_poly
                new_dispatch[mask_max] += diff_th_imp[mask_max] - self._epsilon_poly

                # current dispatch doesn't respect pmin/pmax / ramp_min / ramp_max
                # for polynomial stability
                minimum_redisp[mask_max] = new_dispatch[mask_max] - self._epsilon_poly
                maximum_redisp[mask_min] = new_dispatch[mask_min] + self._epsilon_poly

            new_redisp, except_ = self._aux_aux_redisp(minimum_redisp,maximum_redisp,
                                                       gen_redispatchable,
                                                       new_dispatch,
                                                       0.)

            return new_redisp, except_
        return self.actual_dispatch, except_

    def _get_new_prod_setpoint(self, action):
        except_ = None
        redisp_act = 1. * action._redispatch

        # get the modification of generator active setpoint from the action
        new_p = 1. * self.gen_activeprod_t
        if "prod_p" in action._dict_inj:
            tmp = action._dict_inj["prod_p"]
            indx_ok = np.isfinite(tmp)
            new_p[indx_ok] = tmp[indx_ok]

        # modification of the environment always override the modification of the agents (if any)
        # TODO have a flag there if this is the case.
        if "prod_p" in self.env_modification._dict_inj:
            # modification of the production setpoint value
            tmp = self.env_modification._dict_inj["prod_p"]
            indx_ok = np.isfinite(tmp)
            new_p[indx_ok] = tmp[indx_ok]
        return new_p, except_

    def _make_redisp_0sum(self, action, new_p):
        """
        Test the redispatching is valid, then make it a 0 sum action.

        This method updates actual_dispatch and target_dispatch

        Parameters
        ----------
        action
        new_p

        Returns
        -------

        """
        # Redispatching process the redispatching actions here, get a redispatching vector with 0-sum
        # from the environment.

        except_ = None

        # get the redispatching action (if any)
        redisp_act_orig = 1. * action._redispatch
        previous_redisp = 1. * self.actual_dispatch

        if np.all(redisp_act_orig == 0.) and np.all(self.target_dispatch == 0.) and np.all(self.actual_dispatch == 0.):
            return except_

        # check that everything is consistent with pmin, pmax:
        if np.any(self.target_dispatch > self.gen_pmax):
            # action is invalid, the target redispatching would be above pmax for at least a generator
            except_ = InvalidRedispatching("Target redispatching above pmax for generators "
                                           "{}".format(np.where(self.target_dispatch > self.gen_pmax)[0]))
            return except_

        if np.any(self.target_dispatch < self.gen_pmin):
            # action is invalid, the target redispatching would be below pmin for at least a generator
            except_ = InvalidRedispatching("Target redispatching bellow pmin for generators "
                                           "{}".format(np.where(self.target_dispatch < self.gen_pmin)[0]))
            return except_

        # i can't redispatch turned off generators [turned off generators need to be turned on before redispatching]
        if np.any(redisp_act_orig[new_p == 0.]):
            # action is invalid, a generator has been redispatched, but it's turned off
            except_ = InvalidRedispatching("Impossible to dispatched a turned off generator")
            return except_

        redisp_act_orig[new_p == 0.] = 0.
        # TODO add a flag here too, like before (the action has been "cut")

        # get the target redispatching (cumulation starting from the first element of the scenario)
        self.target_dispatch += redisp_act_orig
        if np.abs(np.sum(self.actual_dispatch)) >= self._tol_poly or \
                np.sum(np.abs(self.actual_dispatch - self.target_dispatch)) >= self._tol_poly:
            # make sure the redispatching action is zero sum
            new_redisp, except_ = self._get_redisp_zero_sum(self.target_dispatch,
                                                            self.gen_activeprod_t,
                                                            redisp_act_orig)
            if except_ is not None:
                # if there is an error, then remove the above "action" and propagate it
                self.actual_dispatch = previous_redisp
                self.target_dispatch -= redisp_act_orig
                return except_
            else:
                self.actual_dispatch = new_redisp
        return except_

    def _handle_updown_times(self, gen_up_before, redisp_act):
        # get the generators that are not connected after the action
        except_ = None

        # computes which generator will be turned on after the action
        gen_up_after = 1.0 * self.gen_activeprod_t
        if "prod_p" in self.env_modification._dict_inj:
            tmp = self.env_modification._dict_inj["prod_p"]
            indx_ok = np.isfinite(tmp)
            gen_up_after[indx_ok] = self.env_modification._dict_inj["prod_p"][indx_ok]
        gen_up_after += redisp_act
        gen_up_after = gen_up_after > 0.

        # update min down time, min up time etc.
        gen_disconnected_this = gen_up_before & (~gen_up_after)
        gen_connected_this_timestep = (~gen_up_before) & (gen_up_after)
        gen_still_connected = gen_up_before & gen_up_after
        gen_still_disconnected = (~gen_up_before) & (~gen_up_after)

        if np.any(self.gen_downtime[gen_connected_this_timestep] < self.gen_min_downtime[gen_connected_this_timestep]):
            # i reconnected a generator before the minimum time allowed
            id_gen = self.gen_downtime[gen_connected_this_timestep] < self.gen_min_downtime[gen_connected_this_timestep]
            id_gen = np.where(id_gen)[0]
            id_gen = np.where(gen_connected_this_timestep[id_gen])[0]
            except_ = GeneratorTurnedOnTooSoon("Some generator has been connected too early ({})".format(id_gen))
            return except_
        else:
            self.gen_downtime[gen_connected_this_timestep] = -1
            self.gen_uptime[gen_connected_this_timestep] = 1

        if np.any(self.gen_uptime[gen_disconnected_this] < self.gen_min_uptime[gen_disconnected_this]):
            # i disconnected a generator before the minimum time allowed
            id_gen = self.gen_uptime[gen_disconnected_this] < self.gen_min_uptime[gen_disconnected_this]
            id_gen = np.where(id_gen)[0]
            id_gen = np.where(gen_connected_this_timestep[id_gen])[0]
            except_ = GeneratorTurnedOffTooSoon("Some generator has been disconnected too early ({})".format(id_gen))
            return except_
        else:
            self.gen_downtime[gen_connected_this_timestep] = 0
            self.gen_uptime[gen_connected_this_timestep] = 1

        self.gen_uptime[gen_still_connected] += 1
        self.gen_downtime[gen_still_disconnected] += 1
        return except_

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).

        If the :class:`grid2op.Action.Action` is illegal or ambiguous, the step is performed, but the action is
        replaced with a "do nothing" action.

        Parameters
        ----------
            action: :class:`grid2op.Action.Action`
                an action provided by the agent that is applied on the underlying through the backend.

        Returns
        -------
            observation: :class:`grid2op.Observation.Observation`
                agent's observation of the current environment

            reward: ``float``
                amount of reward returned after previous action

            done: ``bool``
                whether the episode has ended, in which case further step() calls will return undefined results

            info: ``dict``
                contains auxiliary diagnostic information (helpful for debugging, and sometimes learning). It is a
                dicitonnary with keys:

                    - "disc_lines": a numpy array (or ``None``) saying, for each powerline if it has been disconnected
                        due to overflow
                    - "is_illegal" (``bool``) whether the action given as input was illegal
                    - "is_ambiguous" (``bool``) whether the action given as input was ambiguous.
                    - "is_illegal_redisp" (``bool``) was the action illegal due to redispatching
                    - "is_illegal_reco" (``bool``) was the action illegal due to a powerline reconnection
                    - "exception" (:class:`Exceptions.Exceptions.Grid2OpException` if an exception was raised
                       or ``None`` if everything was fine.)

        """
        has_error = True
        is_done = False
        disc_lines = None
        is_illegal = False
        is_ambiguous = False
        is_illegal_redisp = False
        is_illegal_reco = False
        except_ = []
        init_disp = 1.0 * action._redispatch
        previous_disp = 1.0 * self.actual_dispatch
        previous_target_disp = 1.0 * self.target_dispatch
        try:
            beg_ = time.time()
            is_illegal = not self.game_rules(action=action, env=self)
            if is_illegal:
                # action is replace by do nothing
                action = self.helper_action_player({})
                except_.append(IllegalAction("Action illegal"))

            ambiguous, except_tmp = action.is_ambiguous()
            if ambiguous:
                # action is replace by do nothing
                action = self.helper_action_player({})
                has_error = True
                is_ambiguous = True
                except_.append(except_tmp)

            # get the modification of generator active setpoint from the environment
            self.env_modification = self._update_actions()
            if self.redispatching_unit_commitment_availble:
                # remember generator that were "up" before the action
                gen_up_before = self.gen_activeprod_t > 0.

                # compute the redispatching and the new productions active setpoint
                new_p, except_tmp = self._get_new_prod_setpoint(action)
                if except_tmp is not None:
                    action = self.helper_action_player({})
                    is_illegal_redisp = True
                    new_p, _ = self._get_new_prod_setpoint(action)
                    except_.append(except_tmp)

                except_tmp = self._make_redisp_0sum(action, new_p)
                if except_tmp is not None:
                    action = self.helper_action_player({})
                    is_illegal_redisp = True
                    except_.append(except_tmp)

                # and now compute the actual dispatch that is consistent with pmin, pmax, ramp min, ramp max
                # this emulates the "frequency control" that is automatic.
                new_dispatch, except_tmp = self._compute_actual_dispatch(new_p)
                if except_tmp is not None:
                    action = self.helper_action_player({})
                    is_illegal_redisp = True
                    except_.append(except_tmp)
                    self.actual_dispatch = previous_disp
                    self.target_dispatch = previous_target_disp
                    new_dispatch, except_tmp = self._compute_actual_dispatch(new_p)
                    if except_tmp is None:
                        self.actual_dispatch = new_dispatch
                    else:
                        pass
                        # TODO what can i do if do nothing cannot be performed.
                        # probably a game over !
                else:
                    self.actual_dispatch = new_dispatch

                # check the validity of min downtime and max uptime
                except_tmp = self._handle_updown_times(gen_up_before, self.actual_dispatch)
                if except_tmp is not None:
                    is_illegal_reco = True
                    action = self.helper_action_player({})
                    except_.append(except_tmp)

            # make sure the dispatching action is not implemented "as is" by the backend.
            # the environment must make sure it's a zero-sum action.
            action._redispatch[:] = 0.
            try:
                self.backend.apply_action(action)
            except AmbiguousAction as e:
                # action has not been implemented on the powergrid because it's ambiguous, it's equivalent to
                # "do nothing"
                is_ambiguous = True
                except_.append(e)
            action._redispatch[:] = init_disp

            # now get the new generator voltage setpoint
            # TODO "automaton" to do that!

            self.env_modification._redispatch = self.actual_dispatch
            # action, for redispatching is composed of multiple actions, so basically i won't check
            # ramp_min and ramp_max
            self.env_modification._single_act = False

            # have the opponent here
            # TODO code the opponent part here

            self.backend.apply_action(self.env_modification)
            self._time_apply_act += time.time() - beg_

            self.nb_time_step += 1
            try:
                # compute the next _grid state
                beg_ = time.time()
                disc_lines, infos = self.backend.next_grid_state(env=self, is_dc=self.env_dc)
                self._time_powerflow += time.time() - beg_

                beg_ = time.time()
                self.backend.update_thermal_limit(self)  # update the thermal limit, for DLR for example
                overflow_lines = self.backend.get_line_overflow()
                # overflow_lines = np.full(self.n_line, fill_value=False, dtype=np.bool)

                # one timestep passed, i can maybe reconnect some lines
                self.time_remaining_before_line_reconnection[self.time_remaining_before_line_reconnection > 0] -= 1
                # update the vector for lines that have been disconnected
                self.time_remaining_before_line_reconnection[disc_lines] = int(self.parameters.NB_TIMESTEP_RECONNECTION)
                self._update_time_reconnection_hazards_maintenance()

                # for the powerline that are on overflow, increase this time step
                self.timestep_overflow[overflow_lines] += 1

                # set to 0 the number of timestep for lines that are not on overflow
                self.timestep_overflow[~overflow_lines] = 0

                # build the topological action "cooldown"
                aff_lines, aff_subs = action.get_topological_impact()
                if self.max_timestep_line_status_deactivated > 0:
                    # this is a feature I want to consider in the parameters
                    self.times_before_line_status_actionable[self.times_before_line_status_actionable > 0] -= 1
                    self.times_before_line_status_actionable[aff_lines] = self.max_timestep_line_status_deactivated
                if self.max_timestep_topology_deactivated > 0:
                    # this is a feature I want to consider in the parameters
                    self.times_before_topology_actionable[self.times_before_topology_actionable > 0] -= 1
                    self.times_before_topology_actionable[aff_subs] = self.max_timestep_topology_deactivated

                # build the observation
                self.current_obs = self.get_obs()
                self._time_extract_obs += time.time() - beg_

                # extract production active value at this time step (should be independant of action class)
                self.gen_activeprod_t, *_ = self.backend.generators_info()

                has_error = False
            except Grid2OpException as e:
                except_.append(e)
                if self.logger is not None:
                    self.logger.error("Impossible to compute next _grid state with error \"{}\"".format(e))

        except StopIteration:
            # episode is over
            is_done = True

        infos = {"disc_lines": disc_lines,
                 "is_illegal": is_illegal,
                 "is_ambiguous": is_ambiguous,
                 "is_dipatching_illegal": is_illegal_redisp,
                 "is_illegal_reco": is_illegal_reco,
                 "exception": except_}
        self.done = self._is_done(has_error, is_done)
        self.current_reward = self._get_reward(action,
                                               has_error,
                                               is_done,
                                               is_illegal or is_illegal_redisp or is_illegal_reco,
                                               is_ambiguous)

        # TODO documentation on all the possible way to be illegal now
        return self.current_obs, self.current_reward, self.done, infos

    def _reset_vectors_and_timings(self):
        """
        Maintenance are not reset, otherwise the data are not read properly (skip the first time step)

        Returns
        -------

        """
        self.no_overflow_disconnection = self.parameters.NO_OVERFLOW_DISCONNECTION
        self.timestep_overflow = np.zeros(shape=(self.n_line,), dtype=np.int)
        self.nb_timestep_overflow_allowed = np.full(shape=(self.n_line,),
                                                    fill_value=self.parameters.NB_TIMESTEP_POWERFLOW_ALLOWED)
        self.nb_time_step = 0
        self.hard_overflow_threshold = self.parameters.HARD_OVERFLOW_THRESHOLD
        self.env_dc = self.parameters.ENV_DC

        self.times_before_line_status_actionable = np.zeros(shape=(self.n_line,), dtype=np.int)
        self.max_timestep_line_status_deactivated = self.parameters.NB_TIMESTEP_LINE_STATUS_REMODIF

        self.times_before_topology_actionable = np.zeros(shape=(self.n_sub,), dtype=np.int)
        self.max_timestep_topology_deactivated = self.parameters.NB_TIMESTEP_TOPOLOGY_REMODIF

        self.time_remaining_before_line_reconnection = np.zeros(shape=(self.n_line,), dtype=np.int)

        # # maintenance and hazards
        # self.time_next_maintenance = np.zeros(shape=(self.n_line,), dtype=np.int) - 1
        # self.duration_next_maintenance = np.zeros(shape=(self.n_line,), dtype=np.int)
        # self._hazard_duration = np.zeros(shape=(self.n_line,), dtype=np.int)
        # self.time_remaining_before_line_reconnection = np.full(shape=(self.n_line,), fill_value=0, dtype=np.int)

        # reset timings
        self._time_apply_act = 0
        self._time_powerflow = 0
        self._time_extract_obs = 0

        # reward and others
        self.current_reward = self.reward_range[0]
        self.done = False

    def _reset_redispatching(self):
        # redispatching
        self.target_dispatch = np.full(shape=self.n_gen, dtype=np.float, fill_value=0.)
        self.actual_dispatch = np.full(shape=self.n_gen, dtype=np.float, fill_value=0.)
        self.gen_uptime = np.full(shape=self.n_gen, dtype=np.int, fill_value=0)
        # if self.redispatching_unit_commitment_availble:
        #     # pretend that all generator has been turned off for a suffcient number of timestep,
        #     # otherwise when reconnecting them at first step it's complicated
        #     self.gen_downtime = self.gen_min_downtime
        # else:
        #     self.gen_downtime = np.full(shape=self.n_gen, dtype=np.int, fill_value=0)
        self.gen_downtime = np.full(shape=self.n_gen, dtype=np.int, fill_value=0)
        self.gen_activeprod_t = np.zeros(self.n_gen, dtype=np.float)

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

    def close(self):
        # todo there might be some side effect
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        self.backend.close()

    @staticmethod
    def _get_poly(t, tmp_p, pmin, pmax):
        return tmp_p + 0.5 * (pmax - pmin) * t + 0.5 * (pmax + pmin - 2 * tmp_p) * t ** 2

    @staticmethod
    def _get_poly_coeff(tmp_p, pmin, pmax):
        p_s = tmp_p.sum()
        p_min_s = pmin.sum()
        p_max_s = pmax.sum()

        p_0 = p_s
        p_1 = 0.5 * (p_max_s - p_min_s)
        p_2 = 0.5 * (p_max_s + p_min_s - 2 * p_s)
        return p_0, p_1, p_2

    @staticmethod
    def _get_t(tmp_p, pmin, pmax, total_dispatch):
        # to_dispatch = too_much.sum() + not_enough.sum()
        p_0, p_1, p_2 = Environment._get_poly_coeff(tmp_p, pmin, pmax)

        res = np.roots((p_2, p_1, p_0-(total_dispatch)))
        res = res[np.isreal(res)]
        res = res[(res <= 1) & (res >= -1)]
        if res.shape[0] == 0:
            raise Grid2OpException("Impossible to solve for this equilibrium, not enough production")
        else:
            res = res[0]
        return res

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
            runner = Runner(**env.init_backend(), agentClass=DoNothingAgent)

            # now you can run
            runner.run(nb_episode=1)  # run for 1 episode

        """
        res = {}
        res["init_grid_path"] = self.init_grid_path
        res["path_chron"] = self.chronics_handler.path
        res["parameters_path"] # TODO
        res["names_chronics_to_backend"] = self.names_chronics_to_backend
        res["actionClass"] = self.actionClass
        res["observationClass"] = self.observationClass
        res["rewardClass"] = self.rewardClass
        res["legalActClass"] = self.legalActClass
        res["envClass"] = Environment
        res["gridStateclass"] = self.chronics_handler.chronicsClass
        res["backendClass"] = type(self.backend)  # TODO
        res["verbose"] = False
        res["gridStateclass_kwargs"] = self.chronics_handler.kwargs
        res["thermal_limit_a"] = self._thermal_limit_a
        return res