# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import time
import numpy as np
import copy
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from abc import ABC, abstractmethod

from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Space import GridObjects, RandomObject
from grid2op.Exceptions import *
from grid2op.Parameters import Parameters
from grid2op.Reward import BaseReward
from grid2op.Reward import RewardHelper
from grid2op.Opponent import OpponentSpace, NeverAttackBudget
from grid2op.Action import DontAct, BaseAction
from grid2op.Rules import AlwaysLegal
from grid2op.Opponent import BaseOpponent
from grid2op.Action._BackendAction import _BackendAction

# TODO put in a separate class the redispatching function

DETAILED_REDISP_ERR_MSG = "\nThis is an attempt to explain why the dispatch did not succeed and caused a game over.\n" \
                          "To compensate the {increase} of loads (and / or {decrease} of " \
                          "renewable energy), " \
                          "the generators should {increase} their total production of {sum_move:.2f}MW (in total).\n" \
                          "But, if you take into account the generator constraints ({pmax} and {max_ramp_up}) you " \
                          "can have at most {avail_up_sum:.2f}MW.\n" \
                          "Indeed at time t, generators are in state:\n\t{gen_setpoint}\ntheir ramp max is:" \
                          "\n\t{ramp_up}\n and pmax is:\n\t{gen_pmax}\n" \
                          "Wrapping up, each generator can {increase} at {maximum} of:\n\t{avail_up}\n" \
                          "NB: if you did not do any dispatch during this episode, it would have been possible to " \
                          "meet these constraints. This situation is caused by not having enough degree of freedom " \
                          "to \"compensate\" the variation of the load due to (most likely) an \"over usage\" of " \
                          "redispatching feature (some generators stuck at {pmax} as a consequence of your " \
                          "redispatching. They can't increase their productions to meet the {increase} in demand or " \
                          "{decrease} of renewables)"


class BaseEnv(GridObjects, RandomObject, ABC):
    """
    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    This class represent some usefull abstraction that is re used by :class:`Environment` and
    :class:`grid2op.Observation._Obsenv` for example.

    The documentation is showed here to document the common attributes of an "BaseEnvironment".

    Attributes
    ----------

    parameters: :class:`grid2op.Parameters.Parameters`
        The parameters of the game (to expose more control on what is being simulated)

    with_forecast: ``bool``
        Whether the chronics allow to have some kind of "forecast". See :func:`BaseEnv.activate_forceast`
        for more information

    logger:
        TO BE DONE: a way to log what is happening (**currently not implemented**)

    time_stamp: ``datetime.datetime``
        The actual time stamp of the current observation.

    nb_time_step: ``int``
        Number of time steps played in the current environment

    current_obs: :class:`grid2op.Observation.BaseObservation`
        The current observation (or None if it's not intialized)

    backend: :class:`grid2op.Backend.Backend`
        The backend used to compute the powerflows and cascading failures.

    done: ``bool``
        Whether the environment is "done". If ``True`` you need to call :func:`Environment.reset` in order
        to continue.

    current_reward: ``float``
        The last computed reward (reward of the current step)

    other_rewards: ``dict``
        Dictionary with key being the name (identifier) and value being some RewardHelper. At each time step, all the
        values will be computed by the :class:`Environment` and the information about it will be returned in the
        "reward" key of the "info" dictionnary of the :func:`Environment.step`.

    chronics_handler: :class:`grid2op.Chronics.ChronicsHandler`
        The object in charge managing the "chronics", which store the information about load and generator for example.

    reward_range: ``tuple``
        For open ai gym compatibility. It represents the range of the rewards: reward min, reward max

    viewer
        For open ai gym compatibility.

    viewer_fig
        For open ai gym compatibility.

    _gen_activeprod_t:
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Should be initialized at 0. for "step" to properly recognize it's the first time step of the game

    _no_overflow_disconnection: ``bool``
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Whether or not cascading failures are computed or not (TRUE = the powerlines above their thermal limits will
        not be disconnected). This is initialized based on the attribute
        :attr:`grid2op.Parameters.Parameters.NO_OVERFLOW_DISCONNECTION`.

    _timestep_overflow: ``numpy.ndarray``, dtype: int
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Number of consecutive timesteps each powerline has been on overflow.

    _nb_timestep_overflow_allowed: ``numpy.ndarray``, dtype: int
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Number of consecutive timestep each powerline can be on overflow. It is usually read from
        :attr:`grid2op.Parameters.Parameters.NB_TIMESTEP_POWERFLOW_ALLOWED`.

    _hard_overflow_threshold: ``float``
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Number of timestep before an :class:`grid2op.BaseAgent.BaseAgent` can reconnet a powerline that has been
        disconnected
        by the environment due to an overflow.

    _env_dc: ``bool``
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Whether the environment computes the powerflow using the DC approximation or not. It is usually read from
        :attr:`grid2op.Parameters.Parameters.ENV_DC`.

    _names_chronics_to_backend: ``dict``
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Configuration file used to associated the name of the objects in the backend
        (both extremities of powerlines, load or production for
        example) with the same object in the data (:attr:`Environment.chronics_handler`). The idea is that, usually
        data generation comes from a different software that does not take into account the powergrid infrastructure.
        Hence, the same "object" can have a different name. This mapping is present to avoid the need to rename
        the "object" when providing data. A more detailed description is available at
        :func:`grid2op.ChronicsHandler.GridValue.initialize`.

    _env_modification: :class:`grid2op.Action.Action`
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Representation of the actions of the environment for the modification of the powergrid.

    _rewardClass: ``type``
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Type of reward used. Should be a subclass of :class:`grid2op.BaseReward.BaseReward`

    _init_grid_path: ``str``
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        The path where the description of the powergrid is located.

    _game_rules: :class:`grid2op.Rules.RulesChecker`
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        The rules of the game (define which actions are legal and which are not)

    _helper_action_player: :class:`grid2op.Action.ActionSpace`
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Helper used to manipulate more easily the actions given to / provided by the :class:`grid2op.Agent.BaseAgent`
        (player)

    _helper_action_env: :class:`grid2op.Action.ActionSpace`
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Helper used to manipulate more easily the actions given to / provided by the environment to the backend.

    _helper_observation: :class:`grid2op.Observation.ObservationSpace`
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Helper used to generate the observation that will be given to the :class:`grid2op.BaseAgent`

    _reward_helper: :class:`grid2p.BaseReward.RewardHelper`
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Helper that is called to compute the reward at each time step.

    """
    def __init__(self,
                 parameters,
                 voltagecontrolerClass,
                 thermal_limit_a=None,
                 epsilon_poly=1e-4,  # precision of the redispatching algorithm
                 tol_poly=1e-2,  # i need to compute a redispatching if the actual values are "more than tol_poly" the values they should be
                 other_rewards={},
                 with_forecast=True,
                 opponent_action_class=DontAct,
                 opponent_class=BaseOpponent,
                 opponent_init_budget=0.,
                 opponent_budget_per_ts=0.,
                 opponent_budget_class=NeverAttackBudget,
                 opponent_attack_duration=0,
                 opponent_attack_cooldown=99999,
                 kwargs_opponent={}
                 ):
        GridObjects.__init__(self)
        RandomObject.__init__(self)

        self._DEBUG = False
        # specific to power system
        if not isinstance(parameters, Parameters):
            raise Grid2OpException("Parameter \"parameters\" used to build the Environment should derived form the "
                                   "grid2op.Parameters class, type provided is \"{}\"".format(type(parameters)))
        self.parameters = parameters
        self.with_forecast = with_forecast

        # some timers
        self._time_apply_act = dt_float(0)
        self._time_powerflow = dt_float(0)
        self._time_extract_obs = dt_float(0)
        self._time_opponent = dt_float(0)
        self._time_redisp = dt_float(0)

        # data relative to interpolation
        self._epsilon_poly = dt_float(epsilon_poly)
        self._tol_poly = dt_float(tol_poly)

        # define logger
        self.logger = None

        # the voltage controler

        # class used for the action spaces
        self._helper_action_class = None
        self._helper_observation_class = None

        # and calendar data
        self.time_stamp = None
        self.nb_time_step = dt_int(0)

        # observation
        self.current_obs = None
        self._line_status = None

        self._ignore_min_up_down_times = self.parameters.IGNORE_MIN_UP_DOWN_TIME
        self._forbid_dispatch_off = not self.parameters.ALLOW_DISPATCH_GEN_SWITCH_OFF

        # type of power flow to play
        # if True, then it will not disconnect lines above their thermal limits
        self._no_overflow_disconnection = self.parameters.NO_OVERFLOW_DISCONNECTION
        self._timestep_overflow = None
        self._nb_timestep_overflow_allowed = None
        self._hard_overflow_threshold = self.parameters.HARD_OVERFLOW_THRESHOLD

        # store actions "cooldown"
        self._times_before_line_status_actionable = None
        self._max_timestep_line_status_deactivated = self.parameters.NB_TIMESTEP_COOLDOWN_LINE
        self._times_before_topology_actionable = None
        self._max_timestep_topology_deactivated = self.parameters.NB_TIMESTEP_COOLDOWN_SUB

        # for maintenance operation
        self._time_next_maintenance = None
        self._duration_next_maintenance = None

        # hazard (not used outside of this class, information is given in `times_before_line_status_actionable`
        self._hazard_duration = None

        self._env_dc = self.parameters.ENV_DC

        # redispatching data
        self._target_dispatch = None
        self._actual_dispatch = None
        self._gen_uptime = None
        self._gen_downtime = None
        self._gen_activeprod_t = None
        self._gen_activeprod_t_redisp = None

        self._thermal_limit_a = thermal_limit_a

        # store environment modifications
        self._injection = None
        self._maintenance = None
        self._hazards = None
        self._env_modification = None

        # to use the data
        self.done = False
        self.current_reward = None
        self._helper_action_env = None
        self.chronics_handler = None
        self._game_rules = None
        self._helper_action_player = None

        self._rewardClass = None
        self._actionClass = None
        self._observationClass = None
        self._legalActClass = None
        self._helper_observation = None
        self._names_chronics_to_backend = None
        self._reward_helper = None

        # gym compatibility
        self.reward_range = None, None
        self.viewer = None
        self.viewer_fig = None

        # other rewards
        self.other_rewards = {}
        for k, v in other_rewards.items():
            if not issubclass(v, BaseReward):
                raise Grid2OpException("All values of \"rewards\" key word argument should be classes that inherit "
                                       "from \"grid2op.BaseReward\"")
            if not isinstance(k, str):
                raise Grid2OpException("All keys of \"rewards\" should be of string type.")
            self.other_rewards[k] = RewardHelper(v)

        # opponent
        self._opponent_action_class = opponent_action_class  # class of the action of the opponent
        self._opponent_class = opponent_class  # class of the opponent
        self._opponent_init_budget = dt_float(opponent_init_budget)
        self._opponent_attack_duration = dt_int(opponent_attack_duration)
        self._opponent_attack_cooldown = dt_int(opponent_attack_cooldown)
        self._opponent_budget_per_ts = dt_float(opponent_budget_per_ts)
        self._kwargs_opponent = kwargs_opponent
        self._opponent_budget_class = opponent_budget_class

        # below initialized by _create_env, above: need to be called
        self._opponent_action_space = None
        self._compute_opp_budget = None
        self._opponent = None
        self._oppSpace = None

        # voltage
        self._voltagecontrolerClass = voltagecontrolerClass
        self._voltage_controler = None

        # backend
        self._init_grid_path = None

        # backend action
        self._backend_action_class = None
        self._backend_action = None

        # specific to Basic Env, do not change
        self.backend = None
        self.__is_init = False
        self.debug_dispatch = False

    def _create_opponent(self):
        if not self.__is_init:
            raise EnvError("Impossible to create an opponent with a non initialized environment!")

        if not issubclass(self._opponent_action_class, BaseAction):
            raise EnvError("Impossible to make an environment with an opponent action class not derived from BaseAction")
        try:
            self._opponent_init_budget = dt_float(self._opponent_init_budget)
        except Exception as e:
            raise EnvError("Impossible to convert \"opponent_init_budget\" to a float with error {}".format(e))
        if self._opponent_init_budget < 0.:
            raise EnvError("If you want to deactivate the opponent, please don't set its budget to a negative number."
                           "Prefer the use of the DontAct action type (\"opponent_action_class=DontAct\" "
                           "and / or set its budget to 0.")
        if not issubclass(self._opponent_class, BaseOpponent):
            raise EnvError("Impossible to make an opponent with a type that does not inherit from BaseOpponent.")

        self._opponent_action_space = self._helper_action_class(gridobj=self.backend,
                                                                legal_action=AlwaysLegal,
                                                                actionClass=self._opponent_action_class)

        self._compute_opp_budget = self._opponent_budget_class(self._opponent_action_space)
        self._opponent = self._opponent_class(self._opponent_action_space)
        self._oppSpace = OpponentSpace(compute_budget=self._compute_opp_budget,
                                       init_budget=self._opponent_init_budget,
                                       attack_duration=self._opponent_attack_duration,
                                       attack_cooldown=self._opponent_attack_cooldown,
                                       budget_per_timestep=self._opponent_budget_per_ts,
                                       opponent=self._opponent
                                       )
        self._oppSpace.init_opponent(**self._kwargs_opponent)
        self._oppSpace.reset()

    def _has_been_initialized(self):
        # type of power flow to play
        # if True, then it will not disconnect lines above their thermal limits
        self.__class__ = self.init_grid(self.backend)  # create the proper environment class for this specific environment
        if np.min([self.n_line, self.n_gen, self.n_load, self.n_sub]) <= 0:
            raise EnvironmentError("Environment has not been initialized properly")
        self._backend_action_class = _BackendAction.init_grid(self.backend)
        self._backend_action = self._backend_action_class()

        self._no_overflow_disconnection = self.parameters.NO_OVERFLOW_DISCONNECTION
        self._timestep_overflow = np.zeros(shape=(self.n_line,), dtype=dt_int)
        self._nb_timestep_overflow_allowed = np.full(shape=(self.n_line,),
                                                    fill_value=self.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED,
                                                    dtype=dt_int)
        # store actions "cooldown"
        self._times_before_line_status_actionable = np.zeros(shape=(self.n_line,), dtype=dt_int)
        self._max_timestep_line_status_deactivated = self.parameters.NB_TIMESTEP_COOLDOWN_LINE

        self._times_before_topology_actionable = np.zeros(shape=(self.n_sub,), dtype=dt_int)
        self._max_timestep_topology_deactivated = self.parameters.NB_TIMESTEP_COOLDOWN_SUB

        # hazard (not used outside of this class, information is given in `times_before_line_status_actionable`
        self._hazard_duration = np.zeros(shape=(self.n_line,), dtype=dt_int)

        # hard overflow part
        self._hard_overflow_threshold = self.parameters.HARD_OVERFLOW_THRESHOLD
        self._env_dc = self.parameters.ENV_DC

        # initialize maintenance / hazards
        self._time_next_maintenance = np.full(self.n_line, -1, dtype=dt_int)
        self._duration_next_maintenance = np.zeros(shape=(self.n_line,), dtype=dt_int)
        self._times_before_line_status_actionable = np.full(shape=(self.n_line,), fill_value=0, dtype=dt_int)

        # create the vector to the proper shape
        self._target_dispatch = np.zeros(self.n_gen, dtype=dt_float)
        self._actual_dispatch = np.zeros(self.n_gen, dtype=dt_float)
        self._gen_uptime = np.zeros(self.n_gen, dtype=dt_int)
        self._gen_downtime = np.zeros(self.n_gen, dtype=dt_int)
        self._gen_activeprod_t = np.zeros(self.n_gen, dtype=dt_float)
        self._gen_activeprod_t_redisp = np.zeros(self.n_gen, dtype=dt_float)

        self._reset_redispatching()
        self.__is_init = True

    def reset(self):
        """
        Reset the base environment (set the appropriate variables to correct initialization).
        It is (and must be) overloaded in other :class:`grid2op.Environment`
        """
        self.__is_init = True
        self.current_obs = None
        self._line_status[:] = True

    def seed(self, seed=None):
        """
        Set the seed of this :class:`Environment` for a better control and to ease reproducible experiments.

        Parameters
        ----------
        seed: ``int``
           The seed to set.

        Returns
        ---------
        seed: ``tuple``
            The seed used to set the prng (pseudo random number generator) for the environment
        seed_chron: ``tuple``
            The seed used to set the prng for the chronics_handler (if any), otherwise ``None``
        seed_obs: ``tuple``
            The seed used to set the prng for the observation space (if any), otherwise ``None``
        seed_action_space: ``tuple``
            The seed used to set the prng for the action space (if any), otherwise ``None``
        seed_env_modif: ``tuple``
            The seed used to set the prng for the modification of th environment (if any otherwise ``None``)
        seed_volt_cont: ``tuple``
            The seed used to set the prng for voltage controler (if any otherwise ``None``)
        seed_opponent: ``tuple``
            The seed used to set the prng for the opponent (if any otherwise ``None``)

        Examples
        ---------

        Seeding an environment should be done with:

        .. code-block:: python

            import grid2op
            env = grid2op.make()
            env.seed(0)
            obs = env.reset()

        As long as the environment instance (variable `env` in the above code) is not `reset` the `env.seed` has no
        real effect (but can have side effect).

        For a full control on the seed mechanism it is more than advised to reset it after it has been seeded.

        """
        try:
            seed = np.array(seed).astype(dt_int)
        except Exception as exc_:
            raise Grid2OpException("Impossible to seed with the seed provided. Make sure it can be converted to a"
                                   "numpy 64 integer.")
        # example from gym
        # self.np_random, seed = seeding.np_random(seed)
        # inspiration from @ https://github.com/openai/gym/tree/master/gym/utils

        super().seed(seed)
        seed_chron = None
        seed_obs = None
        seed_action_space = None
        seed_env_modif = None
        seed_volt_cont = None
        seed_opponent = None
        max_int = np.iinfo(dt_int).max
        if self.chronics_handler is not None:
            seed = self.space_prng.randint(max_int)
            seed_chron = self.chronics_handler.seed(seed)
        if self._helper_observation is not None:
            seed = self.space_prng.randint(max_int)
            seed_obs = self._helper_observation.seed(seed)
        if self._helper_action_player is not None:
            seed = self.space_prng.randint(max_int)
            seed_action_space = self._helper_action_player.seed(seed)
        if self._helper_action_env is not None:
            seed = self.space_prng.randint(max_int)
            seed_env_modif = self._helper_action_env.seed(seed)
        if self._voltage_controler is not None:
            seed = self.space_prng.randint(max_int)
            seed_volt_cont = self._voltage_controler.seed(seed)
        if self._opponent is not None:
            seed = self.space_prng.randint(max_int)
            seed_opponent = self._opponent.seed(seed)
        return (seed, seed_chron, seed_obs, seed_action_space, seed_env_modif, seed_volt_cont, seed_opponent)

    def deactivate_forecast(self):
        """
        This function will have the effect to deactivate the `obs.simulate`, the forecast will not be updated
        in the observation space.

        This will most likely lead to some performance increase (~10-15% faster) if you don't use the
        `obs.simulate` function.

        Notes
        ------
        If you really don't want to use the `obs.simulate` functionality, you should rather disable it at the creation
        of the environment. For example, if you use the recommended `make` function, you can pass an argument
        that will ignore the chronics even when reading it (using `GridStateFromFile` instead of
        `GridStateFromFileWithForecast` for example) this would give something like:

        .. code-block:: python

            import grid2op
            from grid2op.Chronics import GridStateFromFile
            # tell grid2op not to read the "forecast"
            env = grid2op.make("rte_case14_realistic", data_feeding_kwargs={"gridvalueClass": GridStateFromFile})

            do_nothing_action = env.action_space()

            # improve speed ups to not even try to use forecast
            env.deactivate_forecast()

            # this is normal behavior
            obs = env.reset()

            # but this will make the programm stop working
            # obs.simulate(do_nothing_action)  # DO NOT RUN IT RAISES AN ERROR

        """
        if self._helper_observation is not None:
            self._helper_observation.with_forecast = False
        self.with_forecast = False

    def reactivate_forecast(self):
        """
        This function will have the effect to reactivate the `obs.simulate`, the forecast will be updated
        in the observation space.

        This will most likely lead to some performance decrease but you will be able to use `obs.simulate` function.

        Notes
        ------
        You can use this function as followed:

        .. code-block:: python

            import grid2op
            from grid2op.Chronics import GridStateFromFile
            # tell grid2op not to read the "forecast"
            env = grid2op.make("rte_case14_realistic", data_feeding_kwargs={"gridvalueClass": GridStateFromFile})

            do_nothing_action = env.action_space()

            # improve speed ups to not even try to use forecast
            env.deactivate_forecast()

            # this is normal behavior
            obs = env.reset()

            # but this will make the programm stop working
            # obs.simulate(do_nothing_action)  # DO NOT RUN IT RAISES AN ERROR

            env.reactivate_forecast()
            obs, reward, done, info = env.step(do_nothing_action)

            # and now forecast are available again
            simobs, sim_r, sim_d, sim_info = obs.simulate(do_nothing_action)

        """
        if self._helper_observation is not None:
            self._helper_observation.with_forecast = True
        self.with_forecast = True

    @abstractmethod
    def _init_backend(self, init_grid_path, chronics_handler, backend,
                      names_chronics_to_backend, actionClass, observationClass,
                      rewardClass, legalActClass):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        This method is used for Environment specific implementation. Only use it if you know exactly what
        you are doing.

        """
        pass

    def set_thermal_limit(self, thermal_limit):
        """
        Set the thermal limit effectively.

        Parameters
        ----------
        thermal_limit: ``numpy.ndarray``
            The new thermal limit. It must be a numpy ndarray vector (or convertible to it). For each powerline it
            gives the new thermal limit.

        Examples
        ---------

        This function can be used like this:

        .. code-block:: python

            import grid2op

            # I create an environment
            env = grid2op.make("rte_case5_example", test=True)

            # i set the thermal limit of each powerline to 20000 amps
            env.set_thermal_limit([20000 for _ in range(env.n_line)])


        """
        if not self.__is_init:
            raise Grid2OpException("Impossible to set the thermal limit to a non initialized Environment")
        try:
            tmp = np.array(thermal_limit).flatten().astype(dt_float)
        except Exception as e:
            raise Grid2OpException("Impossible to convert the vector as input into a 1d numpy float array.")
        if tmp.shape[0] != self.n_line:
            raise Grid2OpException("Attempt to set thermal limit on {} powerlines while there are {}"
                                   "on the grid".format(tmp.shape[0], self.n_line))
        if np.any(~np.isfinite(tmp)):
            raise Grid2OpException("Impossible to use non finite value for thermal limits.")

        self._thermal_limit_a = tmp
        self.backend.set_thermal_limit(self._thermal_limit_a)

    def _reset_redispatching(self):
        # redispatching
        self._target_dispatch[:] = 0.
        self._actual_dispatch[:] = 0.
        self._gen_uptime[:] = 0
        self._gen_downtime[:] = 0
        self._gen_activeprod_t[:] = 0.
        self._gen_activeprod_t_redisp[:] = 0.

    def _get_new_prod_setpoint(self, action):
        # get the modification of generator active setpoint from the action
        new_p = 1. * self._gen_activeprod_t
        if "prod_p" in action._dict_inj:
            tmp = action._dict_inj["prod_p"]
            indx_ok = np.isfinite(tmp)
            new_p[indx_ok] = tmp[indx_ok]

        # modification of the environment always override the modification of the agents (if any)
        # TODO have a flag there if this is the case.
        if "prod_p" in self._env_modification._dict_inj:
            # modification of the production setpoint value
            tmp = self._env_modification._dict_inj["prod_p"]
            indx_ok = np.isfinite(tmp)
            new_p[indx_ok] = tmp[indx_ok]
        return new_p

    def _get_already_modified_gen(self, action):

        redisp_act_orig = 1. * action._redispatch

        already_modified_gen = self._target_dispatch != 0.
        self._target_dispatch[already_modified_gen] += redisp_act_orig[already_modified_gen]
        first_modified = (~already_modified_gen) & (redisp_act_orig != 0)
        self._target_dispatch[first_modified] = self._actual_dispatch[first_modified] + redisp_act_orig[first_modified]
        already_modified_gen |= first_modified
        return already_modified_gen

    def _prepare_redisp(self, action, new_p, already_modified_gen):
        # trying with an optimization method
        except_ = None
        info_ = []
        valid = True

        # get the redispatching action (if any)
        redisp_act_orig = 1. * action._redispatch

        if np.all(redisp_act_orig == 0.) and np.all(self._target_dispatch == 0.) and np.all(self._actual_dispatch == 0.):
            return valid, except_, info_

        # check that everything is consistent with pmin, pmax:
        if np.any(self._target_dispatch > self.gen_pmax - self.gen_pmin):
            # action is invalid, the target redispatching would be above pmax for at least a generator
            cond_invalid = self._target_dispatch > self.gen_pmax - self.gen_pmin
            except_ = InvalidRedispatching("You cannot ask for a dispatch higher than pmax - pmin  [it would be always "
                                           "invalid because, even if the sepoint is pmin, this dispatch would set it "
                                           "to a number higher than pmax, which is impossible]. Invalid dispatch for "
                                           "generator(s): "
                                           "{}".format(np.where(cond_invalid)[0]))
            self._target_dispatch -= redisp_act_orig
            return valid, except_, info_
        if np.any(self._target_dispatch < self.gen_pmin - self.gen_pmax):
            # action is invalid, the target redispatching would be below pmin for at least a generator
            cond_invalid = self._target_dispatch < self.gen_pmin - self.gen_pmax
            except_ = InvalidRedispatching("You cannot ask for a dispatch lower than pmin - pmax  [it would be always "
                                           "invalid because, even if the sepoint is pmax, this dispatch would set it "
                                           "to a number bellow pmin, which is impossible]. Invalid dispatch for "
                                           "generator(s): "
                                           "{}".format(np.where(cond_invalid)[0]))
            self._target_dispatch -= redisp_act_orig
            return valid, except_, info_

        # i can't redispatch turned off generators [turned off generators need to be turned on before redispatching]
        if np.any(redisp_act_orig[new_p == 0.]) and self._forbid_dispatch_off:
            # action is invalid, a generator has been redispatched, but it's turned off
            except_ = InvalidRedispatching("Impossible to dispatch a turned off generator")
            self._target_dispatch -= redisp_act_orig
            return valid, except_, info_

        if self._forbid_dispatch_off is True:
            redisp_act_orig_cut = 1.0 * redisp_act_orig
            redisp_act_orig_cut[new_p == 0.] = 0.
            if np.any(redisp_act_orig_cut != redisp_act_orig):
                info_.append({"INFO: redispatching cut because generator will be turned_off":
                              np.where(redisp_act_orig_cut != redisp_act_orig)[0]})
        return valid, except_, info_

    def _make_redisp(self, already_modified_gen, new_p):
        except_ = None
        info_ = []
        valid = True
        mismatch = self._actual_dispatch - self._target_dispatch
        mismatch = np.abs(mismatch)
        if np.abs(np.sum(self._actual_dispatch)) >= self._tol_poly or \
           np.max(mismatch) >= self._tol_poly:
            except_ = self._compute_dispatch_vect(already_modified_gen, new_p)
            valid = except_ is None
        return valid, except_, info_

    def _compute_dispatch_vect(self, already_modified_gen, new_p):
        except_ = None
        # first i define the participating generators
        # these are the generators that will be adjusted for redispatching
        gen_participating = (new_p > 0.) | (self._actual_dispatch != 0.) | (self._target_dispatch != self._actual_dispatch)
        gen_participating[~self.gen_redispatchable] = False

        # define the objective value
        target_vals = self._target_dispatch[gen_participating] - self._actual_dispatch[gen_participating]
        already_modified_gen_me = already_modified_gen[gen_participating]
        target_vals_me = target_vals[already_modified_gen_me]
        nb_dispatchable = np.sum(gen_participating)
        tmp_zeros = np.zeros((1, nb_dispatchable), dtype=dt_float)
        coeffs = 1.0 / (self.gen_max_ramp_up + self.gen_max_ramp_down + self._epsilon_poly)
        weights = np.ones(nb_dispatchable) * coeffs[gen_participating]
        weights /= weights.sum()

        if target_vals_me.shape[0] == 0:
            # no dispatch means all dispatchable, otherwise i will never get to 0
            already_modified_gen_me[:] = True
            target_vals_me = target_vals[already_modified_gen_me]

        # for numeric stability
        # to scale the input also:
        # see https://stackoverflow.com/questions/11155721/positive-directional-derivative-for-linesearch
        scale_x = max(np.max(np.abs(self._actual_dispatch)), 1.0)
        scale_x = dt_float(scale_x)
        target_vals_me_optim = 1.0 * (target_vals_me / scale_x)
        target_vals_me_optim = target_vals_me_optim.astype(dt_float)

        # see https://stackoverflow.com/questions/11155721/positive-directional-derivative-for-linesearch
        # where they advised to scale the function
        scale_objective = max(0.5 * np.sum(np.abs(target_vals_me_optim))**2, 1.0)
        scale_objective = np.round(scale_objective, decimals=4)
        scale_objective = dt_float(scale_objective)

        # add the "sum to 0"
        mat_sum_0_no_turn_on = np.ones((1, nb_dispatchable), dtype=dt_float)
        const_sum_0_no_turn_on = np.zeros(1, dtype=dt_float)

        # gen increase in the chronics
        new_p_th = new_p[gen_participating] + self._actual_dispatch[gen_participating]
        incr_in_chronics = new_p - (self._gen_activeprod_t_redisp - self._actual_dispatch)

        # minimum value available for disp
        ## first limit delta because of pmin
        p_min_const = self.gen_pmin[gen_participating] - new_p_th
        ## second limit delta because of ramps
        ramp_down_const = -self.gen_max_ramp_down[gen_participating] - incr_in_chronics[gen_participating]
        ## take max of the 2
        min_disp = np.maximum(p_min_const, ramp_down_const)
        min_disp = min_disp.astype(dt_float)

        # maximum value available for disp
        ## first limit delta because of pmin
        p_max_const = self.gen_pmax[gen_participating] - new_p_th
        ## second limit delta because of ramps
        ramp_up_const = self.gen_max_ramp_up[gen_participating] - incr_in_chronics[gen_participating]
        ## take min of the 2
        max_disp = np.minimum(p_max_const, ramp_up_const)
        max_disp = max_disp.astype(dt_float)

        # add everything into a linear constraint object
        # equality
        added = 0.5 * self._epsilon_poly
        equality_const = LinearConstraint(mat_sum_0_no_turn_on,  # do the sum
                                          (const_sum_0_no_turn_on ) / scale_x,  # lower bound
                                          (const_sum_0_no_turn_on ) / scale_x  # upper bound
                                          )
        mat_pmin_max_ramps = np.eye(nb_dispatchable)
        ineq_const = LinearConstraint(mat_pmin_max_ramps,
                                      (min_disp - added) / scale_x,
                                      (max_disp + added) / scale_x)

        # check if the constraints are violated
        ## total available "juice" to go down (incl ramp and pmin / pmax)
        p_min_down = self.gen_pmin[gen_participating] - self._gen_activeprod_t_redisp[gen_participating]
        avail_down = np.maximum(p_min_down, -self.gen_max_ramp_down[gen_participating])
        ## total available "juice" to go up (incl. ramp and pmin / pmax)
        p_max_up = self.gen_pmax[gen_participating] - self._gen_activeprod_t_redisp[gen_participating]
        avail_up = np.minimum(p_max_up, self.gen_max_ramp_up[gen_participating])
        except_ = self._detect_infeasible_dispatch(incr_in_chronics[gen_participating], avail_down, avail_up)
        if except_ is not None:
            return except_

        # choose a good initial point (close to the solution)
        # the idea here is to chose a initial point that would be close to the
        # desired solution (split the (sum of the) dispatch to the available generators)
        x0 = (self._target_dispatch[gen_participating] - self._actual_dispatch[gen_participating]) / scale_x
        can_adjust = x0 == 0.
        if np.any(can_adjust):
            init_sum = np.sum(x0)
            denom_adjust = np.sum(1. / weights[can_adjust])
            if denom_adjust <= 1e-2:
                # i don't want to divide by something to cloose to 0.
                denom_adjust = 1.0
            x0[can_adjust] = - init_sum / (weights[can_adjust] * denom_adjust)

        def target(actual_dispatchable):
            # define my real objective
            quad_ = (actual_dispatchable[already_modified_gen_me] - target_vals_me_optim) ** 2
            coeffs_quads = weights[already_modified_gen_me] * quad_
            coeffs_quads_const = coeffs_quads.sum()
            coeffs_quads_const /= scale_objective  # scaling the function
            return coeffs_quads_const

        def jac(actual_dispatchable):
            res_jac = 1.0 * tmp_zeros
            res_jac[0, already_modified_gen_me] = 2.0 * weights[already_modified_gen_me] * \
                                              (actual_dispatchable[already_modified_gen_me] - target_vals_me_optim)
            res_jac /= scale_objective  # scaling the function
            return res_jac

        # objective function
        def f(init):
            this_res = minimize(target,
                                init,
                                method="SLSQP",
                                constraints=[equality_const, ineq_const],
                                options={'eps': max(self._epsilon_poly / scale_x, 1e-6),
                                         "ftol": max(self._epsilon_poly / scale_x, 1e-6),
                                         'disp': False},
                                jac=jac
                                # hess=hess  # not used for SLSQP
                                )
            return this_res
        res = f(x0)
        if res.success:
            self._actual_dispatch[gen_participating] += res.x * scale_x
        else:
            # check if constraints are "approximately" met
            mat_const = np.concatenate((mat_sum_0_no_turn_on, mat_pmin_max_ramps))
            downs = np.concatenate((const_sum_0_no_turn_on / scale_x, (min_disp - added) / scale_x))
            ups = np.concatenate((const_sum_0_no_turn_on / scale_x, (max_disp + added) / scale_x))
            vals = np.matmul(mat_const, res.x)
            ok_down = np.all(vals - downs >= -self._tol_poly)  # i don't violate "down" constraints
            ok_up = np.all(vals - ups <= self._tol_poly)
            if ok_up and ok_down:
                # it's ok i can tolerate "small" perturbations
                self._actual_dispatch[gen_participating] += res.x * scale_x
            else:
                # TODO try with another method here, maybe
                error_dispatch = "Redispatching automaton terminated with error (no more information available " \
                                 "at this point):\n\"{}\"".format(res.message)
                except_ = InvalidRedispatching(error_dispatch)
        return except_

    def _detect_infeasible_dispatch(self, incr_in_chronics, avail_down, avail_up):
        """This function is an attempt to give more detailed log by detecting infeasible dispatch"""
        except_ = None
        sum_move = np.sum(incr_in_chronics)
        avail_down_sum = np.sum(avail_down)
        avail_up_sum = np.sum(avail_up)
        gen_setpoint = self._gen_activeprod_t_redisp[self.gen_redispatchable]
        if sum_move > avail_up_sum:
            # infeasible because too much is asked
            msg = DETAILED_REDISP_ERR_MSG.format(sum_move=sum_move,
                                                 avail_up_sum=avail_up_sum,
                                                 gen_setpoint=np.round(gen_setpoint, decimals=2),
                                                 ramp_up=self.gen_max_ramp_up[self.gen_redispatchable],
                                                 gen_pmax=self.gen_pmax[self.gen_redispatchable],
                                                 avail_up=np.round(avail_up, decimals=2),
                                                 increase="increase",
                                                 decrease="decrease",
                                                 maximum="maximum",
                                                 pmax="pmax",
                                                 max_ramp_up="max_ramp_up")
            except_ = InvalidRedispatching(msg)
        elif sum_move < avail_down_sum:
            # infeasible because not enough is asked
            msg = DETAILED_REDISP_ERR_MSG.format(sum_move=sum_move,
                                                 avail_up_sum=avail_down_sum,
                                                 gen_setpoint=np.round(gen_setpoint, decimals=2),
                                                 ramp_up=self.gen_max_ramp_down[self.gen_redispatchable],
                                                 gen_pmax=self.gen_pmin[self.gen_redispatchable],
                                                 avail_up=np.round(avail_up, decimals=2),
                                                 increase="decrease",
                                                 decrease="increase",
                                                 maximum="minimum",
                                                 pmax="pmin",
                                                 max_ramp_up="max_ramp_down"
                                                 )
            except_ = InvalidRedispatching(msg)
        return except_

    def _update_actions(self):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Retrieve the actions to perform the update of the underlying powergrid represented by
        the :class:`grid2op.Backend`in the next time step.

        A call to this function will also read the next state of :attr:`chronics_handler`, so it must be called only
        once per time step.

        Returns
        --------
        res: :class:`grid2op.Action.Action`
            The action representing the modification of the powergrid induced by the Backend.
        """
        timestamp, tmp, maintenance_time, maintenance_duration, hazard_duration, prod_v = self.chronics_handler.next_time_step()
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
        self._duration_next_maintenance = maintenance_duration
        self._time_next_maintenance = maintenance_time
        self._hazard_duration = hazard_duration
        return self._helper_action_env({"injection": self._injection, "maintenance": self._maintenance,
                                       "hazards": self._hazards}), prod_v

    def _update_time_reconnection_hazards_maintenance(self):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        This supposes that :attr:`Environment.times_before_line_status_actionable` is already updated
        with the cascading failure, soft overflow and hard overflow.

        It also supposes that :func:`Environment._update_actions` has been called, so that the vectors
        :attr:`Environment.duration_next_maintenance`, :attr:`Environment._time_next_maintenance` and
        :attr:`Environment._hazard_duration` are updated with the most recent values.
        Finally the Environment supposes that this method is called before calling :func:`Environment.get_obs`
        This function integrates the hazards and maintenance in the
        :attr:`Environment.times_before_line_status_actionable` vector.

        For example, if a powerline `i` has no problem
        of overflow, but is affected by a hazard, :attr:`Environment.times_before_line_status_actionable`
        should be updated with the duration of this hazard (stored in one of the three vector mentionned in the
        above paragraph)
        For this Environment, we suppose that the maximum of the 3 values are taken into account. The reality would
        be more complicated.

        """
        first_time_maintenance = self._time_next_maintenance == 0
        self._times_before_line_status_actionable[first_time_maintenance] = np.maximum(
            self._times_before_line_status_actionable[first_time_maintenance],
            self._duration_next_maintenance[first_time_maintenance])
        self._times_before_line_status_actionable[:] = np.maximum(self._times_before_line_status_actionable,
                                                                  self._hazard_duration)

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
        res = self._helper_action_env()
        if prod_v_chronics is not None:
            res.update({"injection": {"prod_v": prod_v_chronics}})
        return res

    def _handle_updown_times(self, gen_up_before, redisp_act):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Handles the up and down tims for the generators.
        """
        # get the generators that are not connected after the action
        except_ = None

        # computes which generator will be turned on after the action
        gen_up_after = 1.0 * self._gen_activeprod_t
        if "prod_p" in self._env_modification._dict_inj:
            tmp = self._env_modification._dict_inj["prod_p"]
            indx_ok = np.isfinite(tmp)
            gen_up_after[indx_ok] = self._env_modification._dict_inj["prod_p"][indx_ok]
        gen_up_after += redisp_act
        gen_up_after = gen_up_after > 0.

        # update min down time, min up time etc.
        gen_disconnected_this = gen_up_before & (~gen_up_after)
        gen_connected_this_timestep = (~gen_up_before) & (gen_up_after)
        gen_still_connected = gen_up_before & gen_up_after
        gen_still_disconnected = (~gen_up_before) & (~gen_up_after)

        if np.any(self._gen_downtime[gen_connected_this_timestep] < self.gen_min_downtime[gen_connected_this_timestep]) \
                and not self._ignore_min_up_down_times:
            # i reconnected a generator before the minimum time allowed
            id_gen = self._gen_downtime[gen_connected_this_timestep] < self.gen_min_downtime[gen_connected_this_timestep]
            id_gen = np.where(id_gen)[0]
            id_gen = np.where(gen_connected_this_timestep[id_gen])[0]
            except_ = GeneratorTurnedOnTooSoon("Some generator has been connected too early ({})".format(id_gen))
            return except_
        else:
            self._gen_downtime[gen_connected_this_timestep] = -1
            self._gen_uptime[gen_connected_this_timestep] = 1

        if np.any(self._gen_uptime[gen_disconnected_this] < self.gen_min_uptime[gen_disconnected_this]) and \
                not self._ignore_min_up_down_times:
            # i disconnected a generator before the minimum time allowed
            id_gen = self._gen_uptime[gen_disconnected_this] < self.gen_min_uptime[gen_disconnected_this]
            id_gen = np.where(id_gen)[0]
            id_gen = np.where(gen_connected_this_timestep[id_gen])[0]
            except_ = GeneratorTurnedOffTooSoon("Some generator has been disconnected too early ({})".format(id_gen))
            return except_
        else:
            self._gen_downtime[gen_connected_this_timestep] = 0
            self._gen_uptime[gen_connected_this_timestep] = 1

        self._gen_uptime[gen_still_connected] += 1
        self._gen_downtime[gen_still_disconnected] += 1
        return except_

    def get_obs(self):
        """
        Return the observations of the current environment made by the :class:`grid2op.BaseAgent.BaseAgent`.

        Returns
        -------
        res: :class:`grid2op.Observation.Observation`
            The current BaseObservation given to the :class:`grid2op.BaseAgent.BaseAgent` / bot / controler.

        Examples
        ---------

        This function can be use at any moment, even if the actual observation is not present.

        .. code-block:: python

            import grid2op

            # I create an environment
            env = grid2op.make()

            obs = env.reset()

            # have a big piece of code
            obs2 = env.get_obs()

            # obs2 and obs are identical.

        """
        res = self._helper_observation(env=self)
        return res

    def get_thermal_limit(self):
        """
        Get the current thermal limit in amps registered for the environment.

        Examples
        ---------

        It can be used like this:

        .. code-block:: python

            import grid2op

            # I create an environment
            env = grid2op.make()

            thermal_limits = env.get_thermal_limit()

        """
        return 1.0 * self._thermal_limit_a

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).

        If the :class:`grid2op.BaseAction.BaseAction` is illegal or ambiguous, the step is performed, but the action is
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
                dictionary with keys:

                    - "disc_lines": a numpy array (or ``None``) saying, for each powerline if it has been disconnected
                        due to overflow
                    - "is_illegal" (``bool``) whether the action given as input was illegal
                    - "is_ambiguous" (``bool``) whether the action given as input was ambiguous.
                    - "is_dispatching_illegal" (``bool``) was the action illegal due to redispatching
                    - "is_illegal_reco" (``bool``) was the action illegal due to a powerline reconnection
                    - "opponent_attack_line" (``np.ndarray``, ``bool``) for each powerline, say if the opponent
                      attacked it (``True``) or not (``False``).
                    - "opponent_attack_sub" (``np.ndarray``, ``bool``) for each substation, say if the opponent
                      attacked it (``True``) or not (``False``).
                    - "opponent_attack_duration" (``int``) the duration of the current attack (if any)
                    - "exception" (``list`` of :class:`Exceptions.Exceptions.Grid2OpException` if an exception was
                      raised  or ``[]`` if everything was fine.)
                    - "detailed_infos_for_cascading_failures" (optional, only if the backend has been create with
                      `detailed_infos_for_cascading_failures=True`) the list of the intermediate steps computed during
                      the simulation of the "cascading failures".

        Examples
        ---------

        As any openAI gym environment, this is used like:

        .. code-block:: python

            import grid2op
            from grid2op.Agent import RandomAgent

            # I create an environment
            env = grid2op.make()

            # define an agent here, this is an example
            agent = RandomAgent(env.action_space)

            # environment need to be "reset" before usage:
            obs = env.reset()
            reward = env.reward_range[0]
            done = False

            # now run through each steps like this
            while not done:
                action = agent.act(obs, reward, done)
                obs, reward, done, info = env.step(action)

        """

        if not self.__is_init:
            raise Grid2OpException("Impossible to make a step with a non initialized backend. Have you called "
                                   "\"env.reset()\" after the last game over ?")

        has_error = True
        is_done = False
        disc_lines = None
        is_illegal = False
        is_ambiguous = False
        is_illegal_redisp = False
        is_illegal_reco = False
        attack = None
        except_ = []
        detailed_info = []
        init_disp = 1.0 * action._redispatch
        attack_duration = 0
        lines_attacked, subs_attacked = None, None
        conv_ = None
        init_line_status = copy.deepcopy(self.backend.get_line_status())
        try:
            beg_ = time.time()
            is_legal, reason = self._game_rules(action=action, env=self)
            if not is_legal:
                # action is replace by do nothing
                action = self._helper_action_player({})
                except_.append(reason)
                is_illegal = True

            ambiguous, except_tmp = action.is_ambiguous()
            if ambiguous:
                # action is replace by do nothing
                action = self._helper_action_player({})
                is_ambiguous = True
                except_.append(except_tmp)

            # get the modification of generator active setpoint from the environment
            self._env_modification, prod_v_chronics = self._update_actions()
            self._env_modification._single_act = False  # because it absorbs all redispatching actions
            new_p = self._get_new_prod_setpoint(action)

            if self.redispatching_unit_commitment_availble:
                # remember generator that were "up" before the action
                gen_up_before = self._gen_activeprod_t > 0.

                # compute the redispatching and the new productions active setpoint
                beg__redisp = time.time()
                already_modified_gen = self._get_already_modified_gen(action)
                valid_disp, except_tmp, info_ = self._prepare_redisp(action, new_p, already_modified_gen)

                if except_tmp is not None:
                    action = self._helper_action_player({})
                    is_illegal_redisp = True
                    except_.append(except_tmp)

                valid_disp, except_tmp, info_ = self._make_redisp(already_modified_gen, new_p)
                if not valid_disp or except_tmp is not None:
                    # game over case (divergence of the scipy routine to compute redispatching)
                    action = self._helper_action_player({})
                    is_illegal_redisp = True
                    except_.append(except_tmp)
                    is_done = True
                    except_.append("Game over due to infeasible redispatching state. "
                                   "The routine used to compute the \"next state\" has diverged. "
                                   "This means that there is no way to compute a physically valid generator state "
                                   "(one that meets all pmin / pmax - ramp min / ramp max with the information "
                                   "provided. As one of the physical constraints would be violated, this means that "
                                   "a generator would be damaged in real life. This is a game over.")

                # check the validity of min downtime and max uptime
                except_tmp = self._handle_updown_times(gen_up_before, self._actual_dispatch)
                if except_tmp is not None:
                    is_illegal_reco = True
                    action = self._helper_action_player({})
                    except_.append(except_tmp)
                self._time_redisp += time.time() - beg__redisp

            # make sure the dispatching action is not implemented "as is" by the backend.
            # the environment must make sure it's a zero-sum action.
            action._redispatch[:] = 0.
            self._backend_action += action
            action._redispatch[:] = init_disp

            self._backend_action += self._env_modification
            self._backend_action.set_redispatch(self._actual_dispatch)

            # now get the new generator voltage setpoint
            voltage_control_act = self._voltage_control(action, prod_v_chronics)
            self._backend_action += voltage_control_act

            # have the opponent here
            # TODO code the opponent part here and split more the timings! here "opponent time" is
            # included in time_apply_act
            tick = time.time()
            attack, attack_duration = self._oppSpace.attack(observation=self.current_obs,
                                                            agent_action=action,
                                                            env_action=self._env_modification)
            if attack is not None:
                # the opponent choose to attack
                # i update the "cooldown" on these things
                lines_attacked, subs_attacked = attack.get_topological_impact()
                self._times_before_line_status_actionable[lines_attacked] = \
                                np.maximum(attack_duration, self._times_before_line_status_actionable[lines_attacked])
                self._times_before_topology_actionable[subs_attacked] = \
                                np.maximum(attack_duration, self._times_before_topology_actionable[subs_attacked])
                self._backend_action += attack
            self._time_opponent += time.time() - tick
            self.backend.apply_action(self._backend_action)

            self._time_apply_act += time.time() - beg_

            self.nb_time_step += 1
            try:
                # compute the next _grid state
                beg_ = time.time()
                disc_lines, detailed_info, conv_ = self.backend.next_grid_state(env=self, is_dc=self._env_dc)
                self._time_powerflow += time.time() - beg_
                if conv_ is None:
                    beg_ = time.time()
                    self.backend.update_thermal_limit(self)  # update the thermal limit, for DLR for example
                    overflow_lines = self.backend.get_line_overflow()
                    self._backend_action.update_state(disc_lines)

                    # one timestep passed, i can maybe reconnect some lines
                    self._times_before_line_status_actionable[self._times_before_line_status_actionable > 0] -= 1
                    # update the vector for lines that have been disconnected
                    self._times_before_line_status_actionable[disc_lines] = int(self.parameters.NB_TIMESTEP_RECONNECTION)
                    self._update_time_reconnection_hazards_maintenance()

                    # for the powerline that are on overflow, increase this time step
                    self._timestep_overflow[overflow_lines] += 1

                    # set to 0 the number of timestep for lines that are not on overflow
                    self._timestep_overflow[~overflow_lines] = 0

                    # build the topological action "cooldown"
                    aff_lines, aff_subs = action.get_topological_impact(init_line_status)
                    if self._max_timestep_line_status_deactivated > 0:
                        # i update the cooldown only when this does not impact the line disconnected for the
                        # opponent or by maitnenance for example
                        cond = aff_lines  # powerlines i modified
                        # powerlines that are not affected by any other "forced disconnection"
                        cond &= self._times_before_line_status_actionable < self._max_timestep_line_status_deactivated
                        self._times_before_line_status_actionable[cond] = self._max_timestep_line_status_deactivated
                    if self._max_timestep_topology_deactivated > 0:
                        self._times_before_topology_actionable[self._times_before_topology_actionable > 0] -= 1
                        self._times_before_topology_actionable[aff_subs] = self._max_timestep_topology_deactivated

                    # build the observation
                    self.current_obs = self.get_obs()
                    self._time_extract_obs += time.time() - beg_

                    # extract production active value at this time step (should be independant of action class)
                    self._gen_activeprod_t[:], *_ = self.backend.generators_info()
                    # problem with the gen_activeprod_t above, is that the slack bus absorbs alone all the losses
                    # of the system. So basically, when it's too high (higher than the ramp) it can
                    # mess up the rest of the environment
                    self._gen_activeprod_t_redisp[:] = new_p + self._actual_dispatch

                    # set the line status
                    self._line_status[:] = self.backend.get_line_status()

                    has_error = False
            except Grid2OpException as e:
                except_.append(e)
                if self.logger is not None:
                    self.logger.error("Impossible to compute next _grid state with error \"{}\"".format(e))

        except StopIteration:
            # episode is over
            is_done = True

        self._backend_action.reset()
        if conv_ is not None:
            except_.append(conv_)
        infos = {"disc_lines": disc_lines,
                 "is_illegal": is_illegal,
                 "is_ambiguous": is_ambiguous,
                 "is_dispatching_illegal": is_illegal_redisp,
                 "is_illegal_reco": is_illegal_reco,
                 "opponent_attack_line": lines_attacked,
                 "opponent_attack_sub": subs_attacked,
                 "opponent_attack_duration": attack_duration,
                 "exception": except_}
        if self.backend.detailed_infos_for_cascading_failures:
            infos["detailed_infos_for_cascading_failures"] = detailed_info

        self.done = self._is_done(has_error, is_done)
        self.current_reward, other_reward = self._get_reward(action,
                                                             has_error,
                                                             is_done,
                                                             is_illegal or is_illegal_redisp or is_illegal_reco,
                                                             is_ambiguous)
        infos["rewards"] = other_reward
        if has_error:
            # update the observation so when it's plotted everything is "down"
            # generators information
            self.current_obs.set_game_over()

        # TODO documentation on all the possible way to be illegal now
        if self.done:
            self.__is_init = False
        return self.current_obs, self.current_reward, self.done, infos

    def _get_reward(self, action, has_error, is_done, is_illegal, is_ambiguous):
        res = self._reward_helper(action, self, has_error, is_done, is_illegal, is_ambiguous)
        other_rewards = {k: v(action, self, has_error, is_done, is_illegal, is_ambiguous)
                         for k, v in self.other_rewards.items()
                         }
        return res, other_rewards

    def get_reward_instance(self):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Returns the instance of the object that is used to compute the reward.
        """
        return self._reward_helper.template_reward

    def _is_done(self, has_error, is_done):
        no_more_data = self.chronics_handler.done()
        return has_error or is_done or no_more_data

    def _reset_vectors_and_timings(self):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Maintenance are not reset, otherwise the data are not read properly (skip the first time step)
        """
        self._no_overflow_disconnection = self.parameters.NO_OVERFLOW_DISCONNECTION
        self._timestep_overflow[:] = 0
        self._nb_timestep_overflow_allowed[:] = self.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED

        self.nb_time_step = 0
        self._hard_overflow_threshold = self.parameters.HARD_OVERFLOW_THRESHOLD
        self._env_dc = self.parameters.ENV_DC

        self._times_before_line_status_actionable[:] = 0
        self._max_timestep_line_status_deactivated = self.parameters.NB_TIMESTEP_COOLDOWN_LINE

        self._times_before_topology_actionable[:] = 0
        self._max_timestep_topology_deactivated = self.parameters.NB_TIMESTEP_COOLDOWN_SUB

        # reset timings
        self._time_apply_act = dt_float(0.)
        self._time_powerflow = dt_float(0.)
        self._time_extract_obs = dt_float(0.)
        self._time_opponent = dt_float(0.)
        self._time_redisp = dt_float(0.)

        # reward and others
        self.current_reward = self.reward_range[0]
        self.done = False

    def _reset_maintenance(self):
        self._time_next_maintenance[:] = -1
        self._duration_next_maintenance[:] = 0

    def __enter__(self):
        """
        Support *with-statement* for the environment.

        Examples
        --------

        .. code-block:: python

            import grid2op
            import grid2op.BaseAgent
            with grid2op.make() as env:
                agent = grid2op.BaseAgent.DoNothingAgent(env.action_space)
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

    def close(self):
        # todo there might be some side effect
        if self.viewer is not None:
            self.viewer = None
            self.viewer_fig = None

        if self.backend is not None:
            self.backend.close()
        self.backend = None
        self.__is_init = False

    def attach_layout(self, grid_layout):
        """
        Compare to the method of the base class, this one performs a check.
        This method must be called after initialization.

        Parameters
        ----------
        grid_layout: ``dict``
            The layout of the grid (*i.e* the coordinates (x,y) of all substations). The keys
            should be the substation names, and the values a tuple (with two float) representing
            the coordinate of the substation.

        Examples
        ---------
        Here is an example on how to attach a layout for an environment:

        .. code-block:: python

            import grid2op

            # create the environment
            env = grid2op.make()

            # assign coordinates (0., 0.) to all substations (this is a dummy thing to do here!)
            layout = {sub_name: (0., 0.) for sub_name in env.name_sub}
            env.attach_layout(layout)

        """
        if isinstance(grid_layout, dict):
            pass
        elif isinstance(grid_layout, list):
            grid_layout = {k: v for k, v in zip(self.name_sub, grid_layout)}
        else:
            raise EnvError("Attempt to set a layout from something different than a dictionary or a list. "
                           "This is for now not supported.")

        if self.__is_init:
            res = {}
            for el in self.name_sub:
                if not el in grid_layout:
                    raise EnvError("The substation \"{}\" is not present in grid_layout while in the powergrid."
                                   "".format(el))
                tmp = grid_layout[el]
                try:
                    x, y = tmp
                    x = dt_float(x)
                    y = dt_float(y)
                    res[el] = (x, y)
                except Exception as e_:
                    raise EnvError("attach_layout: impossible to convert the value of \"{}\" to a pair of float "
                                   "that will be used the grid layout. The error is: \"{}\""
                                   "".format(el, e_))
            super().attach_layout(res)
            if self._helper_action_player is not None:
                self._helper_action_player.attach_layout(res)
            if self._helper_action_env is not None:
                self._helper_action_env.attach_layout(res)
            if self._helper_observation is not None:
                self._helper_observation.attach_layout(res)
            if self._voltage_controler is not None:
                self._voltage_controler.attach_layout(res)
            if self._opponent_action_space is not None:
                self._opponent_action_space.attach_layout(res)

    def fast_forward_chronics(self, nb_timestep):
        """
        This method allows you to skip some time step at the beginning of the chronics.

        This is usefull at the beginning of the training, if you want your agent to learn on more diverse scenarios.
        Indeed, the data provided in the chronics usually starts always at the same date time (for example Jan 1st at
        00:00). This can lead to suboptimal exploration, as during this phase, only a few time steps are managed by
        the agent, so in general these few time steps will correspond to grid state around Jan 1st at 00:00.

        Parameters
        ----------
        nb_timestep: ``int``
            Number of time step to "fast forward"

        Examples
        ---------
        This can be used like this:

        .. code-block:: python

            import grid2op

            # create the environment
            env = grid2op.make()

            # skip the first 150 steps of the chronics
            env.fast_forward_chronics(150)
            done = env.is_done
            if not done:
                obs = env.get_obs()
                # do something
            else:
                # there was a "game over"
                # you need to reset the env (which will "cancel" the fast_forward)
                pass
                # do something else

        Notes
        -----
        This method can set the state of the environment in a 'game over' state (`done=True`) for example if the
        chronics last `xxx` time steps and you ask to "fast foward" more than `xxx` steps. This is why we advise to
        check the state of the environment after the call to this method if you use it (see the "Examples" paragaph)

        """
        # Go to the timestep requested minus one
        nb_timestep = max(1, nb_timestep - 1)
        self.chronics_handler.fast_forward(nb_timestep)
        self.nb_time_step += nb_timestep

        # Update the timing vectors
        min_time_line_reco = np.zeros(self.n_line, dtype=dt_int)
        min_time_topo = np.zeros(self.n_sub, dtype=dt_int)
        ff_time_line_act = self._times_before_line_status_actionable - nb_timestep
        ff_time_topo_act = self._times_before_topology_actionable - nb_timestep
        self._times_before_line_status_actionable[:] = np.maximum(ff_time_line_act, min_time_line_reco)
        self._times_before_topology_actionable[:] = np.maximum(ff_time_topo_act, min_time_topo)

        # Update to the fast forward state using a do nothing action
        self.step(self._helper_action_player({}))

    def get_current_line_status(self):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            prefer using :attr:`grid2op.BaseObservation.line_status`

        This method allows to retrieve the line status.
        """
        if self.current_obs is not None:
            powerline_status = self._line_status
        else:
            # at first time step, every powerline is connected
            powerline_status = np.full(self.n_line, fill_value=True, dtype=dt_bool)
        return powerline_status