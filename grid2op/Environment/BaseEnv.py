# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import time
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from abc import ABC, abstractmethod

from grid2op.dtypes import dt_int, dt_float
from grid2op.Space import GridObjects, RandomObject
from grid2op.Exceptions import *
from grid2op.Parameters import Parameters
from grid2op.Reward import BaseReward
from grid2op.Reward import RewardHelper
from grid2op.Opponent import OpponentSpace, UnlimitedBudget
from grid2op.Action import DontAct, BaseAction
from grid2op.Rules import AlwaysLegal
from grid2op.Opponent import BaseOpponent
from grid2op.Action._BackendAction import _BackendAction


class BaseEnv(GridObjects, RandomObject, ABC):
    """
    Internal class, do not use.

    This class represent some usefull abstraction that is re used by :class:`Environment` and
    :class:`grid2op.Observation._Obsenv` for example.

    Attributes
    ----------

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
        Number of timestep before an :class:`grid2op.BaseAgent.BaseAgent` can reconnet a powerline that has been
        disconnected
        by the environment due to an overflow.

    env_dc: ``bool``
        Whether the environment computes the powerflow using the DC approximation or not. It is usually read from
        :attr:`grid2op.Parameters.Parameters.ENV_DC`.


    TODO update with maintenance, hazards etc. see below
    # store actions "cooldown"
    times_before_line_status_actionable
    max_timestep_line_status_deactivated
    times_before_topology_actionable
    max_timestep_topology_deactivated
    time_next_maintenance
    duration_next_maintenance
    hard_overflow_threshold

    # redispacthing
    target_dispatch
    actual_dispatch

    gen_activeprod_t:
        Should be initialized at 0. for "step" to properly recognize it's the first time step of the game

    other_rewards: ``dict``
        Dictionnary with key being the name (identifier) and value being some RewardHelper. At each time step, all the
        values will be computed by the :class:`Environment` and the information about it will be returned in the
        "reward" key of the "info" dictionnary of the :func:`Environment.step`.

    """
    def __init__(self,
                 parameters,
                 thermal_limit_a=None,
                 epsilon_poly=1e-2,
                 tol_poly=1e-6,
                 other_rewards={}
                 ):
        GridObjects.__init__(self)
        RandomObject.__init__(self)

        # specific to power system
        if not isinstance(parameters, Parameters):
            raise Grid2OpException("Parameter \"parameters\" used to build the Environment should derived form the "
                                   "grid2op.Parameters class, type provided is \"{}\"".format(type(parameters)))
        self.parameters = parameters

        # some timers
        self._time_apply_act = dt_float(0)
        self._time_powerflow = dt_float(0)
        self._time_extract_obs = dt_float(0)
        self._time_opponent = dt_float(0)

        # data relative to interpolation
        self._epsilon_poly = dt_float(epsilon_poly)
        self._tol_poly = dt_float(tol_poly)

        # define logger
        self.logger = None

        # class used for the action spaces
        self.helper_action_class = None

        # and calendar data
        self.time_stamp = None
        self.nb_time_step = dt_int(0)

        # observation
        self.current_obs = None


        self.ignore_min_up_down_times = self.parameters.IGNORE_MIN_UP_DOWN_TIME
        self.forbid_dispatch_off = not self.parameters.ALLOW_DISPATCH_GEN_SWITCH_OFF

        # type of power flow to play
        # if True, then it will not disconnect lines above their thermal limits
        self.no_overflow_disconnection = self.parameters.NO_OVERFLOW_DISCONNECTION
        self.timestep_overflow = None
        self.nb_timestep_overflow_allowed = None

        # store actions "cooldown"
        self.times_before_line_status_actionable = None
        self.max_timestep_line_status_deactivated = self.parameters.NB_TIMESTEP_COOLDOWN_LINE

        self.times_before_topology_actionable = None
        self.max_timestep_topology_deactivated = self.parameters.NB_TIMESTEP_COOLDOWN_SUB

        # for maintenance operation
        self.time_next_maintenance = None
        self.duration_next_maintenance = None

        # hazard (not used outside of this class, information is given in `times_before_line_status_actionable`
        self._hazard_duration = None

        # hard overflow part
        self.hard_overflow_threshold = self.parameters.HARD_OVERFLOW_THRESHOLD
        self.env_dc = self.parameters.ENV_DC

        # Remember last line buses
        self.last_bus_line_or = None
        self.last_bus_line_ex = None

        # redispatching data
        self.target_dispatch = None
        self.actual_dispatch = None
        self.gen_uptime = None
        self.gen_downtime = None
        self.gen_activeprod_t = None
        self.gen_activeprod_t_redisp = None

        self._thermal_limit_a = thermal_limit_a

        # maintenance / hazards
        self.time_next_maintenance = None
        self.duration_next_maintenance = None

        # store environment modifications
        self._injection = None
        self._maintenance = None
        self._hazards = None
        self.env_modification = None

        # to use the data
        self.done = False
        self.current_reward = None
        self.helper_action_env = None
        self.chronics_handler = None
        self.game_rules = None
        self.helper_action_player = None

        self.rewardClass = None
        self.actionClass = None
        self.observationClass = None
        self.legalActClass = None
        self.helper_observation = None
        self.names_chronics_to_backend = None
        self.reward_helper = None
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
        self.opponent_action_class = DontAct  # class of the action of the opponent
        self.opponent_class = BaseOpponent  # class of the opponent
        self.opponent_init_budget = 0

        ## below initialized by _create_env, above: need to be called
        self.opponent_action_space = None  # ActionSpace(gridobj=)
        self.compute_opp_budg = None  # UnlimitedBudget(self.opponent_act_space)
        self.opponent = None  # OpponentSpace()
        self.oppSpace = None

        # voltage
        self.voltage_controler = None

        # backend
        self.init_grid_path = None

        # backend action
        self._backend_action_class = None
        self._backend_action = None

        # specific to Basic Env, do not change
        self.backend = None
        self.__is_init = False

    def _create_opponent(self):
        if not self.__is_init:
            raise EnvError("Impossible to create an opponent with a non initialized environment!")

        if not issubclass(self.opponent_action_class, BaseAction):
            raise EnvError("Impossible to make an environment with an opponent action class not derived from BaseAction")
        try:
            self.opponent_init_budget = dt_float(self.opponent_init_budget)
        except Exception as e:
            raise EnvError("Impossible to convert \"opponent_init_budget\" to a float with error {}".format(e))
        if self.opponent_init_budget < 0.:
            raise EnvError("If you want to deactive the opponent, please don't set its budget to a negative number."
                           "Prefer the use of the DontAct action type (\"opponent_action_class=DontAct\" "
                           "and / or set its budget to 0.")
        if not issubclass(self.opponent_class, BaseOpponent):
            raise EnvError("Impossible to make an opponent with a type that does not inherit from BaseOpponent.")

        self.opponent_action_space = self.helper_action_class(gridobj=self.backend,
                                                              legal_action=AlwaysLegal,
                                                              actionClass=self.opponent_action_class)
        self.compute_opp_budg = UnlimitedBudget(self.opponent_action_space)
        self.opponent = self.opponent_class(self.opponent_action_space)
        self.oppSpace = OpponentSpace(compute_budget=self.compute_opp_budg,
                                      init_budget=self.opponent_init_budget,
                                      opponent=self.opponent
                                      )
        self.oppSpace.init()
        self.oppSpace.reset()

    def _has_been_initialized(self):
        # type of power flow to play
        # if True, then it will not disconnect lines above their thermal limits
        self.__class__ = self.init_grid(self.backend)  # create the proper environment class for this specific environment
        if np.min([self.n_line, self.n_gen, self.n_load, self.n_sub]) <= 0:
            raise EnvironmentError("Environment has not been initialized properly")
        self._backend_action_class = _BackendAction.init_grid(self.backend)
        self._backend_action = self._backend_action_class()

        self.no_overflow_disconnection = self.parameters.NO_OVERFLOW_DISCONNECTION
        self.timestep_overflow = np.zeros(shape=(self.n_line,), dtype=dt_int)
        self.nb_timestep_overflow_allowed = np.full(shape=(self.n_line,),
                                                    fill_value=self.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED,
                                                    dtype=dt_int)
        # store actions "cooldown"
        self.times_before_line_status_actionable = np.zeros(shape=(self.n_line,), dtype=dt_int)
        self.max_timestep_line_status_deactivated = self.parameters.NB_TIMESTEP_COOLDOWN_LINE

        self.times_before_topology_actionable = np.zeros(shape=(self.n_sub,), dtype=dt_int)
        self.max_timestep_topology_deactivated = self.parameters.NB_TIMESTEP_COOLDOWN_SUB

        # hazard (not used outside of this class, information is given in `times_before_line_status_actionable`
        self._hazard_duration = np.zeros(shape=(self.n_line,), dtype=dt_int)

        # hard overflow part
        self.hard_overflow_threshold = self.parameters.HARD_OVERFLOW_THRESHOLD
        self.env_dc = self.parameters.ENV_DC

        # Remember lines last bus
        self.last_bus_line_or = np.full(shape=self.n_line, fill_value=1, dtype=dt_int)
        self.last_bus_line_ex = np.full(shape=self.n_line, fill_value=1, dtype=dt_int)

        # initialize maintenance / hazards
        self.time_next_maintenance = np.full(self.n_line, -1, dtype=dt_int)
        self.duration_next_maintenance = np.zeros(shape=(self.n_line,), dtype=dt_int)
        self.times_before_line_status_actionable = np.full(shape=(self.n_line,), fill_value=0, dtype=dt_int)

        # create the vector to the proper shape
        self.target_dispatch = np.zeros(self.n_gen, dtype=dt_float)
        self.actual_dispatch = np.zeros(self.n_gen, dtype=dt_float)
        self.gen_uptime = np.zeros(self.n_gen, dtype=dt_int)
        self.gen_downtime = np.zeros(self.n_gen, dtype=dt_int)
        self.gen_activeprod_t = np.zeros(self.n_gen, dtype=dt_float)
        self.gen_activeprod_t_redisp = np.zeros(self.n_gen, dtype=dt_float)

        self._reset_redispatching()
        self.__is_init = True

    def reset(self):
        self.__is_init = True

    def seed(self, seed=None):
        """
        Set the seed of this :class:`Environment` for a better control and to ease reproducible experiments.

        Parameters
        ----------
            seed: ``int``
               The seed to set.

        Returns
        ---------
        seed: ``int``
            The seed used to set the prng (pseudo random number generator) for the environment
        seed_chron: ``int``
            The seed used to set the prng for the chronics_handler (if any), otherwise ``None``
        seed_obs: ``int``
            The seed used to set the prng for the observation space (if any), otherwise ``None``
        seed_action_space: ``int``
            The seed used to set the prng for the action space (if any), otherwise ``None``
        seed_env_modif: ``int``
            The seed used to set the prng for the modification of th environment (if any otherwise ``None``)

        """
        try:
            seed = np.array(seed).astype(dt_int)
        except Exception as e:
            raise Grid2OpException("Impossible to seed with the seed provided. Make sure it can be converted to a"
                                   "numpy 64 integer.")
        # example from gym
        # self.np_random, seed = seeding.np_random(seed)
        # TODO make that more clean, see example of seeding @ https://github.com/openai/gym/tree/master/gym/utils

        super().seed(seed)
        seed_chron = None
        seed_obs = None
        seed_action_space = None
        seed_env_modif = None
        max_int = np.iinfo(dt_int).max
        if self.chronics_handler is not None:
            seed = self.space_prng.randint(max_int)
            seed_chron = self.chronics_handler.seed(seed)
        if self.helper_observation is not None:
            seed = self.space_prng.randint(max_int)
            seed_obs = self.helper_observation.seed(seed)
        if self.helper_action_player is not None:
            seed = self.space_prng.randint(max_int)
            seed_action_space = self.helper_action_player.seed(seed)
        if self.helper_action_env is not None:
            seed = self.space_prng.randint(max_int)
            seed_env_modif = self.helper_action_env.seed(seed)
        return (seed, seed_chron, seed_obs, seed_action_space, seed_env_modif)

    @abstractmethod
    def init_backend(self, init_grid_path, chronics_handler, backend,
                     names_chronics_to_backend, actionClass, observationClass,
                     rewardClass, legalActClass):
        pass

    def set_thermal_limit(self, thermal_limit):
        """
        Set the thermal limit effectively.

        Parameters
        ----------
        thermal_limit: ``numpy.ndarray``
            The new thermal limit. It must be a numpy ndarray vector (or convertible to it). For each powerline it
            gives the new thermal limit.
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
        self.target_dispatch[:] = 0.
        self.actual_dispatch[:] = 0.
        self.gen_uptime[:] = 0
        self.gen_downtime[:] = 0
        self.gen_activeprod_t[:] = 0.
        self.gen_activeprod_t_redisp[:] = 0.

    def _get_new_prod_setpoint(self, action):
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
        return new_p

    def _make_redisp(self, action, new_p):
        # trying with an optimization method
        except_ = None
        info_ = []
        valid = True

        # get the redispatching action (if any)
        redisp_act_orig = 1. * action._redispatch
        previous_redisp = 1. * self.actual_dispatch

        if np.all(redisp_act_orig == 0.) and np.all(self.target_dispatch == 0.) and np.all(self.actual_dispatch == 0.):
            return valid, except_, info_

        # I update the target dispatch of generator i have never modified
        already_modified_gen = self.target_dispatch != 0.
        self.target_dispatch[already_modified_gen] += redisp_act_orig[already_modified_gen]
        first_modified = (~already_modified_gen) & (redisp_act_orig != 0)
        self.target_dispatch[first_modified] = self.actual_dispatch[first_modified] + redisp_act_orig[first_modified]
        already_modified_gen |= first_modified

        # check that everything is consistent with pmin, pmax:
        if np.any(self.target_dispatch > self.gen_pmax - self.gen_pmin):
            # action is invalid, the target redispatching would be above pmax for at least a generator
            cond_invalid = self.target_dispatch > self.gen_pmax - self.gen_pmin
            except_ = InvalidRedispatching("You cannot ask for a dispatch higher than pmax - pmin  [it would be always "
                                           "invalid because, even if the sepoint is pmin, this dispatch would set it "
                                           "to a number higher than pmax, which is impossible]. Invalid dispatch for "
                                           "generator(s): "
                                           "{}".format(np.where(cond_invalid)[0]))
            self.target_dispatch -= redisp_act_orig
            return valid, except_, info_
        if np.any(self.target_dispatch < self.gen_pmin - self.gen_pmax):
            # action is invalid, the target redispatching would be below pmin for at least a generator
            cond_invalid = self.target_dispatch < self.gen_pmin - self.gen_pmax
            except_ = InvalidRedispatching("You cannot ask for a dispatch lower than pmin - pmax  [it would be always "
                                           "invalid because, even if the sepoint is pmax, this dispatch would set it "
                                           "to a number bellow pmin, which is impossible]. Invalid dispatch for "
                                           "generator(s): "
                                           "{}".format(np.where(cond_invalid)[0]))
            self.target_dispatch -= redisp_act_orig
            return valid, except_, info_

        # i can't redispatch turned off generators [turned off generators need to be turned on before redispatching]
        if np.any(redisp_act_orig[new_p == 0.]) and self.forbid_dispatch_off:
            # action is invalid, a generator has been redispatched, but it's turned off
            except_ = InvalidRedispatching("Impossible to dispatch a turned off generator")
            self.target_dispatch -= redisp_act_orig
            return valid, except_, info_

        if self.forbid_dispatch_off is True:
            redisp_act_orig_cut = 1.0 * redisp_act_orig
            redisp_act_orig_cut[new_p == 0.] = 0.
            # TODO add a flag here too, like before (the action has been "cut")
            if np.any(redisp_act_orig_cut != redisp_act_orig):
                info_.append({"INFO: redispatching cut because generator will be turned_off":
                              np.where(redisp_act_orig_cut != redisp_act_orig)[0]})
        else:
            redisp_act_orig_cut = redisp_act_orig

        mismatch = self.actual_dispatch - self.target_dispatch
        mismatch = np.abs(mismatch)
        if np.abs(np.sum(self.actual_dispatch)) >= self._tol_poly or \
                   np.sum(mismatch) >= self._tol_poly:
            except_ = self._compute_dispatch_vect(already_modified_gen, new_p)
            valid = except_ is None
        return valid, except_, info_

    def _compute_dispatch_vect(self, already_modified_gen, new_p):
        except_ = None
        # first i define the participating generator
        gen_participating = (new_p > 0.) | (self.actual_dispatch != 0.) | (self.target_dispatch != self.actual_dispatch)
        gen_participating[~self.gen_redispatchable] = False

        # define the objective value
        target_vals = self.target_dispatch[gen_participating] - self.actual_dispatch[gen_participating]
        already_modified_gen_me = already_modified_gen[gen_participating]
        target_vals_me = target_vals[already_modified_gen_me]
        nb_dispatchable = np.sum(gen_participating)
        tmp_zeros = np.zeros((1, nb_dispatchable))
        coeffs = 1.0 / (self.gen_max_ramp_up + self.gen_max_ramp_down + self._epsilon_poly)
        weights = np.ones(nb_dispatchable) * coeffs[gen_participating]
        weights /= weights.sum()

        if target_vals_me.shape[0] == 0:
            # no dispatch means all dispatchable, otherwise i will never get to 0
            already_modified_gen_me[:] = True
            target_vals_me = target_vals[already_modified_gen_me]

        def target(actual_dispatchable):
            # define my real objective
            quad_ = (actual_dispatchable[already_modified_gen_me] - target_vals_me) ** 2
            coeffs_quads = weights[already_modified_gen_me] * quad_
            coeffs_quads_const = coeffs_quads.sum()
            return coeffs_quads_const

        def jac(actual_dispatchable):
            res = 1.0 * tmp_zeros
            res[0, already_modified_gen_me] = 2.0 * weights[already_modified_gen_me] * \
                                              (actual_dispatchable[already_modified_gen_me] - target_vals_me)
            return res

        # hessian is not used for the optimization method
        # hess_mat = np.zeros((nb_dispatchable, nb_dispatchable))
        # hess_mat[already_modified_gen_me, already_modified_gen_me] = 2.0 * weights[already_modified_gen_me]
        # def hess(actual_dispatchable):
        #     return hess_mat

        mat_sum_0_no_turn_on = np.ones((1, nb_dispatchable))
        const_sum_O_no_turn_on = np.zeros(1)
        equality_const = LinearConstraint(mat_sum_0_no_turn_on,
                                          const_sum_O_no_turn_on,
                                          const_sum_O_no_turn_on)
        # gen increase in the chronics
        incr_in_chronics = new_p - (self.gen_activeprod_t_redisp - self.actual_dispatch)

        # minmum value available for disp
        ## first limit delta because of pmin
        p_min_const = self.gen_pmin[gen_participating] - new_p[gen_participating] - self.actual_dispatch[
            gen_participating]
        ## second limit delta because of ramps
        ramp_down_const = -self.gen_max_ramp_down[gen_participating] - incr_in_chronics[gen_participating]
        min_disp = np.maximum(p_min_const, ramp_down_const)
        # maximum value available for disp
        ## first limit delta because of pmin
        p_max_const = self.gen_pmax[gen_participating] - new_p[gen_participating] - self.actual_dispatch[
            gen_participating]
        ## second limit delta because of ramps
        ramp_up_const = self.gen_max_ramp_up[gen_participating] - incr_in_chronics[gen_participating]
        max_disp = np.minimum(p_max_const, ramp_up_const)

        # add everything into a linear constraint object
        mat_pmin_max_ramps = np.eye(nb_dispatchable)
        lower_pmin_max_ramps = min_disp + self._epsilon_poly
        upper_pmin_max_ramps = max_disp - self._epsilon_poly
        linear_constraint = LinearConstraint(mat_pmin_max_ramps,
                                             lower_pmin_max_ramps,
                                             upper_pmin_max_ramps)

        x0 = np.zeros(nb_dispatchable)
        def f(init):
            res = minimize(target,
                           init,
                           method="SLSQP",
                           constraints=[equality_const, linear_constraint],
                           options={'eps': self._tol_poly, "ftol": self._tol_poly, 'disp': False},
                           jac=jac
                           # hess=hess  # not used for SLSQP
                           )
            return res
        res = f(x0)
        if res.success:
            self.actual_dispatch[gen_participating] += res.x
        else:
            except_ = InvalidRedispatching("Redispatching automaton terminated with error:\n{}".format(res.message))
        return except_

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
        self.duration_next_maintenance = maintenance_duration
        self.time_next_maintenance = maintenance_time
        self._hazard_duration = hazard_duration
        return self.helper_action_env({"injection": self._injection, "maintenance": self._maintenance,
                                       "hazards": self._hazards}), prod_v

    def _update_time_reconnection_hazards_maintenance(self):
        """
        This supposes that :attr:`Environment.times_before_line_status_actionable` is already updated
        with the cascading failure, soft overflow and hard overflow.
        It also supposes that :func:`Environment._update_actions` has been called, so that the vectors
        :attr:`Environment.duration_next_maintenance`, :attr:`Environment.time_next_maintenance` and
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
        Returns
        -------
        """
        self.times_before_line_status_actionable[:] = np.maximum(self.times_before_line_status_actionable,
                                                                  self.duration_next_maintenance)
        self.times_before_line_status_actionable[:] = np.maximum(self.times_before_line_status_actionable,
                                                                  self._hazard_duration)

    def _voltage_control(self, agent_action, prod_v_chronics):
        """
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
        res = self.helper_action_env()
        if prod_v_chronics is not None:
            res.update({"injection": {"prod_v": prod_v_chronics}})
        return res

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

        if np.any(self.gen_downtime[gen_connected_this_timestep] < self.gen_min_downtime[gen_connected_this_timestep]) \
                and not self.ignore_min_up_down_times:
            # i reconnected a generator before the minimum time allowed
            id_gen = self.gen_downtime[gen_connected_this_timestep] < self.gen_min_downtime[gen_connected_this_timestep]
            id_gen = np.where(id_gen)[0]
            id_gen = np.where(gen_connected_this_timestep[id_gen])[0]
            except_ = GeneratorTurnedOnTooSoon("Some generator has been connected too early ({})".format(id_gen))
            return except_
        else:
            self.gen_downtime[gen_connected_this_timestep] = -1
            self.gen_uptime[gen_connected_this_timestep] = 1

        if np.any(self.gen_uptime[gen_disconnected_this] < self.gen_min_uptime[gen_disconnected_this]) and \
                not self.ignore_min_up_down_times:
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

    def get_obs(self):
        """
        Return the observations of the current environment made by the :class:`grid2op.BaseAgent.BaseAgent`.

        Returns
        -------
        res: :class:`grid2op.Observation.Observation`
            The current BaseObservation given to the :class:`grid2op.BaseAgent.BaseAgent` / bot / controler.
        """
        res = self.helper_observation(env=self)
        return res

    def get_thermal_limit(self):
        """
        get the current thermal limit in amps
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
                dicitonnary with keys:

                    - "disc_lines": a numpy array (or ``None``) saying, for each powerline if it has been disconnected
                        due to overflow
                    - "is_illegal" (``bool``) whether the action given as input was illegal
                    - "is_ambiguous" (``bool``) whether the action given as input was ambiguous.
                    - "is_illegal_redisp" (``bool``) was the action illegal due to redispatching
                    - "is_illegal_reco" (``bool``) was the action illegal due to a powerline reconnection
                    - "exception" (``list`` of :class:`Exceptions.Exceptions.Grid2OpException` if an exception was raised
                       or ``[]`` if everything was fine.)

        """
        # TODO update the documentation

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
        except_ = []
        init_disp = 1.0 * action._redispatch

        try:
            # "smart" reconnecting
            beg_ = time.time()
            is_illegal = not self.game_rules(action=action, env=self)
            if is_illegal:
                # action is replace by do nothing
                action = self.helper_action_player({})
                except_.append(IllegalAction("BaseAction illegal"))

            ambiguous, except_tmp = action.is_ambiguous()
            if ambiguous:
                # action is replace by do nothing
                action = self.helper_action_player({})
                is_ambiguous = True
                except_.append(except_tmp)

            # get the modification of generator active setpoint from the environment
            self.env_modification, prod_v_chronics = self._update_actions()
            self.env_modification._single_act = False  # because it absorbs all redispatching actions
            new_p = self._get_new_prod_setpoint(action)

            if self.redispatching_unit_commitment_availble:
                # remember generator that were "up" before the action
                gen_up_before = self.gen_activeprod_t > 0.

                # compute the redispatching and the new productions active setpoint
                valid_disp, except_tmp, info_ = self._make_redisp(action, new_p)
                if not valid_disp:
                    # game over case
                    action = self.helper_action_player({})
                    is_illegal_redisp = True
                    except_.append(except_tmp)
                    is_done = True
                    except_.append("Game over due to infeasible redispatching state. A generator would "
                                               "\"behave abnormally\" in a real system.")
                if except_tmp is not None:
                    action = self.helper_action_player({})
                    is_illegal_redisp = True
                    except_.append(except_tmp)

                # check the validity of min downtime and max uptime
                except_tmp = self._handle_updown_times(gen_up_before, self.actual_dispatch)
                if except_tmp is not None:
                    is_illegal_reco = True
                    action = self.helper_action_player({})
                    except_.append(except_tmp)

            # make sure the dispatching action is not implemented "as is" by the backend.
            # the environment must make sure it's a zero-sum action.
            action._redispatch[:] = 0.
            self._backend_action += action
            action._redispatch[:] = init_disp

            self.env_modification._redispatch[:] = self.actual_dispatch
            self._backend_action += self.env_modification

            # action, for redispatching is composed of multiple actions, so basically i won't check
            # ramp_min and ramp_max
            self.env_modification._single_act = False

            # now get the new generator voltage setpoint
            voltage_control_act = self._voltage_control(action, prod_v_chronics)
            self._backend_action += voltage_control_act

            # have the opponent here
            # TODO code the opponent part here and split more the timings! here "opponent time" is
            # included in time_apply_act
            tick = time.time()
            attack = self.oppSpace.attack(observation=self.current_obs,
                                          agent_action=action,
                                          env_action=self.env_modification)
            self._backend_action += attack
            self._time_opponent += time.time() - tick
            self.backend.apply_action(self._backend_action)

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
                self._backend_action.update_state(disc_lines)

                # one timestep passed, i can maybe reconnect some lines
                self.times_before_line_status_actionable[self.times_before_line_status_actionable > 0] -= 1
                # update the vector for lines that have been disconnected
                self.times_before_line_status_actionable[disc_lines] = int(self.parameters.NB_TIMESTEP_RECONNECTION)
                self._update_time_reconnection_hazards_maintenance()

                # for the powerline that are on overflow, increase this time step
                self.timestep_overflow[overflow_lines] += 1

                # set to 0 the number of timestep for lines that are not on overflow
                self.timestep_overflow[~overflow_lines] = 0

                # build the topological action "cooldown"
                aff_lines, aff_subs = action.get_topological_impact()
                if self.max_timestep_line_status_deactivated > 0:
                    # i update the cooldown only when this does not impact the line disconnected for the
                    # opponent or by maitnenance for example
                    cond = aff_lines  # powerlines i modified
                    # powerlines that are not affected by any other "forced disconnection"
                    cond &= self.times_before_line_status_actionable < self.max_timestep_line_status_deactivated
                    self.times_before_line_status_actionable[cond] = self.max_timestep_line_status_deactivated
                if self.max_timestep_topology_deactivated > 0:
                    self.times_before_topology_actionable[self.times_before_topology_actionable > 0] -= 1
                    self.times_before_topology_actionable[aff_subs] = self.max_timestep_topology_deactivated

                # build the observation
                self.current_obs = self.get_obs()
                self._time_extract_obs += time.time() - beg_

                # extract production active value at this time step (should be independant of action class)
                self.gen_activeprod_t[:], *_ = self.backend.generators_info()
                # problem with the gen_activeprod_t above, is that the slack bus absorbs alone all the losses
                # of the system. So basically, when it's too high (higher than the ramp) it can
                # mess up the rest of the environment
                self.gen_activeprod_t_redisp[:] = new_p + self.actual_dispatch
                has_error = False
            except Grid2OpException as e:
                except_.append(e)
                if self.logger is not None:
                    self.logger.error("Impossible to compute next _grid state with error \"{}\"".format(e))

        except StopIteration:
            # episode is over
            is_done = True

        self._backend_action.reset()

        infos = {"disc_lines": disc_lines,
                 "is_illegal": is_illegal,
                 "is_ambiguous": is_ambiguous,
                 "is_dispatching_illegal": is_illegal_redisp,
                 "is_illegal_reco": is_illegal_reco,
                 "exception": except_}
        self.done = self._is_done(has_error, is_done)
        self.current_reward, other_reward = self._get_reward(action,
                                                             has_error,
                                                             is_done,
                                                             is_illegal or is_illegal_redisp or is_illegal_reco,
                                                             is_ambiguous)
        infos["rewards"] = other_reward
        # TODO documentation on all the possible way to be illegal now
        if self.done:
            self.__is_init = False
        return self.current_obs, self.current_reward, self.done, infos

    def _get_reward(self, action, has_error, is_done, is_illegal, is_ambiguous):
        res = self.reward_helper(action, self, has_error, is_done, is_illegal, is_ambiguous)
        other_rewards = {k: v(action, self, has_error, is_done, is_illegal, is_ambiguous)
                         for k, v in self.other_rewards.items()
                         }
        return res, other_rewards

    def _is_done(self, has_error, is_done):
        no_more_data = self.chronics_handler.done()
        return has_error or is_done or no_more_data

    def _reset_vectors_and_timings(self):
        """
        Maintenance are not reset, otherwise the data are not read properly (skip the first time step)

        Returns
        -------

        """
        self.no_overflow_disconnection = self.parameters.NO_OVERFLOW_DISCONNECTION
        self.timestep_overflow[:] = 0
        self.nb_timestep_overflow_allowed[:] = self.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED

        self.nb_time_step = 0
        self.hard_overflow_threshold = self.parameters.HARD_OVERFLOW_THRESHOLD
        self.env_dc = self.parameters.ENV_DC

        self.times_before_line_status_actionable[:] = 0
        self.max_timestep_line_status_deactivated = self.parameters.NB_TIMESTEP_COOLDOWN_LINE

        self.times_before_topology_actionable[:] = 0
        self.max_timestep_topology_deactivated = self.parameters.NB_TIMESTEP_COOLDOWN_SUB

        # reset timings
        self._time_apply_act = 0
        self._time_powerflow = 0
        self._time_extract_obs = 0
        self._time_opponent = 0

        # reward and others
        self.current_reward = self.reward_range[0]
        self.done = False

    def _reset_maintenance(self):
        self.time_next_maintenance[:] = -1
        self.duration_next_maintenance[:] = 0

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
        self.backend.close()

    def attach_layout(self, grid_layout):
        """
        Compare to the method of the base class, this one performs a check.
        This method must be called after initialization.

        Parameters
        ----------
        grid_layout

        Returns
        -------

        """
        if isinstance(grid_layout, dict):
            pass
        elif isinstance(grid_layout, list):
            grid_layout = {k: v for k, v in zip(self.name_sub, grid_layout)}
        else:
            raise EnvError("Attempt to set a layout from something different than a dictionnary or a list. "
                           "This is for now not supported.")

        if self.__is_init:
            res = {}
            for el in self.name_sub:
                if not el in grid_layout:
                    raise EnvError("The substation \"{}\" is not present in grid_layout while in the powergrid."
                                   "".format(el))
                tmp = grid_layout[el]
                try:
                    x,y = tmp
                    x = dt_float(x)
                    y = dt_float(y)
                    res[el] = (x, y)
                except Exception as e_:
                    raise EnvError("attach_layout: impossible to convert the value of \"{}\" to a pair of float "
                                   "that will be used the grid layout. The error is: \"{}\""
                                   "".format(el, e_))
            super().attach_layout(res)
            if self.helper_action_player is not None:
                self.helper_action_player.attach_layout(res)
            if self.helper_action_env is not None:
                self.helper_action_env.attach_layout(res)
            if self.helper_observation is not None:
                self.helper_observation.attach_layout(res)
            if self.voltage_controler is not None:
                self.voltage_controler.attach_layout(res)
            if self.opponent_action_space is not None:
                self.opponent_action_space.attach_layout(res)

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

        """
        # Go to the timestep requested minus one
        nb_timestep = max(1, nb_timestep - 1)
        self.chronics_handler.fast_forward(nb_timestep)
        self.nb_time_step += nb_timestep

        # Update the timing vectors
        min_time_line_reco = np.zeros(self.n_line, dtype=dt_int)
        min_time_topo = np.zeros(self.n_sub, dtype=dt_int)
        ff_time_line_act = self.times_before_line_status_actionable - nb_timestep
        ff_time_topo_act = self.times_before_topology_actionable - nb_timestep
        self.times_before_line_status_actionable[:] = np.maximum(ff_time_line_act, min_time_line_reco)
        self.times_before_topology_actionable[:] = np.maximum(ff_time_topo_act, min_time_topo)

        # Update to the fast forward state using a do nothing action
        self.step(self.helper_action_player({}))
        # return self.current_obs
