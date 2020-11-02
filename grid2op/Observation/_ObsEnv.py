# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import numpy as np
from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Environment.BaseEnv import BaseEnv
from grid2op.Chronics import ChangeNothing
from grid2op.Rules import RulesChecker, BaseRules
from grid2op.Exceptions import Grid2OpException


class _ObsCH(ChangeNothing):
    """
    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    This class is reserved to internal use. Do not attempt to do anything with it.
    """
    def forecasts(self):
        return []


class _ObsEnv(BaseEnv):
    """
    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    This class is an 'Emulator' of a :class:`grid2op.Environment.Environment` used to be able to 'simulate'
    forecasted grid states.
    It should not be used outside of an :class:`grid2op.Observation.BaseObservation` instance, or one of its derivative.

    It contains only the most basic element of an Environment. See :class:`grid2op.Environment.Environment` for more
    details.

    This class is reserved for internal use. Do not attempt to do anything with it.
    """
    def __init__(self,
                 backend_instanciated,
                 completeActionClass,
                 parameters,
                 reward_helper,
                 obsClass,
                 action_helper,
                 thermal_limit_a,
                 legalActClass,
                 donothing_act,
                 helper_action_class,
                 helper_action_env,
                 other_rewards={}):
        BaseEnv.__init__(self, parameters, thermal_limit_a, other_rewards=other_rewards)
        self._helper_action_class = helper_action_class
        self._reward_helper = reward_helper
        self._obsClass = None

        self.gen_activeprod_t_init = np.zeros(self.n_gen, dtype=dt_float)
        self.gen_activeprod_t_redisp_init = np.zeros(self.n_gen, dtype=dt_float)
        self.times_before_line_status_actionable_init = np.zeros(self.n_line, dtype=dt_int)
        self.times_before_topology_actionable_init = np.zeros(self.n_sub, dtype=dt_int)
        self.time_next_maintenance_init = np.zeros(self.n_line, dtype=dt_int)
        self.duration_next_maintenance_init = np.zeros(self.n_line, dtype=dt_int)
        self.target_dispatch_init = np.zeros(self.n_gen, dtype=dt_float)
        self.actual_dispatch_init = np.zeros(self.n_gen, dtype=dt_float)

        self._init_backend(init_grid_path=None,
                           chronics_handler=_ObsCH(),
                           backend=backend_instanciated,
                           names_chronics_to_backend=None,
                           actionClass=action_helper.actionClass,
                           observationClass=obsClass,
                           rewardClass=None,
                           legalActClass=legalActClass)
        self.no_overflow_disconnection = parameters.NO_OVERFLOW_DISCONNECTION

        self._load_p, self._load_q, self._load_v =  None, None, None
        self._prod_p, self._prod_q, self._prod_v = None, None, None
        self._topo_vect = None

        # convert line status to -1 / 1 instead of false / true
        self._line_status = None
        self.is_init = False
        self._helper_action_env = helper_action_env
        self.env_modification = self._helper_action_env()
        self._do_nothing_act = self._helper_action_env()
        self._backend_action_set = self._backend_action_class()

        # opponent
        self.opp_space_state = None
        self.opp_state = None

    def _init_backend(self,
                      init_grid_path,
                      chronics_handler,
                      backend,
                      names_chronics_to_backend,
                      actionClass,
                      observationClass,
                      rewardClass, legalActClass):
        self._env_dc = self.parameters.FORECAST_DC
        self.chronics_handler = chronics_handler
        self.backend = backend
        self._has_been_initialized()
        self._obsClass = observationClass

        if not issubclass(legalActClass, BaseRules):
            raise Grid2OpException(
                "Parameter \"legalActClass\" used to build the Environment should derived form the "
                "grid2op.BaseRules class, type provided is \"{}\"".format(
                    type(legalActClass)))
        self._game_rules = RulesChecker(legalActClass=legalActClass)
        self._legalActClass = legalActClass
        self._helper_action_player = self._do_nothing
        self.backend.set_thermal_limit(self._thermal_limit_a)
        self._create_opponent()

        self.current_obs_init = self._obsClass(seed=None,
                                               obs_env=None,
                                               action_helper=None)
        self.current_obs = self.current_obs_init

        # backend has loaded everything
        self._line_status = np.ones(shape=self.n_line, dtype=dt_bool)

    def _do_nothing(self, x):
        return self._do_nothing_act

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
        # TODO consider disconnecting maintenance forecasted :-)
        # This "environment" doesn't modify anything
        return self._do_nothing_act, None

    def copy(self):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Implement the deep copy of this instance.

        Returns
        -------
        res: :class:`ObsEnv`
            A deep copy of this instance.
        """
        backend = self.backend
        self.backend = None
        res = copy.deepcopy(self)
        res.backend = backend.copy()
        self.backend = backend
        return res

    def init(self, new_state_action, time_stamp, timestep_overflow, topo_vect, time_step=1):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Initialize a "forecasted grid state" based on the new injections, possibly new topological modifications etc.

        Parameters
        ----------
        new_state_action: :class:`grid2op.Action`
            The action that is performed on the powergrid to get the forecast at the current date. This "action" is
            NOT performed by the user, it's performed internally by the BaseObservation to have a "forecasted" powergrid
            with the forecasted values present in the chronics.

        time_stamp: ``datetime.datetime``
            The time stamp of the forecast, as a datetime.datetime object. NB this is not the time stamp at which the
            forecast is produced, but the time stamp of the powergrid forecasted.

        timestep_overflow: ``numpy.ndarray``
            The see :attr:`grid2op.Env.timestep_overflow` for a better description of this argument.

        Returns
        -------
        ``None``

        """
        self._reset_to_orig_state()
        self._reset_vect()
        self._topo_vect[:] = topo_vect
        # TODO update maintenance time, duration and cooldown accordingly (see all todos in `update_grid`)

        # TODO set the shunts here
        # update the action that set the grid to the real value
        still_in_maintenance, reconnected, first_ts_maintenance = self._update_vector_with_timestep(time_step)
        if np.any(first_ts_maintenance):
            set_status = np.array(self._line_status, dtype=dt_int)
            set_status[first_ts_maintenance] = -1
            topo_vect = np.array(self._topo_vect, dtype=dt_int)
            topo_vect[self.line_or_pos_topo_vect[first_ts_maintenance]] = -1
            topo_vect[self.line_ex_pos_topo_vect[first_ts_maintenance]] = -1
        else:
            set_status = self._line_status
            topo_vect = self._topo_vect

        self._backend_action_set += self._helper_action_env({"set_line_status": set_status,
                                                             "set_bus": topo_vect,
                                                             "injection": {"prod_p": self._prod_p,
                                                                           "prod_v": self._prod_v,
                                                                           "load_p": self._load_p,
                                                                           "load_q": self._load_q}
                                                             })
        self._backend_action_set += new_state_action
        self.is_init = True
        self.current_obs.reset()
        self.time_stamp = time_stamp
        self._timestep_overflow[:] = timestep_overflow

    def _update_vector_with_timestep(self, time_step):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        update the value of the "time dependant" attributes
        """
        self._times_before_line_status_actionable[:] = np.maximum(self._times_before_line_status_actionable - time_step,
                                                                  0.)
        self._times_before_topology_actionable[:] = np.maximum(self._times_before_topology_actionable - time_step,
                                                               0.)

        still_in_maintenance = (self._duration_next_maintenance > time_step) & (self._time_next_maintenance == 0)
        reconnected = (self._duration_next_maintenance < time_step) & (self._time_next_maintenance == 0)
        first_ts_maintenance = self._time_next_maintenance == time_step

        # powerline that are still in maintenance at this time step
        self._time_next_maintenance[still_in_maintenance] = 0
        self._duration_next_maintenance[still_in_maintenance] -= 1

        # powerline that will be in maintenance at this time step
        self._time_next_maintenance[first_ts_maintenance] = 0

        # powerline that won't be in maintenance at this time step
        self._time_next_maintenance[reconnected] = -1
        self._duration_next_maintenance[reconnected] = 0
        return still_in_maintenance, reconnected, first_ts_maintenance

    def reset(self):
        super().reset()
        self.current_obs = self.current_obs_init

    def _reset_vect(self):
        self._gen_activeprod_t[:] = self.gen_activeprod_t_init
        self._gen_activeprod_t_redisp[:] = self.gen_activeprod_t_redisp_init
        self._times_before_line_status_actionable[:] = self.times_before_line_status_actionable_init
        self._times_before_topology_actionable[:] = self.times_before_topology_actionable_init
        self._time_next_maintenance[:] = self.time_next_maintenance_init
        self._duration_next_maintenance[:] = self.duration_next_maintenance_init
        self._target_dispatch[:] = self.target_dispatch_init
        self._actual_dispatch[:] = self.actual_dispatch_init

    def _reset_to_orig_state(self):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        reset this "environment" to the state it should be
        """
        self.reset()  # reset the "BaseEnv"
        self.backend.set_thermal_limit(self._thermal_limit_a)
        self._backend_action_set.all_changed()
        self._backend_action = copy.deepcopy(self._backend_action_set)
        self._oppSpace._set_state(self.opp_space_state, self.opp_state)

    def simulate(self, action):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            Prefer using `obs.simulate(action)`

        This function is the core method of the :class:`ObsEnv`. It allows to perform a simulation of what would
        give and action if it were to be implemented on the "forecasted" powergrid.

        It has the same signature as :func:`grid2op.Environment.Environment.step`. One of the major difference is that
        it doesn't
        check whether the action is illegal or not (but an implementation could be provided for this method). The
        reason for this is that there is not one single and unique way to "forecast" how the thermal limit will behave,
        which lines will be available or not, which actions will be done or not between the time stamp at which
        "simulate" is called, and the time stamp that is simulated.

        Parameters
        ----------
        action: :class:`grid2op.Action.Action`
            The action to test

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

        """
        self._reset_to_orig_state()
        obs, reward, done, info = self.step(action)
        return obs, reward, done, info

    def get_obs(self):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Method to retrieve the "forecasted grid" as a valid observation object.

        Returns
        -------
        res: :class:`grid2op.Observation.Observation`
            The observation available.
        """
        self.current_obs.update(self, with_forecast=False)
        res = copy.deepcopy(self.current_obs)
        return res

    def update_grid(self, env):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Update this "emulated" environment with the real powergrid.

        Parameters
        ----------
        env: :class:`grid2op.Environment.BaseEnv`
            A reference to the environment
        """
        real_backend = env.backend
        self._reward_helper = env._reward_helper

        self._load_p, self._load_q, self._load_v = real_backend.loads_info()
        self._prod_p, self._prod_q, self._prod_v = real_backend.generators_info()
        self._topo_vect = real_backend.get_topo_vect()

        # convert line status to -1 / 1 instead of false / true
        self._line_status = real_backend.get_line_status().astype(dt_int)  # false -> 0 true -> 1
        self._line_status *= 2  # false -> 0 true -> 2
        self._line_status -= 1  # false -> -1; true -> 1
        self.is_init = False

        # Make a copy of env state for simulation
        # TODO this depends on the datetime simulated, so find a way to have it independant of that !!!
        self._thermal_limit_a = env._thermal_limit_a.astype(dt_float)
        self.gen_activeprod_t_init[:] = env._gen_activeprod_t
        self.gen_activeprod_t_redisp_init[:] = env._gen_activeprod_t_redisp
        self.times_before_line_status_actionable_init[:] = env._times_before_line_status_actionable
        self.times_before_topology_actionable_init[:] = env._times_before_topology_actionable
        self.time_next_maintenance_init[:] = env._time_next_maintenance
        self.duration_next_maintenance_init[:] = env._duration_next_maintenance
        self.target_dispatch_init[:] = env._target_dispatch
        self.actual_dispatch_init[:] = env._actual_dispatch
        self.opp_space_state, self.opp_state = env._oppSpace._get_state()
        # TODO check redispatching and simulate are working as intended
        # TODO also update the status of hazards, maintenance etc.
        # TODO and simulate also when a maintenance is forcasted!
        # TODO add the opponent budget here (should decrease with the time step :scared:) -> we really need to address
        # all that before 1.0.0

    def get_current_line_status(self):
        return self._line_status == 1