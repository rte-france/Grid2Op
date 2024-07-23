# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import numpy as np
import warnings
from typing import Dict, Union, Tuple, List, Optional, Any, Literal

import grid2op
from grid2op.Exceptions.envExceptions import EnvError
from grid2op.typing_variables import STEP_INFO_TYPING
from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Environment.baseEnv import BaseEnv
from grid2op.Chronics import ChangeNothing
from grid2op.Rules import RulesChecker
from grid2op.operator_attention import LinearAttentionBudget


class _ObsCH(ChangeNothing):
    """
    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    This class is reserved to internal use. Do not attempt to do anything with it.
    """

    def forecasts(self):
        return []


class _ObsEnv(BaseEnv):
    """
    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

    This class is an 'Emulator' of a :class:`grid2op.Environment.Environment` used to be able to 'simulate'
    forecasted grid states.
    It should not be used outside of an :class:`grid2op.Observation.BaseObservation` instance, or one of its derivative.

    It contains only the most basic element of an Environment. See :class:`grid2op.Environment.Environment` for more
    details.

    This class is reserved for internal use. Do not attempt to do anything with it.
    """

    def __init__(
        self,
        init_env_path,
        init_grid_path,
        backend_instanciated,
        parameters,
        reward_helper,
        obsClass,  # not initialized :-/
        action_helper,
        thermal_limit_a,
        legalActClass,
        helper_action_class,
        helper_action_env,
        epsilon_poly,
        tol_poly,
        max_episode_duration,
        delta_time_seconds,
        other_rewards={},
        has_attention_budget=False,
        attention_budget_cls=LinearAttentionBudget,
        kwargs_attention_budget={},
        logger=None,
        highres_sim_counter=None,
        _complete_action_cls=None,
        _ptr_orig_obs_space=None,
        _local_dir_cls=None,  # only set at the first call to `make(...)` after should be false
        _read_from_local_dir=None,
    ):
        BaseEnv.__init__(
            self,
            init_env_path,
            init_grid_path,
            copy.deepcopy(parameters),
            thermal_limit_a,
            other_rewards=other_rewards,
            epsilon_poly=epsilon_poly,
            tol_poly=tol_poly,
            has_attention_budget=has_attention_budget,
            attention_budget_cls=attention_budget_cls,
            kwargs_attention_budget=kwargs_attention_budget,
            kwargs_observation=None,
            logger=logger,
            highres_sim_counter=highres_sim_counter,
            update_obs_after_reward=False,
            _local_dir_cls=_local_dir_cls,
            _read_from_local_dir=_read_from_local_dir
        )
        self._do_not_erase_local_dir_cls = True
        self.__unusable = False  # unsuable if backend cannot be copied
        
        self._reward_helper = reward_helper
        self._helper_action_class = helper_action_class
        # TODO init reward and other reward

        # initialize the observation space
        self._obsClass = None
        
        cls = type(self)
        # line status (inherited from BaseEnv)
        self._line_status = np.full(cls.n_line, dtype=dt_bool, fill_value=True)
        # line status (for this usage)
        self._line_status_me = np.ones(
            shape=cls.n_line, dtype=dt_int
        )  # this is "line status" but encode in +1 / -1

        if self._thermal_limit_a is None:
            self._thermal_limit_a = 1.0 * thermal_limit_a.astype(dt_float)
        else:
            self._thermal_limit_a[:] = thermal_limit_a
        
        self.current_obs_init = None
        self.current_obs = None
        self._init_backend(
            chronics_handler=_ObsCH(),
            backend=backend_instanciated,
            names_chronics_to_backend=None,
            actionClass=action_helper.actionClass,
            observationClass=obsClass,
            rewardClass=None,
            legalActClass=legalActClass,
        )

        self.delta_time_seconds = delta_time_seconds
        ####
        # to be able to save and import (using env.generate_classes) correctly
        self._actionClass = action_helper.subtype
        self._complete_action_cls = _complete_action_cls
        self._action_space = (
            action_helper  # obs env and env share the same action space
        )
        
        self._ptr_orig_obs_space = _ptr_orig_obs_space
        
        ####

        self.no_overflow_disconnection = parameters.NO_OVERFLOW_DISCONNECTION
        self._topo_vect = np.zeros(type(backend_instanciated).dim_topo, dtype=dt_int)

        # other stuff
        self.is_init = False
        self._helper_action_env = helper_action_env
        self.env_modification = self._helper_action_env()
        self._do_nothing_act = self._helper_action_env()
        
        if self.__unusable:
            self._backend_action_set = None
        else:
            self._backend_action_set = self._backend_action_class()

        if self.__unusable:
            self._disc_lines = np.zeros(shape=0, dtype=dt_int) - 1
        else:
            self._disc_lines = np.zeros(shape=self.n_line, dtype=dt_int) - 1
            
        self._max_episode_duration = max_episode_duration

    def max_episode_duration(self):
        return self._max_episode_duration

    def _init_myclass(self):
        """this class has already all the powergrid information: it is initialized in the obs space !"""
        pass

    def _init_backend(
        self,
        chronics_handler,
        backend,
        names_chronics_to_backend,
        actionClass,
        observationClass,  # base grid2op type
        rewardClass,
        legalActClass,
    ):
        if backend is None:
            self.__unusable = True
            return
        self._actionClass_orig = actionClass
        self._observationClass_orig = observationClass
        
        self.__unusable = False
        self._env_dc = self.parameters.ENV_DC
        self.chronics_handler = chronics_handler
        self.backend = backend
        self._has_been_initialized()  # really important to include this piece of code! and just here after the
        
        self._check_rules_correct(legalActClass)
        self._game_rules = RulesChecker(legalActClass=legalActClass)
        self._game_rules.initialize(self)
        self._legalActClass = legalActClass
        
        # self._action_space = self._do_nothing
        self.backend.set_thermal_limit(self._thermal_limit_a)

        from grid2op.Observation import ObservationSpace
        from grid2op.Reward import FlatReward
        ob_sp_cls = ObservationSpace.init_grid(type(backend), _local_dir_cls=self._local_dir_cls)
        self._observation_space = ob_sp_cls(type(backend),
                                            env=self,
                                            with_forecast=False,
                                            rewardClass=FlatReward,
                                            _with_obs_env=False,
                                            _local_dir_cls=self._local_dir_cls
                                            )
        self._observationClass = self._observation_space.subtype  # not used anyway
        
        # create the opponent
        self._create_opponent()

        # create the attention budget
        self._create_attention_budget()
        self._obsClass = observationClass.init_grid(type(self.backend), _local_dir_cls=self._local_dir_cls)
        self._obsClass._INIT_GRID_CLS = observationClass
        self.current_obs_init = self._obsClass(obs_env=None, action_helper=None)
        self.current_obs = self.current_obs_init

        # init the alert relate attributes
        self._init_alert_data()
        
        # backend has loaded everything
        self._hazard_duration = np.zeros(shape=type(self).n_line, dtype=dt_int)

    def _do_nothing(self, x):
        """
        this is should be only called within _Obsenv.step, and there, only return the "do nothing"
        action.

        This is why this function is used as the "obsenv action space"
        """
        return self._do_nothing_act

    def _update_actions(self):
        """
        INTERNAL

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

    def copy(self, env=None, new_obs_space=None):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Implement the deep copy of this instance.

        Returns
        -------
        res: :class:`ObsEnv`
            A deep copy of this instance.
        """
        if self.__unusable:
            raise EnvError("Impossible to use a Observation backend with an "
                           "environment that cannot be copied.")
            
        my_cls = type(self)
        res = my_cls.__new__(my_cls)
        
        # fill its attribute
        res.__unusable = self.__unusable
        res._obsClass = self._obsClass
        res._line_status = copy.deepcopy(self._line_status)
        res._line_status_me = copy.deepcopy(self._line_status_me)
        if env is not None:
            # res._ptr_orig_obs_space = env._observation_space  # this is not created when this function is called
            # so this is why i pass the `new_obs_space` as argument
            res._ptr_orig_obs_space = new_obs_space
        else:
            res._ptr_orig_obs_space = self._ptr_orig_obs_space
        res.no_overflow_disconnection = self.parameters.NO_OVERFLOW_DISCONNECTION
        res._topo_vect = copy.deepcopy(self._topo_vect)
        res.is_init = self.is_init
        if env is not None:
            res._helper_action_env = env._helper_action_env
        else:
            res._helper_action_env = self._helper_action_env
        res._disc_lines = copy.deepcopy(self._disc_lines)
        res._highres_sim_counter = self._highres_sim_counter
        res._max_episode_duration = self._max_episode_duration
        
        res.current_obs_init = self._obsClass(obs_env=None, action_helper=None)
        res.current_obs_init.reset()
        res.current_obs = res.current_obs_init
        
        # copy attribute of "super"
        super()._custom_deepcopy_for_copy(res)
        
        # finish to initialize res
        res.env_modification = res._helper_action_env()
        res._do_nothing_act = res._helper_action_env()
        res._backend_action_set = res._backend_action_class()
        res.current_obs = res.current_obs_init
        return res

    def _reset_to_orig_state(self, obs):
        super()._reset_to_orig_state(obs)
        self._line_status_me[:] = obs._env_internal_params["_line_status_env"]
        
    def init(
        self,
        new_state_action,
        time_stamp,
        obs,
        time_step=1
    ):
        """
        INTERNAL

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
        if self.__unusable:
            raise EnvError("Impossible to use a Observation backend with an "
                           "environment that cannot be copied.")
        
        self.reset()  # reset the "BaseEnv"
        self._reset_to_orig_state(obs)
        self._topo_vect[:] = obs.topo_vect
        
        if time_step >= 1:
            is_overflow = obs.rho > 1.
            
            # handle the components that depends on the time
            (
                still_in_maintenance,
                reconnected,
                first_ts_maintenance,
            ) = self._update_vector_with_timestep(time_step, is_overflow)
            if first_ts_maintenance.any():
                set_status = np.array(self._line_status_me, dtype=dt_int)
                set_status[first_ts_maintenance] = -1
                topo_vect = np.array(self._topo_vect, dtype=dt_int)
                topo_vect[self.line_or_pos_topo_vect[first_ts_maintenance]] = -1
                topo_vect[self.line_ex_pos_topo_vect[first_ts_maintenance]] = -1
            else:
                set_status = self._line_status_me
                topo_vect = self._topo_vect
            
            if still_in_maintenance.any():
                set_status[still_in_maintenance] = -1
                topo_vect = np.array(self._topo_vect, dtype=dt_int)
                topo_vect[self.line_or_pos_topo_vect[still_in_maintenance]] = -1
                topo_vect[self.line_ex_pos_topo_vect[still_in_maintenance]] = -1
        else:
            set_status = self._line_status_me
            topo_vect = self._topo_vect
            
        # TODO set the shunts here
        # update the action that set the grid to the real value
        self._backend_action_set += self._helper_action_env(
            {
                "set_line_status": set_status,
                "set_bus": topo_vect,
                "injection": {
                    "prod_p": obs.gen_p,
                    "prod_v": obs.gen_v,
                    "load_p": obs.load_p,
                    "load_q": obs.load_q,
                },
            }
        )
        self._backend_action_set += new_state_action
        # for storage unit
        if time_step > 0:
            self._backend_action_set.storage_power.values[:] = 0.0
        self._backend_action_set.all_changed()
        self._backend_action = copy.deepcopy(self._backend_action_set)
        
        # for curtailment
        if self._env_modification is not None:
            self._env_modification._dict_inj = {}
        
        self.is_init = True
        self.current_obs.reset()
        self.time_stamp = time_stamp

    def _get_new_prod_setpoint(self, action):
        new_p = 1.0 * self._backend_action_set.prod_p.values
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
    
    def _get_new_load_setpoint(self, action):
        new_p = 1.0 * self._backend_action_set.prod_p.values
        if "load_p" in action._dict_inj:
            tmp = action._dict_inj["load_p"]
            indx_ok = np.isfinite(tmp)
            new_p[indx_ok] = tmp[indx_ok]

        # modification of the environment always override the modification of the agents (if any)
        # TODO have a flag there if this is the case.
        if "load_p" in self._env_modification._dict_inj:
            # modification of the production setpoint value
            tmp = self._env_modification._dict_inj["load_p"]
            indx_ok = np.isfinite(tmp)
            new_p[indx_ok] = tmp[indx_ok]
        return new_p

    def reset(self):
        if self.__unusable:
            raise EnvError("Impossible to use a Observation backend with an "
                           "environment that cannot be copied.")
        super().reset()
        self.current_obs = self.current_obs_init

    def simulate(self, action : "grid2op.Action.BaseAction") -> Tuple["grid2op.Observation.BaseObservation", float, bool, STEP_INFO_TYPING]:
        """
        INTERNAL

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
        action: :class:`grid2op.Action.BaseAction`
            The action to test

        Returns
        -------
        observation: :class:`grid2op.Observation.BaseObservation`
            agent's observation of the current environment

        reward: ``float``
            amount of reward returned after previous action

        done: ``bool``
            whether the episode has ended, in which case further step() calls will return undefined results

        info: ``dict``
            contains auxiliary diagnostic information (helpful for debugging, and sometimes learning). See 
            description of :func:`grid2op.Environment.BaseEnv.step` for more information about the
            keys of this dictionary.

        """
        if self.__unusable:
            raise EnvError("Impossible to use a Observation backend with an "
                           "environment that cannot be copied.")
        self._ptr_orig_obs_space.simulate_called()
        maybe_exc = self._ptr_orig_obs_space.can_use_simulate()
        self._highres_sim_counter.add_one()
        if maybe_exc is not None:
            raise maybe_exc
        obs, reward, done, info = self.step(action)
        return obs, reward, done, info

    def get_obs(self, _update_state=True, _do_copy=True):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Method to retrieve the "forecasted grid" as a valid observation object.

        Returns
        -------
        res: :class:`grid2op.Observation.Observation`
            The observation available.
        """
        if self.__unusable:
            raise EnvError("Impossible to use a Observation backend with an "
                           "environment that cannot be copied.")
        if _update_state:
            self.current_obs.update(self, with_forecast=False)
            
        if _do_copy:
            res = copy.deepcopy(self.current_obs)
        else:
            res = self.current_obs
        return res

    def update_grid(self, env):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Update this "emulated" environment with the real powergrid.

        # TODO it should be updated from the observation only, especially if the observation is partially
        # TODO observable. This would lead to data leakage here somehow.

        Parameters
        ----------
        env: :class:`grid2op.Environment.BaseEnv`
            A reference to the environment
        """
        if self.__unusable:
            raise EnvError("Impossible to use a Observation backend with an "
                           "environment that cannot be copied.")
        self.is_init = False
        
    def get_current_line_status(self):
        if self.__unusable:
            raise EnvError("Impossible to use a Observation backend with an "
                           "environment that cannot be copied.")
        return self._line_status == 1

    def is_valid(self):
        """return whether or not the obs_env is valid, *eg* whether
        we could copy the backend of the environment."""
        return not self.__unusable
    
    def close(self):
        """close this environment, once and for all"""
        super().close()

        # clean all the attributes
        for attr_nm in [
            "_obsClass",
            "_line_status",
            "_line_status_me",
            "_max_episode_duration",
            "_ptr_orig_obs_space",
        ]:
            if hasattr(self, attr_nm):
                delattr(self, attr_nm)
            setattr(self, attr_nm, None)
