# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import numpy as np
import pdb

from grid2op.Environment.BaseEnv import BaseEnv
from grid2op.Chronics import ChangeNothing
from grid2op.Rules import RulesChecker, BaseRules
from grid2op.Action import BaseAction
from grid2op.Exceptions import Grid2OpException


class _ObsCH(ChangeNothing):
    """
    This class is reserved to internal use. Do not attempt to do anything with it.
    """
    def forecasts(self):
        return []


class _ObsEnv(BaseEnv):
    """
    This class is an 'Emulator' of a :class:`grid2op.Environment.Environment` used to be able to 'simulate'
    forecasted grid states.
    It should not be used outside of an :class:`grid2op.Observation.BaseObservation` instance, or one of its derivative.

    It contains only the most basic element of an Environment. See :class:`grid2op.Environment.Environment` for more
    details.

    This class is reserved for internal use. Do not attempt to do anything with it.
    """
    def __init__(self, backend_instanciated, parameters, reward_helper, obsClass,
                 action_helper, thermal_limit_a, legalActClass, donothing_act,
                 other_rewards={}):
        BaseEnv.__init__(self, parameters, thermal_limit_a, other_rewards=other_rewards)
        self.donothing_act = donothing_act
        self.reward_helper = reward_helper
        self.obsClass = None
        self._action = None
        self.init_grid(backend_instanciated)
        self.init_backend(init_grid_path=None,
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

    def init_backend(self,
                     init_grid_path, chronics_handler, backend,
                     names_chronics_to_backend, actionClass, observationClass,
                     rewardClass, legalActClass):
        """
        backend should not be the backend of the environment!!!

        Parameters
        ----------
        init_grid_path
        chronics_handler
        backend
        names_chronics_to_backend
        actionClass
        observationClass
        rewardClass
        legalActClass

        Returns
        -------

        """
        self.env_dc = self.parameters.FORECAST_DC
        self.chronics_handler = chronics_handler
        self.backend = backend
        self.init_grid(self.backend)
        self._has_been_initialized()
        self.obsClass = observationClass

        if not issubclass(legalActClass, BaseRules):
            raise Grid2OpException(
                "Parameter \"legalActClass\" used to build the Environment should derived form the "
                "grid2op.BaseRules class, type provided is \"{}\"".format(
                    type(legalActClass)))
        self.game_rules = RulesChecker(legalActClass=legalActClass)
        self.legalActClass = legalActClass
        self.helper_action_player = self._do_nothing
        self.backend.set_thermal_limit(self._thermal_limit_a)
        self._create_opponent()

    def _do_nothing(self, x):
        return self.donothing_act

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
        # TODO consider disconnecting maintenance forecasted :-)
        # This "environment" doesn't modify anything
        return self.donothing_act, None

    def copy(self):
        """
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

    def init(self, new_state_action, time_stamp, timestep_overflow):
        """
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
        if self.is_init:
            return

        # update the action that set the grid to the real value
        self._action = BaseAction(gridobj=self)
        self._action.update({"set_line_status": np.array(self._line_status),
                             "set_bus": self._topo_vect,
                             "injection": {"prod_p": self._prod_p, "prod_v": self._prod_v,
                                           "load_p": self._load_p, "load_q": self._load_q}})

        self._action += new_state_action

        self.is_init = True
        self.current_obs = None
        self.time_stamp = time_stamp
        self.timestep_overflow = timestep_overflow

    def simulate(self, action):
        """
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
            dictionnary with keys:

                - "disc_lines": a numpy array (or ``None``) saying, for each powerline if it has been disconnected
                    due to overflow
                - "is_illegal" (``bool``) whether the action given as input was illegal
                - "is_ambiguous" (``bool``) whether the action given as input was ambiguous.

        """
        self.backend.set_thermal_limit(self._thermal_limit_a)
        self.backend.apply_action(self._action)
        return self.step(action)

    def get_obs(self):
        """
        Method to retrieve the "forecasted grid" as a valid observation object.

        Returns
        -------
        res: :class:`grid2op.Observation.Observation`
            The observation available.
        """

        self.current_obs = self.obsClass(gridobj=self.backend,
                                         seed=None,
                                         obs_env=None,
                                         action_helper=None)

        self.current_obs.update(self)
        res = self.current_obs
        return res

    def update_grid(self, env):
        """
        Update this "emulated" environment with the real powergrid.

        Parameters
        ----------
        env: :class:`grid2op.Environement.BaseEnv`
            A reference to the environement

        Returns
        -------

        """
        real_backend = env.backend
        self._load_p, self._load_q, self._load_v = real_backend.loads_info()
        self._prod_p, self._prod_q, self._prod_v = real_backend.generators_info()
        self._topo_vect = real_backend.get_topo_vect()

        # convert line status to -1 / 1 instead of false / true
        self._line_status = real_backend.get_line_status().astype(np.int)  # false -> 0 true -> 1
        self._line_status *= 2  # false -> 0 true -> 2
        self._line_status -= 1  # false -> -1; true -> 1
        self.is_init = False

        # Make a copy of env state for simulation
        self._thermal_limit_a = env._thermal_limit_a
        self.gen_activeprod_t[:] = env.gen_activeprod_t[:]
        self.gen_activeprod_t_redisp[:] = env.gen_activeprod_t_redisp[:]
        self.times_before_line_status_actionable[:] = env.times_before_line_status_actionable[:]
        self.times_before_topology_actionable[:]  = env.times_before_topology_actionable[:]
        self.time_remaining_before_line_reconnection[:] = env.time_remaining_before_line_reconnection[:]
        self.time_next_maintenance[:] = env.time_next_maintenance[:]
        self.duration_next_maintenance[:] = env.duration_next_maintenance[:]
        self.target_dispatch[:] = env.target_dispatch[:]
        self.actual_dispatch[:] = env.actual_dispatch[:]
        self.last_bus_line_or = env.last_bus_line_or[:]
        self.last_bus_line_ex = env.last_bus_line_ex[:]
        # TODO check redispatching and simulate are working as intended
        # TODO also update the status of hazards, maintenance etc.
        # TODO and simulate also when a maintenance is forcasted!
