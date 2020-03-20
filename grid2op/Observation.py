"""
In a "reinforcement learning" framework, an :class:`grid2op.Agent` receive two information before taking any action on
the :class:`grid2op.Environment`. One of them is the :class:`grid2op.Reward` that tells it how well the past action
performed. The second main input received from the environment is the :class:`Observation`. This is gives the Agent
partial, noisy, or complete information about the current state of the environment. This module implement a generic
:class:`Observation`  class and an example of a complete observation in the case of the Learning
To Run a Power Network (`l2RPN <https://l2rpn.chalearn.org/>`_ ) competition.

Compared to other Reinforcement Learning problems the L2PRN competition allows another flexibility. Today, when
operating a powergrid, operators have "forecasts" at their disposal. We wanted to make them available in the
L2PRN competition too. In the  first edition of the L2PRN competition, was offered the
functionality to simulate the effect of an action on a forecasted powergrid.
This forecasted powergrid used:

  - the topology of the powergrid of the last know time step
  - all the injections of given in files.

This functionality was originally attached to the Environment and could only be used to simulate the effect of an action
on this unique time step. We wanted in this recoding to change that:

  - in an RL setting, an :class:`grid2op.Agent` should not be able to look directly at the :class:`grid2op.Environment`.
    The only information about the Environment the Agent should have is through the :class:`grid2op.Observation` and
    the :class:`grid2op.Reward`. Having this principle implemented will help enforcing this principle.
  - In some wider context, it is relevant to have these forecasts available in multiple way, or modified by the
    :class:`grid2op.Agent` itself (for example having forecast available for the next 2 or 3 hours, with the Agent able
    not only to change the topology of the powergrid with actions, but also the injections if he's able to provide
    more accurate predictions for example.

The :class:`Observation` class implement the two above principles and is more flexible to other kind of forecasts,
or other methods to build a power grid based on the forecasts of injections.
"""

import copy
import numpy as np
import re
import warnings
import json
import os
import time

from abc import ABC, abstractmethod

import pdb

try:
    from .Exceptions import *
    from .Space import SerializableSpace, GridObjects
    from .BasicEnv import _BasicEnv
    from .Reward import ConstantReward, RewardHelper
    from .Action import Action
    from .GameRules import GameRules, LegalAction
    from .ChronicsHandler import ChangeNothing
except (ModuleNotFoundError, ImportError):
    from Exceptions import *
    from Space import SerializableSpace, GridObjects
    from BasicEnv import _BasicEnv
    from Reward import ConstantReward, RewardHelper
    from Action import Action
    from GameRules import GameRules, LegalAction
    from ChronicsHandler import ChangeNothing

# TODO be able to change reward here

# TODO make an action with the difference between the observation that would be an action.
# TODO have a method that could do "forecast" by giving the _injection by the agent, if he wants to make custom forecasts

# TODO finish documentation


# TODO fix "bug" when action not initalized, return nan in to_vect

class ObsCH(ChangeNothing):
    """
    This class is reserved to internal use. Do not attempt to do anything with it.
    """
    def forecasts(self):
        return []


class ObsEnv(_BasicEnv):
    """
    This class is an 'Emulator' of a :class:`grid2op.Environment` used to be able to 'simulate' forecasted grid states.
    It should not be used outside of an :class:`grid2op.Observation` instance, or one of its derivative.

    It contains only the most basic element of an Environment. See :class:`grid2op.Environment` for more details.

    This class is reserved for internal use. Do not attempt to do anything with it.
    """
    def __init__(self, backend_instanciated, parameters, reward_helper, obsClass,
                 action_helper, thermal_limit_a, legalActClass, donothing_act):
        _BasicEnv.__init__(self, parameters, thermal_limit_a)
        self.donothing_act = donothing_act
        self.reward_helper = reward_helper
        self.obsClass = None
        self._action = None
        self.init_grid(backend_instanciated)
        self.init_backend(init_grid_path=None,
                          chronics_handler=ObsCH(),
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

        if not issubclass(legalActClass, LegalAction):
            raise Grid2OpException(
                "Parameter \"legalActClass\" used to build the Environment should derived form the "
                "grid2op.LegalAction class, type provided is \"{}\"".format(
                    type(legalActClass)))
        self.game_rules = GameRules(legalActClass=legalActClass)
        self.legalActClass = legalActClass
        self.helper_action_player = lambda x: self.donothing_act
        self.backend.set_thermal_limit(self._thermal_limit_a)

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
            NOT performed by the user, it's performed internally by the Observation to have a "forecasted" powergrid
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
        self._action = Action(gridobj=self)
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
        # return self.current_obs, reward, has_error, {}

    def _get_reward(self, action, has_error, is_done,is_illegal, is_ambiguous):
        if has_error:
            res = self.reward_helper.range()[0]
        else:
            res = self.reward_helper(action, self, has_error, is_done, is_illegal, is_ambiguous)
        return res

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
        env: :class:`grid2op.Environement._BasicEnv`
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
        self.times_before_line_status_actionable[:] = env.times_before_line_status_actionable[:]
        self.times_before_topology_actionable[:]  = env.times_before_topology_actionable[:]
        self.time_remaining_before_line_reconnection[:] = env.time_remaining_before_line_reconnection[:]
        self.time_next_maintenance[:] = env.time_next_maintenance[:]
        self.duration_next_maintenance[:] = env.duration_next_maintenance[:]
        self.target_dispatch[:] = env.target_dispatch[:]
        self.actual_dispatch[:] = env.actual_dispatch[:]
        # TODO check redispatching and simulate are working as intended
        # TODO also update the status of hazards, maintenance etc.
        # TODO and simulate also when a maintenance is forcasted!


class Observation(GridObjects):
    """
    Basic class representing an observation.

    All observation must derive from this class and implement all its abstract methods.

    Attributes
    ----------
    action_helper: :class:`grid2op.Action.HelperAction`
        A reprensentation of the possible action space.

    year: ``int``
        The current year

    month: ``int``
        The current month (0 = january, 11 = december)

    day: ``int``
        The current day of the month

    hour_of_day: ``int``
        The current hour of the day

    minute_of_hour: ``int``
        The current minute of the current hour

    day_of_week: ``int``
        The current day of the week. Monday = 0, Sunday = 6

    prod_p: :class:`numpy.ndarray`, dtype:float
        The active production value of each generator (expressed in MW).

    prod_q: :class:`numpy.ndarray`, dtype:float
        The reactive production value of each generator (expressed in MVar).

    prod_v: :class:`numpy.ndarray`, dtype:float
        The voltage magnitude of the bus to which each generator is connected (expressed in kV).

    load_p: :class:`numpy.ndarray`, dtype:float
        The active load value of each consumption (expressed in MW).

    load_q: :class:`numpy.ndarray`, dtype:float
        The reactive load value of each consumption (expressed in MVar).

    load_v: :class:`numpy.ndarray`, dtype:float
        The voltage magnitude of the bus to which each consumption is connected (expressed in kV).

    p_or: :class:`numpy.ndarray`, dtype:float
        The active power flow at the origin end of each powerline (expressed in MW).

    q_or: :class:`numpy.ndarray`, dtype:float
        The reactive power flow at the origin end of each powerline (expressed in MVar).

    v_or: :class:`numpy.ndarray`, dtype:float
        The voltage magnitude at the bus to which the origin end of each powerline is connected (expressed in kV).

    a_or: :class:`numpy.ndarray`, dtype:float
        The current flow at the origin end of each powerline (expressed in A).

    p_ex: :class:`numpy.ndarray`, dtype:float
        The active power flow at the extremity end of each powerline (expressed in MW).

    q_ex: :class:`numpy.ndarray`, dtype:float
        The reactive power flow at the extremity end of each powerline (expressed in MVar).

    v_ex: :class:`numpy.ndarray`, dtype:float
        The voltage magnitude at the bus to which the extremity end of each powerline is connected (expressed in kV).

    a_ex: :class:`numpy.ndarray`, dtype:float
        The current flow at the extremity end of each powerline (expressed in A).

    rho: :class:`numpy.ndarray`, dtype:float
        The capacity of each powerline. It is defined at the observed current flow divided by the thermal limit of each
        powerline (no unit)

    connectivity_matrix_: :class:`numpy.ndarray`, dtype:float
        The connectivityt matrix (if computed, or None) see definition of :func:`connectivity_matrix` for
        more information

    bus_connectivity_matrix_: :class:`numpy.ndarray`, dtype:float
        The `bus_connectivity_matrix_` matrix (if computed, or None) see definition of
          :func:`Observation.bus_connectivity_matrix` for more information

    vectorized: :class:`numpy.ndarray`, dtype:float
        The vector representation of this Observation (if computed, or None) see definition of
        :func:`to_vect` for more information.

    topo_vect:  :class:`numpy.ndarray`, dtype:int
        For each object (load, generator, ends of a powerline) it gives on which bus this object is connected
        in its substation. See :func:`grid2op.Backend.Backend.get_topo_vect` for more information.

    line_status: :class:`numpy.ndarray`, dtype:bool
        Gives the status (connected / disconnected) for every powerline (``True`` at position `i` means the powerline
        `i` is connected)

    timestep_overflow: :class:`numpy.ndarray`, dtype:int
        Gives the number of time steps since a powerline is in overflow.

    time_before_cooldown_line:  :class:`numpy.ndarray`, dtype:int
        For each powerline, it gives the number of time step the powerline is unavailable due to "cooldown"
        (see :attr:`grid2op.Parameters.Parameters.NB_TIMESTEP_LINE_STATUS_REMODIF` for more information). 0 means the
        an action will be able to act on this same powerline, a number > 0 (eg 1) means that an action at this time step
        cannot act on this powerline (in the example the agent have to wait 1 time step)

    time_before_cooldown_sub: :class:`numpy.ndarray`, dtype:int
        Same as :attr:`Observation.time_before_cooldown_line` but for substations. For each substation, it gives the
        number of timesteps to wait before acting on this substation (see
        see :attr:`grid2op.Parameters.Parameters.NB_TIMESTEP_TOPOLOGY_REMODIF` for more information).

    time_before_line_reconnectable: :class:`numpy.ndarray`, dtype:int
        For each powerline, it gives the number of timesteps before the powerline can be reconnected. This only
        concerns the maintenance, outage (hazards) and disconnection due to cascading failures (including overflow). The
        same convention as for :attr:`Observation.time_before_cooldown_line` and
        :attr:`Observation.time_before_cooldown_sub` is adopted: 0 at position `i` means that the powerline can be
        reconnected. It there is 2 (for example) it means that the powerline `i` is unavailable for 2 timesteps (we
        will be able to re connect it not this time, not next time, but the following timestep)

    time_next_maintenance: :class:`numpy.ndarray`, dtype:int
        For each powerline, it gives the time of the next planned maintenance. For example if there is:

            - `1` at position `i` it means that the powerline `i` will be disconnected for maintenance operation at the next time step. A
            - `0` at position `i` means that powerline `i` is disconnected from the powergrid for maintenance operation
              at the current time step.
            - `-1` at position `i` means that powerline `i` will not be disconnected for maintenance reason for this
              episode.

    duration_next_maintenance: :class:`numpy.ndarray`, dtype:int
        For each powerline, it gives the number of time step that the maintenance will last (if any). This means that,
        if at position `i` of this vector:

            - there is a `0`: the powerline is not disconnected from the grid for maintenance
            - there is a `1`, `2`, ... the powerline will be disconnected for at least `1`, `2`, ... timestep (**NB**
              in all case, the powerline will stay disconnected until a :class:`grid2op.Agent.Agent` performs the
              proper :class:`grid2op.Action.Action` to reconnect it).

    target_dispatch: :class:`numpy.ndarray`, dtype:float
        For **each** generators, it gives the target redispatching, asked by the agent. This is the sum of all
        redispatching asked by the agent for during all the episode. It for each generator it is a number between:
        - pmax and pmax. Note that there is information about all generators there, even the one that are not
        dispatchable.

    actual_dispatch: :class:`numpy.ndarray`, dtype:float
        For **each** generators, it gives the redispatching currently implemented by the environment.
        Indeed, the environment tries to implement at best the :attr:`Observation.target_dispatch`, but sometimes,
        due to physical limitation (pmin, pmax, ramp min and ramp max) it cannot. In this case, only the best possible
        redispatching is implemented at the current time step, and this is what this vector stores. Note that there is
        information about all generators there, even the one that are not
        dispatchable.

    """
    def __init__(self, gridobj,
                 obs_env=None,
                 action_helper=None,
                 seed=None):
        GridObjects.__init__(self)
        self.init_grid(gridobj)

        self.action_helper = action_helper

        # time stamp information
        self.year = 1970
        self.month = 0
        self.day = 0
        self.hour_of_day = 0
        self.minute_of_hour = 0
        self.day_of_week = 0

        # for non deterministic observation that would not use default np.random module
        self.seed = None

        # handles the forecasts here
        self._forecasted_grid = []
        self._forecasted_inj = []

        self._obs_env = obs_env

        self.timestep_overflow = None

        # 0. (line is disconnected) / 1. (line is connected)
        self.line_status = None

        # topological vector
        self.topo_vect = None

        # generators information
        self.prod_p = None
        self.prod_q = None
        self.prod_v = None
        # loads information
        self.load_p = None
        self.load_q = None
        self.load_v = None
        # lines origin information
        self.p_or = None
        self.q_or = None
        self.v_or = None
        self.a_or = None
        # lines extremity information
        self.p_ex = None
        self.q_ex = None
        self.v_ex = None
        self.a_ex = None
        # lines relative flows
        self.rho = None

        # cool down and reconnection time after hard overflow, soft overflow or cascading failure
        self.time_before_cooldown_line = None
        self.time_before_cooldown_sub = None
        self.time_before_line_reconnectable = None
        self.time_next_maintenance = None
        self.duration_next_maintenance = None

        # matrices
        self.connectivity_matrix_ = None
        self.bus_connectivity_matrix_ = None
        self.vectorized = None

        # redispatching
        self.target_dispatch = None
        self.actual_dispatch = None

        # value to assess if two observations are equal
        self._tol_equal = 5e-1

        self.attr_list_vect = None
        self.reset()

    def state_of(self, _sentinel=None, load_id=None, gen_id=None, line_id=None, substation_id=None):
        """
        Return the state of this action on a give unique load, generator unit, powerline of substation.
        Only one of load, gen, line or substation should be filled.

        The querry of these objects can only be done by id here (ie by giving the integer of the object in the backed).
        The :class:`HelperAction` has some utilities to access them by name too.

        Parameters
        ----------
        _sentinel: ``None``
            Used to prevent positional parameters. Internal, do not use.

        load_id: ``int``
            ID of the load we want to inspect

        gen_id: ``int``
            ID of the generator we want to inspect

        line_id: ``int``
            ID of the powerline we want to inspect

        substation_id: ``int``
            ID of the powerline we want to inspect

        Returns
        -------
        res: :class:`dict`
            A dictionnary with keys and value depending on which object needs to be inspected:

            - if a load is inspected, then the keys are:

                - "p" the active value consumed by the load
                - "q" the reactive value consumed by the load
                - "v" the voltage magnitude of the bus to which the load is connected
                - "bus" on which bus the load is connected in the substation
                - "sub_id" the id of the substation to which the load is connected

            - if a generator is inspected, then the keys are:

                - "p" the active value produced by the generator
                - "q" the reactive value consumed by the generator
                - "v" the voltage magnitude of the bus to which the generator is connected
                - "bus" on which bus the generator is connected in the substation
                - "sub_id" the id of the substation to which the generator is connected
                - "actual_dispatch" the actual dispatch implemented for this generator
                - "target_dispatch" the target dispatch (cumulation of all previously asked dispatch by the agent)
                  for this generator

            - if a powerline is inspected then the keys are "origin" and "extremity" each being dictionnary with keys:

                - "p" the active flow on line end (extremity or origin)
                - "q" the reactive flow on line end (extremity or origin)
                - "v" the voltage magnitude of the bus to which the line end (extremity or origin) is connected
                - "bus" on which bus the line end (extremity or origin) is connected in the substation
                - "sub_id" the id of the substation to which the generator is connected
                - "a" the current flow on the line end (extremity or origin)

                In the case of a powerline, additional information are:

                - "maintenance": information about the maintenance operation (time of the next maintenance and duration
                  of this next maintenance.
                - "cooldown_time": for how many timestep i am not supposed to act on the powerline due to cooldown
                  (see :attr:`grid2op.Parameters.Parameters.NB_TIMESTEP_LINE_STATUS_REMODIF` for more information)
                - "indisponibility": for how many timestep the powerline is unavailable (disconnected, and it's
                  impossible to reconnect it) due to hazards, maintenance or overflow (incl. cascading failure)

            - if a substation is inspected, it returns the topology to this substation in a dictionary with keys:

                - "topo_vect": the representation of which object is connected where
                - "nb_bus": number of active buses in this substations
                - "cooldown_time": for how many timestep i am not supposed to act on the substation due to cooldown
                  (see :attr:`grid2op.Parameters.Parameters.NB_TIMESTEP_TOPOLOGY_REMODIF` for more information)

        Raises
        ------
        Grid2OpException
            If _sentinel is modified, or if None of the arguments are set or alternatively if 2 or more of the
            parameters are being set.

        """
        if _sentinel is not None:
            raise Grid2OpException("action.effect_on should only be called with named argument.")

        if load_id is None and gen_id is None and line_id is None and substation_id is None:
            raise Grid2OpException("You ask the state of an object in a observation without specifying the object id. "
                                   "Please provide \"load_id\", \"gen_id\", \"line_id\" or \"substation_id\"")

        if load_id is not None:
            if gen_id is not None or line_id is not None or substation_id is not None:
                raise Grid2OpException("You can only the inspect the effect of an action on one single element")
            if load_id >= len(self.load_p):
                raise Grid2OpException("There are no load of id \"load_id={}\" in this grid.".format(load_id))

            res = {"p": self.load_p[load_id],
                   "q": self.load_q[load_id],
                   "v": self.load_v[load_id],
                   "bus": self.topo_vect[self.load_pos_topo_vect[load_id]],
                   "sub_id": self.load_to_subid[load_id]
                   }
        elif gen_id is not None:
            if line_id is not None or substation_id is not None:
                raise Grid2OpException("You can only the inspect the effect of an action on one single element")
            if gen_id >= len(self.prod_p):
                raise Grid2OpException("There are no generator of id \"gen_id={}\" in this grid.".format(gen_id))

            res = {"p": self.prod_p[gen_id],
                   "q": self.prod_q[gen_id],
                   "v": self.prod_v[gen_id],
                   "bus": self.topo_vect[self.gen_pos_topo_vect[gen_id]],
                   "sub_id": self.gen_to_subid[gen_id],
                   "target_dispatch": self.target_dispatch[gen_id],
                   "actual_dispatch": self.target_dispatch[gen_id]
                   }
        elif line_id is not None:
            if substation_id is not None:
                raise Grid2OpException("You can only the inspect the effect of an action on one single element")
            if line_id >= len(self.p_or):
                raise Grid2OpException("There are no powerline of id \"line_id={}\" in this grid.".format(line_id))

            res = {}
            # origin information
            res["origin"] = {
                "p": self.p_or[line_id],
                "q": self.q_or[line_id],
                "v": self.v_or[line_id],
                "a": self.a_or[line_id],
                "bus": self.topo_vect[self.line_or_pos_topo_vect[line_id]],
                "sub_id": self.line_or_to_subid[line_id]
            }
            # extremity information
            res["extremity"] = {
                "p": self.p_ex[line_id],
                "q": self.q_ex[line_id],
                "v": self.v_ex[line_id],
                "a": self.a_ex[line_id],
                "bus": self.topo_vect[self.line_ex_pos_topo_vect[line_id]],
                "sub_id": self.line_ex_to_subid[line_id]
            }

            # maintenance information
            res["maintenance"] = {"next": self.time_next_maintenance[line_id],
                                  "duration_next": self.duration_next_maintenance[line_id]}

            # cooldown
            res["cooldown_time"] = self.time_before_cooldown_line[line_id]

            # indisponibility
            res["indisponibility"] = self.time_before_line_reconnectable[line_id]

        else:
            if substation_id >= len(self.sub_info):
                raise Grid2OpException("There are no substation of id \"substation_id={}\" in this grid.".format(substation_id))

            beg_ = int(np.sum(self.sub_info[:substation_id]))
            end_ = int(beg_ + self.sub_info[substation_id])
            topo_sub = self.topo_vect[beg_:end_]
            if np.any(topo_sub > 0):
                nb_bus = np.max(topo_sub[topo_sub > 0]) - np.min(topo_sub[topo_sub > 0]) + 1
            else:
                nb_bus = 0
            res = {
                "topo_vect": topo_sub,
                "nb_bus": nb_bus,
                "cooldown_time": self.time_before_cooldown_sub[substation_id]
                   }

        return res

    def reset(self):
        """
        Reset the :class:`Observation` to a blank state, where everything is set to either ``None`` or to its default
        value.

        """
        # vecorized _grid
        self.timestep_overflow = np.zeros(shape=(self.n_line,), dtype=np.int)

        # 0. (line is disconnected) / 1. (line is connected)
        self.line_status = np.ones(shape=self.n_line, dtype=np.bool)

        # topological vector
        self.topo_vect = np.full(shape=self.dim_topo, dtype=np.int, fill_value=0)

        # generators information
        self.prod_p = np.full(shape=self.n_gen, dtype=np.float, fill_value=np.NaN)
        self.prod_q = np.full(shape=self.n_gen, dtype=np.float, fill_value=np.NaN)
        self.prod_v = np.full(shape=self.n_gen, dtype=np.float, fill_value=np.NaN)
        # loads information
        self.load_p = np.full(shape=self.n_load, dtype=np.float, fill_value=np.NaN)
        self.load_q = np.full(shape=self.n_load, dtype=np.float, fill_value=np.NaN)
        self.load_v = np.full(shape=self.n_load, dtype=np.float, fill_value=np.NaN)
        # lines origin information
        self.p_or = np.full(shape=self.n_line, dtype=np.float, fill_value=np.NaN)
        self.q_or = np.full(shape=self.n_line, dtype=np.float, fill_value=np.NaN)
        self.v_or = np.full(shape=self.n_line, dtype=np.float, fill_value=np.NaN)
        self.a_or = np.full(shape=self.n_line, dtype=np.float, fill_value=np.NaN)
        # lines extremity information
        self.p_ex = np.full(shape=self.n_line, dtype=np.float, fill_value=np.NaN)
        self.q_ex = np.full(shape=self.n_line, dtype=np.float, fill_value=np.NaN)
        self.v_ex = np.full(shape=self.n_line, dtype=np.float, fill_value=np.NaN)
        self.a_ex = np.full(shape=self.n_line, dtype=np.float, fill_value=np.NaN)
        # lines relative flows
        self.rho = np.full(shape=self.n_line, dtype=np.float, fill_value=np.NaN)

        # cool down and reconnection time after hard overflow, soft overflow or cascading failure
        self.time_before_cooldown_line = np.full(shape=self.n_line, dtype=np.int, fill_value=-1)
        self.time_before_cooldown_sub = np.full(shape=self.n_sub, dtype=np.int, fill_value=-1)
        self.time_before_line_reconnectable = np.full(shape=self.n_line, dtype=np.int, fill_value=-1)
        self.time_next_maintenance = np.full(shape=self.n_line, dtype=np.int, fill_value=-1)
        self.duration_next_maintenance = np.full(shape=self.n_line, dtype=np.int, fill_value=-1)

        # calendar data
        self.year = 1970
        self.month = 0
        self.day = 0
        self.hour_of_day = 0
        self.minute_of_hour = 0
        self.day_of_week = 0

        # forecasts
        self._forecasted_inj = []
        self._forecasted_grid = []

        # redispatching
        self.target_dispatch = np.full(shape=self.n_gen, dtype=np.float, fill_value=np.NaN)
        self.actual_dispatch = np.full(shape=self.n_gen, dtype=np.float, fill_value=np.NaN)

    def __compare_stats(self, other, name):
        if self.__dict__[name] is None and other.__dict__[name] is not None:
            return False
        if self.__dict__[name] is not None and other.__dict__[name] is None:
            return False
        if self.__dict__[name] is not None:
            if self.__dict__[name].shape != other.__dict__[name].shape:
                return False
            if self.__dict__[name].dtype != other.__dict__[name].dtype:
                return False
            if np.issubdtype(self.__dict__[name].dtype, np.dtype(float).type):
                # special case of floating points, otherwise vector are never equal
                if not np.all(np.abs(self.__dict__[name] - other.__dict__[name]) <= self._tol_equal):
                    return False
            else:
                if not np.all(self.__dict__[name] == other.__dict__[name]):
                    return False
        return True

    def __eq__(self, other):
        """
        Test the equality of two actions.

        2 actions are said to be identical if the have the same impact on the powergrid. This is unlrelated to their
        respective class. For example, if an Action is of class :class:`Action` and doesn't act on the _injection, it
        can be equal to a an Action of derived class :class:`TopologyAction` (if the topological modification are the
        same of course).

        This implies that the attributes :attr:`Action.authorized_keys` is not checked in this method.

        Note that if 2 actions doesn't act on the same powergrid, or on the same backend (eg number of loads, or
        generators is not the same in *self* and *other*, or they are not in the same order) then action will be
        declared as different.

        **Known issue** if two backend are different, but the description of the _grid are identical (ie all
        n_gen, n_load, n_line, sub_info, dim_topo, all vectors \*_to_subid, and \*_pos_topo_vect are
        identical) then this method will not detect the backend are different, and the action could be declared
        as identical. For now, this is only a theoretical behaviour: if everything is the same, then probably, up to
        the naming convention, then the powergrid are identical too.

        Parameters
        ----------
        other: :class:`Observation`
            An instance of class Action to which "self" will be compared.

        Returns
        -------

        """
        # TODO doc above

        if self.year != other.year:
            return False
        if self.month != other.month:
            return False
        if self.day != other.day:
            return False
        if self.day_of_week != other.day_of_week:
            return False
        if self.hour_of_day != other.hour_of_day:
            return False
        if self.minute_of_hour != other.minute_of_hour:
            return False

        # check that the _grid is the same in both instances
        same_grid = True
        same_grid = same_grid and self.n_gen == other.n_gen
        same_grid = same_grid and self.n_load == other.n_load
        same_grid = same_grid and self.n_line == other.n_line
        same_grid = same_grid and np.all(self.sub_info == other.sub_info)
        same_grid = same_grid and self.dim_topo == other.dim_topo
        # to which substation is connected each element
        same_grid = same_grid and np.all(self.load_to_subid == other.load_to_subid)
        same_grid = same_grid and np.all(self.gen_to_subid == other.gen_to_subid)
        same_grid = same_grid and np.all(self.line_or_to_subid == other.line_or_to_subid)
        same_grid = same_grid and np.all(self.line_ex_to_subid == other.line_ex_to_subid)
        # which index has this element in the substation vector
        same_grid = same_grid and np.all(self.load_to_sub_pos == other.load_to_sub_pos)
        same_grid = same_grid and np.all(self.gen_to_sub_pos == other.gen_to_sub_pos)
        same_grid = same_grid and np.all(self.line_or_to_sub_pos == other.line_or_to_sub_pos)
        same_grid = same_grid and np.all(self.line_ex_to_sub_pos == other.line_ex_to_sub_pos)
        # which index has this element in the topology vector
        same_grid = same_grid and np.all(self.load_pos_topo_vect == other.load_pos_topo_vect)
        same_grid = same_grid and np.all(self.gen_pos_topo_vect == other.gen_pos_topo_vect)
        same_grid = same_grid and np.all(self.line_or_pos_topo_vect == other.line_or_pos_topo_vect)
        same_grid = same_grid and np.all(self.line_ex_pos_topo_vect == other.line_ex_pos_topo_vect)

        if not same_grid:
            return False

        for stat_nm in ["line_status", "topo_vect",
                        "timestep_overflow",
                        "prod_p", "prod_q", "prod_v",
                        "load_p", "load_q", "load_v",
                        "p_or", "q_or", "v_or", "a_or",
                        "p_ex", "q_ex", "v_ex", "a_ex",
                        "time_before_cooldown_line",
                        "time_before_cooldown_sub",
                        "time_before_line_reconnectable",
                        "time_next_maintenance",
                        "duration_next_maintenance",
                        "target_dispatch", "actual_dispatch"
                        ]:
            if not self.__compare_stats(other, stat_nm):
                # one of the above stat is not equal in this and in other
                return False

        return True

    @abstractmethod
    def update(self, env):
        """
        Update the actual instance of Observation with the new received value from the environment.

        An observation is a description of the powergrid perceived by an agent. The agent takes his decision based on
        the current observation and the past rewards.

        This method `update` receive complete detailed information about the powergrid, but that does not mean an
        agent sees everything.
        For example, it is possible to derive this class to implement some noise in the generator or load, or flows to
        mimic sensor inaccuracy.

        It is also possible to give fake information about the topology, the line status etc.

        In the Grid2Op framework it's also through the observation that the agent has access to some forecast (the way
        forecast are handled depends are implemented in this class). For example, forecast data (retrieved thanks to
        `chronics_handler`) are processed, but can be processed differently. One can apply load / production forecast to
        each _grid state, or to make forecast for one "reference" _grid state valid a whole day and update this one
        only etc.
        All these different mechanisms can be implemented in Grid2Op framework by overloading the `update` observation
        method.

        This class is really what a dispatcher observes from it environment.
        It can also include some temperatures, nebulosity, wind etc. can also be included in this class.
        """
        pass

    def connectivity_matrix(self):
        """
        Computes and return the "connectivity matrix" `con_mat`.
        if "dim_topo = 2 * n_line + n_prod + n_conso"
        It is a matrix of size dim_topo, dim_topo, with values 0 or 1.
        For two objects (lines extremity, generator unit, load) i,j :

            - if i and j are connected on the same substation:
                - if `conn_mat[i,j] = 0` it means the objects id'ed i and j are not connected to the same bus.
                - if `conn_mat[i,j] = 1` it means the objects id'ed i and j are connected to the same bus, are both end
                  of the same powerline

            - if i and j are not connected on the same substation then`conn_mat[i,j] = 0` except if i and j are
              the two extremities of the same power line, in this case `conn_mat[i,j] = 1`.

        By definition, the diagonal is made of 0.

        Returns
        -------
        res: ``numpy.ndarray``, shape:dim_topo,dim_topo, dtype:float
            The connectivity matrix, as defined above
        """
        raise NotImplementedError("This method is not implemented")

    def bus_connectivity_matrix(self):
        """
        If we denote by `nb_bus` the total number bus of the powergrid.

        The `bus_connectivity_matrix` will have a size nb_bus, nb_bus and will be made of 0 and 1.

        If `bus_connectivity_matrix[i,j] = 1` then at least a power line connects bus i and bus j.
        Otherwise, nothing connects it.

        Returns
        -------
        res: ``numpy.ndarray``, shape:nb_bus,nb_bus dtype:float
            The bus connectivity matrix
        """
        raise NotImplementedError("This method is not implemented. ")

    def simulate(self, action, time_step=0):
        """
        This method is used to simulate the effect of an action on a forecasted powergrid state. It has the same return
        value as the :func:`grid2op.Environment.Environment.step` function.


        Parameters
        ----------
        action: :class:`grid2op.Action.Action`
            The action to simulate

        time_step: ``int``
            The time step of the forecasted grid to perform the action on. If no forecast are available for this
            time step, a :class:`grid2op.Exceptions.NoForecastAvailable` is thrown.

        Raises
        ------
        :class:`grid2op.Exceptions.NoForecastAvailable`
            if no forecast are available for the time_step querried.

        Returns
        -------
            observation: :class:`grid2op.Observation.Observation`
                agent's observation of the current environment
            reward: ``float``
                amount of reward returned after previous action
            done: ``bool``
                whether the episode has ended, in which case further step() calls will return undefined results
            info: ``dict``
                contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)

        """
        if self.action_helper is None or self._obs_env is None:
            raise NoForecastAvailable("No forecasts are available for this instance of Observation (no action_space "
                                      "and no simulated environment are set).")

        if time_step >= len(self._forecasted_inj):
            raise NoForecastAvailable("Forecast for {} timestep ahead is not possible with your chronics.".format(time_step))

        timestamp, inj_forecasted = self._forecasted_inj[time_step]
        inj_action = self.action_helper(inj_forecasted)
        # initialize the "simulation environment" with the proper injections
        self._forecasted_grid[time_step] = self._obs_env.copy()
        # TODO avoid un necessary copy above. Have one backend for all "simulate" and save instead the
        # TODO obs_env._action that set the backend to the sate we want to simulate
        self._forecasted_grid[time_step].init(inj_action, time_stamp=timestamp,
                                              timestep_overflow=self.timestep_overflow)

        return self._forecasted_grid[time_step].simulate(action)

    def copy(self):
        """
        Make a (deep) copy of the observation.

        Returns
        -------
        res: :class:`Observation`
            The deep copy of the observation

        """
        obs_env = self._obs_env
        self._obs_env = None
        res = copy.deepcopy(self)
        self._obs_env = obs_env
        res._obs_env = obs_env.copy()
        return res


class CompleteObservation(Observation):
    """
    This class represent a complete observation, where everything on the powergrid can be observed without
    any noise.

    This is the only :class:`Observation` implemented (and used) in Grid2Op. Other type of observation, for other
    usage can of course be implemented following this example.

    It has the same attributes as the :class:`Observation` class. Only one is added here.

    For a :class:`CompleteObservation` the unique representation as a vector is:

        1. the year [1 element]
        2. the month [1 element]
        3. the day [1 element]
        4. the day of the week. Monday = 0, Sunday = 6 [1 element]
        5. the hour of the day [1 element]
        6. minute of the hour  [1 element]
        7. :attr:`Observation.prod_p` the active value of the productions [:attr:`Observation.n_gen` elements]
        8. :attr:`Observation.prod_q` the reactive value of the productions [:attr:`Observation.n_gen` elements]
        9. :attr:`Observation.prod_q` the voltage setpoint of the productions [:attr:`Observation.n_gen` elements]
        10. :attr:`Observation.load_p` the active value of the loads [:attr:`Observation.n_load` elements]
        11. :attr:`Observation.load_q` the reactive value of the loads [:attr:`Observation.n_load` elements]
        12. :attr:`Observation.load_v` the voltage setpoint of the loads [:attr:`Observation.n_load` elements]
        13. :attr:`Observation.p_or` active flow at origin of powerlines [:attr:`Observation.n_line` elements]
        14. :attr:`Observation.q_or` reactive flow at origin of powerlines [:attr:`Observation.n_line` elements]
        15. :attr:`Observation.v_or` voltage at origin of powerlines [:attr:`Observation.n_line` elements]
        16. :attr:`Observation.a_or` current flow at origin of powerlines [:attr:`Observation.n_line` elements]
        17. :attr:`Observation.p_ex` active flow at extremity of powerlines [:attr:`Observation.n_line` elements]
        18. :attr:`Observation.q_ex` reactive flow at extremity of powerlines [:attr:`Observation.n_line` elements]
        19. :attr:`Observation.v_ex` voltage at extremity of powerlines [:attr:`Observation.n_line` elements]
        20. :attr:`Observation.a_ex` current flow at extremity of powerlines [:attr:`Observation.n_line` elements]
        21. :attr:`Observation.rho` line capacity used (current flow / thermal limit) [:attr:`Observation.n_line` elements]
        22. :attr:`Observation.line_status` line status [:attr:`Observation.n_line` elements]
        23. :attr:`Observation.timestep_overflow` number of timestep since the powerline was on overflow
            (0 if the line is not on overflow)[:attr:`Observation.n_line` elements]
        24. :attr:`Observation.topo_vect` representation as a vector of the topology [for each element
            it gives its bus]. See :func:`grid2op.Backend.Backend.get_topo_vect` for more information.
        25. :attr:`Observation.time_before_cooldown_line` representation of the cooldown time on the powerlines
            [:attr:`Observation.n_line` elements]
        26. :attr:`Observation.time_before_cooldown_sub` representation of the cooldown time on the substations
            [:attr:`Observation.n_sub` elements]
        27. :attr:`Observation.time_before_line_reconnectable` number of timestep to wait before a powerline
            can be reconnected (it is disconnected due to maintenance, cascading failure or overflow)
            [:attr:`Observation.n_line` elements]
        28. :attr:`Observation.time_next_maintenance` number of timestep before the next maintenance (-1 means
            no maintenance are planned, 0 a maintenance is in operation) [:attr:`Observation.n_line` elements]
        29. :attr:`Observation.duration_next_maintenance` duration of the next maintenance. If a maintenance
            is taking place, this is the number of timestep before it ends. [:attr:`Observation.n_line` elements]
        30. :attr:`Observation.target_dispatch` the target dispatch for each generator
            [:attr:`Observation.n_gen` elements]
        31. :attr:`Observation.actual_dispatch` the actual dispatch for each generator
            [:attr:`Observation.n_gen` elements]

    This behavior is specified in the :attr:`Observation.attr_list_vect` vector.

    Attributes
    ----------
    dictionnarized: ``dict``
        The representation of the action in a form of a dictionnary. See the definition of
        :func:`CompleteObservation.to_dict` for a description of this dictionnary.

    """
    def __init__(self, gridobj,
                 obs_env=None,action_helper=None,
                 seed=None):

        Observation.__init__(self, gridobj,
                             obs_env=obs_env,
                             action_helper=action_helper,
                             seed=seed)
        self.dictionnarized = None
        self.attr_list_vect = ["year", "month", "day", "hour_of_day", "minute_of_hour", "day_of_week",
                               "prod_p", "prod_q", "prod_v",
                               "load_p", "load_q", "load_v",
                               "p_or", "q_or", "v_or", "a_or",
                               "p_ex", "q_ex", "v_ex", "a_ex",
                               "rho",
                               "line_status", "timestep_overflow",
                               "topo_vect", "time_before_cooldown_line",
                               "time_before_cooldown_line", "time_before_cooldown_sub",
                               "time_before_line_reconnectable",
                               "time_next_maintenance", "duration_next_maintenance",
                               "target_dispatch", "actual_dispatch"
                               ]

    def _reset_matrices(self):
        self.connectivity_matrix_ = None
        self.bus_connectivity_matrix_ = None
        self.vectorized = None
        self.dictionnarized = None

    def update(self, env):
        """
        This use the environement to update properly the Observation.

        Parameters
        ----------
        env: :class:`grid2op.Environment.Environment`
            The environment from which to update this observation.

        """
        # reset the matrices
        self._reset_matrices()
        self.reset()

        # extract the time stamps
        self.year = env.time_stamp.year
        self.month = env.time_stamp.month
        self.day = env.time_stamp.day
        self.hour_of_day = env.time_stamp.hour
        self.minute_of_hour = env.time_stamp.minute
        self.day_of_week = env.time_stamp.weekday()

        # get the values related to topology
        self.timestep_overflow = copy.copy(env.timestep_overflow)
        self.line_status = copy.copy(env.backend.get_line_status())
        self.topo_vect = copy.copy(env.backend.get_topo_vect())

        # get the values related to continuous values
        self.prod_p[:], self.prod_q[:], self.prod_v[:] = env.backend.generators_info()
        self.load_p[:], self.load_q[:], self.load_v[:] = env.backend.loads_info()
        self.p_or[:], self.q_or[:], self.v_or[:], self.a_or[:] = env.backend.lines_or_info()
        self.p_ex[:], self.q_ex[:], self.v_ex[:], self.a_ex[:] = env.backend.lines_ex_info()

        # handles forecasts here
        self._forecasted_inj = env.chronics_handler.forecasts()
        for i in range(len(self._forecasted_grid)):
            # in the action, i assign the lat topology known, it's a choice here...
            self._forecasted_grid[i]["setbus"] = self.topo_vect

        self._forecasted_grid = [None for _ in self._forecasted_inj]
        self.rho = env.backend.get_relative_flow()

        # cool down and reconnection time after hard overflow, soft overflow or cascading failure
        self.time_before_cooldown_line[:] = env.times_before_line_status_actionable
        self.time_before_cooldown_sub[:] = env.times_before_topology_actionable
        self.time_before_line_reconnectable[:] = env.time_remaining_before_line_reconnection
        self.time_next_maintenance[:] = env.time_next_maintenance
        self.duration_next_maintenance[:] = env.duration_next_maintenance

        # redispatching
        self.target_dispatch[:] = env.target_dispatch
        self.actual_dispatch[:] = env.actual_dispatch

    def from_vect(self, vect):
        """
        Convert back an observation represented as a vector into a proper observation.

        Some convertion are done silently from float to the type of the corresponding observation attribute.

        Parameters
        ----------
        vect: ``numpy.ndarray``
            A representation of an Observation in the form of a vector that is used to convert back the current
            observation to be equal to the vect.

        """

        # reset the matrices
        self._reset_matrices()
        # and ensure everything is reloaded properly
        super().from_vect(vect)

    def to_dict(self):
        """

        Returns
        -------

        """
        # TODO doc
        if self.dictionnarized is None:
            self.dictionnarized = {}
            self.dictionnarized["timestep_overflow"] = self.timestep_overflow
            self.dictionnarized["line_status"] = self.line_status
            self.dictionnarized["topo_vect"] = self.topo_vect
            self.dictionnarized["loads"] = {}
            self.dictionnarized["loads"]["p"] = self.load_p
            self.dictionnarized["loads"]["q"] = self.load_q
            self.dictionnarized["loads"]["v"] = self.load_v
            self.dictionnarized["prods"] = {}
            self.dictionnarized["prods"]["p"] = self.prod_p
            self.dictionnarized["prods"]["q"] = self.prod_q
            self.dictionnarized["prods"]["v"] = self.prod_v
            self.dictionnarized["lines_or"] = {}
            self.dictionnarized["lines_or"]["p"] = self.p_or
            self.dictionnarized["lines_or"]["q"] = self.q_or
            self.dictionnarized["lines_or"]["v"] = self.v_or
            self.dictionnarized["lines_or"]["a"] = self.a_or
            self.dictionnarized["lines_ex"] = {}
            self.dictionnarized["lines_ex"]["p"] = self.p_ex
            self.dictionnarized["lines_ex"]["q"] = self.q_ex
            self.dictionnarized["lines_ex"]["v"] = self.v_ex
            self.dictionnarized["lines_ex"]["a"] = self.a_ex
            self.dictionnarized["rho"] = self.rho

            self.dictionnarized["maintenance"] = {}
            self.dictionnarized["maintenance"]['time_next_maintenance'] = self.time_next_maintenance
            self.dictionnarized["maintenance"]['duration_next_maintenance'] = self.duration_next_maintenance
            self.dictionnarized["cooldown"] = {}
            self.dictionnarized["cooldown"]['line'] = self.time_before_cooldown_line
            self.dictionnarized["cooldown"]['substation'] = self.time_before_cooldown_sub
            self.dictionnarized["time_before_line_reconnectable"] = self.time_before_line_reconnectable
            self.dictionnarized["redispatching"] = {}
            self.dictionnarized["redispatching"]["target_redispatch"] = self.target_dispatch
            self.dictionnarized["redispatching"]["actual_dispatch"] = self.actual_dispatch

        return self.dictionnarized

    def connectivity_matrix(self):
        """
        Computes and return the "connectivity matrix" `con_mat`.
        if "dim_topo = 2 * n_line + n_prod + n_conso"
        It is a matrix of size dim_topo, dim_topo, with values 0 or 1.
        For two objects (lines extremity, generator unit, load) i,j :

            - if i and j are connected on the same substation:
                - if `conn_mat[i,j] = 0` it means the objects id'ed i and j are not connected to the same bus.
                - if `conn_mat[i,j] = 1` it means the objects id'ed i and j are connected to the same bus, are both end
                  of the same powerline

            - if i and j are not connected on the same substation then`conn_mat[i,j] = 0` except if i and j are
              the two extremities of the same power line, in this case `conn_mat[i,j] = 1`.

        By definition, the diagonal is made of 0.

        Returns
        -------
        res: ``numpy.ndarray``, shape:dim_topo,dim_topo, dtype:float
            The connectivity matrix, as defined above
        """
        if self.connectivity_matrix_ is None:
            self.connectivity_matrix_ = np.zeros(shape=(self.dim_topo, self.dim_topo),dtype=np.float)
            # fill it by block for the objects
            beg_ = 0
            end_ = 0
            for sub_id, nb_obj in enumerate(self.sub_info):
                # it must be a vanilla python integer, otherwise it's not handled by some backend
                # especially if written in c++
                nb_obj = int(nb_obj)
                end_ += nb_obj
                tmp = np.zeros(shape=(nb_obj, nb_obj), dtype=np.float)
                for obj1 in range(nb_obj):
                    for obj2 in range(obj1+1, nb_obj):
                        if self.topo_vect[beg_+obj1] == self.topo_vect[beg_+obj2]:
                            # objects are on the same bus
                            tmp[obj1, obj2] = 1
                            tmp[obj2, obj1] = 1

                self.connectivity_matrix_[beg_:end_, beg_:end_] = tmp
                beg_ += nb_obj
            # connect the objects together with the lines (both ends of a lines are connected together)
            for q_id in range(self.n_line):
                self.connectivity_matrix_[self.line_or_pos_topo_vect[q_id], self.line_ex_pos_topo_vect[q_id]] = 1
                self.connectivity_matrix_[self.line_ex_pos_topo_vect[q_id], self.line_or_pos_topo_vect[q_id]] = 1

        return self.connectivity_matrix_

    def bus_connectivity_matrix(self):
        """
        If we denote by `nb_bus` the total number bus of the powergrid.

        The `bus_connectivity_matrix` will have a size nb_bus, nb_bus and will be made of 0 and 1.

        If `bus_connectivity_matrix[i,j] = 1` then at least a power line connects bus i and bus j.
        Otherwise, nothing connects it.

        Returns
        -------
        res: ``numpy.ndarray``, shape:nb_bus,nb_bus dtype:float
            The bus connectivity matrix
        """
        # TODO voir avec Antoine pour les r,x,h ici !! (surtout les x)
        if self.bus_connectivity_matrix_ is None:
            # computes the number of buses in the powergrid.
            nb_bus = 0
            nb_bus_per_sub = np.zeros(self.sub_info.shape[0])
            beg_ = 0
            end_ = 0
            for sub_id, nb_obj in enumerate(self.sub_info):
                nb_obj = int(nb_obj)
                end_ += nb_obj

                tmp = len(np.unique(self.topo_vect[beg_:end_]))
                nb_bus_per_sub[sub_id] = tmp
                nb_bus += tmp

                beg_ += nb_obj

            # define the bus_connectivity_matrix
            self.bus_connectivity_matrix_ = np.zeros(shape=(nb_bus, nb_bus), dtype=np.float)
            np.fill_diagonal(self.bus_connectivity_matrix_, 1)

            for q_id in range(self.n_line):
                bus_or = int(self.topo_vect[self.line_or_pos_topo_vect[q_id]])
                sub_id_or = int(self.line_or_to_subid[q_id])

                bus_ex = int(self.topo_vect[self.line_ex_pos_topo_vect[q_id]])
                sub_id_ex = int(self.line_ex_to_subid[q_id])

                # try:
                bus_id_or = int(np.sum(nb_bus_per_sub[:sub_id_or])+(bus_or-1))
                bus_id_ex = int(np.sum(nb_bus_per_sub[:sub_id_ex])+(bus_ex-1))

                self.bus_connectivity_matrix_[bus_id_or, bus_id_ex] = 1
                self.bus_connectivity_matrix_[bus_id_ex, bus_id_or] = 1
                # except:
                #     pdb.set_trace()
        return self.bus_connectivity_matrix_


class SerializableObservationSpace(SerializableSpace):
    """
    This class allows to serialize / de serialize the action space.

    It should not be used inside an Environment, as some functions of the action might not be compatible with
    the serialization, especially the checking of whether or not an Observation is legal or not.

    Attributes
    ----------

    observationClass: ``type``
        Type used to build the :attr:`SerializableActionSpace._template_act`

    _empty_obs: :class:`Observation`
        An instance of the "*observationClass*" provided used to provide higher level utilities

    """
    def __init__(self, gridobj, observationClass=CompleteObservation):
        """

        Parameters
        ----------
        gridobj: :class:`grid2op.Space.GridObjects`
            Representation of the objects in the powergrid.

        observationClass: ``type``
            Type of action used to build :attr:`Space.SerializableSpace._template_obj`

        """
        SerializableSpace.__init__(self, gridobj=gridobj, subtype=observationClass)

        self.observationClass = self.subtype
        self._empty_obs = self._template_obj

    @staticmethod
    def from_dict(dict_):
        """
        Allows the de-serialization of an object stored as a dictionnary (for example in the case of json saving).

        Parameters
        ----------
        dict_: ``dict``
            Representation of an Observation Space (aka SerializableObservationSpace) as a dictionnary.

        Returns
        -------
        res: :class:``SerializableObservationSpace``
            An instance of an action space matching the dictionnary.

        """
        tmp = SerializableSpace.from_dict(dict_)
        res = SerializableObservationSpace(gridobj=tmp,
                                           observationClass=tmp.subtype)
        return res


class ObservationHelper(SerializableObservationSpace):
    """
    Helper that provides usefull functions to manipulate :class:`Observation`.

    Observation should only be built using this Helper. It is absolutely not recommended to make an observation
    directly form its constructor.

    This class represents the same concept as the "Observation Space" in the OpenAI gym framework.

    Attributes
    ----------

    observationClass: ``type``
        Class used to build the observations. It defaults to :class:`CompleteObservation`

    _empty_obs: ``Observation.Observation``
        An empty observation with the proper dimensions.

    parameters: :class:`grid2op.Parameters.Parameters`
        Type of Parameters used to compute powerflow for the forecast.

    rewardClass: ``type``
        Class used by the :class:`grid2op.Environment.Environment` to send information about its state to the
        :class:`grid2op.Agent.Agent`. You can change this class to differentiate between the reward of output of
        :func:`Observation.simulate`  and the reward used to train the Agent.

    action_helper_env: :class:`grid2op.Action.HelperAction`
        Action space used to create action during the :func:`Observation.simulate`

    reward_helper: :class:`grid2op.Reward.HelperReward`
        Reward function used by the the :func:`Observation.simulate` function.

    obs_env: :class:`ObsEnv`
        Instance of the environenment used by the Observation Helper to provide forcecast of the grid state.

    _empty_obs: :class:`Observation`
        An instance of the observation that is updated and will be sent to he Agent.

    """
    def __init__(self, gridobj,
                 env,
                 rewardClass=None,
                 observationClass=CompleteObservation):
        """
        Env: requires :attr:`grid2op.Environment.parameters` and :attr:`grid2op.Environment.backend` to be valid
        """

        SerializableObservationSpace.__init__(self, gridobj, observationClass=observationClass)

        # TODO DOCUMENTATION !!!

        # print("ObservationHelper init with rewardClass: {}".format(rewardClass))
        self.parameters = copy.deepcopy(env.parameters)
        # for the observation, I switch between the _parameters for the environment and for the simulation
        self.parameters.ENV_DC = self.parameters.FORECAST_DC

        if rewardClass is None:
            self.rewardClass = env.rewardClass
        else:
            self.rewardClass = rewardClass

        # helpers
        self.action_helper_env = env.helper_action_env
        self.reward_helper = RewardHelper(rewardClass=self.rewardClass)
        self.reward_helper.initialize(env)

        # TODO here: have another backend maybe
        self.backend_obs = env.backend.copy()

        self.obs_env = ObsEnv(backend_instanciated=self.backend_obs, obsClass=self.observationClass,
                              parameters=env.parameters,
                              reward_helper=self.reward_helper,
                              action_helper=self.action_helper_env,
                              thermal_limit_a=env._thermal_limit_a,
                              legalActClass=env.legalActClass,
                              donothing_act=env.helper_action_player())

        self._empty_obs = self.observationClass(gridobj=self,
                                                obs_env=self.obs_env,
                                                action_helper=self.action_helper_env)
        self._update_env_time = 0.

    def __call__(self, env):
        self.obs_env.update_grid(env)

        res = self.observationClass(gridobj=self,
                                    obs_env=self.obs_env,
                                    action_helper=self.action_helper_env)

        # TODO how to make sure that whatever the number of time i call "simulate" i still get the same observations
        # TODO use self.obs_prng when updating actions
        res.update(env=env)
        return res

    def size_obs(self):
        """
        Size if the observation vector would be flatten
        :return:
        """
        return self.n
