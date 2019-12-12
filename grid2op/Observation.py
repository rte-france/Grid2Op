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
import math

from abc import ABC, abstractmethod

import pdb

try:
    from .Exceptions import *
    from .Reward import ConstantReward, RewardHelper
    from ._utils import extract_from_dict, save_to_dict
except (ModuleNotFoundError, ImportError):
    from Exceptions import *
    from Reward import ConstantReward, RewardHelper
    from _utils import extract_from_dict, save_to_dict

# TODO be able to change reward here

# TODO refactor, Observation and Action, they are really close in their actual form, especially the Helpers, if
# TODO that make sense.

# TODO make an action with the difference between the observation that would be an action.
# TODO have a method that could do "forecast" by giving the _injection by the agent, if he wants to make custom forecasts

# TODO finish documentation


class ObsCH(object):
    """
    This class is reserved to internal use. Do not attempt to do anything with it.
    """
    def forecasts(self):
        return []


class ObsEnv(object):
    """
    This class is an 'Emulator' of a :class:`grid2op.Environment` used to be able to 'simulate' forecasted grid states.
    It should not be used outside of an :class:`grid2op.Observation` instance, or one of its derivative.

    It contains only the most basic element of an Environment. See :class:`grid2op.Environment` for more details.

    This class is reserved for internal use. Do not attempt to do anything with it.
    """
    def __init__(self, backend_instanciated, parameters, reward_helper, obsClass, action_helper):
        self.timestep_overflow = None
        # self.action_helper = action_helper
        self.hard_overflow_threshold = parameters.HARD_OVERFLOW_THRESHOLD
        self.nb_timestep_overflow_allowed = np.full(shape=(backend_instanciated.n_lines,),
                                                    fill_value=parameters.NB_TIMESTEP_POWERFLOW_ALLOWED)
        self.no_overflow_disconnection = parameters.NO_OVERFLOW_DISCONNECTION
        self.backend = backend_instanciated.copy()
        self.is_init = False
        self.env_dc = parameters.FORECAST_DC
        self.current_obs = None
        self.reward_helper = reward_helper
        self.obsClass = obsClass
        self.parameters = parameters
        self.dim_topo = np.sum(self.backend.subs_elements)
        self.time_stamp = None

        self.chronics_handler = ObsCH()

        self.times_before_line_status_actionable = np.zeros(shape=(self.backend.n_lines,), dtype=np.int)
        self.times_before_topology_actionable = np.zeros(shape=(self.backend.n_substations,), dtype=np.int)
        self.time_remaining_before_line_reconnection = np.zeros(shape=(self.backend.n_lines,), dtype=np.int)

        # TODO handle that in forecast!
        self.time_next_maintenance = np.zeros(shape=(self.backend.n_lines,), dtype=np.int) - 1
        self.duration_next_maintenance = np.zeros(shape=(self.backend.n_lines,), dtype=np.int)

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
        self.backend.apply_action(new_state_action)
        self.is_init = True
        self.current_obs = None
        self.time_stamp = time_stamp
        self.timestep_overflow = timestep_overflow

    def simulate(self, action):
        """
        This function is the core method of the :class:`ObsEnv`. It allows to perform a simulation of what would
        give and action if it were to be implemented on the "forecasted" powergrid.

        It has the same signature as :func:`grid2op.Environment.step`. One of the major difference is that it doesn't
        check whether the action is illegal or not (but an implementation could be provided for this method). The
        reason for this is that there is not one single and unique way to "forecast" how the thermal limit will behave,
        which lines will be available or not, which actions will be done or not between the time stamp at which
        "simulate" is called, and the time stamp that is simulated.

        Parameters
        ----------
        action: :class:`grid2op.Action`
            The action to test

        Returns
        -------
        observation: :class:`grid2op.Observation`
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
        has_error = True
        tmp_backend = self.backend.copy()
        is_done = False
        reward = None

        is_illegal = False
        is_ambiguous = False

        try:
            self.backend.apply_action(action)
        except AmbiguousAction:
            # action has not been implemented on the powergrid because it's ambiguous, it's equivalent to
            # "do nothing"
            is_ambiguous = True

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                disc_lines, infos = self.backend.next_grid_state(env=self, is_dc=self.env_dc)
            self.current_obs = self.obsClass(self.backend.n_generators, self.backend.n_loads, self.backend.n_lines,
                                             self.backend.subs_elements, self.dim_topo,
                                             self.backend.load_to_subid, self.backend.gen_to_subid,
                                             self.backend.lines_or_to_subid, self.backend.lines_ex_to_subid,
                                             self.backend.load_to_sub_pos, self.backend.gen_to_sub_pos,
                                             self.backend.lines_or_to_sub_pos,
                                             self.backend.lines_ex_to_sub_pos,
                                             self.backend.load_pos_topo_vect, self.backend.gen_pos_topo_vect,
                                             self.backend.lines_or_pos_topo_vect,
                                             self.backend.lines_ex_pos_topo_vect,
                                             seed=None,
                                             obs_env=None,
                                             action_helper=None)
            self.current_obs.update(self)
            has_error = False

        except Grid2OpException as e:
            has_error = True
            reward = self.reward_helper.range()[0]
        # print("reward_helper in ObsEnv: {}".format(self.reward_helper.template_reward))
        if reward is None:
            reward = self._get_reward(action, has_error, is_done, is_illegal, is_ambiguous)
        self.backend = tmp_backend

        return self.current_obs, reward, has_error, {}

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
        res = self.current_obs
        return res

    def update_grid(self, real_backend):
        """
        Update this "emulated" environment with the real powergrid.

        Parameters
        ----------
        real_backend: :class:`grid2op.Backend.Backend`
            The real grid of the environment.

        Returns
        -------

        """
        self.backend = real_backend.copy()
        self.is_init = False


class Observation(ABC):
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

    n_lines: :class:`int`
        number of powerline in the powergrid

    n_gen: :class:`int`
        number of generators in the powergrid

    n_load: :class:`int`
        number of loads in the powergrid

    n_sub: ``int``
        Number of susbtations on the powergrid

    subs_info: :class:`numpy.array`, dtype:int
        for each substation, gives the number of elements connected to it

    dim_topo: ``int``
        The number of objects (= powerline extremity, load or generator) on the powergrid.

    prod_p: :class:`numpy.ndarray`, dtype:float
        The active production value of each generator

    prod_q: :class:`numpy.ndarray`, dtype:float
        The reactive production value of each generator

    prod_v: :class:`numpy.ndarray`, dtype:float
        The voltage magnitude of the bus to which each generator is connected

    load_p: :class:`numpy.ndarray`, dtype:float
        The active load value of each consumption

    load_q: :class:`numpy.ndarray`, dtype:float
        The reactive load value of each consumption

    load_v: :class:`numpy.ndarray`, dtype:float
        The voltage magnitude of the bus to which each consumption is connected

    p_or: :class:`numpy.ndarray`, dtype:float
        The active power flow at the origin end of each powerline

    q_or: :class:`numpy.ndarray`, dtype:float
        The reactive power flow at the origin end of each powerline

    v_or: :class:`numpy.ndarray`, dtype:float
        The voltage magnitude at the bus to which the origin end of each powerline is connected

    a_or: :class:`numpy.ndarray`, dtype:float
        The current flow at the origin end of each powerline

    p_ex: :class:`numpy.ndarray`, dtype:float
        The active power flow at the extremity end of each powerline

    q_ex: :class:`numpy.ndarray`, dtype:float
        The reactive power flow at the extremity end of each powerline

    v_ex: :class:`numpy.ndarray`, dtype:float
        The voltage magnitude at the bus to which the extremity end of each powerline is connected

    a_ex: :class:`numpy.ndarray`, dtype:float
        The current flow at the extremity end of each powerline

    rho: :class:`numpy.ndarray`, dtype:float
        The capacity of each powerline. It is defined at the observed current flow divided by the thermal limit of each
        powerline.

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

    """
    def __init__(self,
                 n_gen, n_load, n_lines, subs_info, dim_topo,
                 load_to_subid, gen_to_subid, lines_or_to_subid, lines_ex_to_subid,
                 load_to_sub_pos, gen_to_sub_pos, lines_or_to_sub_pos, lines_ex_to_sub_pos,
                 load_pos_topo_vect, gen_pos_topo_vect, lines_or_pos_topo_vect, lines_ex_pos_topo_vect,
                 obs_env=None, action_helper=None,
                 seed=None):
        self.action_helper = action_helper

        # powergrid static information
        self.n_gen = n_gen
        self.n_load = n_load
        self.n_lines = n_lines
        self.subs_info = subs_info
        self.dim_topo = dim_topo
        self.n_sub = subs_info.shape[0]

        # to which substation is connected each element
        self._load_to_subid = load_to_subid
        self._gen_to_subid = gen_to_subid
        self._lines_or_to_subid = lines_or_to_subid
        self._lines_ex_to_subid = lines_ex_to_subid
        # which index has this element in the substation vector
        self._load_to_sub_pos = load_to_sub_pos
        self._gen_to_sub_pos = gen_to_sub_pos
        self._lines_or_to_sub_pos = lines_or_to_sub_pos
        self._lines_ex_to_sub_pos = lines_ex_to_sub_pos
        # which index has this element in the topology vector
        self._load_pos_topo_vect = load_pos_topo_vect
        self._gen_pos_topo_vect = gen_pos_topo_vect
        self._lines_or_pos_topo_vect = lines_or_pos_topo_vect
        self._lines_ex_pos_topo_vect = lines_ex_pos_topo_vect

        # Game over
        self.game_over = None

        # time stamp information
        self.year = None
        self.month = None
        self.day = None
        self.hour_of_day = None
        self.minute_of_hour = None
        self.day_of_week = None

        # for non deterministic observation that would not use default np.random module
        self.seed = seed

        # handles the forecasts here
        self._forecasted_grid = []
        self._forecasted_inj = []

        self._obs_env = obs_env

        self.timestep_overflow = np.zeros(shape=(self.n_lines,))

        # 0. (line is disconnected) / 1. (line is connected)
        self.line_status = np.ones(shape=self.n_lines, dtype=np.float)

        # topological vector
        self.topo_vect = np.full(shape=self.dim_topo, dtype=np.float, fill_value=1.)

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

        # value to assess if two observations are equal
        self._tol_equal = 5e-1

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
                   "bus": self.topo_vect[self._load_pos_topo_vect[load_id]],
                   "sub_id": self._load_to_subid[load_id]
                   }
        elif gen_id is not None:
            if line_id is not None or substation_id is not None:
                raise Grid2OpException("You can only the inspect the effect of an action on one single element")
            if gen_id >= len(self.prod_p):
                raise Grid2OpException("There are no generator of id \"gen_id={}\" in this grid.".format(gen_id))

            res = {"p": self.prod_p[gen_id],
                   "q": self.prod_q[gen_id],
                   "v": self.prod_v[gen_id],
                   "bus": self.topo_vect[self._gen_pos_topo_vect[gen_id]],
                   "sub_id": self._gen_to_subid[gen_id]
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
                "bus": self.topo_vect[self._lines_or_pos_topo_vect[line_id]],
                "sub_id": self._lines_or_to_subid[line_id]
            }
            # extremity information
            res["extremity"] = {
                "p": self.p_ex[line_id],
                "q": self.q_ex[line_id],
                "v": self.v_ex[line_id],
                "a": self.a_ex[line_id],
                "bus": self.topo_vect[self._lines_ex_pos_topo_vect[line_id]],
                "sub_id": self._lines_ex_to_subid[line_id]
            }

            # maintenance information
            res["maintenance"] = {"next": self.time_next_maintenance[line_id],
                                  "duration_next": self.duration_next_maintenance[line_id]}

            # cooldown
            res["cooldown_time"] = self.time_before_cooldown_line[line_id]

            # indisponibility
            res["indisponibility"] = self.time_before_line_reconnectable[line_id]

        else:
            if substation_id >= len(self.subs_info):
                raise Grid2OpException("There are no substation of id \"substation_id={}\" in this grid.".format(substation_id))

            beg_ = int(np.sum(self.subs_info[:substation_id]))
            end_ = int(beg_ + self.subs_info[substation_id])
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
        # 0. (line is disconnected) / 1. (line is connected)
        self.line_status = np.ones(shape=self.n_lines, dtype=np.float)
        self.topo_vect = np.full(shape=self.dim_topo, dtype=np.float, fill_value=1.)

        # vecorized _grid
        self.timestep_overflow = None

        # generators information
        self.prod_p = None
        self.prod_q = None
        self.prod_v = None
        # loads information
        self.load_p = None
        self.prod_q = None
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

        # matrices
        self.connectivity_matrix_ = None
        self.bus_connectivity_matrix_ = None
        self.vectorized = None

        # calendar data
        self.year = None
        self.month = None
        self.day = None
        self.day_of_week = None
        self.hour_of_day = None
        self.minute_of_hour = None

        # cool down and reconnection time after hard overflow, soft overflow or cascading failure
        self.time_before_cooldown_line = None
        self.time_before_cooldown_sub = None
        self.time_before_line_reconnectable = None
        self.time_next_maintenance = None
        self.duration_next_maintenance = None

        # forecasts
        self._forecasted_inj = []
        self._forecasted_grid = []

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
        _n_gen, _n_load, _n_lines, _subs_info, _dim_topo, all vectors \*_to_subid, and \*_pos_topo_vect are
        identical) then this method will not detect the backend are different, and the action could be declared
        as identical. For now, this is only a theoretical behaviour: if everything is the same, then probably, up to
        the naming convention, then the powergrid are identical too.

        Parameters
        ----------
        other: :class:`Action`
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
        same_grid = same_grid and self.n_lines == other.n_lines
        same_grid = same_grid and np.all(self.subs_info == other.subs_info)
        same_grid = same_grid and self.dim_topo == other.dim_topo
        # to which substation is connected each element
        same_grid = same_grid and np.all(self._load_to_subid == other._load_to_subid)
        same_grid = same_grid and np.all(self._gen_to_subid == other._gen_to_subid)
        same_grid = same_grid and np.all(self._lines_or_to_subid == other._lines_or_to_subid)
        same_grid = same_grid and np.all(self._lines_ex_to_subid == other._lines_ex_to_subid)
        # which index has this element in the substation vector
        same_grid = same_grid and np.all(self._load_to_sub_pos == other._load_to_sub_pos)
        same_grid = same_grid and np.all(self._gen_to_sub_pos == other._gen_to_sub_pos)
        same_grid = same_grid and np.all(self._lines_or_to_sub_pos == other._lines_or_to_sub_pos)
        same_grid = same_grid and np.all(self._lines_ex_to_sub_pos == other._lines_ex_to_sub_pos)
        # which index has this element in the topology vector
        same_grid = same_grid and np.all(self._load_pos_topo_vect == other._load_pos_topo_vect)
        same_grid = same_grid and np.all(self._gen_pos_topo_vect == other._gen_pos_topo_vect)
        same_grid = same_grid and np.all(self._lines_or_pos_topo_vect == other._lines_or_pos_topo_vect)
        same_grid = same_grid and np.all(self._lines_ex_pos_topo_vect == other._lines_ex_pos_topo_vect)

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
                        "duration_next_maintenance"
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

    @abstractmethod
    def to_vect(self):
        """
        Convert this instance of Observation to a numpy array.
        The size of the array is always the same and is determined by the `size` method.

        This method is an "abstract" method, and should be overridden each time the base class :class:`Observation`
        is overidden.

        Returns
        -------
        res: ``numpy.ndarray``
            The respresentation of this action as a numpy array

        """
        pass

    @abstractmethod
    def from_vect(self, vect):
        """
        Convert a observation, represented as a vector, into an observation object.

        This method is an "abstract" method, and should be overridden each time the base class :class:`Observation`
        is overridden.


        Only the size is checked. If it does not match, an :class:`grid2op.Exceptions.AmbiguousAction` is thrown.
        Otherwise the component of the vector are coerced into the proper type silently.

        It may results in an non deterministic behaviour if the input vector is not a real action, or cannot be
        converted to one.

        Parameters
        ----------
        vect: ``numpy.ndarray``
            A vector representing an Action.

        Returns
        -------
        ``None``

        """
        pass

    @abstractmethod
    def size(self):
        """
        When the action is converted to a vector, this method return its size.

        NB that it is a requirement that converting an observation gives a vector of a fixed size throughout a training.

        Returns
        -------
        size: ``int``
            The size of the Action.

        """
        pass

    def connectivity_matrix(self):
        """
        Computes and return the "connectivity matrix" `con_mat`.
        if "_dim_topo = 2 * _n_lines + n_prod + n_conso"
        It is a matrix of size _dim_topo, _dim_topo, with values 0 or 1.
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
        res: ``numpy.ndarray``, shape:_dim_topo,_dim_topo, dtype:float
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

        if self._forecasted_grid[time_step] is None:
            # initialize the "simulation environment" with the proper injections
            self._forecasted_grid[time_step] = self._obs_env.copy()
            timestamp, inj_forecasted = self._forecasted_inj[time_step]
            inj_action = self.action_helper(inj_forecasted)
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

    Attributes
    ----------
    dictionnarized: ``dict``
        The representation of the action in a form of a dictionnary. See the definition of
        :func:`CompleteObservation.to_dict` for a description of this dictionnary.

    """
    def __init__(self, n_gen, n_load, n_lines, subs_info, dim_topo,
                 load_to_subid, gen_to_subid, lines_or_to_subid, lines_ex_to_subid,
                 load_to_sub_pos, gen_to_sub_pos, lines_or_to_sub_pos, lines_ex_to_sub_pos,
                 load_pos_topo_vect, gen_pos_topo_vect, lines_or_pos_topo_vect, lines_ex_pos_topo_vect,
                 obs_env,action_helper,
                 seed=None):

        Observation.__init__(self, n_gen, n_load, n_lines, subs_info, dim_topo,
                 load_to_subid, gen_to_subid, lines_or_to_subid, lines_ex_to_subid,
                 load_to_sub_pos, gen_to_sub_pos, lines_or_to_sub_pos, lines_ex_to_sub_pos,
                 load_pos_topo_vect, gen_pos_topo_vect, lines_or_pos_topo_vect, lines_ex_pos_topo_vect,
                             obs_env=obs_env, action_helper=action_helper,
                             seed=seed)
        self.dictionnarized = None

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
        self.prod_p, self.prod_q, self.prod_v = env.backend.generators_info()
        self.load_p, self.load_q, self.load_v = env.backend.loads_info()
        self.p_or, self.q_or, self.v_or, self.a_or = env.backend.lines_or_info()
        self.p_ex, self.q_ex, self.v_ex, self.a_ex = env.backend.lines_ex_info()

        # handles forecasts here
        self._forecasted_inj = env.chronics_handler.forecasts()
        for i in range(len(self._forecasted_grid)):
            # in the action, i assign the lat topology known, it's a choice here...
            self._forecasted_grid[i]["setbus"] = self.topo_vect

        self._forecasted_grid = [None for _ in self._forecasted_inj]
        self.rho = env.backend.get_relative_flow()

        # TODO
        # cool down and reconnection time after hard overflow, soft overflow or cascading failure
        self.time_before_cooldown_line = env.times_before_line_status_actionable
        self.time_before_cooldown_sub = env.times_before_topology_actionable
        self.time_before_line_reconnectable = env.time_remaining_before_line_reconnection
        self.time_next_maintenance = env.time_next_maintenance
        self.duration_next_maintenance = env.duration_next_maintenance

    def to_vect(self):
        """
        Representation of an :class:`CompleteObservation` into a flat floating point vector.

        Some conversion are done to the internal data representation to floating point. This may cause some data loss
        and / or  corruption (eg. using :func:`Observation.to_vect` and then :func:`Observation.from_vect` does
        not guarantee to be exactly the same object.

        Note that the way and the order of the attributes returned by the method are class dependant. All instance
        of :class:`CompleteObservation` will return the data in the same order. But if another Observation class is
        used, no guarantee is given as to the order in which the data are serialized.

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
            13. :attr:`Observation.p_or` active flow at origin of powerlines [:attr:`Observation.n_lines` elements]
            14. :attr:`Observation.q_or` reactive flow at origin of powerlines [:attr:`Observation.n_lines` elements]
            15. :attr:`Observation.v_or` voltage at origin of powerlines [:attr:`Observation.n_lines` elements]
            16. :attr:`Observation.a_or` current flow at origin of powerlines [:attr:`Observation.n_lines` elements]
            17. :attr:`Observation.p_ex` active flow at extremity of powerlines [:attr:`Observation.n_lines` elements]
            18. :attr:`Observation.q_ex` reactive flow at extremity of powerlines [:attr:`Observation.n_lines` elements]
            19. :attr:`Observation.v_ex` voltage at extremity of powerlines [:attr:`Observation.n_lines` elements]
            20. :attr:`Observation.a_ex` current flow at extremity of powerlines [:attr:`Observation.n_lines` elements]
            21. :attr:`Observation.rho` line capacity used (current flow / thermal limit) [:attr:`Observation.n_lines` elements]
            22. :attr:`Observation.line_status` line status [:attr:`Observation.n_lines` elements]
            23. :attr:`Observation.timestep_overflow` number of timestep since the powerline was on overflow
                (0 if the line is not on overflow)[:attr:`Observation.n_lines` elements]
            24. :attr:`Observation.topo_vect` representation as a vector of the topology [for each element
                it gives its bus]. See :func:`grid2op.Backend.Backend.get_topo_vect` for more information.
            25. :attr:`Observation.time_before_cooldown_line` representation of the cooldown time on the powerlines
                [:attr:`Observation.n_lines` elements]
            26. :attr:`Observation.time_before_cooldown_sub` representation of the cooldown time on the substations
                [:attr:`Observation.n_sub` elements]
            27. :attr:`Observation.time_before_line_reconnectable` number of timestep to wait before a powerline
                can be reconnected (it is disconnected due to maintenance, cascading failure or overflow)
                [:attr:`Observation.n_lines` elements]
            28. :attr:`Observation.time_next_maintenance` number of timestep before the next maintenance (-1 means
                no maintenance are planned, 0 a maintenance is in operation) [:attr:`Observation.n_lines` elements]
            29. :attr:`Observation.duration_next_maintenance` duration of the next maintenance. If a maintenance
                is taking place, this is the number of timestep before it ends. [:attr:`Observation.n_lines` elements]

        Returns
        -------
        res: ``numpy.ndarray``
            The vector representing the topology (see above)
        """
        #TODO fix "bug" when action not initalized, return nan in this case
        if self.vectorized is None:
            self.vectorized = np.concatenate((
                (self.year, ),
                (self.month, ),
                (self.day, ),
                (self.day_of_week, ),
                (self.hour_of_day, ),
                (self.minute_of_hour, ),
                self.prod_p.flatten(),
                self.prod_q.flatten(),
                self.prod_v.flatten(),
                self.load_p.flatten(),
                self.load_q.flatten(),
                self.load_v.flatten(),
                self.p_or.flatten(),
                self.q_or.flatten(),
                self.v_or.flatten(),
                self.a_or.flatten(),
                self.p_ex.flatten(),
                self.q_ex.flatten(),
                self.v_ex.flatten(),
                self.a_ex.flatten(),
                self.rho.flatten(),
                self.line_status.flatten(),
                self.timestep_overflow.flatten(),
                self.topo_vect.flatten(),
                self.time_before_cooldown_line.flatten(),
                self.time_before_cooldown_sub.flatten(),
                self.time_before_line_reconnectable.flatten(),
                self.time_next_maintenance.flatten(),
                self.duration_next_maintenance.flatten()
            ))
        return self.vectorized

    def from_vect(self, vect):
        """

        Parameters
        ----------
        vect

        Returns
        -------

        """
        # TODO explain that some conversion are done silently from float to int or bool!!

        # reset the matrices
        self._reset_matrices()

        if vect.shape[0] != self.size():
            raise IncorrectNumberOfElements("Incorrect number of elements found while load an Observation from a vector. Found {} elements instead of {}".format(vect.shape[1], self.size()))

        if math.isnan(vect[0]):
            self.game_over = True
            return
        self.year = int(vect[0])
        self.month = int(vect[1])
        self.day = int(vect[2])
        self.day_of_week = int(vect[3])
        self.hour_of_day = int(vect[4])
        self.minute_of_hour = int(vect[5])

        prev_ = 6
        next_ = 6 + self.n_gen
        self.prod_p = vect[prev_:next_]; prev_ += self.n_gen; next_ += self.n_gen
        self.prod_q = vect[prev_:next_]; prev_ += self.n_gen; next_ += self.n_gen
        self.prod_v = vect[prev_:next_]; prev_ += self.n_gen; next_ += self.n_load

        self.load_p = vect[prev_:next_]; prev_ += self.n_load; next_ += self.n_load
        self.load_q = vect[prev_:next_]; prev_ += self.n_load; next_ += self.n_load
        self.load_v = vect[prev_:next_]; prev_ += self.n_load; next_ += self.n_lines

        self.p_or = vect[prev_:next_]; prev_ += self.n_lines; next_ += self.n_lines
        self.q_or = vect[prev_:next_]; prev_ += self.n_lines; next_ += self.n_lines
        self.v_or = vect[prev_:next_]; prev_ += self.n_lines; next_ += self.n_lines
        self.a_or = vect[prev_:next_]; prev_ += self.n_lines; next_ += self.n_lines
        self.p_ex = vect[prev_:next_]; prev_ += self.n_lines; next_ += self.n_lines
        self.q_ex = vect[prev_:next_]; prev_ += self.n_lines; next_ += self.n_lines
        self.v_ex = vect[prev_:next_]; prev_ += self.n_lines; next_ += self.n_lines
        self.a_ex = vect[prev_:next_]; prev_ += self.n_lines; next_ += self.n_lines
        self.rho = vect[prev_:next_]; prev_ += self.n_lines; next_ += self.n_lines

        self.line_status = vect[prev_:next_]; prev_ += self.n_lines; next_ += self.n_lines
        self.line_status = self.line_status.astype(np.bool)
        self.timestep_overflow = vect[prev_:next_]; prev_ += self.n_lines; next_ += self.dim_topo
        self.timestep_overflow = self.timestep_overflow.astype(np.int)
        self.topo_vect = vect[prev_:next_]; prev_ += self.dim_topo; next_ += self.n_lines
        self.topo_vect = self.topo_vect.astype(np.int)

        # cooldown
        self.time_before_cooldown_line = vect[prev_:next_]; prev_ += self.n_lines; next_ += self.n_sub
        self.time_before_cooldown_line = self.time_before_cooldown_line.astype(np.int)
        self.time_before_cooldown_sub = vect[prev_:next_]; prev_ += self.n_sub; next_ += self.n_lines
        self.time_before_cooldown_sub = self.time_before_cooldown_sub.astype(np.int)

        # maintenance and hazards
        self.time_before_line_reconnectable = vect[prev_:next_]; prev_ += self.n_lines; next_ += self.n_lines
        self.time_before_line_reconnectable = self.time_before_line_reconnectable.astype(np.int)
        self.time_next_maintenance = vect[prev_:next_]; prev_ += self.n_lines; next_ += self.n_lines
        self.time_next_maintenance = self.time_next_maintenance.astype(np.int)
        self.duration_next_maintenance = vect[prev_:next_]; prev_ += self.n_lines; next_ += self.n_lines
        self.duration_next_maintenance = self.duration_next_maintenance.astype(np.int)

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
            self.dictionnarized["maintenance"]['time_next_maintenance'] = self.duration_next_maintenance
            self.dictionnarized["cooldown"] = {}
            self.dictionnarized["cooldown"]['line'] = self.time_before_cooldown_line
            self.dictionnarized["cooldown"]['substation'] = self.time_before_cooldown_sub
            self.dictionnarized["time_before_line_reconnectable"] = self.time_before_line_reconnectable

        return self.dictionnarized

    def connectivity_matrix(self):
        """
        Computes and return the "connectivity matrix" `con_mat`.
        if "_dim_topo = 2 * _n_lines + n_prod + n_conso"
        It is a matrix of size _dim_topo, _dim_topo, with values 0 or 1.
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
        res: ``numpy.ndarray``, shape:_dim_topo,_dim_topo, dtype:float
            The connectivity matrix, as defined above
        """
        if self.connectivity_matrix_ is None:
            self.connectivity_matrix_ = np.zeros(shape=(self.dim_topo, self.dim_topo),dtype=np.float)
            # fill it by block for the objects
            beg_ = 0
            end_ = 0
            for sub_id, nb_obj in enumerate(self.subs_info):
                nb_obj = int(nb_obj)  # i must be a vanilla python integer, otherwise it's not handled by boost python method to index substations for example.
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
            for q_id in range(self.n_lines):
                self.connectivity_matrix_[self._lines_or_pos_topo_vect[q_id], self._lines_ex_pos_topo_vect[q_id]] = 1
                self.connectivity_matrix_[self._lines_ex_pos_topo_vect[q_id], self._lines_or_pos_topo_vect[q_id]] = 1

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
            nb_bus_per_sub = np.zeros(self.subs_info.shape[0])
            beg_ = 0
            end_ = 0
            for sub_id, nb_obj in enumerate(self.subs_info):
                nb_obj = int(nb_obj)
                end_ += nb_obj

                tmp = len(np.unique(self.topo_vect[beg_:end_]))
                nb_bus_per_sub[sub_id] = tmp
                nb_bus += tmp

                beg_ += nb_obj

            # define the bus_connectivity_matrix
            self.bus_connectivity_matrix_ = np.zeros(shape=(nb_bus, nb_bus), dtype=np.float)
            np.fill_diagonal(self.bus_connectivity_matrix_, 1)

            for q_id in range(self.n_lines):
                bus_or = int(self.topo_vect[self._lines_or_pos_topo_vect[q_id]])
                sub_id_or = int(self._lines_or_to_subid[q_id])

                bus_ex = int(self.topo_vect[self._lines_ex_pos_topo_vect[q_id]])
                sub_id_ex = int(self._lines_ex_to_subid[q_id])

                # try:
                bus_id_or = int(np.sum(nb_bus_per_sub[:sub_id_or])+(bus_or-1))
                bus_id_ex = int(np.sum(nb_bus_per_sub[:sub_id_ex])+(bus_ex-1))

                self.bus_connectivity_matrix_[bus_id_or, bus_id_ex] = 1
                self.bus_connectivity_matrix_[bus_id_ex, bus_id_or] = 1
                # except:
                #     pdb.set_trace()
        return self.bus_connectivity_matrix_

    def size(self):
        """
        Return the size of the flatten observation vector.
        For this CompletObservation:

            - 6 calendar data
            - each generator is caracterized by 3 values: p, q and v
            - each load is caracterized by 3 values: p, q and v
            - each end of a powerline by 4 values: flow p, flow q, v, current flow
            - each line have also a status
            - each line can also be impossible to reconnect
            - the topology vector of dim `dim_topo`

        :return: the size of the flatten observation vector.
        """
        # TODO documentation
        res = 6 + 3*self.n_gen + 3*self.n_load + 2 * 4*self.n_lines + 3*self.n_lines
        res += self.dim_topo + 4*self.n_lines + self.n_sub
        return res


class SerializableObservationSpace:
    """
    This class allows to serialize / de serialize the observation space.

    It should not be used inside an Environment, as some functions of the Observation might not be compatible with
    the serialization, for example the "forecast" method.

    Attributes
    ----------
    n_lines: :class:`int`
        number of powerline in the _grid

    n_gen: :class:`int`
        number of generators in the _grid

    n_load: :class:`int`
        number of loads in the powergrid

    subs_info: :class:`numpy.array`, dtype:int
        for each substation, gives the number of elements connected to it

    load_to_subid: :class:`numpy.array`, dtype:int
        for each load, gives the id the substation to which it is connected

    gen_to_subid: :class:`numpy.array`, dtype:int
        for each generator, gives the id the substation to which it is connected

    lines_or_to_subid: :class:`numpy.array`, dtype:int
        for each lines, gives the id the substation to which its "origin" end is connected

    lines_ex_to_subid: :class:`numpy.array`, dtype:int
        for each lines, gives the id the substation to which its "extremity" end is connected

    load_to_sub_pos: :class:`numpy.array`, dtype:int
        The topology if of the subsation *i* is given by a vector, say *sub_topo_vect* of size
        :attr:`Backend.subs_info`\[i\]. For a given load of id *l*, :attr:`Backend._load_to_sub_pos`\[l\] is the index
        of the load *l* in the vector *sub_topo_vect*. This means that, if
        *sub_topo_vect\[ action._load_to_sub_pos\[l\] \]=2*
        then load of id *l* is connected to the second bus of the substation.

    gen_to_sub_pos: :class:`numpy.array`, dtype:int
        same as :attr:`Backend._load_to_sub_pos` but for generators.

    lines_or_to_sub_pos: :class:`numpy.array`, dtype:int
        same as :attr:`Backend._load_to_sub_pos`  but for "origin" end of powerlines.

    lines_ex_to_sub_pos: :class:`numpy.array`, dtype:int
        same as :attr:`Backend._load_to_sub_pos` but for "extremity" end of powerlines.

    load_pos_topo_vect: :class:`numpy.array`, dtype:int
        It has a similar role as :attr:`Backend._load_to_sub_pos` but it gives the position in the vector representing
        the whole topology. More concretely, if the complete topology of the powergrid is represented here by a vector
        *full_topo_vect* resulting of the concatenation of the topology vector for each substation
        (see :attr:`Backend._load_to_sub_pos`for more information). For a load of id *l* in the powergrid,
        :attr:`Backend._load_pos_topo_vect`\[l\] gives the index, in this *full_topo_vect* that concerns load *l*.
        More formally, if *_topo_vect\[ backend._load_pos_topo_vect\[l\] \]=2* then load of id l is connected to the
        second bus of the substation.

    gen_pos_topo_vect: :class:`numpy.array`, dtype:int
        same as :attr:`Backend._load_pos_topo_vect` but for generators.

    lines_or_pos_topo_vect: :class:`numpy.array`, dtype:int
        same as :attr:`Backend._load_pos_topo_vect` but for "origin" end of powerlines.

    lines_ex_pos_topo_vect: :class:`numpy.array`, dtype:int
        same as :attr:`Backend._load_pos_topo_vect` but for "extremity" end of powerlines.

    observationClass: ``type``
        Class used to build the observations. It defaults to :class:`CompleteObservation`

    """
    def __init__(self,
                 n_gen, n_load, n_lines, subs_info,
                 load_to_subid, gen_to_subid, lines_or_to_subid, lines_ex_to_subid,
                 load_to_sub_pos, gen_to_sub_pos, lines_or_to_sub_pos, lines_ex_to_sub_pos,
                 load_pos_topo_vect, gen_pos_topo_vect, lines_or_pos_topo_vect, lines_ex_pos_topo_vect,
                 observationClass=CompleteObservation):

        # print("ObservationHelper init with reward_helper class: {}".format(self.reward_helper.template_reward))
        self.n_gen = n_gen
        self.n_load = n_load
        self.n_lines = n_lines
        self.subs_info = subs_info
        self.dim_topo = np.sum(subs_info)

        # to which substation is connected each element
        self.load_to_subid = load_to_subid
        self.gen_to_subid = gen_to_subid
        self.lines_or_to_subid = lines_or_to_subid
        self.lines_ex_to_subid = lines_ex_to_subid
        # which index has this element in the substation vector
        self.load_to_sub_pos = load_to_sub_pos
        self.gen_to_sub_pos = gen_to_sub_pos
        self.lines_or_to_sub_pos = lines_or_to_sub_pos
        self.lines_ex_to_sub_pos = lines_ex_to_sub_pos
        # which index has this element in the topology vector
        self.load_pos_topo_vect = load_pos_topo_vect
        self.gen_pos_topo_vect = gen_pos_topo_vect
        self.lines_or_pos_topo_vect = lines_or_pos_topo_vect
        self.lines_ex_pos_topo_vect = lines_ex_pos_topo_vect

        self.observationClass = observationClass

        self.empty_obs = self.observationClass(n_gen=self.n_gen, n_load=self.n_load, n_lines=self.n_lines,
                                               subs_info=self.subs_info, dim_topo=self.dim_topo,
                                               load_to_subid=self.load_to_subid,
                                               gen_to_subid=self.gen_to_subid,
                                               lines_or_to_subid=self.lines_or_to_subid,
                                               lines_ex_to_subid=self.lines_ex_to_subid,
                                               load_to_sub_pos=self.load_to_sub_pos,
                                               gen_to_sub_pos=self.gen_to_sub_pos,
                                               lines_or_to_sub_pos=self.lines_or_to_sub_pos,
                                               lines_ex_to_sub_pos=self.lines_ex_to_sub_pos,
                                               load_pos_topo_vect=self.load_pos_topo_vect,
                                               gen_pos_topo_vect=self.gen_pos_topo_vect,
                                               lines_or_pos_topo_vect=self.lines_or_pos_topo_vect,
                                               lines_ex_pos_topo_vect=self.lines_ex_pos_topo_vect,
                                               obs_env=None,
                                               action_helper=None)

        self.n = self.empty_obs.size()

    @staticmethod
    def from_dict(dict_):
        """
        Allows the de-serialization of an object stored as a dictionnary (for example in the case of json saving).

        Parameters
        ----------
        dict_: ``dict``
            Representation of an Observation Space (aka ObservartionHelper) as a dictionnary.

        Returns
        -------
        res: :class:``SerializableObservationSpace``
            An instance of an observationHelper matching the dictionnary.

        """
        if isinstance(dict_, str):
            path = dict_
            if not os.path.exists(path):
                raise Grid2OpException("Unable to find the file \"{}\" to load the ObservationSpace".format(path))
            with open(path, "r", encoding="utf-8") as f:
                dict_ = json.load(fp=f)

        n_gen = extract_from_dict(dict_, "n_gen", int)
        n_load = extract_from_dict(dict_, "n_load", int)
        n_lines = extract_from_dict(dict_, "n_lines", int)

        subs_info = extract_from_dict(dict_, "subs_info", lambda x: np.array(x).astype(np.int))
        load_to_subid = extract_from_dict(dict_, "load_to_subid", lambda x: np.array(x).astype(np.int))
        gen_to_subid = extract_from_dict(dict_, "gen_to_subid", lambda x: np.array(x).astype(np.int))
        lines_or_to_subid = extract_from_dict(dict_, "lines_or_to_subid", lambda x: np.array(x).astype(np.int))
        lines_ex_to_subid = extract_from_dict(dict_, "lines_ex_to_subid", lambda x: np.array(x).astype(np.int))

        load_to_sub_pos = extract_from_dict(dict_, "load_to_sub_pos", lambda x: np.array(x).astype(np.int))
        gen_to_sub_pos = extract_from_dict(dict_, "gen_to_sub_pos", lambda x: np.array(x).astype(np.int))
        lines_or_to_sub_pos = extract_from_dict(dict_, "lines_or_to_sub_pos", lambda x: np.array(x).astype(np.int))
        lines_ex_to_sub_pos = extract_from_dict(dict_, "lines_ex_to_sub_pos", lambda x: np.array(x).astype(np.int))

        load_pos_topo_vect = extract_from_dict(dict_, "load_pos_topo_vect", lambda x: np.array(x).astype(np.int))
        gen_pos_topo_vect = extract_from_dict(dict_, "gen_pos_topo_vect", lambda x: np.array(x).astype(np.int))
        lines_or_pos_topo_vect = extract_from_dict(dict_, "lines_or_pos_topo_vect", lambda x: np.array(x).astype(np.int))
        lines_ex_pos_topo_vect = extract_from_dict(dict_, "lines_ex_pos_topo_vect", lambda x: np.array(x).astype(np.int))

        observationClass_str = extract_from_dict(dict_, "observationClass", str)
        observationClass_li = observationClass_str.split('.')

        if observationClass_li[-1] in globals():
            observationClass = globals()[observationClass_li[-1]]
        else:
            try:
                observationClass = eval(observationClass_str)
            except NameError:
                msg_err_ = "Impossible to find the module \"{}\" to load back the observation space. Try \"from {} import {}\""
                raise Grid2OpException(msg_err_.format(observationClass_str, ".".join(observationClass_li[:-1]), observationClass_li[-1]))
            except AttributeError:
                try:
                    observationClass = eval(observationClass_li[-1])
                except:
                    if len(observationClass_li) > 1:
                        msg_err_ = "Impossible to find the class named \"{}\" to load back the observation " \
                                   "(module is found but not the class in it) Please import it via \"from {} import {}\"."
                        msg_err_ = msg_err_.format(observationClass_str,
                                                   ".".join(observationClass_li[:-1]),
                                                   observationClass_li[-1])
                    else:
                        msg_err_ = "Impossible to import the class named \"{}\" to load back the observation space (the " \
                                   "module is found but not the class in it)"
                        msg_err_ = msg_err_.format(observationClass_str)
                    raise Grid2OpException(msg_err_)

        res = SerializableObservationSpace(n_gen, n_load, n_lines, subs_info,
                                           load_to_subid, gen_to_subid, lines_or_to_subid, lines_ex_to_subid,
                                           load_to_sub_pos, gen_to_sub_pos, lines_or_to_sub_pos, lines_ex_to_sub_pos,
                                           load_pos_topo_vect, gen_pos_topo_vect, lines_or_pos_topo_vect,
                                           lines_ex_pos_topo_vect,
                                           observationClass=observationClass)
        return res

    def to_dict(self):
        """
        Serialize this object as a dictionnary.

        Returns
        -------
        res: ``dict``
            A dictionnary representing this object content. It can be loaded back with
             :func:`SerializableObservationSpace.from_dict`
        """
        res = {}
        save_to_dict(res, self, "n_gen", int)
        save_to_dict(res, self, "n_load", int)
        save_to_dict(res, self, "n_lines", int)
        save_to_dict(res, self, "subs_info", lambda li: [int(el) for el in li])
        save_to_dict(res, self, "load_to_subid", lambda li: [int(el) for el in li])
        save_to_dict(res, self, "gen_to_subid", lambda li: [int(el) for el in li])
        save_to_dict(res, self, "lines_or_to_subid", lambda li: [int(el) for el in li])
        save_to_dict(res, self, "lines_ex_to_subid", lambda li: [int(el) for el in li])

        save_to_dict(res, self, "load_to_sub_pos", lambda li: [int(el) for el in li])
        save_to_dict(res, self, "gen_to_sub_pos", lambda li: [int(el) for el in li])
        save_to_dict(res, self, "lines_or_to_sub_pos", lambda li: [int(el) for el in li])
        save_to_dict(res, self, "lines_ex_to_sub_pos", lambda li: [int(el) for el in li])

        save_to_dict(res, self, "load_pos_topo_vect", lambda li: [int(el) for el in li])
        save_to_dict(res, self, "gen_pos_topo_vect", lambda li: [int(el) for el in li])
        save_to_dict(res, self, "lines_or_pos_topo_vect", lambda li: [int(el) for el in li])
        save_to_dict(res, self, "lines_ex_pos_topo_vect", lambda li: [int(el) for el in li])

        save_to_dict(res, self, "observationClass", lambda x: re.sub("(<class ')|('>)", "", "{}".format(x)))

        return res

    def from_vect(self, obs):
        """
        Convert a observation, represented as a vector to a valid :class:`Observation` instance

        Parameters
        ----------
        obs: ``numpy.ndarray``
            The observation (represented as a numpy array) to convert to
            :class:`grid2op.Observation.Observation` instance.

        Returns
        -------
        res: :class:`grid2op.Observation.Observation`
            The converted observation (converted from vector to :class:`grid2op.Observation.Observation` )

        """
        res = copy.deepcopy(self.empty_obs)
        res.from_vect(obs)
        return res

    def get_obj_connect_to(self, _sentinel=None, substation_id=None):
        """
        Get all the object connected to a given substation:

        Parameters
        ----------
        _sentinel: ``None``
            Used to prevent positional parameters. Internal, do not use.

        substation_id: ``int``
            ID of the substation we want to inspect

        Returns
        -------
        res: ``dict``
            A dictionnary with keys:

              - "loads_id": a vector giving the id of the loads connected to this substation, empty if none
              - "generators_id": a vector giving the id of the generators connected to this substation, empty if none
              - "lines_or_id": a vector giving the id of the origin end of the powerlines connected to this substation,
                empty if none
              - "lines_ex_id": a vector giving the id of the extermity end of the powerlines connected to this
                substation, empty if none.
              - "nb_elements" : number of elements connected to this substation

        """

        if substation_id is None:
            raise Grid2OpException("You ask the composition of a substation without specifying its id."
                                   "Please provide \"substation_id\"")
        if substation_id >= len(self.subs_info):
            raise Grid2OpException("There are no substation of id \"substation_id={}\" in this grid.".format(substation_id))

        res = {}
        res["loads_id"] = np.where(self.load_to_subid == substation_id)[0]
        res["generators_id"] = np.where(self.gen_to_subid == substation_id)[0]
        res["lines_or_id"] = np.where(self.lines_or_to_subid == substation_id)[0]
        res["lines_ex_id"] = np.where(self.lines_ex_to_subid == substation_id)[0]
        res["nb_elements"] = self.subs_info[substation_id]
        return res

    def get_lines_id(self, _sentinel=None, from_=None, to_=None):
        """
        Returns the list of all the powerlines id in the backend going from "from_" to "to_"

        Parameters
        ----------
        _sentinel: ``None``
            Internal, do not use

        from_: ``int``
            Id the substation to which the origin end of the powerline to look for should be connected to

        to_: ``int``
            Id the substation to which the extremity end of the powerline to look for should be connected to

        Returns
        -------
        res: ``list``
            Id of the powerline looked for.

        Raises
        ------
        :class:`grid2op.Exceptions.BackendError` if no match is found.

        """
        res = []
        if from_ is None:
            raise BackendError("ObservationSpace.get_lines_id: impossible to look for a powerline with no origin substation. Please modify \"from_\" parameter")
        if to_ is None:
            raise BackendError("ObservationSpace.get_lines_id: impossible to look for a powerline with no extremity substation. Please modify \"to_\" parameter")

        for i, (ori, ext) in enumerate(zip(self.lines_or_to_subid, self.lines_ex_to_subid)):
            if ori == from_ and ext == to_:
                res.append(i)

        if res is []:
            raise BackendError("ObservationSpace.get_line_id: impossible to find a powerline with connected at origin at {} and extremity at {}".format(from_, to_))

        return res

    def get_generators_id(self, sub_id):
        """
        Returns the list of all generators id in the backend connected to the substation sub_id

        Parameters
        ----------
        sub_id: ``int``
            The substation to which we look for the generator

        Returns
        -------
        res: ``list``
            Id of the generator id looked for.

        Raises
        ------
        :class:`grid2op.Exceptions.BackendError` if no match is found.


        """
        res = []
        if sub_id is None:
            raise BackendError(
                "ObservationSpace.get_generators_id: impossible to look for a generator not connected to any substation. Please modify \"sub_id\" parameter")

        for i, s_id_gen in enumerate(self.gen_to_subid):
            if s_id_gen == sub_id:
                res.append(i)

        if res is []:
            raise BackendError(
                "ObservationSpace.get_generators_id: impossible to find a generator connected at substation {}".format(sub_id))

        return res

    def get_loads_id(self, sub_id):
        """
        Returns the list of all generators id in the backend connected to the substation sub_id

        Parameters
        ----------
        sub_id: ``int``
            The substation to which we look for the generator

        Returns
        -------
        res: ``list``
            Id of the generator id looked for.

        Raises
        ------
        :class:`grid2op.Exceptions.BackendError` if no match found.

        """
        res = []
        if sub_id is None:
            raise BackendError(
                "ObservationSpace.get_loads_id: impossible to look for a load not connected to any substation. Please modify \"sub_id\" parameter")

        for i, s_id_gen in enumerate(self.load_to_subid):
            if s_id_gen == sub_id:
                res.append(i)

        if res is []:
            raise BackendError(
                "ObservationSpace.get_loads_id: impossible to find a load connected at substation {}".format(sub_id))

        return res


class ObservationHelper(SerializableObservationSpace):
    """
    Helper that provides usefull functions to manipulate :class:`Observation`.

    Observation should only be built using this Helper. It is absolutely not recommended to make an observation
    directly form its constructor.

    Attributes
    ----------
    parameters: :class:`grid2op.Parameters.Parameters`
        Type of Parameters used to compute powerflow for the forecast.

    rewardClass: ``type``
        Class used by the :class:`grid2op.Environment.Environment` to send information about its state to the
        :class:`grid2op.Agent.Agent`

    action_helper_env: :class:`grid2op.Action.HelperAction`
        Action space used to create action during the :func:`Observation.simulate`

    reward_helper: :class:`grid2op.Reward.HelperReward`
        Reward function used by the the :func:`Observation.simulate` function.

    obs_env: :class:`ObsEnv`
        Instance of the environenment used by the Observation Helper to provide forcecast of the grid state.

    empty_obs: :class:`Observation`
        An instance of the observation that is updated and will be sent to he Agent.

    """
    def __init__(self,
                 n_gen, n_load, n_lines, subs_info,
                 load_to_subid, gen_to_subid, lines_or_to_subid, lines_ex_to_subid,
                 load_to_sub_pos, gen_to_sub_pos, lines_or_to_sub_pos, lines_ex_to_sub_pos,
                 load_pos_topo_vect, gen_pos_topo_vect, lines_or_pos_topo_vect, lines_ex_pos_topo_vect,
                 env,
                 rewardClass=None,
                 observationClass=CompleteObservation):
        """
        Env: requires :attr:`grid2op.Environment.parameters` and :attr:`grid2op.Environment.backend` to be valid
        """

        SerializableObservationSpace.__init__(self, n_gen, n_load, n_lines, subs_info,
                 load_to_subid, gen_to_subid, lines_or_to_subid, lines_ex_to_subid,
                 load_to_sub_pos, gen_to_sub_pos, lines_or_to_sub_pos, lines_ex_to_sub_pos,
                 load_pos_topo_vect, gen_pos_topo_vect, lines_or_pos_topo_vect, lines_ex_pos_topo_vect,
                                              observationClass=observationClass)

        # TODO DOCUMENTATION !!!

        # print("ObservationHelper init with rewardClass: {}".format(rewardClass))
        self.parameters = copy.deepcopy(env.parameters)
        # for the observation, I switch betwween the _parameters for the environment and for the simulation
        self.parameters.ENV_DC = self.parameters.FORECAST_DC

        if rewardClass is None:
            self.rewardClass = env.rewardClass
        else:
            self.rewardClass = rewardClass

        # helpers
        self.action_helper_env = env.helper_action_env
        self.reward_helper = RewardHelper(rewardClass=self.rewardClass)

        self.obs_env = ObsEnv(backend_instanciated=env.backend, obsClass=self.observationClass,
                              parameters=env.parameters, reward_helper=self.reward_helper,
                              action_helper=self.action_helper_env)

        self.empty_obs = self.observationClass(n_gen=self.n_gen, n_load=self.n_load, n_lines=self.n_lines,
                                               subs_info=self.subs_info, dim_topo=self.dim_topo,
                                               load_to_subid=self.load_to_subid,
                                               gen_to_subid=self.gen_to_subid,
                                               lines_or_to_subid=self.lines_or_to_subid,
                                               lines_ex_to_subid=self.lines_ex_to_subid,
                                               load_to_sub_pos=self.load_to_sub_pos,
                                               gen_to_sub_pos=self.gen_to_sub_pos,
                                               lines_or_to_sub_pos=self.lines_or_to_sub_pos,
                                               lines_ex_to_sub_pos=self.lines_ex_to_sub_pos,
                                               load_pos_topo_vect=self.load_pos_topo_vect,
                                               gen_pos_topo_vect=self.gen_pos_topo_vect,
                                               lines_or_pos_topo_vect=self.lines_or_pos_topo_vect,
                                               lines_ex_pos_topo_vect=self.lines_ex_pos_topo_vect,
                                               obs_env=self.obs_env,
                                               action_helper=self.action_helper_env)

        self.seed = None

    def __call__(self, env):
        if self.seed is not None:
            # in this case i have specific seed set. So i force the seed to be deterministic.
            # TODO seed handling
            self.seed = np.random.randint(4294967295)
        self.obs_env.update_grid(env.backend)

        res = self.observationClass(n_gen=self.n_gen, n_load=self.n_load, n_lines=self.n_lines,
                                    subs_info=self.subs_info, dim_topo=self.dim_topo,
                                    load_to_subid=self.load_to_subid,
                                    gen_to_subid=self.gen_to_subid,
                                    lines_or_to_subid=self.lines_or_to_subid,
                                    lines_ex_to_subid=self.lines_ex_to_subid,
                                    load_to_sub_pos=self.load_to_sub_pos,
                                    gen_to_sub_pos=self.gen_to_sub_pos,
                                    lines_or_to_sub_pos=self.lines_or_to_sub_pos,
                                    lines_ex_to_sub_pos=self.lines_ex_to_sub_pos,
                                    load_pos_topo_vect=self.load_pos_topo_vect,
                                    gen_pos_topo_vect=self.gen_pos_topo_vect,
                                    lines_or_pos_topo_vect=self.lines_or_pos_topo_vect,
                                    lines_ex_pos_topo_vect=self.lines_ex_pos_topo_vect,
                                    seed=self.seed,
                                    obs_env=self.obs_env,
                                    action_helper=self.action_helper_env)
        res.update(env=env)
        return res

    def seed(self, seed):
        """
        Use to set the seed in case of non determinitics observation.
        :param seed:
        :return:
        """
        self.seed = seed

    def size_obs(self):
        """
        Size if the observation vector would be flatten
        :return:
        """
        return self.n

    def size(self):
        """
        Size if the observation vector would be flatten. That's also the dimension of the observation space.

        Returns
        -------
        size: ``int``
            The size defined above.

        """
        return self.n