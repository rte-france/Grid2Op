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

from abc import ABC, abstractmethod

import pdb

try:
    from .Exceptions import *
    from .Reward import ConstantReward, RewardHelper
except (ModuleNotFoundError, ImportError):
    from Exceptions import *
    from Reward import ConstantReward, RewardHelper

# TODO be able to change reward here
# TODO finish call to simulate, _grid ObsEnv should never see the right _grid
# TODO Finish the action checking too

# TODO code "convert for" to be able to change the backend
# TODO refactor, Observation and Action, they are really close in their actual form, especially the Helpers, if
# TODO that make sense.
# TODO make an action with the difference between the observation that would be an action.
# TODO have a method that could do "forecast" by giving the _injection by the agent, if he wants to make custom forecasts

# TODO finish documentation

# TODO add serialization of ObservationSpace to json or yaml

# TODO catch warnings in simulate


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
        self.timestep_overflow = None  # TODO
        self.action_helper = action_helper
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

        It has the same signature as :func:`grid2op.Environment.step`

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
            contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)

        """
        has_error = True
        tmp_backend = self.backend.copy()
        is_done = False
        reward = None

        self.backend.apply_action(action)
        try:
            disc_lines, infos = self.backend.next_grid_state(env=self, is_dc=self.env_dc)
            self.current_obs = self.obsClass(self.parameters,
                                             self.backend.n_generators, self.backend.n_loads, self.backend.n_lines,
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
                                             action_helper=self.action_helper)
            self.current_obs.update(self)
            has_error = False

        except Grid2OpException as e:
            has_error = True
            reward = self.reward_helper.range()[0]
        # print("reward_helper in ObsEnv: {}".format(self.reward_helper.template_reward))
        if reward is None:
            reward = self._get_reward(action, has_error, is_done)
        self.backend = tmp_backend

        return self.current_obs, reward, has_error, {}

    def _get_reward(self, action, has_error, is_done):
        if has_error:
            res = self.reward_helper.range()[0]
        else:
            res = self.reward_helper(action, self, has_error, is_done)
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

    """
    def __init__(self, parameters,
                 n_gen, n_load, n_lines, subs_info, dim_topo,
                 load_to_subid, gen_to_subid, lines_or_to_subid, lines_ex_to_subid,
                 load_to_sub_pos, gen_to_sub_pos, lines_or_to_sub_pos, lines_ex_to_sub_pos,
                 load_pos_topo_vect, gen_pos_topo_vect, lines_or_pos_topo_vect, lines_ex_pos_topo_vect,
                 obs_env, action_helper,
                 seed=None):

        self._parameters = parameters
        self.action_helper = action_helper

        # time stamp information
        self.year = None
        self.month = None
        self.day = None
        self.hour_of_day = None
        self.minute_of_hour = None
        self.day_of_week = None

        # powergrid information
        self.n_gen = n_gen
        self.n_load = n_load
        self.n_lines = n_lines
        self.subs_info = subs_info
        self.dim_topo = dim_topo

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
            Used to prevent positional _parameters. Internal, do not use.

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

            - if a substation is inspected, it returns the topology to this substation in a dictionary with keys:

                - "topo_vect": the representation of which object is connected where
                - "nb_bus": number of active buses in this substations

        Raises
        ------
        Grid2OpException
            If _sentinel is modified, or if None of the arguments are set or alternatively if 2 or more of the
            _parameters are being set.

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
                "nb_bus": nb_bus
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
                        ]:
            if not self.__compare_stats(other, stat_nm):
                # one of the above stat is not equal in this and in other
                return False

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

        :param backend: an instance of Backend
        :type backend: :class:`grid2op.Backend`
        :param _timestep_overflow:
        :param chronics_handler:
        :return:
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
        res: ``numpyp.ndarray``
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
    def __init__(self, parameters, n_gen, n_load, n_lines, subs_info, dim_topo,
                 load_to_subid, gen_to_subid, lines_or_to_subid, lines_ex_to_subid,
                 load_to_sub_pos, gen_to_sub_pos, lines_or_to_sub_pos, lines_ex_to_sub_pos,
                 load_pos_topo_vect, gen_pos_topo_vect, lines_or_pos_topo_vect, lines_ex_pos_topo_vect,
                 obs_env,action_helper,
                 seed=None):

        Observation.__init__(self, parameters, n_gen, n_load, n_lines, subs_info, dim_topo,
                 load_to_subid, gen_to_subid, lines_or_to_subid, lines_ex_to_subid,
                 load_to_sub_pos, gen_to_sub_pos, lines_or_to_sub_pos, lines_ex_to_sub_pos,
                 load_pos_topo_vect, gen_pos_topo_vect, lines_or_pos_topo_vect, lines_ex_pos_topo_vect,
                             obs_env=obs_env,action_helper=action_helper,
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

    def to_vect(self):

        # TODO explain in which order the object are given.

        #TODO fix bug when action not initalized, return nan in this case
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
                self.topo_vect.flatten()
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

        self.year = int(vect[0])
        self.month = int(vect[1])
        self.day = int(vect[2])
        self.day_of_week = int(vect[3])
        self.hour_of_day = int(vect[4])
        self.minute_of_hour = int(vect[5])

        prev_ = 6
        next_ = 6+self.n_gen
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
        self.timestep_overflow = vect[prev_:next_]; prev_ += self.n_lines; next_ += self.n_lines
        self.timestep_overflow = self.timestep_overflow.astype(np.int)
        self.topo_vect = vect[prev_:]
        self.topo_vect = self.topo_vect.astype(np.int)

    def to_dict(self):
        """

        Returns
        -------

        """
        if self.dictionnarized is None:
            self.dictionnarized = {}
            self.dictionnarized["_timestep_overflow"] = self.timestep_overflow
            self.dictionnarized["line_status"] = self.line_status
            self.dictionnarized["_topo_vect"] = self.topo_vect
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
            - the topology vector of dim `_dim_topo`

        :return: the size of the flatten observation vector.
        """
        return 6 + 3*self.n_gen + 3*self.n_load + 2 * 4*self.n_lines + 3*self.n_lines + self.dim_topo


class ObservationHelper:
    """
    Helper that provides usefull functions to manipulate :class:`Observation`.

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
        Env: requires :attr:`grid2op.Environment._parameters` and :attr:`grid2op.Environment.backend`

        Parameters
        ----------
        _parameters
        n_gen
        n_load
        n_lines
        subs_info
        load_to_subid
        gen_to_subid
        lines_or_to_subid
        lines_ex_to_subid
        load_to_sub_pos
        gen_to_sub_pos
        lines_or_to_sub_pos
        lines_ex_to_sub_pos
        load_pos_topo_vect
        gen_pos_topo_vect
        lines_or_pos_topo_vect
        lines_ex_pos_topo_vect
        env
        rewardClass
        observationClass
        """

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

        # print("ObservationHelper init with reward_helper class: {}".format(self.reward_helper.template_reward))
        self.n_gen = n_gen
        self.n_load = n_load
        self.n_lines = n_lines
        self.subs_info = subs_info
        self.dim_topo = np.sum(subs_info)
        self.observationClass = observationClass

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

        self.obs_env = ObsEnv(backend_instanciated=env.backend, obsClass=self.observationClass,
                              parameters=env.parameters, reward_helper=self.reward_helper,
                              action_helper=self.action_helper_env)

        self.empty_obs = self.observationClass(parameters = self.parameters,
                                               n_gen=self.n_gen, n_load=self.n_load, n_lines=self.n_lines,
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
        self.n = self.empty_obs.size()

    def __call__(self, env):
        if self.seed is not None:
            # in this case i have specific seed set. So i force the seed to be deterministic.
            self.seed = np.random.randint(4294967295)
        self.obs_env.update_grid(env.backend)

        res = self.observationClass(parameters = self.parameters,
                                    n_gen=self.n_gen, n_load=self.n_load, n_lines=self.n_lines,
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
        res: ``dict`` with keys:

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