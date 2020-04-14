# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import numpy as np
import warnings
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import pdb

from grid2op.Exceptions import EnvError

# TODO sous echantillonner ou sur echantilloner les scenario: need to modify everything that affect the number
# TODO of time steps there, for example "Space.gen_min_time_on" or "params.NB_TIMESTEP_POWERFLOW_ALLOWED" for
# TODO example. And more generally, it would be better to have all of this attributes exported / imported in
# TODO time interval, instead of time steps.

# TODO add a class to sample "online" the data.

# TODO add a method 'skip' that can skip a given number of timestep or a until a specific date.


class GridValue(ABC):
    """
    This is the base class for every kind of data for the _grid.

    It allows the :class:`grid2op.Environment` to perform powergrid modification that make the "game" time dependant.

    It is not recommended to directly create :class:`GridValue` object, but to use the
    :attr:`grid2op.Environment.chronics_handler" for such a purpose. This is made in an attempt to make sure the
    :func:`GridValue.initialize` is called. Before this initialization, it is not recommended to use any
    :class:`GridValue` object.

    The method :func:`GridValue.next_chronics` should be used between two epoch of the game. If there are no more
    data to be generated from this object, then :func:`GridValue.load_next` should raise a :class:`StopIteration`
    exception and a call to :func:`GridValue.done` should return True.

    Attributes
    ----------
    time_interval: :class:`.datetime.timedelta`
        Time interval between 2 consecutive timestamps. Default 5 minutes.

    start_datetime:  :class:`datetime.datetime`
        The datetime of the first timestamp of the scenario.

    current_datetime: :class:`datetime.datetime`
        The timestamp of the current scenario.

    n_gen: ``int``
        Number of generators in the powergrid

    n_load: ``int``
        Number of loads in the powergrid

    n_line: ``int``
        Number of powerline in the powergrid

    max_iter: ``int``
        Number maximum of data to generate for one episode.

    curr_iter: ``int``
        Duration of the current episode.

    maintenance_time: ``numpy.ndarray``, dtype:``int``
        Number of time steps the next maintenance will take place with the following convention:

            - -1 no maintenance are planned for the forseeable future
            - 0 a maintenance is taking place
            - 1, 2, 3 ... a maintenance will take place in 1, 2, 3, ... time step

        Some examples are given in :func:`GridValue.maintenance_time_1d`.

    maintenance_duration: ``numpy.ndarray``, dtype:``int``
        Duration of the next maintenance. 0 means no maintenance is happening. If a maintenance is planned for a
        given powerline, this number decreases each time step, up until arriving at 0 when the maintenance is over. Note
        that if a maintenance is planned (see :attr:`GridValue.maintenance_time`) this number indicates how long
        the maintenance will last, and does not suppose anything on the maintenance taking place or not (= there can be
        positive number here without a powerline being removed from the grid for maintenance reason). Some examples are
        given in :func:`GridValue.maintenance_duration_1d`.

    hazard_duration: ``numpy.ndarray``, dtype:``int``
        Duration of the next hzard. 0 means no maintenance is happening. If a hazard is taking place for a
        given powerline, this number decreases each time step, up until arriving at 0 when the maintenance is over. On
        the contrary to :attr:`GridValue.maintenance_duration`, if a component of this vector is higher than 1, it
        means that the powerline is out of service. Some examples are
        given in :func:`GridValue.get_hazard_duration_1d`.


    """
    def __init__(self, time_interval=timedelta(minutes=5), max_iter=-1,
                 start_datetime=datetime(year=2019, month=1, day=1),
                 chunk_size=None):
        self.time_interval = time_interval
        self.current_datetime = start_datetime
        self.start_datetime = start_datetime
        self.n_gen = None
        self.n_load = None
        self.n_line = None
        self.max_iter = max_iter
        self.curr_iter = 0

        self.maintenance_time = None
        self.maintenance_duration = None
        self.hazard_duration = None

    @abstractmethod
    def initialize(self, order_backend_loads, order_backend_prods, order_backend_lines, order_backend_subs,
                   names_chronics_to_backend):
        """
        This function is used to initialize the data generator.
        It can be use to load scenarios, or to initialize noise if scenarios are generated on the fly. It must also
        initialize :attr:`GridValue.maintenance_time`, :attr:`GridValue.maintenance_duration` and
        :attr:`GridValue.hazard_duration`.

        This function should also increment :attr:`GridValue.curr_iter` of 1 each time it is called.

        The :class:`GridValue` is what makes the connection between the data (generally in a shape of files on the
        hard drive) and the power grid. One of the main advantage of the Grid2Op package is its ability to change
        the tool that computes the load flows. Generally, such :class:`grid2op.Backend` expects data in a specific
        format that is given by the way their internal powergrid is represented, and in particular, the "same"
        objects can have different name and different position. To ensure that the same chronics would
        produce the same results on every backend (**ie** regardless of the order of which the Backend is expecting
        the data, the outcome of the powerflow is the same) we encourage the user to provide a file that maps the name
        of the object in the chronics to the name of the same object in the backend.

        This is done with the "names_chronics_to_backend" dictionnary that has the following keys:

          - "loads"
          - "prods"
          - "lines"

        The value associated to each of these keys is in turn a mapping dictionnary from the chronics to the backend.
        This means that each *keys* of these subdictionnary is a name of one column in the files, and each values
        is the corresponding name of this same object in the dictionnary. An example is provided bellow.

        Parameters
        ----------
        order_backend_loads: ``numpy.ndarray``, dtype:str
            Ordered name, in the Backend, of the loads. It is required that a :class:`grid2op.Backend` object always
            output the informations in the same order. This array gives the name of the loads following this order.
            See the documentation of :mod:`grid2op.Backend` for more information about this.

        order_backend_prods: ``numpy.ndarray``, dtype:str
            Same as order_backend_loads, but for generators.

        order_backend_lines: ``numpy.ndarray``, dtype:str
            Same as order_backend_loads, but for powerline.

        order_backend_subs: ``numpy.ndarray``, dtype:str
            Same as order_backend_loads, but for powerline.

        names_chronics_to_backend: ``dict``
            See in the description of the method for more information about its format.

        Returns
        -------
        ``None``

        Examples
        --------
        For example, suppose we have a :class:`grid2op.Backend` with:

          - substations ids strart from 0 to N-1 (N being the number of substations in the powergrid)
          - loads named "load_i" with "i" the subtations to which it is connected
          - generators units named "gen_i" (i still the substation id to which it is connected)
          - powerlnes are named "i_j" if it connected substations i to substation j

        And on the other side, we have some files with the following conventions:

          - substations are numbered from 1 to N
          - loads are named "i_C" with i being the substation to which it is connected
          - generators are named "i_G" with is being the id of the substations to which it is connected
          - powerlines are namesd "i_j_k" where i is the origin substation, j the extremity substations and "k"
            is a unique identifier of this powerline in the powergrid.

        In this case, instead of renaming the powergrid (in the backend) of the data files, it is advised to build the
        following elements and initialize the object gridval of type :class:`GridValue` with:

        .. code-block:: python

            gridval = GridValue()  # Note: this code won't execute because "GridValue" is an abstract class
            order_backend_loads = ['load_1', 'load_2', 'load_13', 'load_3', 'load_4', 'load_5', 'load_8', 'load_9',
                                     'load_10', 'load_11', 'load_12']
            order_backend_prods = ['gen_1', 'gen_2', 'gen_5', 'gen_7', 'gen_0']
            order_backend_lines = ['0_1', '0_4', '8_9', '8_13', '9_10', '11_12', '12_13', '1_2', '1_3', '1_4', '2_3',
                                       '3_4', '5_10', '5_11', '5_12', '3_6', '3_8', '4_5', '6_7', '6_8']
            order_backend_subs = ['sub_0', 'sub_1', 'sub_10', 'sub_11', 'sub_12', 'sub_13', 'sub_2', 'sub_3', 'sub_4',
                                      'sub_5', 'sub_6', 'sub_7', 'sub_8', 'sub_9']
            names_chronics_to_backend = {"loads": {"2_C": 'load_1', "3_C": 'load_2',
                                                       "14": 'load_13', "4_C": 'load_3', "5_C": 'load_4',
                                                       "6_C": 'load_5', "9_C": 'load_8', "10_C": 'load_9',
                                                       "11_C": 'load_10', "12_C": 'load_11',
                                                       "13_C": 'load_12'},
                                             "lines": {'1_2_1': '0_1', '1_5_2': '0_4', '9_10_16': '8_9', '9_14_17': '8_13',
                                                      '10_11_18': '9_10', '12_13_19': '11_12', '13_14_20': '12_13',
                                                       '2_3_3': '1_2', '2_4_4': '1_3', '2_5_5': '1_4', '3_4_6': '2_3',
                                                       '4_5_7': '3_4', '6_11_11': '5_10', '6_12_12': '5_11',
                                                       '6_13_13': '5_12', '4_7_8': '3_6', '4_9_9': '3_8', '5_6_10': '4_5',
                                                      '7_8_14': '6_7', '7_9_15': '6_8'},
                                             "prods": {"1_G": 'gen_0', "3_G": "gen_2", "6_G": "gen_5",
                                                       "2_G": "gen_1", "8_G": "gen_7"},
                                            }
            gridval.initialize(order_backend_loads, order_backend_prods, order_backend_lines, names_chronics_to_backend)

        """
        self.curr_iter += 1
        self.current_datetime += self.time_interval

    @staticmethod
    def get_maintenance_time_1d(maintenance):
        """
        This function allows to transform a 1d numpy aarray maintenance, where is specify:

            - 0 there is no maintenance at this time step
            - 1 there is a maintenance at this time step

        Into the representation in terms of "next maintenance time" as specified in
        :attr:`GridValue.maintenance_time` which is:

            - `-1` no foreseeable maintenance operation will be performed
            - `0` a maintenance operation is being performed
            - `1`, `2` etc. is the number of time step the next maintenance will be performed.

        Parameters
        ----------
        maintenance: ``numpy.ndarray``
            1 dimensional array representing the time series of the maintenance (0 there is no maintenance, 1 there
            is a maintenance at this time step)

        Returns
        -------
        maintenance_duration: ``numpy.ndarray``
            Array representing the time series of the duration of the next maintenance forseeable.

        Examples
        --------

        If no maintenance are planned:

        .. code-block:: python

            maintenance_time = GridValue.get_maintenance_time_1d(np.array([0 for _ in range(10)]))
            assert np.all(maintenance_time == np.array([-1  for _ in range(10)]))


        If a maintenance planned of 3 time steps starting at timestep 6 (index 5 - index starts at 0)

        .. code-block:: python

            maintenance = np.array([0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0])
            maintenance_time = GridValue.get_maintenance_time_1d(maintenance)
            assert np.all(maintenance_time == np.array([5,4,3,2,1,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1]))

        If a maintenance planned of 3 time steps starting at timestep 6
        (index 5 - index starts at 0), and a second one for 2 time steps at time step 13

        .. code-block:: python

            maintenance = np.array([0,0,0,0,0,1,1,1,0,0,0,0,1,1,0,0,0])
            maintenance_time = GridValue.get_maintenance_time_1d(maintenance)
            assert np.all(maintenance_time == np.array([5,4,3,2,1,0,0,0,4,3,2,1,0,0,-1,-1,-1]))

        """

        res = np.full(maintenance.shape, fill_value=np.NaN, dtype=np.int)
        maintenance = np.concatenate((maintenance, (0, 0)))
        a = np.diff(maintenance)
        # +1 is because numpy does the diff `t+1` - `t` so to get index of the initial array
        # I need to "+1"
        start = np.where(a == 1)[0] + 1  # start of maintenance
        end = np.where(a == -1)[0] + 1  # end of maintenance
        prev_ = 0
        # it's efficient here as i do a loop only on the number of time there is a maintenance
        # and maintenance are quite rare
        for beg_, end_ in zip(start, end):
            res[prev_:beg_] = list(range(beg_ - prev_, 0, -1))
            res[beg_:end_] = 0
            prev_ = end_

        # no maintenance are planned in the forseeable future
        res[prev_:] = -1
        return res

    @staticmethod
    def get_maintenance_duration_1d(maintenance):
        """
        This function allows to transform a 1d numpy aarray maintenance (or hazards), where is specify:

            - 0 there is no maintenance at this time step
            - 1 there is a maintenance at this time step

        Into the representation in terms of "next maintenance duration" as specified in
        :attr:`GridValue.maintenance_duration` which is:

            - `0` no forseeable maintenance operation will be performed
            - `1`, `2` etc. is the number of time step the next maintenance will last (it can be positive even in the
                case that no maintenance is currently being performed.

        Parameters
        ----------
        maintenance: ``numpy.ndarray``
            1 dimensional array representing the time series of the maintenance (0 there is no maintenance, 1 there
            is a maintenance at this time step)

        Returns
        -------
        maintenance_duration: ``numpy.ndarray``
            Array representing the time series of the duration of the next maintenance forseeable.

        Examples
        --------

        If no maintenance are planned:

        .. code-block:: python

            maintenance = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            maintenance_duration = GridValue.get_maintenance_duration_1d(maintenance)
            assert np.all(maintenance_duration == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))

        If a maintenance planned of 3 time steps starting at timestep 6 (index 5 - index starts at 0)

        .. code-block:: python

            maintenance = np.array([0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0])
            maintenance_duration = GridValue.get_maintenance_duration_1d(maintenance)
            assert np.all(maintenance_duration == np.array([3,3,3,3,3,3,2,1,0,0,0,0,0,0,0,0]))

        If a maintenance planned of 3 time steps starting at timestep 6
        (index 5 - index starts at 0), and a second one for 2 time steps at time step 13

        .. code-block:: python

            maintenance = np.array([0,0,0,0,0,1,1,1,0,0,0,0,1,1,0,0,0])
            maintenance_duration = GridValue.get_maintenance_duration_1d(maintenance)
            assert np.all(maintenance_duration == np.array([3,3,3,3,3,3,2,1,2,2,2,2,2,1,0,0,0]))

        """

        res = np.full(maintenance.shape, fill_value=np.NaN, dtype=np.int)
        maintenance = np.concatenate((maintenance, (0,0)))
        a = np.diff(maintenance)
        # +1 is because numpy does the diff `t+1` - `t` so to get index of the initial array
        # I need to "+1"
        start = np.where(a == 1)[0] + 1  # start of maintenance
        end = np.where(a == -1)[0] + 1  # end of maintenance
        prev_ = 0
        # it's efficient here as i do a loop only on the number of time there is a maintenance
        # and maintenance are quite rare
        for beg_, end_ in zip(start, end):
            res[prev_:beg_] = end_ - beg_
            res[beg_:end_] = list(range(end_ - beg_, 0, -1))
            prev_ = end_

        # no maintenance are planned in the foreseeable future
        res[prev_:] = 0
        return res

    @staticmethod
    def get_hazard_duration_1d(hazard):
        """
        This function allows to transform a 1d numpy aarray maintenance (or hazards), where is specify:

            - 0 there is no maintenance at this time step
            - 1 there is a maintenance at this time step

        Into the representation in terms of "hzard duration" as specified in
        :attr:`GridValue.maintenance_duration` which is:

            - `0` no forseeable hazard operation will be performed
            - `1`, `2` etc. is the number of time step the next hzard will last (it is positive only when a hazard
                affect a given powerline)

        Compared to :func:`GridValue.get_maintenance_duration_1d` we only know when the hazard occurs how long it
        will last.

        Parameters
        ----------
        maintenance: ``numpy.ndarray``
            1 dimensional array representing the time series of the maintenance (0 there is no maintenance, 1 there
            is a maintenance at this time step)

        Returns
        -------
        maintenance_duration: ``numpy.ndarray``
            Array representing the time series of the duration of the next maintenance forseeable.

        Examples
        --------

        If no maintenance are planned:

        .. code-block:: python

            hazard = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            hazard_duration = GridValue.get_hazard_duration_1d(hazard)
            assert np.all(hazard_duration == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))

        If a maintenance planned of 3 time steps starting at timestep 6 (index 5 - index starts at 0)

        .. code-block:: python

            hazard = np.array([0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0])
            hazard_duration = GridValue.get_hazard_duration_1d(hazard)
            assert np.all(hazard_duration == np.array([0,0,0,0,0,3,2,1,0,0,0,0,0,0,0,0]))

        If a maintenance planned of 3 time steps starting at timestep 6
        (index 5 - index starts at 0), and a second one for 2 time steps at time step 13

        .. code-block:: python

            hazard = np.array([0,0,0,0,0,1,1,1,0,0,0,0,1,1,0,0,0])
            hazard_duration = GridValue.get_hazard_duration_1d(hazard)
            assert np.all(hazard_duration == np.array([0,0,0,0,0,3,2,1,0,0,0,0,2,1,0,0,0]))

        """

        res = np.full(hazard.shape, fill_value=np.NaN, dtype=np.int)
        hazard = np.concatenate((hazard, (0, 0)))
        a = np.diff(hazard)
        # +1 is because numpy does the diff `t+1` - `t` so to get index of the initial array
        # I need to "+1"
        start = np.where(a == 1)[0] + 1  # start of maintenance
        end = np.where(a == -1)[0] + 1  # end of maintenance
        prev_ = 0
        # it's efficient here as i do a loop only on the number of time there is a maintenance
        # and maintenance are quite rare
        for beg_, end_ in zip(start, end):
            res[prev_:beg_] = 0
            res[(beg_):(end_)] = list(range(end_ - beg_, 0, -1))
            prev_ = end_

        # no maintenance are planned in the forseeable future
        res[prev_:] = 0
        return res

    @abstractmethod
    def load_next(self):
        """
        Generate the next values, either by reading from a file, or by generating on the fly and return a dictionnary
        compatible with the :class:`grid2op.BaseAction` class allowed for the :class:`Environment`.

        More information about this dictionnary can be found at :func:`grid2op.BaseAction.update`.

        As a (quick) reminder: this dictionnary has for keys:

          - "injection" (optional): a dictionnary with keys (optional) "load_p", "load_q", "prod_p", "prod_v"
          - "hazards" (optional) : the outage suffered from the _grid
          - "maintenance" (optional) : the maintenance operations planned on the grid for the current time step.

        Returns
        -------
        timestamp: ``datetime.datetime``
            The current timestamp for which the modifications have been generated.

        dict_: ``dict``
            A dictionnary understandable by the ::func:`grid2op.BaseAction.update` method. **NB** this function should
            return the dictionnary that will be converted, is should not, in any case, return an action.

        maintenance_time: ``numpy.ndarray``, dtype:``int``
            Information about the next planned maintenance. See :attr:`GridValue.maintenance_time` for more information.

        maintenance_duration: ``numpy.ndarray``, dtype:``int``
            Information about the duration of next planned maintenance. See :attr:`GridValue.maintenance_duration`
            for more information.

        hazard_duration: ``numpy.ndarray``, dtype:``int``
            Information about the current hazard. See :attr:`GridValue.hazard_duration`
            for more information.

        Raises
        ------
        StopIteration
            if the chronics is over
        """
        self.current_datetime += self.time_interval
        return self.current_datetime, {}, self.maintenance_time, self.maintenance_duration, self.hazard_duration

    @abstractmethod
    def check_validity(self, backend):
        """
        To make sure that the data returned by this class are of the proper dimension, a call to this method
        must be performed before actually using the data generated by this class.

        In the grid2op framework, this is ensure because the :class:`grid2op.Environment` calls this method
        in its initialization.

        Parameters
        ----------
        backend: :class:`grid2op.Backend`
            The backend used by the :class;`Environment`.

        Returns
        -------

        """
        raise EnvError("check_validity not implemented")

    def done(self):
        """
        Whether the episode is over or not.

        Returns
        -------
        done: ``bool``
            ``True`` means the episode has arrived to the end (no more data to generate) ``False`` means that the episode
            is not over yet.

        """
        if self.max_iter >= 0:
            return self.curr_iter >= self.max_iter
        else:
            return False

    def forecasts(self):
        """
        This method is used to generate the forecasts that are made available to the :class:`grid2op.BaseAgent`.
        This forecasts are behaving the same way than a list of tuple as the one returned by
        :func:`GridValue.load_next` method.

        The way they are generated depends on the GridValue class. If not forecasts are made available, then
        the empty list should be returned.

        Returns
        -------
        res: ``list``
            Each element of this list having the same type as what is returned by :func:`GridValue.load_next`.
        """
        return []

    @abstractmethod
    def next_chronics(self):
        """
        Load the next batch of chronics. This function is called after an episode has finished by the
        :class:`grid2op.Environment` or the :class:`grid2op.Runner`.

        A call to this function should also reset :attr:`GridValue.curr_iter` to 0.
        Returns
        -------
        ``None``
        """
        pass

    def tell_id(self, id_num):
        """
        Tell the backend to use one folder for the chronics in particular. This method is mainly use when the GridValue
        object can deal with many folder. In this case, this method is used by the :class:`grid2op.Runner` to indicate
        which chronics to load for the current simulated episode.

        This is important to ensure reproducibility, especially in parrallel computation settings.

        This should also be used in case of generation "on the fly" of the chronics to ensure the same property.

        By default it does nothing.

        Returns
        -------
        ``None``
        """
        warnings.warn("Class {} doesn't handle different input folder. \"tell_id\" method has no impact."
                      "".format(type(self).__name__))

    def get_id(self) -> str:
        """
        Utility to get the current name of the path of the data are looked at, if data are files.

        This could also be used to return a unique identifier to the generated chronics even in the case where they are
        generated on the fly, for example by return a hash of the seed.

        Returns
        -------
        res: ``str``
            A unique identifier of the chronics generated for this episode. For example, if the chronics comes from a
            specific folder, this could be the path to this folder.

        """
        warnings.warn("Class {} doesn't handle different input folder. \"get_id\" method will return \"\"."
                      "".format(type(self).__name__))
        return ""

    def max_timestep(self):
        """
        This method returned the maximum timestep that the current episode can last.
        Note that if the :class:`grid2op.BaseAgent` performs a bad action that leads to a game over, then the episode
        can lasts less.

        Returns
        -------
        res: int
            -1 if possibly infinite length of a positive integer representing the maximum duration of this episode

        """
        # warnings.warn("Class {} has possibly and infinite duration.".format(type(self).__name__))
        return self.max_iter

    def shuffle(self, shuffler):
        """
        This method can be overiden if the data that are represented by this object need to be shuffle.

        By default it does nothing.
        Parameters
        ----------
        shuffler: ``object``
            Any function that can be used to shuffle the data.

        Returns
        -------

        """
        pass

    def set_chunk_size(self, new_chunk_size):
        """
        TODO
        Parameters
        ----------
        new_chunk_size

        Returns
        -------

        """
        pass
