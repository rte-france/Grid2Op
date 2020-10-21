# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import os
import warnings
import json

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Exceptions import EnvError, DivergingPowerFlow, IncorrectNumberOfElements, IncorrectNumberOfLoads
from grid2op.Exceptions import IncorrectNumberOfGenerators, BackendError, IncorrectNumberOfLines
from grid2op.Space import GridObjects
from grid2op.Action import CompleteAction

# TODO compute a method to update a backend state from an observation.


class Backend(GridObjects, ABC):
    """
    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Unless if you want to code yourself a backend this is not recommend to alter it
        or use it directly in any way.

        If you want to code a backend, an example is given in :class:`PandaPowerBackend` (
        or in the repository lightsim2grid on github)

    This documentation is present mainly for exhaustivity. It is not recommended to manipulate a Backend
    directly. Prefer using an :class:`grid2op.Environment.Environment`

    This is a base class for each :class:`Backend` object.
    It allows to run power flow smoothly, and abstract the method of computing cascading failures.
    This class allow the user or the agent to interact with an power flow calculator, while relying on dedicated
    methods to change the power grid behaviour.

    It is NOT recommended to use this class outside the Environment.

    An example of a valid backend is provided in the :class:`PandapowerBackend`.

    All the abstract methods (that need to be implemented for a backend to work properly) are:

    - :func:`Backend.load_grid`
    - :func:`Backend.close`
    - :func:`Backend.apply_action`
    - :func:`Backend.runpf`
    - :func:`Backend.copy`
    - :func:`Backend.get_line_status`
    - :func:`Backend.get_line_flow`
    - :func:`Backend.get_topo_vect`
    - :func:`Backend._disconnect_line`
    - :func:`Backend.generators_info`
    - :func:`Backend.loads_info`
    - :func:`Backend.lines_or_info`
    - :func:`Backend.lines_ex_info`

    And, if the flag :attr:Backend.shunts_data_available` is set to ``True`` the method :func:`Backend.shunt_info`
    should also be implemented.

    In order to be valid and carry out some computations, you should call :func:`Backend.load_grid` and later
    :func:`grid2op.Spaces.GridObjects.assert_grid_correct`. It is also more than recommended to call
    :func:`Backend.assert_grid_correct_after_powerflow` after the first powerflow. This is all carried ou in the
    environment properly.

    Attributes
    ----------
    detailed_infos_for_cascading_failures: :class:`bool`
        Whether to be verbose when computing a cascading failure.

    thermal_limit_a: :class:`numpy.array`, dtype:float
        Thermal limit of the powerline in amps for each powerline. Thie thermal limit is relevant on only one
        side of the powerline: the same side returned by :func:`Backend.get_line_overflow`

    """
    env_name = "unknown"

    def __init__(self, detailed_infos_for_cascading_failures=False):
        """
        Initialize an instance of Backend. This does nothing per se. Only the call to :func:`Backend.load_grid`
        should guarantee the backend is properly configured.

        :param detailed_infos_for_cascading_failures: Whether to be detailed (but slow) when computing cascading failures
        :type detailed_infos_for_cascading_failures: :class:`bool`

        """
        GridObjects.__init__(self)

        # the following parameter is used to control the amount of verbosity when computing a cascading failure
        # if it's set to true, it returns all intermediate _grid states. This can slow down the computation!
        self.detailed_infos_for_cascading_failures = detailed_infos_for_cascading_failures

        # the power _grid manipulated. One powergrid per backend.
        self._grid = None

        # thermal limit setting, in ampere, at the same "side" of the powerline than self.get_line_overflow
        self.thermal_limit_a = None

    def assert_grid_correct_after_powerflow(self):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            This is done as it should be by the Environment

        This method is called by the environment. It ensure that the backend remains consistent even after a powerflow
        has be run with :func:`Backend.runpf` method.

        :return: ``None``
        :raise: :class:`grid2op.Exceptions.EnvError` and possibly all of its derived class.
        """
        # test the results gives the proper size
        self.__class__ = self.init_grid(self)
        tmp = self.get_line_status()
        if tmp.shape[0] != self.n_line:
            raise IncorrectNumberOfLines("returned by \"backend.get_line_status()\"")
        if np.any(~np.isfinite(tmp)):
            raise EnvironmentError("Power cannot be computed on the first time step, please check your data.")
        tmp = self.get_line_flow()
        if tmp.shape[0] != self.n_line:
            raise IncorrectNumberOfLines("returned by \"backend.get_line_flow()\"")
        if np.any(~np.isfinite(tmp)):
            raise EnvironmentError("Power cannot be computed on the first time step, please check your data.")
        tmp = self.get_thermal_limit()
        if tmp.shape[0] != self.n_line:
            raise IncorrectNumberOfLines("returned by \"backend.get_thermal_limit()\"")
        if np.any(~np.isfinite(tmp)):
            raise EnvironmentError("Power cannot be computed on the first time step, please check your data.")
        tmp = self.get_line_overflow()
        if tmp.shape[0] != self.n_line:
            raise IncorrectNumberOfLines("returned by \"backend.get_line_overflow()\"")
        if np.any(~np.isfinite(tmp)):
            raise EnvironmentError("Power cannot be computed on the first time step, please check your data.")

        tmp = self.generators_info()
        if len(tmp) != 3:
            raise EnvError("\"generators_info()\" should return a tuple with 3 elements: p, q and v")
        for el in tmp:
            if el.shape[0] != self.n_gen:
                raise IncorrectNumberOfGenerators("returned by \"backend.generators_info()\"")
        tmp = self.loads_info()
        if len(tmp) != 3:
            raise EnvError("\"loads_info()\" should return a tuple with 3 elements: p, q and v")
        for el in tmp:
            if el.shape[0] != self.n_load:
                raise IncorrectNumberOfLoads("returned by \"backend.loads_info()\"")
        tmp = self.lines_or_info()
        if len(tmp) != 4:
            raise EnvError("\"lines_or_info()\" should return a tuple with 4 elements: p, q, v and a")
        for el in tmp:
            if el.shape[0] != self.n_line:
                raise IncorrectNumberOfLines("returned by \"backend.lines_or_info()\"")
        tmp = self.lines_ex_info()
        if len(tmp) != 4:
            raise EnvError("\"lines_ex_info()\" should return a tuple with 4 elements: p, q, v and a")
        for el in tmp:
            if el.shape[0] != self.n_line:
                raise IncorrectNumberOfLines("returned by \"backend.lines_ex_info()\"")

        tmp = self.get_topo_vect()
        if tmp.shape[0] != np.sum(self.sub_info):
            raise IncorrectNumberOfElements("returned by \"backend.get_topo_vect()\"")

        if np.any(~np.isfinite(tmp)):
            raise EnvError("Some components of \"backend.get_topo_vect()\" are not finite. This should be integer.")

    def reset(self, grid_path, grid_filename=None):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            This is done in the `env.reset()` method and should be performed otherwise.

        Reload the power grid.
        For backwards compatibility this method calls `Backend.load_grid`.
        But it is encouraged to overload it in the subclasses.
        """
        self.load_grid(grid_path, filename=grid_filename)

    @abstractmethod
    def load_grid(self, path, filename=None):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            This is called once at the loading of the powergrid.

        Load the powergrid.
        It should first define self._grid.

        And then fill all the helpers used by the backend eg. all the attributes of :class:`Space.GridObjects`.

        After a the call to :func:`Backend.load_grid` has been performed, the backend should be in such a state where
        the :class:`grid2op.Space.GridObjects` is properly set up. See the description of
        :class:`grid2op.Space.GridObjects` to know which attributes should be set here and which should not.

        :param path: the path to find the powergrid
        :type path: :class:`string`

        :param filename: the filename of the powergrid
        :type filename: :class:`string`, optional

        :return: ``None``
        """
        pass

    @abstractmethod
    def close(self):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            This is called by `env.close()` do not attempt to use it otherwise.

        This function is called when the environment is over.
        After calling this function, the backend might not behave properly, and in any case should not be used before
        another call to :func:`Backend.load_grid` is performed

        Returns
        -------
        ``None``
        """

    @abstractmethod
    def apply_action(self, action):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            Don't attempt to apply an action directly to a backend. This function will modify
            the powergrid state given the action in input.

            This is one of the core function if you want to code a backend.

        Modify the powergrid with the action given by an agent or by the envir.
        For the L2RPN project, this action is mainly for topology if it has been sent by the agent.
        Or it can also affect production and loads, if the action is made by the environment.

        The help of :func:`grid2op.BaseAction.BaseAction.__call__` or the code in BaseActiontion.py file give more information about
        the implementation of this method.

        :param action: the action to be implemented on the powergrid.
        :type action: :class:`grid2op.Action._BackendAction._BackendAction`

        :return: ``None``
        """
        pass

    @abstractmethod
    def runpf(self, is_dc=False):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            This is called by :func:`Backend.next_grid_state` (that computes some kind of
            cascading failures).

            This is one of the core function if you want to code a backend. It will carry out
            a powerflow.


        Run a power flow on the underlying _grid.
        Powerflow can be AC (is_dc = False) or DC (is_dc = True)

        :param is_dc: is the powerflow run in DC or in AC
        :type is_dc: :class:`bool`

        :return: True if it has converged, or false otherwise. In case of non convergence, no flows can be inspected on
          the _grid.
        :rtype: :class:`bool`
        """
        pass

    @abstractmethod
    def copy(self):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Performs a deep copy of the backend.

        :return: An instance of Backend equal to :attr:`.self`, but deep copied.
        :rtype: :class:`Backend`
        """
        pass

    def save_file(self, full_path):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Save the current power _grid in a human readable format supported by the backend.
        The format is not modified by this wrapper.

        This function is not mandatory, and if implemented, it is used only as a debugging purpose.

        :param full_path: the full path (path + file name + extension) where *self._grid* is stored.
        :type full_path: :class:`string`

        :return: ``None``
        """
        raise RuntimeError("Class {} does not allow for saving file.".format(self))

    @abstractmethod
    def get_line_status(self):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            Prefer using :attr:`grid2op.Observation.BaseObservation.line_status` instead

        Return the status of each lines (connected : True / disconnected: False )

        It is assume that the order of the powerline is fixed: if the status of powerline "l1" is put at the 42nd element
        of the return vector, then it should always be set at the 42nd element.

        It is also assumed that all the other methods of the backend that allows to retrieve informations on the powerlines
        also respect the same convention, and consistent with one another.
        For example, if powerline "l1" is the 42nd second of the vector returned by :func:`Backend.get_line_status` then information
        about it's flow will be at position *42* of the vector returned by :func:`Backend.get_line_flow` for example.

        :return: an array with the line status of each powerline
        :rtype: np.array, dtype:bool
        """
        pass

    @abstractmethod
    def get_line_flow(self):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            Prefer using :attr:`grid2op.Observation.BaseObservation.a_or` or
            :attr:`grid2op.Observation.BaseObservation.a_ex` for example

        Return the current flow in each lines of the powergrid. Only one value per powerline is returned.

        If the AC mod is used, this shall return the current flow on the end of the powerline where there is a protection.
        For example, if there is a protection on "origin end" of powerline "l2" then this method shall return the current
        flow of at the "origin end" of powerline l2.

        Note that in general, there is no loss of generality in supposing all protections are set on the "origin end" of
        the powerline. So this method will return all origin line flows.
        It is also possible, for a specific application, to return the maximum current flow between both ends of a power
        _grid for more complex scenario.

        For assumption about the order of the powerline flows return in this vector, see the help of the
        :func:`Backend.get_line_status` method.

        :return: an array with the line flows of each powerline
        :rtype: np.array, dtype:float
        """
        pass

    def set_thermal_limit(self, limits):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            You can set the thermal limit directly in the environment.

        This function is used as a convenience function to set the thermal limits :attr:`Backend.thermal_limit_a`
        in amperes.

        It can be used at the beginning of an episode if the thermal limit are not present in the original data files
        or alternatively if the thermal limits depends on the period of the year (one in winter and one in summer
        for example).

        Parameters
        ----------
        limits: ``object``
            It can be understood differently according to its type:

            - If it's a ``numpy.ndarray``, then it is assumed the thermal limits are given in amperes in the same order
              as the powerlines computed in the backend. In that case it modifies all the thermal limits of all
              the powerlines at once.
            - If it's a ``dict`` it must have:

              - as key the powerline names (not all names are mandatory, in that case only the powerlines with the name
                in this dictionnary will be modified)
              - as value the new thermal limit (should be a strictly positive float).

        """
        if isinstance(limits, np.ndarray):
            if limits.shape[0] == self.n_line:
                self.thermal_limit_a = 1. * limits.astype(dt_float)
        elif isinstance(limits, dict):
            for el in limits.keys():
                if not el in self.name_line:
                    raise BackendError("You asked to modify the thermal limit of powerline named \"{}\" that is not "
                                       "on the grid. Names of powerlines are {}".format(el, self.name_line))
            for i, el in self.name_line:
                if el in limits:
                    try:
                        tmp = dt_float(limits[el])
                    except:
                        raise BackendError("Impossible to convert data ({}) for powerline named \"{}\" into float "
                                           "values".format(limits[el], el))
                    if tmp <= 0:
                        raise BackendError("New thermal limit for powerlines \"{}\" is not positive ({})"
                                           "".format(el, tmp))
                    self.thermal_limit_a[i] = tmp

    def update_thermal_limit(self, env):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            This is done in a call to `env.step` in case of DLR for example.

            If you don't want this feature, do not implement it.

        Update the new thermal limit in case of DLR for example.

        By default it does nothing.

        Depending on the operational strategy, it is also possible to implement some
        `Dynamic Line Rating <https://en.wikipedia.org/wiki/Dynamic_line_rating_for_electric_utilities>`_ (DLR)
        strategies.
        In this case, this function will give the thermal limit for a given time step provided the flows and the
        weather condition are accessible by the backend. Our methodology doesn't make any assumption on the method
        used to get these thermal limits.

        Parameters
        ----------
        env: :class:`grid2op.Environment.Environment`
            The environment used to compute the thermal limit

        """

        pass

    def get_thermal_limit(self):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            Retrieve the thermal limit directly from the environment instead (with a call
            to :func:`grid2op.Environment.BaseEnc.get_thermal_limit` for example)

        Gives the thermal limit (in amps) for each powerline of the _grid. Only one value per powerline is returned.

        It is assumed that both :func:`Backend.get_line_flow` and *_get_thermal_limit* gives the value of the same
        end of the powerline.

        See the help of *_get_line_flow* for a more detailed description of this problem.

        For assumption about the order of the powerline flows return in this vector, see the help of the
        :func:`Backend.get_line_status` method.

        :return: An array giving the thermal limit of the powerlines.
        :rtype: np.array, dtype:float
        """
        return self.thermal_limit_a

    def get_relative_flow(self):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            Prefer using :attr:`grid2op.Observation.BaseObservation.rho`

        This method return the relative flows, *eg.* the current flow divided by the thermal limits. It has a pretty
        straightforward default implementation, but it can be overriden for example for transformer if the limits are
        on the lower voltage side or on the upper voltage level.

        Returns
        -------
        res: ``numpy.ndarray``, dtype: float
            The relative flow in each powerlines of the grid.
        """
        num_ = self.get_line_flow()
        denom_ = self.get_thermal_limit()
        res = np.divide(num_, denom_)
        return res

    def get_line_overflow(self):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            Prefer using :attr:`grid2op.Observation.BaseObservation.rho` and
            check whether or not the flow is higher tha 1. or have a look at
            :attr:`grid2op.Observation.BaseObservation.timestep_overflow` and check the
            non zero index.

        Prefer using the attribute of the :class:`grid2op.Observation.BaseObservation`

        faster accessor to the line that are on overflow.

        For assumption about the order of the powerline flows return in this vector, see the help of the
        :func:`Backend.get_line_status` method.

        :return: An array saying if a powerline is overflow or not
        :rtype: np.array, dtype:bool
        """
        th_lim = self.get_thermal_limit()
        flow = self.get_line_flow()
        return flow > th_lim

    @abstractmethod
    def get_topo_vect(self):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            Prefer using :attr:`grid2op.Observation.BaseObservation.topo_vect`

        Get the topology vector from the :attr:`Backend._grid`.
        The topology vector defines, for each object, on which bus it is connected.
        It returns -1 if the object is not connected.

        It is a vector with as much elements (productions, loads and lines extremity) as there are in the powergrid.

        For each elements, it gives on which bus it is connected in its substation.

        For example, if the first element of this vector is the load of id 1, then if `res[0] = 2` it means that the load of id 1 is connected to the second bus of its substation.

        You can check which object of the powerlines is represented by each component of this vector by looking at the `*_pos_topo_vect` (*eg.* :attr:`grid2op.Space.GridObjects.load_pos_topo_vect`) vectors.
        For each elements it gives its position in this vector.

        As any function of the backend, it is not advised to use it directly. You can get this information in the :attr:`grid2op.Observation.Observation.topo_vect` instead.

        Returns
        --------

        res: ``numpy.ndarray`` dtype: ``int``
            An array saying to which bus the object is connected.

        """
        pass

    @abstractmethod
    def generators_info(self):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            Prefer using :attr:`grid2op.Observation.BaseObservation.prod_p`,
            :attr:`grid2op.Observation.BaseObservation.prod_q` and
            :attr:`grid2op.Observation.BaseObservation.prod_v` instead.

        This method is used to retrieve information about the generators (active, reactive production
        and voltage magnitude of the bus to which it is connected).

        Returns
        -------
        prod_p ``numpy.ndarray``
            The active power production for each generator (in MW)
        prod_q ``numpy.ndarray``
            The reactive power production for each generator (in MVAr)
        prod_v ``numpy.ndarray``
            The voltage magnitude of the bus to which each generators is connected (in kV)
        """
        pass

    @abstractmethod
    def loads_info(self):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            Prefer using :attr:`grid2op.Observation.BaseObservation.load_p`,
            :attr:`grid2op.Observation.BaseObservation.load_q` and
            :attr:`grid2op.Observation.BaseObservation.load_v` instead.

        This method is used to retrieve information about the loads (active, reactive consumption
        and voltage magnitude of the bus to which it is connected).

        Returns
        -------
        load_p ``numpy.ndarray``
            The active power consumption for each load (in MW)
        load_q ``numpy.ndarray``
            The reactive power consumption for each load (in MVAr)
        load_v ``numpy.ndarray``
            The voltage magnitude of the bus to which each load is connected (in kV)
        """
        pass

    @abstractmethod
    def lines_or_info(self):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            Prefer using :attr:`grid2op.Observation.BaseObservation.p_or`,
            :attr:`grid2op.Observation.BaseObservation.q_or`,
            :attr:`grid2op.Observation.BaseObservation.a_or` and,
            :attr:`grid2op.Observation.BaseObservation.v_or` instead

        It returns the information extracted from the _grid at the origin end of each powerline.

        For assumption about the order of the powerline flows return in this vector, see the help of the
        :func:`Backend.get_line_status` method.

        Returns
        -------
        p_or ``numpy.ndarray``
            the origin active power flowing on the lines (in MW)
        q_or ``numpy.ndarray``
            the origin reactive power flowing on the lines (in MVAr)
        v_or ``numpy.ndarray``
            the voltage magnitude at the origin of each powerlines (in kV)
        a_or ``numpy.ndarray``
            the current flow at the origin of each powerlines (in A)
        """
        pass

    @abstractmethod
    def lines_ex_info(self):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            Prefer using :attr:`grid2op.Observation.BaseObservation.p_ex`,
            :attr:`grid2op.Observation.BaseObservation.q_ex`,
            :attr:`grid2op.Observation.BaseObservation.a_ex` and,
            :attr:`grid2op.Observation.BaseObservation.v_ex` instead

        It returns the information extracted from the _grid at the extremity end of each powerline.

        For assumption about the order of the powerline flows return in this vector, see the help of the
        :func:`Backend.get_line_status` method.

        Returns
        -------
        p_ex ``numpy.ndarray``
            the extremity active power flowing on the lines (in MW)
        q_ex ``numpy.ndarray``
            the extremity reactive power flowing on the lines (in MVAr)
        v_ex ``numpy.ndarray``
            the voltage magnitude at the extremity of each powerlines (in kV)
        a_ex ``numpy.ndarray``
            the current flow at the extremity of each powerlines (in A)
        """
        pass

    def shunt_info(self):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        This method is optional. If implemented, it should return the proper information about the shunt in the
        powergrid.

        If not implemented it returns empty list.

        Note that if there are shunt on the powergrid, it is recommended that this method should be implemented before
        calling :func:`Backend.check_kirchoff`.

        If this method is implemented AND :func:`Backend.check_kirchoff` is called, the method
        :func:`Backend.sub_from_bus_id` should also be implemented preferably.

        Returns
        -------
        shunt_p ``numpy.ndarray``
            For each shunt, the active power it withdraw at the bus to which it is connected.
        shunt_q ``numpy.ndarray``
            For each shunt, the reactive power it withdraw at the bus to which it is connected.
        shunt_v ``numpy.ndarray``
            For each shunt, the voltage magnitude of the bus to which it is connected.
        shunt_bus ``numpy.ndarray``
            For each shunt, the bus id to which it is connected.
        """
        return [], [], [], []

    def sub_from_bus_id(self, bus_id):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Optionnal method that allows to get the substation if the bus id is provided.

        :param bus_id:
        :return: the substation to which an object connected to bus with id `bus_id` is connected to.
        """
        raise Grid2OpException("This backend doesn't allow to get the substation from the bus id.")

    @abstractmethod
    def _disconnect_line(self, id_):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            Prefer using the action space to disconnect a powerline.


        Disconnect the line of id "id\\_ " in the backend.
        In this scenario, the *id\\_* of a powerline is its position (counted starting from O) in the vector returned by
        :func:`Backend.get_line_status` or :func:`Backend.get_line_flow` for example.
        For example, if the current flow on powerline "l1" is the 42nd element of the vector returned by :func:`Backend.get_line_flow`
        then :func:`Backend._disconnect_line(42)` will disconnect this same powerline "l1".

        For assumption about the order of the powerline flows return in this vector, see the help of the :func:`Backend.get_line_status` method.

        :param id_: id of the powerline to be disconnected
        :type id_: int

        """
        pass

    def _runpf_with_diverging_exception(self, is_dc):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\


        Computes a power flow on the _grid and raises an exception in case of diverging power flow, or any other
        exception that can be thrown by the backend.

        :param is_dc: mode of the power flow. If *is_dc* is True, then the powerlow is run using the DC approximation otherwise it uses the AC powerflow.
        :type is_dc: bool

        Raises
        ------
        exc_: :class:`grid2op.Exceptions.DivergingPowerFlow`
            In case of divergence of the powerflow

        """
        conv = False
        try:
            conv = self.runpf(is_dc=is_dc)  # run powerflow
        except:
            pass

        res = None
        if not conv:
            res = DivergingPowerFlow("GAME OVER: Powerflow has diverged during computation "
                                     "or a load has been disconnected or a generator has been disconnected.")
        return res

    def next_grid_state(self, env, is_dc=False):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            This is called by `env.step`

        This method is called by the environment to compute the next\\_grid\\_states.
        It allows to compute the powerline and approximate the "cascading failures" if there are some overflows.

        Attributes
        ----------
        env: :class:`grid2op.Environment.Environment`
            the environment in which the powerflow is ran.

        is_dc: ``bool``
            mode of power flow (AC : False, DC: is_dc is True)

        Returns
        --------
        disconnected_during_cf: ``numpy.ndarray``, dtype=bool
            For each powerlines, it returns ``True`` if the powerline has been disconnected due to a cascading failure
            or ``False`` otherwise.

        infos: ``list``
            If :attr:`Backend.detailed_infos_for_cascading_failures` is ``True`` then it returns the different
            state computed by the powerflow (can drastically slow down this function, as it requires
            deep copy of backend object). Otherwise the list is always empty.

        """
        infos = []
        disconnected_during_cf = np.full(self.n_line, fill_value=False, dtype=dt_bool)
        conv_ = self._runpf_with_diverging_exception(is_dc)
        if env._no_overflow_disconnection or conv_ is not None:
            return disconnected_during_cf, infos, conv_

        # the environment disconnect some

        init_time_step_overflow = copy.deepcopy(env._timestep_overflow)
        while True:
            # simulate the cascading failure
            lines_flows = self.get_line_flow()
            thermal_limits = self.get_thermal_limit()
            lines_status = self.get_line_status()

            # a) disconnect lines on hard overflow
            to_disc = lines_flows > env._hard_overflow_threshold * thermal_limits

            # b) deals with soft overflow
            init_time_step_overflow[ (lines_flows >= thermal_limits) & (lines_status)] += 1
            to_disc[init_time_step_overflow > env._nb_timestep_overflow_allowed] = True

            # disconnect the current power lines
            if np.sum(to_disc[lines_status]) == 0:
                # no powerlines have been disconnected at this time step, i stop the computation there
                break
            disconnected_during_cf[to_disc] = True

            # perform the disconnection action
            [self._disconnect_line(i) for i, el in enumerate(to_disc) if el]
            # start a powerflow on this new state
            conv_ = self._runpf_with_diverging_exception(is_dc)
            if self.detailed_infos_for_cascading_failures:
                infos.append(self.copy())

            if conv_ is not None:
                break
        return disconnected_during_cf, infos, conv_

    def check_kirchoff(self):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Check that the powergrid respects kirchhoff's law.
        This function can be called at any moment to make sure a powergrid is in a consistent state, or to perform
        some tests for example.

        In order to function properly, this method requires that :func:`Backend.shunt_info` and
        :func:`Backend.sub_from_bus_id` are properly defined. Otherwise the results might be wrong, especially
        for reactive values (q_subs and q_bus bellow)

        Returns
        -------
        p_subs ``numpy.ndarray``
            sum of injected active power at each substations (MW)
        q_subs ``numpy.ndarray``
            sum of injected reactive power at each substations (MVAr)
        p_bus ``numpy.ndarray``
            sum of injected active power at each buses. It is given in form of a matrix, with number of substations as
            row, and number of columns equal to the maximum number of buses for a substation (MW)
        q_bus ``numpy.ndarray``
            sum of injected reactive power at each buses. It is given in form of a matrix, with number of substations as
            row, and number of columns equal to the maximum number of buses for a substation (MVAr)
        """

        p_or, q_or, v_or, *_ = self.lines_or_info()
        p_ex, q_ex, v_ex, *_ = self.lines_ex_info()
        p_gen, q_gen, v_gen = self.generators_info()
        p_load, q_load, v_load = self.loads_info()

        # fist check the "substation law" : nothing is created at any substation
        p_subs = np.zeros(self.n_sub)
        q_subs = np.zeros(self.n_sub)

        # check for each bus
        p_bus = np.zeros((self.n_sub, 2))
        q_bus = np.zeros((self.n_sub, 2))
        topo_vect = self.get_topo_vect()

        for i in range(self.n_line):
            # for substations
            p_subs[self.line_or_to_subid[i]] += p_or[i]
            p_subs[self.line_ex_to_subid[i]] += p_ex[i]

            q_subs[self.line_or_to_subid[i]] += q_or[i]
            q_subs[self.line_ex_to_subid[i]] += q_ex[i]

            # for bus
            p_bus[self.line_or_to_subid[i], topo_vect[self.line_or_pos_topo_vect[i]] - 1] += p_or[i]
            q_bus[self.line_or_to_subid[i], topo_vect[self.line_or_pos_topo_vect[i]] - 1] += q_or[i]

            p_bus[self.line_ex_to_subid[i], topo_vect[self.line_ex_pos_topo_vect[i]] - 1] += p_ex[i]
            q_bus[self.line_ex_to_subid[i], topo_vect[self.line_ex_pos_topo_vect[i]] - 1] += q_ex[i]

        for i in range(self.n_gen):
            # for substations
            p_subs[self.gen_to_subid[i]] -= p_gen[i]
            q_subs[self.gen_to_subid[i]] -= q_gen[i]

            # for bus
            p_bus[self.gen_to_subid[i],  topo_vect[self.gen_pos_topo_vect[i]]-1] -= p_gen[i]
            q_bus[self.gen_to_subid[i],  topo_vect[self.gen_pos_topo_vect[i]]-1] -= q_gen[i]

        for i in range(self.n_load):
            # for substations
            p_subs[self.load_to_subid[i]] += p_load[i]
            q_subs[self.load_to_subid[i]] += q_load[i]

            # for buses
            p_bus[self.load_to_subid[i],  topo_vect[self.load_pos_topo_vect[i]]-1] += p_load[i]
            q_bus[self.load_to_subid[i],  topo_vect[self.load_pos_topo_vect[i]]-1] += q_load[i]

        if self.shunts_data_available:
            p_s, q_s, v_s, bus_s = self.shunt_info()
            for i in range(self.n_shunt):
                # for substations
                p_subs[self.shunt_to_subid[i]] += p_s[i]
                q_subs[self.shunt_to_subid[i]] += q_s[i]

                # for buses
                p_bus[self.shunt_to_subid[i], bus_s[i] - 1] += p_s[i]
                q_bus[self.shunt_to_subid[i], bus_s[i] - 1] += q_s[i]
        else:
            warnings.warn("Backend.check_kirchoff Impossible to get shunt information. Reactive information might be "
                          "incorrect.")

        return p_subs, q_subs, p_bus, q_bus

    def load_redispacthing_data(self, path, name='prods_charac.csv'):
        """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        This method will load everything needed for the redispatching and unit commitment problem.

        Parameters
        ----------
        path: ``str``
            Location of the datafram containing the redispatching data.

        name: ``str``
            Name of the dataframe containing the redispatching data

        """
        # for redispatching
        fullpath = os.path.join(path, name)
        if not os.path.exists(fullpath):
            self.redispatching_unit_commitment_availble = False
            return
        try:
            df = pd.read_csv(fullpath)
        except Exception as e:
            return

        for el in ["type", "Pmax", "Pmin", "max_ramp_up", "max_ramp_down", "start_cost",
                   "shut_down_cost", "marginal_cost", "min_up_time", "min_down_time"]:
            if el not in df.columns:
                return

        gen_info = {}
        for _, row in df.iterrows():
            gen_info[row["name"]] = {"type": row["type"],
                                     "pmax": row["Pmax"],
                                     "pmin": row["Pmin"],
                                     "max_ramp_up": row["max_ramp_up"],
                                     "max_ramp_down": row["max_ramp_down"],
                                     "start_cost": row["start_cost"],
                                     "shut_down_cost": row["shut_down_cost"],
                                     "marginal_cost": row["marginal_cost"],
                                     "min_up_time": row["min_up_time"],
                                     "min_down_time": row["min_down_time"]
                                     }
        self.redispatching_unit_commitment_availble = True

        self.gen_type = np.full(self.n_gen, fill_value="aaaaaaaaaa")
        self.gen_pmin = np.full(self.n_gen, fill_value=1., dtype=dt_float)
        self.gen_pmax = np.full(self.n_gen, fill_value=1., dtype=dt_float)
        self.gen_redispatchable = np.full(self.n_gen, fill_value=False, dtype=dt_bool)
        self.gen_max_ramp_up = np.full(self.n_gen, fill_value=0., dtype=dt_float)
        self.gen_max_ramp_down = np.full(self.n_gen, fill_value=0., dtype=dt_float)
        self.gen_min_uptime = np.full(self.n_gen, fill_value=-1, dtype=dt_int)
        self.gen_min_downtime = np.full(self.n_gen, fill_value=-1, dtype=dt_int)
        self.gen_cost_per_MW = np.full(self.n_gen, fill_value=1., dtype=dt_float)  # marginal cost
        self.gen_startup_cost = np.full(self.n_gen, fill_value=1., dtype=dt_float)  # start cost
        self.gen_shutdown_cost = np.full(self.n_gen, fill_value=1., dtype=dt_float)  # shutdown cost

        for i, gen_nm in enumerate(self.name_gen):
            tmp_gen = gen_info[gen_nm]
            self.gen_type[i] = str(tmp_gen["type"])
            self.gen_pmin[i] = dt_float(tmp_gen["pmin"])
            self.gen_pmax[i] = dt_float(tmp_gen["pmax"])
            self.gen_redispatchable[i] = dt_bool(tmp_gen["type"] not in ["wind", "solar"])
            tmp = dt_float(tmp_gen["max_ramp_up"])
            if np.isfinite(tmp):
                self.gen_max_ramp_up[i] = tmp
            tmp = dt_float(tmp_gen["max_ramp_down"])
            if np.isfinite(tmp):
                self.gen_max_ramp_down[i] = tmp
            self.gen_min_uptime[i] = dt_int(tmp_gen["min_up_time"])
            self.gen_min_downtime[i] = dt_int(tmp_gen["min_down_time"])
            self.gen_cost_per_MW[i] = dt_float(tmp_gen["marginal_cost"])
            self.gen_startup_cost[i] = dt_float(tmp_gen["start_cost"])
            self.gen_shutdown_cost[i] = dt_float(tmp_gen["shut_down_cost"])

    def load_grid_layout(self, path, name='grid_layout.json'):
        full_fn = os.path.join(path, name)
        if not os.path.exists(full_fn):
            return Exception("File {} does not exist".format(full_fn))
        try:
            with open(full_fn, "r") as f:
                dict_ = json.load(f)
        except Exception as e:
            return e

        new_grid_layout = {}
        for el in self.name_sub:
            if not el in dict_:
                return Exception("substation named {} not in layout".format(el))
            tmp = dict_[el]
            try:
                x, y = tmp
                x = dt_float(x)
                y = dt_float(y)
                new_grid_layout[el] = (x, y)
            except Exception as e_:
                return Exception("fail to convert coordinates for {} into list of coordinates with error {}"
                                 "".format(el, e_))

        self.attach_layout(grid_layout=new_grid_layout)

    def get_action_to_set(self):
        line_status = self.get_line_status()
        line_status = 2 * line_status - 1
        line_status = line_status.astype(dt_int)
        topo_vect = self.get_topo_vect()
        prod_p, _, prod_v = self.generators_info()
        load_p, load_q, _ = self.loads_info()
        complete_action_class = CompleteAction.init_grid(self)
        set_me = complete_action_class(self)
        set_me.update({"set_line_status": line_status,
                       "set_bus": topo_vect})
        return set_me
