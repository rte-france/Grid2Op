# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import os
import sys
import warnings
import json

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Exceptions import (
    EnvError,
    DivergingPowerFlow,
    IncorrectNumberOfElements,
    IncorrectNumberOfLoads,
)
from grid2op.Exceptions import (
    IncorrectNumberOfGenerators,
    BackendError,
    IncorrectNumberOfLines,
)
from grid2op.Space import GridObjects
from grid2op.Exceptions import Grid2OpException


# TODO method to get V and theta at each bus, could be in the same shape as check_kirchoff


class Backend(GridObjects, ABC):
    """
    INTERNAL

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

    All the abstract methods (that need to be implemented for a backend to work properly) are (more information given
    in the :ref:`create-backend-module` page):

    - :func:`Backend.load_grid`
    - :func:`Backend.apply_action`
    - :func:`Backend.runpf`
    - :func:`Backend.get_topo_vect`
    - :func:`Backend.generators_info`
    - :func:`Backend.loads_info`
    - :func:`Backend.lines_or_info`
    - :func:`Backend.lines_ex_info`

    And optionally:

    - :func:`Backend.close` (this is mandatory if your backend implementation (`self._grid`) is relying on some
      c / c++ code that do not free memory automatically.
    - :func:`Backend.copy` (not that this is mandatory if your backend implementation (in `self._grid`) cannot be
      deep copied using the python copy.deepcopy function) [as of grid2op >= 1.7.1 it is no more
      required. If not implemented, you won't be able to use some of grid2op feature however]
    - :func:`Backend.get_line_status`: the default implementation uses the "get_topo_vect()" and then check
      if buses at both ends of powerline are positive. This is rather slow and can most likely be optimized.
    - :func:`Backend.get_line_flow`: the default implementation will retrieve all powerline information
      at the "origin" side and just return the "a_or" vector. You want to do something smarter here.
    - :func:`Backend._disconnect_line`: has a default slow implementation using "apply_action" that might
      can most likely be optimized in your backend.
    - :func:`Backend.reset`  will reload the powergrid from the hard drive by default. This is rather slow and we
      recommend to overload it.

    And, if the flag :attr:Backend.shunts_data_available` is set to ``True`` the method :func:`Backend.shunt_info`
    should also be implemented.

    .. note:: Backend also support "shunts" information if the `self.shunts_data_available` flag is set to
        ``True`` in that case, you also need to implement all the relevant shunt information (attributes `n_shunt`,
        `shunt_to_subid`, `name_shunt` and function `shunt_info` and handle the modification of shunts
        bus, active value and reactive value in the "apply_action" function).


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

    comp_time: ``float``
        Time to compute the powerflow (might be unset, ie stay at 0.0)

    """
    IS_BK_CONVERTER = False

    env_name = "unknown"

    # action to set me
    my_bk_act_class = None
    _complete_action_class = None

    ERR_INIT_POWERFLOW = "Power cannot be computed on the first time step, please check your data."
    def __init__(self,
                 detailed_infos_for_cascading_failures: bool=False,
                 can_be_copied: bool=True,
                 **kwargs):
        """
        Initialize an instance of Backend. This does nothing per se. Only the call to :func:`Backend.load_grid`
        should guarantee the backend is properly configured.

        :param detailed_infos_for_cascading_failures: Whether to be detailed (but slow) when computing cascading failures
        :type detailed_infos_for_cascading_failures: :class:`bool`

        """
        GridObjects.__init__(self)

        # the following parameter is used to control the amount of verbosity when computing a cascading failure
        # if it's set to true, it returns all intermediate _grid states. This can slow down the computation!
        self.detailed_infos_for_cascading_failures = (
            detailed_infos_for_cascading_failures
        )

        # the power _grid manipulated. One powergrid per backend.
        self._grid = None

        # thermal limit setting, in ampere, at the same "side" of the powerline than self.get_line_overflow
        self.thermal_limit_a = None

        # for the shunt (only if supported)
        self._sh_vnkv = None  # for each shunt gives the nominal value at the bus at which it is connected
        # if this information is not present, then "get_action_to_set" might not behave correctly

        self.comp_time = 0.0
        self.can_output_theta = False

        # to prevent the use of the same backend instance in different environment.
        self._is_loaded = False

        self._can_be_copied = can_be_copied
        
        self._my_kwargs = {"detailed_infos_for_cascading_failures": detailed_infos_for_cascading_failures,
                           "can_be_copied": self._can_be_copied}
        for k, v in kwargs.items():
            self._my_kwargs[k] = v
        
    @property
    def is_loaded(self):
        return self._is_loaded

    @is_loaded.setter
    def is_loaded(self, value):
        if value is True:
            self._is_loaded = True
        else:
            raise BackendError('Impossible to unset the "is_loaded" status.')

    @abstractmethod
    def load_grid(self, path, filename=None):
        """
        INTERNAL

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
    def apply_action(self, action):
        """
        INTERNAL

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
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            This is called by :func:`Backend.next_grid_state` (that computes some kind of
            cascading failures).

            This is one of the core function if you want to code a backend. It will carry out
            a powerflow.


        Run a power flow on the underlying _grid.
        Powerflow can be AC (is_dc = False) or DC (is_dc = True)

        :param is_dc: is the powerflow run in DC or in AC
        :type is_dc: :class:`bool`

        :return: ``True`` if it has converged, or false otherwise. In case of non convergence, no flows can be inspected on
          the _grid.
        :rtype: :class:`bool`

        :return: an exception in case of divergence (or none if no particular info are available)
        :rtype: `Exception`
        """
        pass

    @abstractmethod
    def get_topo_vect(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            Prefer using :attr:`grid2op.Observation.BaseObservation.topo_vect`

        Get the topology vector from the :attr:`Backend._grid`.
        The topology vector defines, for each object, on which bus it is connected.
        It returns -1 if the object is not connected.

        It is a vector with as much elements (productions, loads and lines extremity) as there are in the powergrid.

        For each elements, it gives on which bus it is connected in its substation.

        For example, if the first element of this vector is the load of id 1, then if `res[0] = 2` it means that the
        load of id 1 is connected to the second bus of its substation.

        You can check which object of the powerlines is represented by each component of this vector by looking at the
        `*_pos_topo_vect` (*eg.* :attr:`grid2op.Space.GridObjects.load_pos_topo_vect`) vectors.
        For each elements it gives its position in this vector.

        As any function of the backend, it is not advised to use it directly. You can get this information in the
        :attr:`grid2op.Observation.Observation.topo_vect` instead.

        Returns
        --------

        res: ``numpy.ndarray`` dtype: ``int``
            An array saying to which bus the object is connected.

        """
        pass

    @abstractmethod
    def generators_info(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            Prefer using :attr:`grid2op.Observation.BaseObservation.gen_p`,
            :attr:`grid2op.Observation.BaseObservation.gen_q` and
            :attr:`grid2op.Observation.BaseObservation.gen_v` instead.

        This method is used to retrieve information about the generators (active, reactive production
        and voltage magnitude of the bus to which it is connected).
        
        .. note:: 
            The values returned here are the values AFTER the powerflow has been computed and not 
            the target values.

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
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            Prefer using :attr:`grid2op.Observation.BaseObservation.load_p`,
            :attr:`grid2op.Observation.BaseObservation.load_q` and
            :attr:`grid2op.Observation.BaseObservation.load_v` instead.

        This method is used to retrieve information about the loads (active, reactive consumption
        and voltage magnitude of the bus to which it is connected).

        .. note:: 
            The values returned here are the values AFTER the powerflow has been computed and not 
            the target values.
            
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
        INTERNAL

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
        INTERNAL

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

    def close(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            This is called by `env.close()` do not attempt to use it otherwise.

        This function is called when the environment is over.
        After calling this function, the backend might not behave properly, and in any case should not be used before
        another call to :func:`Backend.load_grid` is performed

        """
        pass

    def reset(self, grid_path, grid_filename=None):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            This is done in the `env.reset()` method and should be performed otherwise.

        Reload the power grid.
        For backwards compatibility this method calls `Backend.load_grid`.
        But it is encouraged to overload it in the subclasses.
        """
        self.comp_time = 0.0
        self.load_grid(grid_path, filename=grid_filename)

    def copy(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        .. note::
            As of grid2op 1.7.1 you it is not mandatory to implement this function
            when creating a backend.
            
            If it is not available, then grid2op will automatically
            deactivate the forecast capability and will not use the "backend.copy()"
            function.
            
            When this function is not implement, you will not be able to use (for 
            example) :func:`grid2op.Observation.BaseObservation.simulate` nor
            the :class:`grid2op.simulator.Simulator` for example.

        Performs a deep copy of the backend.

        In the default implementation we explicitly called the deepcopy operator on `self._grid` to make the
        error message more explicit in case there is a problem with this part.

        The implementation is **equivalent** to:

        .. code-block:: python

            def copy(self):
                return copy.deepcopy(self)

        :return: An instance of Backend equal to :attr:`self`, but deep copied.
        :rtype: :class:`Backend`
        """
        if not self._can_be_copied:
            raise BackendError("This backend cannot be copied.")

        start_grid = self._grid
        self._grid = None
        res = copy.deepcopy(self)
        res.__class__ = type(self)  # somehow deepcopy forget the init class... weird
        res._grid = copy.deepcopy(start_grid)
        self._grid = start_grid
        res._is_loaded = False  # i can reload a copy of an environment
        return res

    def save_file(self, full_path):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Save the current power _grid in a human readable format supported by the backend.
        The format is not modified by this wrapper.

        This function is not mandatory, and if implemented, it is used only as a debugging purpose.

        :param full_path: the full path (path + file name + extension) where *self._grid* is stored.
        :type full_path: :class:`string`

        :return: ``None``
        """
        raise RuntimeError("Class {} does not allow for saving file.".format(self))

    def get_line_status(self):
        """
        INTERNAL

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
        topo_vect = self.get_topo_vect()
        return (topo_vect[self.line_or_pos_topo_vect] >= 0) & (
            topo_vect[self.line_ex_pos_topo_vect] >= 0
        )

    def get_line_flow(self):
        """
        INTERNAL

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
        p_or, q_or, v_or, a_or = self.lines_or_info()
        return a_or

    def set_thermal_limit(self, limits):
        """
        INTERNAL

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
                self.thermal_limit_a = 1.0 * limits.astype(dt_float)
        elif isinstance(limits, dict):
            for el in limits.keys():
                if not el in self.name_line:
                    raise BackendError(
                        'You asked to modify the thermal limit of powerline named "{}" that is not '
                        "on the grid. Names of powerlines are {}".format(
                            el, self.name_line
                        )
                    )
            for i, el in self.name_line:
                if el in limits:
                    try:
                        tmp = dt_float(limits[el])
                    except:
                        raise BackendError(
                            'Impossible to convert data ({}) for powerline named "{}" into float '
                            "values".format(limits[el], el)
                        )
                    if tmp <= 0:
                        raise BackendError(
                            'New thermal limit for powerlines "{}" is not positive ({})'
                            "".format(el, tmp)
                        )
                    self.thermal_limit_a[i] = tmp

    def update_thermal_limit_from_vect(self, thermal_limit_a):
        """You can use it if your backend stores the thermal limits
        of the grid in a vector (see PandaPowerBackend for example)
        
        .. warning::
            This is not called by the environment and cannot be used to
            model Dynamic Line Rating. For such purpose please use `update_thermal_limit`
            
            This function is used to create a "Simulator" from a backend for example.
        

        Parameters
        ----------
        vect : np.ndarray
            The thermal limits (in A)
        """
        thermal_limit_a = np.array(thermal_limit_a).astype(dt_float)
        self.thermal_limit_a[:] = thermal_limit_a
    
    def update_thermal_limit(self, env):
        """
        INTERNAL

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
        INTERNAL

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
        INTERNAL

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
        INTERNAL

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

    def shunt_info(self):
        """
        INTERNAL

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
        shunt_p: ``numpy.ndarray``
            For each shunt, the active power it withdraw at the bus to which it is connected.
        shunt_q: ``numpy.ndarray``
            For each shunt, the reactive power it withdraw at the bus to which it is connected.
        shunt_v: ``numpy.ndarray``
            For each shunt, the voltage magnitude of the bus to which it is connected.
        shunt_bus: ``numpy.ndarray``
            For each shunt, the bus id to which it is connected.
        """
        return [], [], [], []

    def get_theta(self):
        """

        Notes
        -----
        Don't forget to set the flag :attr:`Backend.can_output_theta` to ``True`` in the
        :func:`Bakcend.load_grid` if you support this feature.

        Returns
        -------
        line_or_theta: ``numpy.ndarray``
            For each origin side of powerline, gives the voltage angle
        line_ex_theta: ``numpy.ndarray``
            For each extremity side of powerline, gives the voltage angle
        load_theta: ``numpy.ndarray``
            Gives the voltage angle to the bus at which each load is connected
        gen_theta: ``numpy.ndarray``
            Gives the voltage angle to the bus at which each generator is connected
        storage_theta: ``numpy.ndarray``
            Gives the voltage angle to the bus at which each storage unit is connected
        """
        raise NotImplementedError(
            "Your backend does not support the retrieval of the voltage angle theta."
        )

    def sub_from_bus_id(self, bus_id):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\


        Optional method that allows to get the substation if the bus id is provided.

        Parameters
        ----------
        bus_id: ``int``
            The id of the bus where you want to know to which substation it belongs

        Returns
        -------
            The substation to which an object connected to bus with id `bus_id` is connected to.

        """
        raise BackendError(
            "This backend doesn't allow to get the substation from the bus id."
        )

    def _disconnect_line(self, id_):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            Prefer using the action space to disconnect a powerline.


        Disconnect the line of id "id\\_ " in the backend.
        In this scenario, the *id\\_* of a powerline is its position (counted starting from O) in the vector returned by
        :func:`Backend.get_line_status` or :func:`Backend.get_line_flow` for example.
        For example, if the current flow on powerline "l1" is the 42nd element of the vector returned by
        :func:`Backend.get_line_flow`
        then :func:`Backend._disconnect_line(42)` will disconnect this same powerline "l1".

        For assumption about the order of the powerline flows return in this vector, see the help of the
        :func:`Backend.get_line_status` method.

        :param id_: id of the powerline to be disconnected
        :type id_: int

        """
        my_cls = type(self)
        action = my_cls._complete_action_class()
        action.update({"set_line_status": [(id_, -1)]})
        bk_act = my_cls.my_bk_act_class()
        bk_act += action
        self.apply_action(bk_act)

    def _runpf_with_diverging_exception(self, is_dc):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\


        Computes a power flow on the _grid and raises an exception in case of diverging power flow, or any other
        exception that can be thrown by the backend.

        :param is_dc: mode of the power flow. If *is_dc* is True, then the powerlow is run using the DC
                      approximation otherwise it uses the AC powerflow.
        :type is_dc: bool

        Raises
        ------
        exc_: :class:`grid2op.Exceptions.DivergingPowerFlow`
            In case of divergence of the powerflow

        """
        conv = False
        exc_me = None
        try:
            conv, exc_me = self.runpf(is_dc=is_dc)  # run powerflow
        except Grid2OpException as exc_:
            exc_me = exc_
        except Exception as exc_:
            exc_me = DivergingPowerFlow(
                f" An unexpected error occurred during the computation of the powerflow."
                f"The error is: \n {exc_} \n. This is game over"
            )

        if not conv and exc_me is None:
            exc_me = DivergingPowerFlow(
                "GAME OVER: Powerflow has diverged during computation "
                "or a load has been disconnected or a generator has been disconnected."
            )
        return exc_me

    def next_grid_state(self, env, is_dc=False):
        """
        INTERNAL

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
        disconnected_during_cf = np.full(self.n_line, fill_value=-1, dtype=dt_int)
        conv_ = self._runpf_with_diverging_exception(is_dc)
        if env._no_overflow_disconnection or conv_ is not None:
            return disconnected_during_cf, infos, conv_

        # the environment disconnect some powerlines
        init_time_step_overflow = copy.deepcopy(env._timestep_overflow)
        ts = 0
        while True:
            # simulate the cascading failure
            lines_flows = 1.0 * self.get_line_flow()
            thermal_limits = self.get_thermal_limit()
            lines_status = self.get_line_status()

            # a) disconnect lines on hard overflow (that are still connected)
            to_disc = (
                lines_flows > env._hard_overflow_threshold * thermal_limits
            ) & lines_status

            # b) deals with soft overflow (disconnect them if lines still connected)
            init_time_step_overflow[(lines_flows >= thermal_limits) & lines_status] += 1
            to_disc[
                (init_time_step_overflow > env._nb_timestep_overflow_allowed)
                & lines_status
            ] = True

            # disconnect the current power lines
            if np.sum(to_disc[lines_status]) == 0:
                # no powerlines have been disconnected at this time step, i stop the computation there
                break
            disconnected_during_cf[to_disc] = ts
            # perform the disconnection action
            for i, el in enumerate(to_disc):
                if el:
                    self._disconnect_line(i)

            # start a powerflow on this new state
            conv_ = self._runpf_with_diverging_exception(is_dc)
            if self.detailed_infos_for_cascading_failures:
                infos.append(self.copy())

            if conv_ is not None:
                break
            ts += 1
        return disconnected_during_cf, infos, conv_

    def storages_info(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            Prefer using :attr:`grid2op.Observation.BaseObservation.storage_power` instead.

        This method is used to retrieve information about the storage units (active, reactive consumption
        and voltage magnitude of the bus to which it is connected).

        Returns
        -------
        storage_p ``numpy.ndarray``
            The active power consumption for each load (in MW)
        storage_q ``numpy.ndarray``
            The reactive power consumption for each load (in MVAr)
        storage_v ``numpy.ndarray``
            The voltage magnitude of the bus to which each load is connected (in kV)
        """
        if self.n_storage > 0:
            raise BackendError(
                "storages_info method is not implemented yet there is batteries on the grid."
            )

    def storage_deact_for_backward_comaptibility(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        This function is called under a very specific condition: an old environment has been loaded that
        do not take into account the storage units, even though they were possibly some modeled by the backend.

        This function is supposed to "remove" from the backend any reference to the storage units.

        Overloading this function is not necessary (when developing a new backend). If it is not overloaded however,
        some "backward compatibility" (for grid2op <= 1.4.0) might not be working properly depending on
        your backend.
        """
        pass

    def check_kirchoff(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Check that the powergrid respects kirchhoff's law.
        This function can be called at any moment (after a powerflow has been run)
        to make sure a powergrid is in a consistent state, or to perform
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
        diff_v_bus: ``numpy.ndarray`` (2d array)
            difference between maximum voltage and minimum voltage (computed for each elements)
            at each bus. It is an array of two dimension:

            - first dimension represents the the substation (between 1 and self.n_sub)
            - second element represents the busbar in the substation (0 or 1 usually)

        """

        p_or, q_or, v_or, *_ = self.lines_or_info()
        p_ex, q_ex, v_ex, *_ = self.lines_ex_info()
        p_gen, q_gen, v_gen = self.generators_info()
        p_load, q_load, v_load = self.loads_info()
        if self.n_storage > 0:
            p_storage, q_storage, v_storage = self.storages_info()

        # fist check the "substation law" : nothing is created at any substation
        p_subs = np.zeros(self.n_sub, dtype=dt_float)
        q_subs = np.zeros(self.n_sub, dtype=dt_float)

        # check for each bus
        p_bus = np.zeros((self.n_sub, 2), dtype=dt_float)
        q_bus = np.zeros((self.n_sub, 2), dtype=dt_float)
        v_bus = (
            np.zeros((self.n_sub, 2, 2), dtype=dt_float) - 1.0
        )  # sub, busbar, [min,max]
        topo_vect = self.get_topo_vect()

        # bellow i'm "forced" to do a loop otherwise, numpy do not compute the "+=" the way I want it to.
        # for example, if two powerlines are such that line_or_to_subid is equal (eg both connected to substation 0)
        # then numpy do not guarantee that `p_subs[self.line_or_to_subid] += p_or` will add the two "corresponding p_or"
        # TODO this can be vectorized with matrix product, see example in obs.flow_bus_matrix (BaseObervation.py)
        for i in range(self.n_line):
            sub_or_id = self.line_or_to_subid[i]
            sub_ex_id = self.line_ex_to_subid[i]
            loc_bus_or = topo_vect[self.line_or_pos_topo_vect[i]] - 1
            loc_bus_ex = topo_vect[self.line_ex_pos_topo_vect[i]] - 1
            
            # for substations
            p_subs[sub_or_id] += p_or[i]
            p_subs[sub_ex_id] += p_ex[i]

            q_subs[sub_or_id] += q_or[i]
            q_subs[sub_ex_id] += q_ex[i]

            # for bus
            p_bus[sub_or_id, loc_bus_or] += p_or[i]
            q_bus[sub_or_id, loc_bus_or] += q_or[i]

            p_bus[ sub_ex_id, loc_bus_ex] += p_ex[i]
            q_bus[sub_ex_id, loc_bus_ex] += q_ex[i]

            # fill the min / max voltage per bus (initialization)
            if (v_bus[sub_or_id,loc_bus_or,][0] == -1):
                v_bus[sub_or_id,loc_bus_or,][0] = v_or[i]
            if (v_bus[sub_ex_id,loc_bus_ex,][0] == -1):
                v_bus[sub_ex_id,loc_bus_ex,][0] = v_ex[i]
            if (v_bus[sub_or_id, loc_bus_or,][1]== -1):
                v_bus[sub_or_id,loc_bus_or,][1] = v_or[i]
            if (v_bus[sub_ex_id,loc_bus_ex,][1]== -1):
                v_bus[sub_ex_id,loc_bus_ex,][1] = v_ex[i]

            # now compute the correct stuff
            if v_or[i] > 0.0:
                # line is connected
                v_bus[sub_or_id,loc_bus_or,][0] = min(v_bus[sub_or_id,loc_bus_or,][0],v_or[i],)
                v_bus[sub_or_id,loc_bus_or,][1] = max(v_bus[sub_or_id,loc_bus_or,][1],v_or[i],)
                
            if v_ex[i] > 0:
                # line is connected
                v_bus[sub_ex_id,loc_bus_ex,][0] = min(v_bus[sub_ex_id,loc_bus_ex,][0],v_ex[i],)
                v_bus[sub_ex_id,loc_bus_ex,][1] = max(v_bus[sub_ex_id,loc_bus_ex,][1],v_ex[i],)
        
        for i in range(self.n_gen):
            # for substations
            p_subs[self.gen_to_subid[i]] -= p_gen[i]
            q_subs[self.gen_to_subid[i]] -= q_gen[i]

            # for bus
            p_bus[
                self.gen_to_subid[i], topo_vect[self.gen_pos_topo_vect[i]] - 1
            ] -= p_gen[i]
            q_bus[
                self.gen_to_subid[i], topo_vect[self.gen_pos_topo_vect[i]] - 1
            ] -= q_gen[i]

            # compute max and min values
            if v_gen[i]:
                # but only if gen is connected
                v_bus[self.gen_to_subid[i], topo_vect[self.gen_pos_topo_vect[i]] - 1][
                    0
                ] = min(
                    v_bus[
                        self.gen_to_subid[i], topo_vect[self.gen_pos_topo_vect[i]] - 1
                    ][0],
                    v_gen[i],
                )
                v_bus[self.gen_to_subid[i], topo_vect[self.gen_pos_topo_vect[i]] - 1][
                    1
                ] = max(
                    v_bus[
                        self.gen_to_subid[i], topo_vect[self.gen_pos_topo_vect[i]] - 1
                    ][1],
                    v_gen[i],
                )

        for i in range(self.n_load):
            # for substations
            p_subs[self.load_to_subid[i]] += p_load[i]
            q_subs[self.load_to_subid[i]] += q_load[i]

            # for buses
            p_bus[
                self.load_to_subid[i], topo_vect[self.load_pos_topo_vect[i]] - 1
            ] += p_load[i]
            q_bus[
                self.load_to_subid[i], topo_vect[self.load_pos_topo_vect[i]] - 1
            ] += q_load[i]

            # compute max and min values
            if v_load[i]:
                # but only if load is connected
                v_bus[self.load_to_subid[i], topo_vect[self.load_pos_topo_vect[i]] - 1][
                    0
                ] = min(
                    v_bus[
                        self.load_to_subid[i], topo_vect[self.load_pos_topo_vect[i]] - 1
                    ][0],
                    v_load[i],
                )
                v_bus[self.load_to_subid[i], topo_vect[self.load_pos_topo_vect[i]] - 1][
                    1
                ] = max(
                    v_bus[
                        self.load_to_subid[i], topo_vect[self.load_pos_topo_vect[i]] - 1
                    ][1],
                    v_load[i],
                )

        for i in range(self.n_storage):
            p_subs[self.storage_to_subid[i]] += p_storage[i]
            q_subs[self.storage_to_subid[i]] += q_storage[i]
            p_bus[
                self.storage_to_subid[i], topo_vect[self.storage_pos_topo_vect[i]] - 1
            ] += p_storage[i]
            q_bus[
                self.storage_to_subid[i], topo_vect[self.storage_pos_topo_vect[i]] - 1
            ] += q_storage[i]

            # compute max and min values
            if v_storage[i] > 0:
                # the storage unit is connected
                v_bus[
                    self.storage_to_subid[i],
                    topo_vect[self.storage_pos_topo_vect[i]] - 1,
                ][0] = min(
                    v_bus[
                        self.storage_to_subid[i],
                        topo_vect[self.storage_pos_topo_vect[i]] - 1,
                    ][0],
                    v_storage[i],
                )
                v_bus[
                    self.storage_to_subid[i],
                    topo_vect[self.storage_pos_topo_vect[i]] - 1,
                ][1] = max(
                    v_bus[
                        self.storage_to_subid[i],
                        topo_vect[self.storage_pos_topo_vect[i]] - 1,
                    ][1],
                    v_storage[i],
                )

        if self.shunts_data_available:
            p_s, q_s, v_s, bus_s = self.shunt_info()
            for i in range(self.n_shunt):
                # for substations
                p_subs[self.shunt_to_subid[i]] += p_s[i]
                q_subs[self.shunt_to_subid[i]] += q_s[i]

                # for buses
                p_bus[self.shunt_to_subid[i], bus_s[i] - 1] += p_s[i]
                q_bus[self.shunt_to_subid[i], bus_s[i] - 1] += q_s[i]

                # compute max and min values
                v_bus[self.shunt_to_subid[i], bus_s[i] - 1][0] = min(
                    v_bus[self.shunt_to_subid[i], bus_s[i] - 1][0], v_s[i]
                )
                v_bus[self.shunt_to_subid[i], bus_s[i] - 1][1] = max(
                    v_bus[self.shunt_to_subid[i], bus_s[i] - 1][1], v_s[i]
                )
        else:
            warnings.warn(
                "Backend.check_kirchoff Impossible to get shunt information. Reactive information might be "
                "incorrect."
            )
        diff_v_bus = np.zeros((self.n_sub, 2), dtype=dt_float)
        diff_v_bus[:, :] = v_bus[:, :, 1] - v_bus[:, :, 0]
        return p_subs, q_subs, p_bus, q_bus, diff_v_bus

    def load_redispacthing_data(self, path, name="prods_charac.csv"):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        This method will load everything needed for the redispatching and unit commitment problem.

        We don't recommend at all to modify this function.

        Parameters
        ----------
        path: ``str``
            Location of the dataframe containing the redispatching data. This dataframe (csv, coma separated)
            should have at least the columns (other columns are ignored, order of the colums do not matter):

            - "name": identifying the name of the generator (should match the names in self.name_gen)
            - "type": one of "thermal", "nuclear", "wind", "solar" or "hydro" representing the type of the generator
            - "pmax": the maximum value the generator can produce (in MW)
            - "pmin": the minimum value the generator can produce (in MW)
            - "max_ramp_up": maximum value the generator can increase its production between two consecutive
              steps TODO make it independant from the duration of the step
            - "max_ramp_down": maximum value the generator can decrease its production between two consecutive
              steps (is positive) TODO make it independant from the duration of the step
            - "start_cost": starting cost of the generator in $ (or any currency you want)
            - "shut_down_cost": cost associated to the shut down of the generator in $ (or any currency you want)
            - "marginal_cost": "average" marginal cost of the generator. For now we don't allow it to vary across
              different steps or episode in $/(MW.time step duration) and NOT $/MWh  (TODO change that)
            - "min_up_time": minimum time a generator need to stay "connected" before we can disconnect it (
              measured in time step)  (TODO change that)
            - "min_down_time": minimum time a generator need to stay "disconnected" before we can connect it again.(
              measured in time step)  (TODO change that)

        name: ``str``
            Name of the dataframe containing the redispatching data. Defaults to 'prods_charac.csv', we don't advise
            to change it.

        """
        self._fill_names()
        self.redispatching_unit_commitment_availble = False

        # for redispatching
        fullpath = os.path.join(path, name)
        if not os.path.exists(fullpath):
            return
        try:
            df = pd.read_csv(fullpath, sep=",")
        except Exception as exc_:
            warnings.warn(
                f'Impossible to load the redispatching data for this environment with error:\n"{exc_}"\n'
                f"Redispatching will be unavailable.\n"
                f"Please make sure \"{name}\" file is a csv (coma ',') separated file."
            )
            return

        mandatory_columns = [
            "type",
            "Pmax",
            "Pmin",
            "max_ramp_up",
            "max_ramp_down",
            "start_cost",
            "shut_down_cost",
            "marginal_cost",
            "min_up_time",
            "min_down_time",
        ]
        for el in mandatory_columns:
            if el not in df.columns:
                warnings.warn(
                    f"Impossible to load the redispatching data for this environment because"
                    f"one of the mandatory column is not present ({el}). Please check the file "
                    f'"{name}" contains all the mandatory columns: {mandatory_columns}'
                )
                return

        gen_info = {}
        for _, row in df.iterrows():
            gen_info[row["name"]] = {
                "type": row["type"],
                "pmax": row["Pmax"],
                "pmin": row["Pmin"],
                "max_ramp_up": row["max_ramp_up"],
                "max_ramp_down": row["max_ramp_down"],
                "start_cost": row["start_cost"],
                "shut_down_cost": row["shut_down_cost"],
                "marginal_cost": row["marginal_cost"],
                "min_up_time": row["min_up_time"],
                "min_down_time": row["min_down_time"],
            }
        self.redispatching_unit_commitment_availble = True

        self.gen_type = np.full(self.n_gen, fill_value="aaaaaaaaaa")
        self.gen_pmin = np.full(self.n_gen, fill_value=1.0, dtype=dt_float)
        self.gen_pmax = np.full(self.n_gen, fill_value=1.0, dtype=dt_float)
        self.gen_redispatchable = np.full(self.n_gen, fill_value=False, dtype=dt_bool)
        self.gen_max_ramp_up = np.full(self.n_gen, fill_value=0.0, dtype=dt_float)
        self.gen_max_ramp_down = np.full(self.n_gen, fill_value=0.0, dtype=dt_float)
        self.gen_min_uptime = np.full(self.n_gen, fill_value=-1, dtype=dt_int)
        self.gen_min_downtime = np.full(self.n_gen, fill_value=-1, dtype=dt_int)
        self.gen_cost_per_MW = np.full(
            self.n_gen, fill_value=1.0, dtype=dt_float
        )  # marginal cost
        self.gen_startup_cost = np.full(
            self.n_gen, fill_value=1.0, dtype=dt_float
        )  # start cost
        self.gen_shutdown_cost = np.full(
            self.n_gen, fill_value=1.0, dtype=dt_float
        )  # shutdown cost
        self.gen_renewable = np.full(self.n_gen, fill_value=False, dtype=dt_bool)

        for i, gen_nm in enumerate(self.name_gen):
            try:
                tmp_gen = gen_info[gen_nm]
            except KeyError as exc_:
                raise BackendError(
                    f"Impossible to load the redispatching data. The generator {i} with name {gen_nm} "
                    f'could not be located on the description file "{name}".'
                )
            self.gen_type[i] = str(tmp_gen["type"])
            self.gen_pmin[i] = self._aux_check_finite_float(
                tmp_gen["pmin"], f' for gen. "{gen_nm}" and column "pmin"'
            )
            self.gen_pmax[i] = self._aux_check_finite_float(
                tmp_gen["pmax"], f' for gen. "{gen_nm}" and column "pmax"'
            )
            self.gen_redispatchable[i] = dt_bool(
                tmp_gen["type"] not in ["wind", "solar"]
            )
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
            self.gen_renewable[i] = dt_bool(tmp_gen["type"] in ["wind", "solar"])
            
        self.redispatching_unit_commitment_availble = True

    def load_storage_data(self, path, name="storage_units_charac.csv"):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        This method will load everything needed in presence of storage unit on the grid.

        We don't recommend at all to modify this function.

        Parameters
        ----------
        path: ``str``
            Location of the dataframe containing the storage unit data. This dataframe (csv, coma separated)
            should have at least the columns. It is mandatory to have it if there are storage units on the grid,
            but it is ignored if not:

            - "name": identifying the name of the unit storage (should match the names in self.name_storage)
            - "type": one of "battery", "pumped_storage" representing the type of the unit storage
            - "Emax": the maximum energy capacity the unit can store (in MWh)
            - "Emin": the minimum energy capacity the unit can store (in MWh) [it can be >0 if a battery cannot be
              completely empty for example]
            - "max_p_prod": maximum flow the battery can absorb in MW
            - "max_p_absorb": maximum flow the battery can produce in MW
            - "marginal_cost": cost in $ (or any currency, really) of usage of the battery.
            - "power_discharge_loss" (optional): power loss in the battery in MW (the capacity will decrease constantly
              of this amount). Set it to 0.0 to deactivate it. If not present, it is set to 0.
            - "charging_efficiency" (optional):
               Float between 0. and 1. 1. means that if the grid provides 1MW (for ex. 1MW for 1h) to the storage
               capacity, then the
               state of charge of the battery will increase of 1MWh. If this efficiency is 0.5 then if 1MWh
               if provided by the grid, then only 0.5MWh will be stored.
            - "discharging_efficiency" (optional): battery efficiency when it is discharged. 1.0 means if you want to
              get 1MWh on the grid, the battery state of charge will decrease by 1MWh. If this is 33% then it
              means if you want to get (grid point of view) 1MWh on the grid, you need to decrease the
              state of charge of 3MWh.

        name: ``str``
            Name of the dataframe containing the redispatching data. Defaults to 'prods_charac.csv', we don't advise
            to change it.

        Notes
        -----
        The battery efficiency defined as the "AC-AC" round trip efficiency is, with the convention above, defined
        as `charging_efficiency * discharging_efficency` (see
        https://www.greeningthegrid.org/news/new-resource-grid-scale-battery-storage-frequently-asked-questions-1
        for further references)
        """

        if self.n_storage == 0:
            # set the "no battery state" if there are none
            type(self).set_no_storage()
            return

        # for storage unit information
        fullpath = os.path.join(path, name)
        if not os.path.exists(fullpath):
            raise BackendError(
                f"There are storage unit on the grid, yet we could not locate their description."
                f'Please make sure to have a file "{name}" where the environment data are located.'
                f'For this environment the location is "{path}"'
            )

        try:
            df = pd.read_csv(fullpath)
        except Exception as exc_:
            raise BackendError(
                f"There are storage unit on the grid, yet we could not locate their description."
                f'Please make sure to have a file "{name}" where the environment data are located.'
                f'For this environment the location is "{path}"'
            )
        mandatory_colnames = [
            "name",
            "type",
            "Emax",
            "Emin",
            "max_p_prod",
            "max_p_absorb",
            "marginal_cost",
        ]
        for el in mandatory_colnames:
            if el not in df.columns:
                raise BackendError(
                    f"There are storage unit on the grid, yet we could not properly load their "
                    f"description. Please make sure the csv {name} contains all the columns "
                    f"{mandatory_colnames}"
                )

        stor_info = {}
        for _, row in df.iterrows():
            stor_info[row["name"]] = {
                "name": row["name"],
                "type": row["type"],
                "Emax": row["Emax"],
                "Emin": row["Emin"],
                "max_p_prod": row["max_p_prod"],
                "max_p_absorb": row["max_p_absorb"],
                "marginal_cost": row["marginal_cost"],
            }
            if "power_loss" in row:
                stor_info[row["name"]]["power_loss"] = row["power_loss"]
            else:
                stor_info[row["name"]]["power_loss"] = 0.0
            if "charging_efficiency" in row:
                stor_info[row["name"]]["charging_efficiency"] = row[
                    "charging_efficiency"
                ]
            else:
                stor_info[row["name"]]["charging_efficiency"] = 1.0
            if "discharging_efficiency" in row:
                stor_info[row["name"]]["discharging_efficiency"] = row[
                    "discharging_efficiency"
                ]
            else:
                stor_info[row["name"]]["discharging_efficiency"] = 1.0

        self.storage_type = np.full(self.n_storage, fill_value="aaaaaaaaaa")
        self.storage_Emax = np.full(self.n_storage, fill_value=1.0, dtype=dt_float)
        self.storage_Emin = np.full(self.n_storage, fill_value=0.0, dtype=dt_float)
        self.storage_max_p_prod = np.full(
            self.n_storage, fill_value=1.0, dtype=dt_float
        )
        self.storage_max_p_absorb = np.full(
            self.n_storage, fill_value=1.0, dtype=dt_float
        )
        self.storage_marginal_cost = np.full(
            self.n_storage, fill_value=1.0, dtype=dt_float
        )
        self.storage_loss = np.full(self.n_storage, fill_value=0.0, dtype=dt_float)
        self.storage_charging_efficiency = np.full(
            self.n_storage, fill_value=1.0, dtype=dt_float
        )
        self.storage_discharging_efficiency = np.full(
            self.n_storage, fill_value=1.0, dtype=dt_float
        )

        for i, sto_nm in enumerate(self.name_storage):
            try:
                tmp_sto = stor_info[sto_nm]
            except KeyError as exc_:
                raise BackendError(
                    f"Impossible to load the storage data. The storage unit {i} with name {sto_nm} "
                    f'could not be located on the description file "{name}" with error : \n'
                    f"{exc_}."
                )

            self.storage_type[i] = str(tmp_sto["type"])
            self.storage_Emax[i] = self._aux_check_finite_float(
                tmp_sto["Emax"], f' for {sto_nm} and column "Emax"'
            )
            self.storage_Emin[i] = self._aux_check_finite_float(
                tmp_sto["Emin"], f' for {sto_nm} and column "Emin"'
            )
            self.storage_max_p_prod[i] = self._aux_check_finite_float(
                tmp_sto["max_p_prod"], f' for {sto_nm} and column "max_p_prod"'
            )
            self.storage_max_p_absorb[i] = self._aux_check_finite_float(
                tmp_sto["max_p_absorb"], f' for {sto_nm} and column "max_p_absorb"'
            )
            self.storage_marginal_cost[i] = self._aux_check_finite_float(
                tmp_sto["marginal_cost"], f' for {sto_nm} and column "marginal_cost"'
            )
            self.storage_loss[i] = self._aux_check_finite_float(
                tmp_sto["power_loss"], f' for {sto_nm} and column "power_loss"'
            )
            self.storage_charging_efficiency[i] = self._aux_check_finite_float(
                tmp_sto["charging_efficiency"],
                f' for {sto_nm} and column "charging_efficiency"',
            )
            self.storage_discharging_efficiency[i] = self._aux_check_finite_float(
                tmp_sto["discharging_efficiency"],
                f' for {sto_nm} and column "discharging_efficiency"',
            )

    def _aux_check_finite_float(self, nb_, str_=""):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        check and returns if correct that a number is convertible to `dt_float` and that it's finite

        """
        tmp = dt_float(nb_)
        if not np.isfinite(tmp):
            raise BackendError(
                f"Infinite number met for a number that should be finite. Please check your data {str_}"
            )
        return tmp

    def load_grid_layout(self, path, name="grid_layout.json"):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        We don't recommend at all to modify this function.

        This function loads the layout (eg the coordinates of each substation) for the powergrid.

        Parameters
        ----------
        path: ``str``
            TODO
        name: ``str``
            TODO

        """
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
            if el not in dict_:
                return Exception("substation named {} not in layout".format(el))
            tmp = dict_[el]
            try:
                x, y = tmp
                x = dt_float(x)
                y = dt_float(y)
                new_grid_layout[el] = (x, y)
            except Exception as e_:
                return Exception(
                    "fail to convert coordinates for {} into list of coordinates with error {}"
                    "".format(el, e_)
                )

        self.attach_layout(grid_layout=new_grid_layout)
        return None

    def _aux_get_line_status_to_set(self, line_status):
        line_status = 2 * line_status - 1
        line_status = line_status.astype(dt_int)
        return line_status

    def get_action_to_set(self):
        """
        Get the action to set another backend to represent the internal state of this current backend.

        It handles also the information about the shunts if available

        Returns
        -------
        res: :class:`grid2op.Action.CompleteAction`
            The complete action to set a backend to the internal state of `self`

        """
        line_status = self._aux_get_line_status_to_set(self.get_line_status())
        topo_vect = self.get_topo_vect()
        if np.all(topo_vect == -1):
            raise RuntimeError(
                "The get_action_to_set should not be used after a divergence of the powerflow"
            )
        prod_p, _, prod_v = self.generators_info()
        load_p, load_q, _ = self.loads_info()
        set_me = self._complete_action_class()
        dict_ = {
            "set_line_status": line_status,
            "set_bus": 1 * topo_vect,
            "injection": {
                "prod_p": prod_p,
                "prod_v": prod_v,
                "load_p": load_p,
                "load_q": load_q,
            },
        }

        if self.shunts_data_available:
            p_s, q_s, sh_v, bus_s = self.shunt_info()
            dict_["shunt"] = {"shunt_bus": bus_s}
            if np.sum(bus_s >= 1):
                p_s *= (self._sh_vnkv / sh_v) ** 2
                q_s *= (self._sh_vnkv / sh_v) ** 2
                p_s[bus_s == -1] = np.NaN
                q_s[bus_s == -1] = np.NaN
                dict_["shunt"]["shunt_p"] = p_s
                dict_["shunt"]["shunt_q"] = q_s

        if self.n_storage > 0:
            sto_p, *_ = self.storages_info()
            dict_["set_storage"] = 1.0 * sto_p

        set_me.update(dict_)
        return set_me

    def update_from_obs(self, obs, force_update=False):
        """
        Takes an observation as input and update the internal state of `self` to match the state of the backend
        that produced this observation.

        Only the "line_status", "topo_vect", "prod_p", "prod_v", "load_p" and "load_q" attributes of the
        observations are used.

        Notes
        -----
        If the observation is not perfect (for example with noise, or partial) this method will not work. You need
        to pass it a complete observation.

        For example, you might want to consider to have a state estimator if that is the case.

        Parameters
        ----------
        obs: :class:`grid2op.Observation.CompleteObservation`
            A complete observation describing the state of the grid you want this backend to be in.

        """
        # lazy loading to prevent circular references
        from grid2op.Observation import CompleteObservation

        if (not force_update) and (not isinstance(obs, CompleteObservation)):
            raise BackendError(
                "Impossible to set a backend to a state not represented by a "
                '"grid2op.Observation.CompleteObservation".'
            )

        backend_action = self.my_bk_act_class()
        act = self._complete_action_class()
        line_status = self._aux_get_line_status_to_set(obs.line_status)
        # skip the action part and update directly the backend action !
        dict_ = {
            "set_bus": obs.topo_vect,
            "set_line_status": line_status,
            "injection": {
                "prod_p": obs.prod_p,
                "prod_v": obs.prod_v,
                "load_p": obs.load_p,
                "load_q": obs.load_q,
            },
        }

        if self.shunts_data_available and obs.shunts_data_available:
            if "_shunt_bus" not in type(obs).attr_list_set:
                raise BackendError(
                    "Impossible to set the backend to the state given by the observation: shunts data "
                    "are not present in the observation."
                )

            dict_["shunt"] = {"shunt_bus": obs._shunt_bus}
            shunt_co = obs._shunt_bus >= 1
            if np.sum(shunt_co):
                mults = (self._sh_vnkv / obs._shunt_v) ** 2
                sh_p = obs._shunt_p * mults
                sh_q = obs._shunt_q * mults
                sh_p[~shunt_co] = np.NaN
                sh_q[~shunt_co] = np.NaN
                dict_["shunt"]["shunt_p"] = sh_p
                dict_["shunt"]["shunt_q"] = sh_q
            act.update(dict_)
        backend_action += act
        self.apply_action(backend_action)

    def assert_grid_correct(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            This is done as it should be by the Environment
        """
        # lazy loading
        from grid2op.Action import CompleteAction
        from grid2op.Action._BackendAction import _BackendAction

        orig_type = type(self)
        if orig_type.my_bk_act_class is None:
            # class is already initialized
            # and set up the proper class and everything
            self._init_class_attr()

            # hack due to changing class of imported module in the module itself
            self.__class__ = type(self).init_grid(
                type(self), force_module=type(self).__module__
            )
            setattr(
                sys.modules[type(self).__module__],
                self.__class__.__name__,
                self.__class__,
            )
            # reset the attribute of the grid2op.Backend.Backend class
            # that can be messed up with depending on the initialization of the backend
            Backend._clear_class_attribute()
            orig_type._clear_class_attribute()

        my_cls = type(self)
        my_cls.my_bk_act_class = _BackendAction.init_grid(my_cls)
        my_cls._complete_action_class = CompleteAction.init_grid(my_cls)
        my_cls._complete_action_class._add_shunt_data()
        my_cls._complete_action_class._update_value_set()
        my_cls.assert_grid_correct_cls()

    def assert_grid_correct_after_powerflow(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            This is done as it should be by the Environment

        This method is called by the environment. It ensure that the backend remains consistent even after a powerflow
        has be run with :func:`Backend.runpf` method.

        :raise: :class:`grid2op.Exceptions.EnvError` and possibly all of its derived class.
        """

        # test the results gives the proper size
        tmp = self.get_line_status()
        if tmp.shape[0] != self.n_line:
            raise IncorrectNumberOfLines('returned by "backend.get_line_status()"')
        if np.any(~np.isfinite(tmp)):
            raise EnvironmentError(type(self).ERR_INIT_POWERFLOW)
        tmp = self.get_line_flow()
        if tmp.shape[0] != self.n_line:
            raise IncorrectNumberOfLines('returned by "backend.get_line_flow()"')
        if np.any(~np.isfinite(tmp)):
            raise EnvironmentError(type(self).ERR_INIT_POWERFLOW)
        tmp = self.get_thermal_limit()
        if tmp.shape[0] != self.n_line:
            raise IncorrectNumberOfLines('returned by "backend.get_thermal_limit()"')
        if np.any(~np.isfinite(tmp)):
            raise EnvironmentError(type(self).ERR_INIT_POWERFLOW)
        tmp = self.get_line_overflow()
        if tmp.shape[0] != self.n_line:
            raise IncorrectNumberOfLines('returned by "backend.get_line_overflow()"')
        if np.any(~np.isfinite(tmp)):
            raise EnvironmentError(type(self).ERR_INIT_POWERFLOW)

        tmp = self.generators_info()
        if len(tmp) != 3:
            raise EnvError(
                '"generators_info()" should return a tuple with 3 elements: p, q and v'
            )
        for el in tmp:
            if el.shape[0] != self.n_gen:
                raise IncorrectNumberOfGenerators(
                    'returned by "backend.generators_info()"'
                )
        tmp = self.loads_info()
        if len(tmp) != 3:
            raise EnvError(
                '"loads_info()" should return a tuple with 3 elements: p, q and v'
            )
        for el in tmp:
            if el.shape[0] != self.n_load:
                raise IncorrectNumberOfLoads('returned by "backend.loads_info()"')
        tmp = self.lines_or_info()
        if len(tmp) != 4:
            raise EnvError(
                '"lines_or_info()" should return a tuple with 4 elements: p, q, v and a'
            )
        for el in tmp:
            if el.shape[0] != self.n_line:
                raise IncorrectNumberOfLines('returned by "backend.lines_or_info()"')
        tmp = self.lines_ex_info()
        if len(tmp) != 4:
            raise EnvError(
                '"lines_ex_info()" should return a tuple with 4 elements: p, q, v and a'
            )
        for el in tmp:
            if el.shape[0] != self.n_line:
                raise IncorrectNumberOfLines('returned by "backend.lines_ex_info()"')

        if self.n_storage > 0:
            tmp = self.storages_info()
            if len(tmp) != 3:
                raise EnvError(
                    '"storages_info()" should return a tuple with 3 elements: p, q and v'
                )
            for el in tmp:
                if el.shape[0] != self.n_storage:
                    raise IncorrectNumberOfLines(
                        'returned by "backend.storages_info()"'
                    )

        tmp = self.get_topo_vect()
        if tmp.shape[0] != np.sum(self.sub_info):
            raise IncorrectNumberOfElements('returned by "backend.get_topo_vect()"')

        if np.any(~np.isfinite(tmp)):
            raise EnvError(
                'Some components of "backend.get_topo_vect()" are not finite. This should be integer.'
            )
