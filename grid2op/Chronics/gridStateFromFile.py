# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import copy
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta

from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Exceptions import (
    IncorrectNumberOfElements,
    ChronicsError,
    ChronicsNotFoundError,
)
from grid2op.Exceptions import (
    IncorrectNumberOfLoads,
    IncorrectNumberOfGenerators,
    IncorrectNumberOfLines,
)
from grid2op.Exceptions import EnvError, InsufficientData
from grid2op.Chronics.gridValue import GridValue


class GridStateFromFile(GridValue):
    """
    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Do not attempt to create an object of this class. This is initialized by the environment
        at its creation.

    Read the injections values from a file stored on hard drive. More detailed about the files is provided in the
    :func:`GridStateFromFile.initialize` method.

    This class reads only files stored as csv. The header of the csv is mandatory and should represent the name of
    the objects. This names should either be matched to the name of the same object in the backend using the
    `names_chronics_to_backend` argument pass into the :func:`GridStateFromFile.initialize` (see
    :func:`GridValue.initialize` for more information) or match the names of the object in the backend.

    When the grid value is initialized, all present csv are read, sorted in order compatible with the backend and
    extracted as numpy array.

    For now, the current date and times are not read from file. It is mandatory that the chronics starts at 00:00 and
    its first time stamps is corresponds to January, 1st 2019.

    Chronics read from this files don't implement the "forecast" value.

    In this values, only 1 episode is stored. If the end of the episode is reached and another one should start, then
    it will loop from the beginning.

    It reads the following files from the "path" location specified:

    - "prod_p.csv": for each time steps, this file contains the value for the active production of
      each generators of the grid (it counts as many rows as the number of time steps - and its header)
      and as many columns as the number of generators on the grid. The header must contains the names of
      the generators used to map their value on the grid. Values must be convertible to floating point and the
      column separator of this file should be semi-colon `;` (unless you specify a "sep" when loading this class)
    - "prod_v.csv": same as "prod_p.csv" but for the production voltage setpoint.
    - "load_p.csv": same as "prod_p.csv" but for the load active value (number of columns = number of loads)
    - "load_q.csv": same as "prod_p.csv" but for the load reactive value (number of columns = number of loads)
    - "maintenance.csv": that contains whether or not there is a maintenance for a given powerline (column) at
      each time step (row).
    - "hazards.csv": that contains whether or not there is a hazard for a given powerline (column) at
      each time step (row).
    - "start_datetime.info": the time stamp (date and time) at which the chronic is starting.
    - "time_interval.info": the amount of time between two consecutive steps (*e.g.* 5 mins, or 1h)

    If a file is missing, it is understood as "this value will not be modified". For example, if the file
    "prod_v.csv" is not present, it will be equivalent as not modifying the production voltage setpoint, never.

    Except if the attribute :attr:`GridStateFromFile.sep` is modified, the above tables should be "semi colon" (;)
    separated.

    Attributes
    ----------
    path: ``str``
        The path of the folder where the data are stored. It is recommended to set absolute path, and not relative
        paths.

    load_p: ``numpy.ndarray``, dtype: ``float``
        All the values of the load active values

    load_q: ``numpy.ndarray``, dtype: ``float``
        All the values of the load reactive values

    prod_p: ``numpy.ndarray``, dtype: ``float``
        All the productions setpoint active values.

    prod_v: ``numpy.ndarray``, dtype: ``float``
        All the productions setpoint voltage magnitude values.

    hazards: ``numpy.ndarray``, dtype: ``bool``
        This vector represents the possible hazards. It is understood as: ``True`` there is a hazard
        for the given powerline, ``False`` there is not.

    maintenance: ``numpy.ndarray``, dtype: ``bool``
        This vector represents the possible maintenance. It is understood as: ``True`` there is a maintenance
        for the given powerline, ``False`` there is not.

    current_index: ``int``
        The index of the last observation sent to the :class:`grid2op.Environment`.

    sep: ``str``, optional
        The csv columns separator. By defaults it's ";"

    names_chronics_to_backend: ``dict``
        This directory matches the name of the objects (line extremity, generator or load) to the same object in the
        backed. See the help of :func:`GridValue.initialize` for more information).
    """
    MULTI_CHRONICS = False

    def __init__(
        self,
        path,
        sep=";",
        time_interval=timedelta(minutes=5),
        max_iter=-1,
        start_datetime=datetime(year=2019, month=1, day=1),
        chunk_size=None,
    ):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Do not attempt to create an object of this class. This is initialized by the environment
        at its creation.


        Build an instance of GridStateFromFile. Such an instance should be built before an :class:`grid2op.Environment`
        is created.

        Parameters
        ----------
        path: ``str``
            Used to initialize :attr:`GridStateFromFile.path`

        sep: ``str``, optional
            Used to initialize :attr:`GridStateFromFile.sep`

        time_interval: ``datetime.timedelta``
            Used to initialize :attr:`GridValue.time_interval`

        max_iter: int, optional
            Used to initialize :attr:`GridValue.max_iter`

        """
        GridValue.__init__(
            self,
            time_interval=time_interval,
            max_iter=max_iter,
            start_datetime=start_datetime,
            chunk_size=chunk_size,
        )

        self.path = path
        self.n_ = None  # maximum number of rows of the array
        self.tmp_max_index = None  # size maximum of the current tables in memory
        self.load_p = None  # numpy array corresponding to the current active load values in the power _grid. It has the same size as the number of loads
        self.load_q = None  # numpy array corresponding to the current reactive load values in the power _grid. It has the same size as the number of loads
        self.prod_p = None  # numpy array corresponding to the current active production values in the power _grid. It has the same size as the number of generators
        self.prod_v = None  # numpy array corresponding to the current voltage production setpoint values in the power _grid. It has the same size as the number of generators

        # for the two following vector, the convention is the following: False(line is disconnected) / True(line is connected)
        self.hazards = None  # numpy array representing the outage (unplanned), same size as the number of powerlines on the _grid.
        self.maintenance = None  # numpy array representing the _maintenance (planned withdrawal of a powerline), same size as the number of powerlines on the _grid.
        self.maintenance_time = None
        self.maintenance_duration = None

        self.current_index = -1
        self.sep = sep

        self.names_chronics_to_backend = None

        # added to provide an easier access to read data in chunk
        self.chunk_size = chunk_size
        self._data_chunk = {}
        self._order_load_p = None
        self._order_load_q = None
        self._order_prod_p = None
        self._order_prod_v = None
        self._order_hazards = None
        self._order_maintenance = None

        # order of the names in the backend
        self._order_backend_loads = None
        self._order_backend_prods = None
        self._order_backend_lines = None

    def _clear(self):        
        self.n_ = None  # maximum number of rows of the array
        self.tmp_max_index = None  # size maximum of the current tables in memory
        self.load_p = None  # numpy array corresponding to the current active load values in the power _grid. It has the same size as the number of loads
        self.load_q = None  # numpy array corresponding to the current reactive load values in the power _grid. It has the same size as the number of loads
        self.prod_p = None  # numpy array corresponding to the current active production values in the power _grid. It has the same size as the number of generators
        self.prod_v = None  # numpy array corresponding to the current voltage production setpoint values in the power _grid. It has the same size as the number of generators

        # for the two following vector, the convention is the following: False(line is disconnected) / True(line is connected)
        self.hazards = None  # numpy array representing the outage (unplanned), same size as the number of powerlines on the _grid.
        self.maintenance = None  # numpy array representing the _maintenance (planned withdrawal of a powerline), same size as the number of powerlines on the _grid.
        self.maintenance_time = None
        self.maintenance_duration = None

        self.current_index = -1

        self.names_chronics_to_backend = None

        # added to provide an easier access to read data in chunk
        self._data_chunk = {}
        self._order_load_p = None
        self._order_load_q = None
        self._order_prod_p = None
        self._order_prod_v = None
        self._order_hazards = None
        self._order_maintenance = None

        # order of the names in the backend
        self._order_backend_loads = None
        self._order_backend_prods = None
        self._order_backend_lines = None
        
    def _assert_correct(self, dict_convert, order_backend):
        len_backend = len(order_backend)
        len_dict_keys = len(dict_convert)
        vals = set(dict_convert.values())
        lend_dict_values = len(vals)

        if len_dict_keys != len_backend:
            err_msg = "Conversion mismatch between backend data {} elements and converter data {} (keys)"
            raise IncorrectNumberOfElements(err_msg.format(len_backend, len_dict_keys))
        if lend_dict_values != len_backend:
            err_msg = "Conversion mismatch between backend data {} elements and converter data {} (values)"
            raise IncorrectNumberOfElements(
                err_msg.format(len_backend, lend_dict_values)
            )

        for el in order_backend:
            if not el in vals:
                raise ChronicsError(
                    'Impossible to find element "{}" in the original converter data'.format(
                        el
                    )
                )

    def _assert_correct_second_stage(self, pandas_name, dict_convert, key, extra=""):
        for i, el in enumerate(pandas_name):
            if not el in dict_convert[key]:
                raise ChronicsError(
                    "Element named {} is found in the data (column {}) but it is not found on the "
                    'powergrid for data of type "{}".\nData in files  are: {}\n'
                    "Converter data are: {}".format(
                        el,
                        i + 1,
                        key,
                        sorted(list(pandas_name)),
                        sorted(list(dict_convert[key].keys())),
                    )
                )

    def _init_date_time(self):
        if os.path.exists(os.path.join(self.path, "start_datetime.info")):
            with open(os.path.join(self.path, "start_datetime.info"), "r") as f:
                a = f.read().rstrip().lstrip()
            try:
                tmp = datetime.strptime(a, "%Y-%m-%d %H:%M")
            except ValueError:
                tmp = datetime.strptime(a, "%Y-%m-%d")
            except Exception:
                raise ChronicsNotFoundError(
                    'Impossible to understand the content of "start_datetime.info". Make sure '
                    'it\'s composed of only one line with a datetime in the "%Y-%m-%d %H:%M"'
                    "format."
                )
            self.start_datetime = tmp
            self.current_datetime = tmp

        if os.path.exists(os.path.join(self.path, "time_interval.info")):
            with open(os.path.join(self.path, "time_interval.info"), "r") as f:
                a = f.read().rstrip().lstrip()
            try:
                tmp = datetime.strptime(a, "%H:%M")
            except ValueError:
                tmp = datetime.strptime(a, "%M")
            except Exception:
                raise ChronicsNotFoundError(
                    'Impossible to understand the content of "time_interval.info". Make sure '
                    'it\'s composed of only one line with a datetime in the "%H:%M"'
                    "format."
                )
            self.time_interval = timedelta(hours=tmp.hour, minutes=tmp.minute)

    def _get_fileext(self, data_name):
        read_compressed = ".csv"
        if not os.path.exists(os.path.join(self.path, "{}.csv".format(data_name))):
            # try to read compressed data
            if os.path.exists(os.path.join(self.path, "{}.csv.bz2".format(data_name))):
                read_compressed = ".csv.bz2"
            elif os.path.exists(os.path.join(self.path, "{}.zip".format(data_name))):
                read_compressed = ".zip"
            elif os.path.exists(
                os.path.join(self.path, "{}.csv.gzip".format(data_name))
            ):
                read_compressed = ".csv.gzip"
            elif os.path.exists(os.path.join(self.path, "{}.csv.xz".format(data_name))):
                read_compressed = ".csv.xz"
            else:
                read_compressed = None
                # raise ChronicsNotFoundError(
                #     "GridStateFromFile: unable to locate the data files that should be at \"{}\"".format(self.path))
        return read_compressed

    def _get_data(self, data_name, chunksize=-1, nrows=None):
        file_ext = self._get_fileext(data_name)
        
        if nrows is None:
            if self._max_iter > 0:
                nrows = self._max_iter + 1
            
        if file_ext is not None:
            if chunksize == -1:
                chunksize = self.chunk_size
            res = pd.read_csv(
                os.path.join(self.path, "{}{}".format(data_name, file_ext)),
                sep=self.sep,
                chunksize=chunksize,
                nrows=nrows,
            )
        else:
            res = None
        return res

    def _get_orders(
        self,
        load_p,
        load_q,
        prod_p,
        prod_v,
        hazards,
        maintenance,
        order_backend_loads,
        order_backend_prods,
        order_backend_lines,
    ):

        order_chronics_load_p = None
        order_backend_load_q = None
        order_backend_prod_p = None
        order_backend_prod_v = None
        order_backend_hazards = None
        order_backend_maintenance = None

        if load_p is not None:
            self._assert_correct_second_stage(
                load_p.columns, self.names_chronics_to_backend, "loads", "active"
            )
            order_chronics_load_p = np.array(
                [
                    order_backend_loads[self.names_chronics_to_backend["loads"][el]]
                    for el in load_p.columns
                ]
            ).astype(dt_int)
        if load_q is not None:
            self._assert_correct_second_stage(
                load_q.columns, self.names_chronics_to_backend, "loads", "reactive"
            )
            order_backend_load_q = np.array(
                [
                    order_backend_loads[self.names_chronics_to_backend["loads"][el]]
                    for el in load_q.columns
                ]
            ).astype(dt_int)

        if prod_p is not None:
            self._assert_correct_second_stage(
                prod_p.columns, self.names_chronics_to_backend, "prods", "active"
            )
            order_backend_prod_p = np.array(
                [
                    order_backend_prods[self.names_chronics_to_backend["prods"][el]]
                    for el in prod_p.columns
                ]
            ).astype(dt_int)

        if prod_v is not None:
            self._assert_correct_second_stage(
                prod_v.columns,
                self.names_chronics_to_backend,
                "prods",
                "voltage magnitude",
            )
            order_backend_prod_v = np.array(
                [
                    order_backend_prods[self.names_chronics_to_backend["prods"][el]]
                    for el in prod_v.columns
                ]
            ).astype(dt_int)

        if hazards is not None:
            self._assert_correct_second_stage(
                hazards.columns, self.names_chronics_to_backend, "lines", "hazards"
            )
            order_backend_hazards = np.array(
                [
                    order_backend_lines[self.names_chronics_to_backend["lines"][el]]
                    for el in hazards.columns
                ]
            ).astype(dt_int)

        if maintenance is not None:
            self._assert_correct_second_stage(
                maintenance.columns,
                self.names_chronics_to_backend,
                "lines",
                "maintenance",
            )
            order_backend_maintenance = np.array(
                [
                    order_backend_lines[self.names_chronics_to_backend["lines"][el]]
                    for el in maintenance.columns
                ]
            ).astype(dt_int)

        return (
            order_chronics_load_p,
            order_backend_load_q,
            order_backend_prod_p,
            order_backend_prod_v,
            order_backend_hazards,
            order_backend_maintenance,
        )

    def _get_next_chunk(self):
        load_p = None
        load_q = None
        prod_p = None
        prod_v = None
        if self._data_chunk["load_p"] is not None:
            load_p = next(self._data_chunk["load_p"])
            self.tmp_max_index = load_p.shape[0]
        if self._data_chunk["load_q"] is not None:
            load_q = next(self._data_chunk["load_q"])
            self.tmp_max_index = load_q.shape[0]
        if self._data_chunk["prod_p"] is not None:
            prod_p = next(self._data_chunk["prod_p"])
            self.tmp_max_index = prod_p.shape[0]
        if self._data_chunk["prod_v"] is not None:
            prod_v = next(self._data_chunk["prod_v"])
            self.tmp_max_index = prod_v.shape[0]
        return load_p, load_q, prod_p, prod_v

    def initialize(
        self,
        order_backend_loads,
        order_backend_prods,
        order_backend_lines,
        order_backend_subs,
        names_chronics_to_backend=None,
    ):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

            Called at the creation of the environment.

        In this function, the numpy arrays are read from the csv using the panda.dataframe engine.

        In order to be valid, the folder located at :attr:`GridStateFromFile.path` can contain:

          - a file named "load_p.csv" used to initialize :attr:`GridStateFromFile.load_p`
          - a file named "load_q.csv" used to initialize :attr:`GridStateFromFile.load_q`
          - a file named "prod_p.csv" used to initialize :attr:`GridStateFromFile.prod_p`
          - a file named "prod_v.csv" used to initialize :attr:`GridStateFromFile.prod_v`
          - a file named "hazards.csv" used to initialize :attr:`GridStateFromFile.hazards`
          - a file named "maintenance.csv" used to initialize :attr:`GridStateFromFile.maintenance`

        All these csv must have the same separator specified by :attr:`GridStateFromFile.sep`.
        If one of these file is missing, it is equivalent to "change nothing" class.

        If a file named "start_datetime.info" is present, then it will be used to initialized
        :attr:`GridStateFromFile.start_datetime`. If this file exists, it should count only one row, with the
        initial datetime in the "%Y-%m-%d %H:%M" format.

        If a file named "time_interval.info" is present, then it will be used to initialized the
        :attr:`GridStateFromFile.time_interval` attribute.  If this file exists, it should count only one row, with the
        initial datetime in the "%H:%M" format. Only timedelta composed of hours and minutes are supported (time delta
        cannot go above 23 hours 55 minutes and cannot be smaller than 0 hour 1 minutes)

        The first row of these csv is understood as the name of the object concerned by the column. Either this name is
        present in the :class:`grid2op.Backend`, in this case no modification is performed, or in case the name
        is not found in the backend and in this case it must be specified in the "names_chronics_to_backend"
        parameters how to understand it. See the help of :func:`GridValue.initialize` for more information
        about this dictionnary.

        All files should have the same number of rows.

        Parameters
        ----------
        See help of :func:`GridValue.initialize` for a detailed help about the parameters.

        """
        self.n_gen = len(order_backend_prods)
        self.n_load = len(order_backend_loads)
        self.n_line = len(order_backend_lines)

        self._order_backend_loads = order_backend_loads
        self._order_backend_prods = order_backend_prods
        self._order_backend_lines = order_backend_lines

        self.names_chronics_to_backend = copy.deepcopy(names_chronics_to_backend)
        if self.names_chronics_to_backend is None:
            self.names_chronics_to_backend = {}
        if not "loads" in self.names_chronics_to_backend:
            self.names_chronics_to_backend["loads"] = {
                k: k for k in order_backend_loads
            }
        else:
            self._assert_correct(
                self.names_chronics_to_backend["loads"], order_backend_loads
            )
        if not "prods" in self.names_chronics_to_backend:
            self.names_chronics_to_backend["prods"] = {
                k: k for k in order_backend_prods
            }
        else:
            self._assert_correct(
                self.names_chronics_to_backend["prods"], order_backend_prods
            )
        if not "lines" in self.names_chronics_to_backend:
            self.names_chronics_to_backend["lines"] = {
                k: k for k in order_backend_lines
            }
        else:
            self._assert_correct(
                self.names_chronics_to_backend["lines"], order_backend_lines
            )
        if not "subs" in self.names_chronics_to_backend:
            self.names_chronics_to_backend["subs"] = {k: k for k in order_backend_subs}
        else:
            self._assert_correct(
                self.names_chronics_to_backend["subs"], order_backend_subs
            )

        self._init_date_time()

        # read the data
        load_p_iter = self._get_data("load_p")
        load_q_iter = self._get_data("load_q")
        prod_p_iter = self._get_data("prod_p")
        prod_v_iter = self._get_data("prod_v")
        read_compressed = self._get_fileext("hazards")
        nrows = None
        if self._max_iter > 0:
            nrows = self._max_iter + 1

        if read_compressed is not None:
            hazards = pd.read_csv(
                os.path.join(self.path, "hazards{}".format(read_compressed)),
                sep=self.sep,
                nrows=nrows,
            )
        else:
            hazards = None

        read_compressed = self._get_fileext("maintenance")
        if read_compressed is not None:
            maintenance = pd.read_csv(
                os.path.join(self.path, "maintenance{}".format(read_compressed)),
                sep=self.sep,
                nrows=nrows,
            )
        else:
            maintenance = None

        # put the proper name in order
        order_backend_loads = {el: i for i, el in enumerate(order_backend_loads)}
        order_backend_prods = {el: i for i, el in enumerate(order_backend_prods)}
        order_backend_lines = {el: i for i, el in enumerate(order_backend_lines)}

        if self.chunk_size is None:
            load_p = load_p_iter
            load_q = load_q_iter
            prod_p = prod_p_iter
            prod_v = prod_v_iter
            if load_p is not None:
                self.tmp_max_index = load_p.shape[0]
            elif load_q is not None:
                self.tmp_max_index = load_q.shape[0]
            elif prod_p is not None:
                self.tmp_max_index = prod_p.shape[0]
            elif prod_v is not None:
                self.tmp_max_index = prod_v.shape[0]
            else:
                raise ChronicsError(
                    'No files are found in directory "{}". If you don\'t want to load any chronics,'
                    ' use  "ChangeNothing" and not "{}" to load chronics.'
                    "".format(self.path, type(self))
                )

        else:
            self._data_chunk = {
                "load_p": load_p_iter,
                "load_q": load_q_iter,
                "prod_p": prod_p_iter,
                "prod_v": prod_v_iter,
            }
            load_p, load_q, prod_p, prod_v = self._get_next_chunk()

        # get the chronics in order
        (
            order_chronics_load_p,
            order_backend_load_q,
            order_backend_prod_p,
            order_backend_prod_v,
            order_backend_hazards,
            order_backend_maintenance,
        ) = self._get_orders(
            load_p,
            load_q,
            prod_p,
            prod_v,
            hazards,
            maintenance,
            order_backend_loads,
            order_backend_prods,
            order_backend_lines,
        )

        # now "sort" the columns of each chunk of data
        self._order_load_p = np.argsort(order_chronics_load_p)
        self._order_load_q = np.argsort(order_backend_load_q)
        self._order_prod_p = np.argsort(order_backend_prod_p)
        self._order_prod_v = np.argsort(order_backend_prod_v)
        self._order_hazards = np.argsort(order_backend_hazards)
        self._order_maintenance = np.argsort(order_backend_maintenance)

        # retrieve total number of rows
        if maintenance is not None:
            n_ = maintenance.shape[0]
        elif hazards is not None:
            n_ = hazards.shape[0]
        else:
            n_ = None
            for fn in ["prod_p", "load_p", "prod_v", "load_q"]:
                ext_ = self._get_fileext(fn)
                if ext_ is not None:
                    n_ = self._file_len(
                        os.path.join(self.path, "{}{}".format(fn, ext_)), ext_
                    )
                    break
            if n_ is None:
                raise ChronicsError(
                    'No files are found in directory "{}". If you don\'t want to load any chronics,'
                    ' use  "ChangeNothing" and not "{}" to load chronics.'
                    "".format(self.path, type(self))
                )
        self.n_ = n_  # the -1 is present because the initial grid state doesn't count as a "time step"

        if self._max_iter > 0:
            if self.n_ is not None:
                if self._max_iter >= self.n_:
                    self._max_iter = self.n_ - 1
                    # TODO: issue warning in this case
            self.n_ = self._max_iter + 1
        else:
            # if the number of maximum time step is not set yet, we set it to be the number of
            # data in the chronics (number of rows of the files) -1.
            # the -1 is present because the initial grid state doesn't count as a "time step" but is read
            # from these data.
            self._max_iter = self.n_ - 1

        self._init_attrs(
            load_p, load_q, prod_p, prod_v, hazards=hazards, maintenance=maintenance,
            is_init=True
        )

        self.curr_iter = 0

    @staticmethod
    def _file_len(fname, ext_):
        res = pd.read_csv(fname, sep="@", dtype=str).shape[0]
        return res

    def _init_attrs(
        self, load_p, load_q, prod_p, prod_v, hazards=None, maintenance=None,
        is_init=False
    ):
        # this called at the initialization but also each time more data should
        # be read from the disk (at the end of each chunk for example)
        self.load_p = None
        self.load_q = None
        self.prod_p = None
        self.prod_v = None
        if is_init:
            self.hazards = None
            self.hazard_duration = None
            self.maintenance = None
            self.maintenance_time = None
            self.maintenance_duration = None

        if load_p is not None:
            self.load_p = copy.deepcopy(
                load_p.values[:, self._order_load_p].astype(dt_float)
            )
        if load_q is not None:
            self.load_q = copy.deepcopy(
                load_q.values[:, self._order_load_q].astype(dt_float)
            )
        if prod_p is not None:
            self.prod_p = copy.deepcopy(
                prod_p.values[:, self._order_prod_p].astype(dt_float)
            )
        if prod_v is not None:
            self.prod_v = copy.deepcopy(
                prod_v.values[:, self._order_prod_v].astype(dt_float)
            )

        # TODO optimize this piece of code, and the whole laoding process if hazards.csv and maintenance.csv are
        # provided in the proper format.
        if hazards is not None:
            # hazards and maintenance cannot be computed by chunk. So we need to differenciate their behaviour
            self.hazards = copy.deepcopy(hazards.values[:, self._order_hazards])
            self.hazard_duration = np.zeros(
                shape=(self.hazards.shape[0], self.n_line), dtype=dt_int
            )
            for line_id in range(self.n_line):
                self.hazard_duration[:, line_id] = self.get_hazard_duration_1d(
                    self.hazards[:, line_id]
                )

            self.hazards = np.abs(self.hazards) >= 1e-7
        if maintenance is not None:
            self.maintenance = copy.deepcopy(
                maintenance.values[:, self._order_maintenance]
            )
            self.maintenance_time = (
                np.zeros(shape=(self.maintenance.shape[0], self.n_line), dtype=dt_int)
                - 1
            )
            self.maintenance_duration = np.zeros(
                shape=(self.maintenance.shape[0], self.n_line), dtype=dt_int
            )

            # test that with chunk size
            for line_id in range(self.n_line):
                self.maintenance_time[:, line_id] = self.get_maintenance_time_1d(
                    self.maintenance[:, line_id]
                )
                self.maintenance_duration[
                    :, line_id
                ] = self.get_maintenance_duration_1d(self.maintenance[:, line_id])

            # there are _maintenance and hazards only if the value in the file is not 0.
            self.maintenance = np.abs(self.maintenance) >= 1e-7
            self.maintenance = self.maintenance.astype(dt_bool)

    def done(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Compare to :func:`GridValue.done` an episode can be over for 2 main reasons:

          - :attr:`GridValue.max_iter` has been reached
          - There are no data in the csv.

        The episode is done if one of the above condition is met.

        Returns
        -------
        res: ``bool``
            Whether the episode has reached its end or not.

        """
        res = False
        # if self.current_index+1 >= self.tmp_max_index:
        if self.current_index > self.n_:
            res = True
        elif self._max_iter > 0:
            if self.curr_iter > self._max_iter:
                res = True
        return res

    @property
    def max_iter(self):
        return self._max_iter
    
    @max_iter.setter
    def max_iter(self, value : int):
        if value == -1:
            self._max_iter = self.n_ - 1
        else:
            self._max_iter = int(value)
        
    def max_timestep(self):
        if self._max_iter == -1:
            return self.n_ - 1
        return self._max_iter 
    
    def _data_in_memory(self):
        if self.chunk_size is None:
            # if i don't use chunk, all the data are in memory alreay
            return True
        if self.current_index == 0:
            # data are loaded the first iteration
            return True
        if self.current_index % self.chunk_size != 0:
            # data are already in ram
            return True
        return False

    def _load_next_chunk_in_memory(self):
        # print("I loaded another chunk")
        # i load the next chunk as dataframes
        load_p, load_q, prod_p, prod_v = self._get_next_chunk()
        # i put these dataframes in the right order (columns)
        self._init_attrs(load_p, load_q, prod_p, prod_v)
        # i don't forget to reset the reading index to 0
        self.current_index = 0

    def load_next(self):
        self.current_index += 1  # index in the chunk
        # for the "global" index use self.curr_iter

        if not self._data_in_memory():
            try:
                self._load_next_chunk_in_memory()
            except StopIteration as exc_:
                raise StopIteration from exc_

        if self.current_index >= self.tmp_max_index:
            raise StopIteration

        if self._max_iter > 0:
            if self.curr_iter > self._max_iter:
                raise StopIteration

        res = {}
        dict_ = {}
        prod_v = None
        if self.load_p is not None:
            dict_["load_p"] = 1.0 * self.load_p[self.current_index, :]
        if self.load_q is not None:
            dict_["load_q"] = 1.0 * self.load_q[self.current_index, :]
        if self.prod_p is not None:
            dict_["prod_p"] = 1.0 * self.prod_p[self.current_index, :]
        if self.prod_v is not None:
            prod_v = 1.0 * self.prod_v[self.current_index, :]
        if dict_:
            res["injection"] = dict_

        if self.maintenance is not None:
            res["maintenance"] = self.maintenance[self.curr_iter, :]
        if self.hazards is not None:
            res["hazards"] = self.hazards[self.curr_iter, :]

        if self.maintenance_time is not None:
            maintenance_time = dt_int(1 * self.maintenance_time[self.curr_iter, :])
            maintenance_duration = dt_int(
                1 * self.maintenance_duration[self.curr_iter, :]
            )
        else:
            maintenance_time = np.full(self.n_line, fill_value=-1, dtype=dt_int)
            maintenance_duration = np.full(self.n_line, fill_value=0, dtype=dt_int)

        if self.hazard_duration is not None:
            hazard_duration = 1 * self.hazard_duration[self.current_index, :]
        else:
            hazard_duration = np.full(self.n_line, fill_value=-1, dtype=dt_int)

        self.current_datetime += self.time_interval
        self.curr_iter += 1
        return (
            self.current_datetime,
            res,
            maintenance_time,
            maintenance_duration,
            hazard_duration,
            prod_v,
        )

    def check_validity(self, backend):
        at_least_one = False
        if self.load_p is not None:
            if self.load_p.shape[1] != backend.n_load:
                msg_err = "for the active part. It should be {} but is in fact {}"
                raise IncorrectNumberOfLoads(
                    msg_err.format(backend.n_load, self.load_p.shape[1])
                )
            at_least_one = True

        if self.load_q is not None:
            if self.load_q.shape[1] != backend.n_load:
                msg_err = "for the reactive part. It should be {} but is in fact {}"
                raise IncorrectNumberOfLoads(
                    msg_err.format(backend.n_load, self.load_q.shape[1])
                )
            at_least_one = True
        if self.prod_p is not None:
            if self.prod_p.shape[1] != backend.n_gen:
                msg_err = "for the active part. It should be {} but is in fact {}"
                raise IncorrectNumberOfGenerators(
                    msg_err.format(backend.n_gen, self.prod_p.shape[1])
                )
            at_least_one = True

        if self.prod_v is not None:
            if self.prod_v.shape[1] != backend.n_gen:
                msg_err = "for the voltage part. It should be {} but is in fact {}"
                raise IncorrectNumberOfGenerators(
                    msg_err.format(backend.n_gen, self.prod_v.shape[1])
                )
            at_least_one = True

        if self.hazards is not None:
            if self.hazards.shape[1] != backend.n_line:
                msg_err = "for the outage. It should be {} but is in fact {}"
                raise IncorrectNumberOfLines(
                    msg_err.format(backend.n_line, self.hazards.shape[1])
                )
            at_least_one = True

        if self.maintenance is not None:
            if self.maintenance.shape[1] != backend.n_line:
                msg_err = "for the maintenance. It should be {} but is in fact {}"
                raise IncorrectNumberOfLines(
                    msg_err.format(backend.n_line, self.maintenance.shape[1])
                )
            at_least_one = True

        if self.maintenance_time is not None:
            if self.maintenance_time.shape[1] != backend.n_line:
                msg_err = "for the maintenance times. It should be {} but is in fact {}"
                raise IncorrectNumberOfLines(
                    msg_err.format(backend.n_line, self.maintenance_time.shape[1])
                )
            at_least_one = True

        if self.maintenance_duration is not None:
            if self.maintenance_duration.shape[1] != backend.n_line:
                msg_err = (
                    "for the maintenance durations. It should be {} but is in fact {}"
                )
                raise IncorrectNumberOfLines(
                    msg_err.format(backend.n_line, self.maintenance_duration.shape[1])
                )
            at_least_one = True

        if self.hazard_duration is not None:
            if self.hazard_duration.shape[1] != backend.n_line:
                msg_err = "for the hazard durations. It should be {} but is in fact {}"
                raise IncorrectNumberOfLines(
                    msg_err.format(backend.n_line, self.hazard_duration.shape[1])
                )
            at_least_one = True

        if not at_least_one:
            raise ChronicsError(
                'No files are found in directory "{}". If you don\'t want to load any chronics, use '
                '"ChangeNothing" and not "{}" to load chronics.'
                "".format(self.path, type(self))
            )

        for name_arr, arr in zip(
            [
                "load_q",
                "load_p",
                "prod_v",
                "prod_p",
                "maintenance",
                "hazards",
                "maintenance time",
                "maintenance duration",
                "hazard duration",
            ],
            [
                self.load_q,
                self.load_p,
                self.prod_v,
                self.prod_p,
                self.maintenance,
                self.hazards,
                self.maintenance_time,
                self.maintenance_duration,
                self.hazard_duration,
            ],
        ):
            if arr is not None:
                if self.chunk_size is None:
                    if arr.shape[0] != self.n_:
                        msg_err = (
                            "Array {} has not the same number of rows ({}) than the maintenance ({}). "
                            "The chronics cannot be loaded properly."
                        )
                        raise EnvError(msg_err.format(name_arr, arr.shape[0], self.n_))

        if self._max_iter > 0:
            if self._max_iter > self.n_:
                msg_err = "Files count {} rows and you ask this episode to last at {} timestep."
                raise InsufficientData(msg_err.format(self.n_, self._max_iter))

    def next_chronics(self):
        self.current_datetime = self.start_datetime
        self.current_index = -1
        self.curr_iter = 0
        if self.chunk_size is not None:
            self._clear()  # remove previously loaded data [only needed if chunk size is set, I assume]

    def get_id(self) -> str:
        return self.path

    def set_chunk_size(self, new_chunk_size):
        self.chunk_size = new_chunk_size

    def _convert_datetime(self, datetime_beg):
        res = datetime_beg
        if not isinstance(datetime_beg, datetime):
            try:
                res = datetime.strptime(datetime_beg, "%Y-%m-%d %H:%M")
            except Exception as exc_:
                try:
                    res = datetime.strptime(datetime_beg, "%Y-%m-%d")
                except Exception as exc_2:
                    raise ChronicsError(
                        'Impossible to convert "{}" to a valid datetime. Accepted format is '
                        '"%Y-%m-%d %H:%M"'.format(datetime_beg)
                    ) from exc_2
        return res

    def _extract_array(self, nm):
        var = self.__dict__[nm]
        if var is None:
            return None
        else:
            return var[self.current_index, :]

    def _save_array(self, array_, path_out, name, colnames):
        if array_ is None:
            return
        tmp = pd.DataFrame(array_)
        tmp.columns = colnames
        tmp.to_csv(os.path.join(path_out, name), index=False, sep=self.sep)

    def _init_res_split(self, nb_rows):
        res_prod_p = None
        res_prod_v = None
        res_load_p = None
        res_load_q = None
        res_maintenance = None
        res_hazards = None
        if self.prod_p is not None:
            res_prod_p = np.zeros((nb_rows, self.n_gen), dtype=dt_float)
        if self.prod_v is not None:
            res_prod_v = np.zeros((nb_rows, self.n_gen), dtype=dt_float)
        if self.load_p is not None:
            res_load_p = np.zeros((nb_rows, self.n_load), dtype=dt_float)
        if self.load_q is not None:
            res_load_q = np.zeros((nb_rows, self.n_load), dtype=dt_float)
        if self.maintenance is not None:
            res_maintenance = np.zeros((nb_rows, self.n_line), dtype=dt_float)
        if self.hazards is not None:
            res_hazards = np.zeros((nb_rows, self.n_line), dtype=dt_float)
        return (
            res_prod_p,
            res_prod_v,
            res_load_p,
            res_load_q,
            res_maintenance,
            res_hazards,
        )

    def _update_res_split(self, i, tmp, *arrays):
        (
            res_prod_p,
            res_prod_v,
            res_load_p,
            res_load_q,
            res_maintenance,
            res_hazards,
        ) = arrays
        if res_prod_p is not None:
            res_prod_p[i, :] = tmp._extract_array("prod_p")
        if res_prod_v is not None:
            res_prod_v[i, :] = tmp._extract_array("prod_v")
        if res_load_p is not None:
            res_load_p[i, :] = tmp._extract_array("load_p")
        if res_load_q is not None:
            res_load_q[i, :] = tmp._extract_array("load_q")
        if res_maintenance is not None:
            res_maintenance[i, :] = tmp._extract_array("maintenance")
        if res_hazards is not None:
            res_hazards[i, :] = tmp._extract_array("hazards")

    def _clean_arrays(self, i, *arrays):
        (
            res_prod_p,
            res_prod_v,
            res_load_p,
            res_load_q,
            res_maintenance,
            res_hazards,
        ) = arrays
        if res_prod_p is not None:
            res_prod_p = res_prod_p[:i, :]
        if res_prod_v is not None:
            res_prod_v = res_prod_v[:i, :]
        if res_load_p is not None:
            res_load_p = res_load_p[:i, :]
        if res_load_q is not None:
            res_load_q = res_load_q[:i, :]
        if res_maintenance is not None:
            res_maintenance = res_maintenance[:i, :]
        if res_hazards is not None:
            res_hazards = res_hazards[:i, :]
        return (
            res_prod_p,
            res_prod_v,
            res_load_p,
            res_load_q,
            res_maintenance,
            res_hazards,
        )

    def _get_name_arrays_for_saving(self):
        return ["prod_p", "prod_v", "load_p", "load_q", "maintenance", "hazards"]

    def _get_colorder_arrays_for_saving(self):
        return [
            self._order_backend_prods,
            self._order_backend_prods,
            self._order_backend_loads,
            self._order_backend_loads,
            self._order_backend_lines,
            self._order_backend_lines,
        ]

    def split_and_save(self, datetime_beg, datetime_end, path_out):
        """
        You can use this function to save the values of the chronics in a format that will be loadable
        by :class:`GridStateFromFile`

        Notes
        -----
        Prefer using the :func:`Multifolder.split_and_save` that handles different chronics

        Parameters
        ----------
        datetime_beg: ``str``
            Time stamp of the beginning of the data you want to save (time stamp in "%Y-%m-%d %H:%M"
            format)

        datetime_end: ``str``
            Time stamp of the end of the data you want to save (time stamp in "%Y-%m-%d %H:%M"
            format)

        path_out: ``str``
            Location where to save the data

        """
        # work on a copy of myself
        tmp = copy.deepcopy(self)
        datetime_beg = self._convert_datetime(datetime_beg)
        datetime_end = self._convert_datetime(datetime_end)

        nb_rows = datetime_end - datetime_beg
        nb_rows = nb_rows.total_seconds()
        nb_rows = int(nb_rows / self.time_interval.total_seconds()) + 1
        if nb_rows <= 0:
            raise ChronicsError(
                'Invalid time step to be extracted. Make sure "datetime_beg" is lower than '
                '"datetime_end" {} - {}'.format(datetime_beg, datetime_end)
            )

        # prepare folder
        if not os.path.exists(path_out):
            os.mkdir(path_out)

        # skip until datetime_beg starts
        curr_dt = tmp.current_datetime
        if curr_dt > datetime_beg:
            warnings.warn(
                "split_and_save: you ask for a beginning of the extraction of the chronics after the "
                "current datetime of it. If they ever existed, the data in the chronics prior to {}"
                "will be ignored".format(curr_dt)
            )
        # in the chronics we load the first row to initialize the data, so here we stop just a bit before that
        datetime_start = datetime_beg - self.time_interval
        while curr_dt < datetime_start:
            curr_dt, *_ = tmp.load_next()
        real_init_dt = curr_dt
        arrays = self._init_res_split(nb_rows)
        i = 0
        while curr_dt < datetime_end:
            self._update_res_split(i, tmp, *arrays)
            curr_dt, *_ = tmp.load_next()
            i += 1
        if i < nb_rows:
            warnings.warn(
                "split_and_save: chronics goes up to {} but you want to split it up to {}. Results "
                "has been troncated".format(curr_dt, datetime_end)
            )

        arrays = self._clean_arrays(i, *arrays)
        nms = self._get_name_arrays_for_saving()
        orders_columns = self._get_colorder_arrays_for_saving()
        for el, nm, colnames in zip(arrays, nms, orders_columns):
            nm = "{}{}".format(nm, ".csv.bz2")
            self._save_array(el, path_out, nm, colnames)

        with open(os.path.join(path_out, "start_datetime.info"), "w") as f:
            f.write("{:%Y-%m-%d %H:%M}\n".format(real_init_dt))

        tmp_for_time_delta = (
            datetime(year=2018, month=1, day=1, hour=0, minute=0, second=0)
            + self.time_interval
        )
        with open(os.path.join(path_out, "time_interval.info"), "w") as f:
            f.write("{:%H:%M}\n".format(tmp_for_time_delta))
