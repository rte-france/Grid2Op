# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import pandas as pd
import numpy as np
import copy

from grid2op.Exceptions import (
    ChronicsError, HandlerError
)

from grid2op.dtypes import dt_int, dt_float


class CSVHandler:
    """Read the time series from a csv.
    
    Only for Environment data, not for FORECAST
    """
    def __init__(self,
                 array_name,  # eg "load_p"
                 sep=";",
                 chunk_size=None,
                 max_iter=-1) -> None:
        
        self.path = None
        self._file_ext = None
        self.tmp_max_index = None  # size maximum of the current tables in memory
        self.array = None  # numpy array corresponding to the current active load values in the power _grid.

        self.current_index = -1
        self.sep = sep

        self.names_chronics_to_backend = None

        # added to provide an easier access to read data in chunk
        self.chunk_size = chunk_size
        self._data_chunk = {}
        self._order_array = None
        
        # 
        self._order_backend_arrays = None
        
        # 
        self.array_name = array_name
        
        # max iter
        self.max_iter = max_iter
        
        #
        self._nb_row_per_step = 1
    
    def _clear(self):
        """reset to a state as if it was just created"""
        self.path = None
        self._file_ext = None
        self.tmp_max_index = None
        self.array = None
        self.current_index = - 1
        self.names_chronics_to_backend = None
        self._data_chunk = {}
        self._order_array = None
        self._order_backend_arrays = None
        return self
    
    def set_path(self, path):
        self._file_ext = self._get_fileext(path)
        self.path = os.path.join(path, f"{self.array_name}{self._file_ext}")
    
    def initialize(self, order_backend_arrays, names_chronics_to_backend):
        self._order_backend_arrays = copy.deepcopy(order_backend_arrays)
        self.names_chronics_to_backend = copy.deepcopy(names_chronics_to_backend)
        
        # read the data
        array_iter = self._get_data()
        
        if not self.names_chronics_to_backend:
            self.names_chronics_to_backend = {}
            self.names_chronics_to_backend[self.array_name] = {
                k: k for k in self._order_backend_arrays
            }
            
        # put the proper name in order
        order_backend_arrays = {el: i for i, el in enumerate(order_backend_arrays)}

        if self.chunk_size is None:
            array = array_iter
            if array is not None:
                self.tmp_max_index = array.shape[0]
            else:
                raise HandlerError(
                    'No files are found in directory "{}". If you don\'t want to load any chronics,'
                    ' use  "DoNothingHandler" (`from grid2op.Chronics.handlers import DoNothingHandler`) '
                    'and not "{}" to load chronics.'
                    "".format(self.path, type(self))
                )

        else:
            self._data_chunk = {
                self.array_name: array_iter,
            }
            array = self._get_next_chunk()

        # get the chronics in order
        order_chronics = self._get_orders(array, order_backend_arrays)

        # now "sort" the columns of each chunk of data
        self._order_array = np.argsort(order_chronics)

        self._init_attrs(array)

        self.curr_iter = 0

    
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
        if self.max_iter > 0:
            if self.curr_iter > self.max_iter:
                return True
        if self.chunk_size is None:
            if self.current_index >= self.array.shape[0]:
                return True
        return False
    
    def load_next(self, dict_):
        self.current_index += 1

        if not self._data_in_memory():
            try:
                self._load_next_chunk_in_memory()
            except StopIteration as exc_:
                raise StopIteration

        if self.current_index > self.tmp_max_index:
            raise StopIteration

        if self.max_iter > 0:
            if self.curr_iter >= self.max_iter:
                raise StopIteration    
        return 1.0 * self.array[self.current_index, :]
    
    def get_max_iter(self):
        if self.max_iter != -1:
            return self.max_iter
        else:
            if self.chunk_size is None and self.array is not None:
                return self.array.shape[0]
            else:
                import warnings
                warnings.warn("Unable to read the 'max_iter' when there is a chunk size set and no \"max_iter\"")
                return -1 # TODO
                
    def check_validity(self, backend):
        # TODO
        return True
    
    def load_next_maintenance(self, *args, **kwargs):
        # TODO
        raise NotImplementedError()
    
    def load_next_hazard(self, *args, **kwargs):
        # TODO
        raise NotImplementedError()

    def get_kwargs(self, dict_):
        pass
    
    def _init_attrs(
        self, array
    ):
        self.array = None

        if array is not None:
            self.array = copy.deepcopy(
                array.values[:, self._order_array].astype(dt_float)
            )
        
    def _get_fileext(self, path_tmp):  # in csvhandler
        read_compressed = ".csv"
        if not os.path.exists(os.path.join(path_tmp, "{}.csv".format(self.array_name))):
            # try to read compressed data
            if os.path.exists(os.path.join(path_tmp, "{}.csv.bz2".format(self.array_name))):
                read_compressed = ".csv.bz2"
            elif os.path.exists(os.path.join(path_tmp, "{}.zip".format(self.array_name))):
                read_compressed = ".zip"
            elif os.path.exists(
                os.path.join(path_tmp, "{}.csv.gzip".format(self.array_name))
            ):
                read_compressed = ".csv.gzip"
            elif os.path.exists(os.path.join(path_tmp, "{}.csv.xz".format(self.array_name))):
                read_compressed = ".csv.xz"
            else:
                read_compressed = None
        return read_compressed

    def _get_data(self, chunksize=-1, nrows=None):   # in csvhandler     
        if nrows is None:
            if self.max_iter > 0:
                nrows = self.max_iter + self._nb_row_per_step
            
        if self._file_ext is not None:
            if chunksize == -1:
                chunksize = self.chunk_size
            res = pd.read_csv(
                self.path,
                sep=self.sep,
                chunksize=chunksize,
                nrows=nrows,
            )
        else:
            res = None
        return res 
    
    def _get_orders(
        self,
        array,  # eg load_p
        order_arrays,  # eg order_backend_loads
    ):

        order_chronics_arrays = None

        if array is not None:
            self._assert_correct_second_stage(
                array.columns, self.names_chronics_to_backend
            )
            order_chronics_arrays = np.array(
                [
                    order_arrays[self.names_chronics_to_backend[self.array_name][el]]
                    for el in array.columns
                ]
            ).astype(dt_int)

        return order_chronics_arrays

    def _assert_correct_second_stage(self, pandas_name, dict_convert):
        for i, el in enumerate(pandas_name):
            if not el in dict_convert[self.array_name]:
                raise ChronicsError(
                    "Element named {} is found in the data (column {}) but it is not found on the "
                    'powergrid for data of type "{}".\nData in files  are: {}\n'
                    "Converter data are: {}".format(
                        el,
                        i + 1,
                        self.array_name,
                        sorted(list(pandas_name)),
                        sorted(list(dict_convert[self.array_name].keys())),
                    )
                )
    
    def set_chunk_size(self, chunk_size):
        self.chunk_size = int(chunk_size)
    
    def set_max_iter(self, max_iter):
        self.max_iter = int(max_iter)
        
    def _get_next_chunk(self):
        res = None
        if self._data_chunk[self.array_name] is not None:
            res = next(self._data_chunk[self.array_name])
        return res
    
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
        # i load the next chunk as dataframes
        array = self._get_next_chunk()  # array: load_p
        # i put these dataframes in the right order (columns)
        self._init_attrs(array)
        # i don't forget to reset the reading index to 0
        self.current_index = 0
        
    def _get_next_chunk(self):
        array = None
        if self._data_chunk[self.array_name] is not None:
            array = next(self._data_chunk[self.array_name])
            self.tmp_max_index = array.shape[0]
        return array

    def forecast(self, *args, **kwargs):
        raise HandlerError("You should only use this class for ENVIRONMENT data, and not for FORECAST data. "
                           "Please consider using `CSVHandlerForecast` (`from grid2op.Chronics.handlers import CSVHandlerForecast`) "
                           "for your forecast data.")
        
    def set_h_forecast(self, h_forecasts):
        raise HandlerError("You should only use this class for ENVIRONMENT data, and not for FORECAST data. "
                           "Please consider using `CSVHandlerForecast` (`from grid2op.Chronics.handlers import CSVHandlerForecast`) "
                           "for your forecast data.")
    
    def next_chronics(self):
        self.current_index = -1
        self.curr_iter = 0
        if self.chunk_size is not None:
            self._clear()  # we should have to reload everything if all data have been already loaded
        
# handler:
# set_path OK
# done OK
# initialize OK
# get_max_iter OK
# set_max_iter OK
# set_chunk_size OK
# next_chronics
# load_next_maintenance SOON
# load_next_hazard SOON
# load_next OK
# check_validity(backend) SOON