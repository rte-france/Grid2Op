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
from typing import Tuple

from grid2op.Exceptions import (
    ChronicsError, HandlerError
)

from grid2op.dtypes import dt_int, dt_float
from grid2op.Chronics.handlers.baseHandler import BaseHandler


class CSVHandler(BaseHandler):
    """Reads and produce time series if given by a csv file (possibly compressed).
    
    The separator used can be specified as input. 
    
    The file name should match the "array_name":
    for example if the data you want to use for "load_p" in the environment
    are in the file "my_load_p_data.csv.bz2" should name this handler 
    "my_load_p_data" and not "load_p" nor "my_load_p_data.csv" nor
    "my_load_p_data.csv.bz2"
    
    The csv should be structured as follow:
    
    - it should not have any "index" or anything, only data used by 
      grid2op will be used
    - Each element (for example a load) is represented by a `column`.
    - It should have a header with the name of the elements it "handles" and 
      this name should match the one in the environment. For example 
      if "load_1_0" is the name of a load and you read data for "load_p"
      or "load_q" then one column of your csv should be named "load_1_0".
    - each time step is represented as a `row` and in order. For example
      (removing the header), row 1 (first row) will be step 1, row 2 will
      be step 2 etc.
    - only floating point numbers should be present in the data (no bool, string
      and integers will be casted to float)
      
      
    .. warning::
        Use this class only for the ENVIRONMENT data ("load_p", "load_q",
        "prod_p" or "prod_v") and not for maintenance (in this case
        use :class:`CSVMaintenanceHandler`) nor for 
        forecast (in this case use :class:`CSVForecastHandler`) 
        nor for setting the initial state state (in this case use 
        :class:`JSONInitStateHandler`)
    
    This is the default way to provide data to grid2op and its used for
    most l2rpn environments.
    
    """
    def __init__(self,
                 array_name,  # eg "load_p"
                 sep=";",
                 chunk_size=None,
                 max_iter=-1) -> None:
        super().__init__(array_name, max_iter)
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
        self._nb_row_per_step = 1
    
    def _clear(self):
        """reset to a state as if it was just created"""
        super()._clear()
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
        
        if self.chunk_size is None:
            self.max_episode_duration = self.array.shape[0] - 1

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
        if self.max_iter > 0 and self.curr_iter > self.max_iter:
            return True
        if self.chunk_size is None and self.current_index >= self.array.shape[0]:
            return True
        return False
    
    def load_next(self, dict_):
        self.current_index += 1

        if not self._data_in_memory():
            try:
                self._load_next_chunk_in_memory()
            except StopIteration as exc_:
                raise StopIteration from exc_

        if self.current_index > self.tmp_max_index:
            raise StopIteration

        if self.max_iter > 0:
            if self.curr_iter >= self.max_iter:
                raise StopIteration    
        return copy.deepcopy(self.array[self.current_index, :])
    
    def get_max_iter(self):
        if self.max_iter != -1:
            return self.max_iter
        if self.max_episode_duration is not None:
            return self.max_episode_duration
            
        if self.chunk_size is None and self.array is not None:
            return self.array.shape[0] - 1
        
        if self.array is None:
            return -1
        
        import warnings
        warnings.warn("Unable to read the 'max_iter' when there is a chunk size set and no \"max_iter\"")
        return -1 # TODO
                
    def check_validity(self, backend):
        # TODO
        return True
    
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

    def forecast(self,
                 forecast_horizon_id : int,
                 inj_dict_env : dict,
                 inj_dict_previous_forecast : dict,
                 # eg gen_p_handler if this is set to gen_p_for_handler:
                 env_handler : "BaseHandler",  
                 # list of the 4 env handlers: (load_p_handler, load_q_handler, gen_p_handler, gen_v_handler)
                 env_handlers : Tuple["BaseHandler", "BaseHandler", "BaseHandler", "BaseHandler"]):
        raise HandlerError(f"forecast {self.array_name}: You should only use this class for ENVIRONMENT data, and not for FORECAST data. "
                           "Please consider using `CSVForecastHandler` (`from grid2op.Chronics.handlers import CSVForecastHandler`) "
                           "for your forecast data.")
    
    def get_available_horizons(self):
        raise HandlerError(f"get_available_horizons {self.array_name}: You should only use this class for ENVIRONMENT data, and not for FORECAST data. "
                           "Please consider using `CSVForecastHandler` (`from grid2op.Chronics.handlers import CSVForecastHandler`) "
                           "for your forecast data.")
        
    def load_next_maintenance(self):
        raise HandlerError(f"load_next_maintenance {self.array_name}: You should only use this class for ENVIRONMENT data, and not for FORECAST data nor MAINTENANCE data. "
                           "Please consider using `CSVMaintenanceHandler` (`from grid2op.Chronics.handlers import CSVMaintenanceHandler`) "
                           "for your maintenance data.")
    
    def load_next_hazard(self):
        raise HandlerError(f"load_next_hazard {self.array_name}: You should only use this class for ENVIRONMENT data, and not for FORECAST "
                           "data nor MAINTENANCE nor HAZARDS data. (NB HAZARDS data are not yet supported) "
                           "by handlers.")
        
    def next_chronics(self):
        self.current_index = -1
        self.curr_iter = 0
        if self.chunk_size is not None:
            self._clear()  # we should have to reload everything if all data have been already loaded
    
    def get_future_data(self, horizon: int, quiet_warnings : bool=False):
        horizon = int(horizon)
        tmp_index = self.current_index + horizon // (self.time_interval.total_seconds() // 60)
        tmp_index = int(tmp_index)
        if tmp_index < self.array.shape[0]:
            res = self.array[tmp_index, :]
        else:
            if not quiet_warnings:
                import warnings
                warnings.warn(f"{type(self)} {self.array_name}: No more data to get, the last known data is returned.")
            res = self.array[-1, :]
        return copy.deepcopy(res)
