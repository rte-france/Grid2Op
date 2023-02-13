# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np
import warnings
import copy

from grid2op.Exceptions import (
    IncorrectNumberOfElements,
    ChronicsError,
    ChronicsNotFoundError,
)

from grid2op.Chronics.gridValue import GridValue
from grid2op.dtypes import dt_int, dt_float, dt_bool


class CSVHandler:
    """Read the time series from a csv.
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
                raise ChronicsError(
                    'No files are found in directory "{}". If you don\'t want to load any chronics,'
                    ' use  "ChangeNothing" and not "{}" to load chronics.'
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
        res = False
        if self.current_index > self.n_:
            res = True
        return res
    
    def load_next(self, dict_):
        self.current_index += 1

        if not self._data_in_memory():
            try:
                self._load_next_chunk_in_memory()
            except StopIteration as exc_:
                raise exc_

        if self.current_index >= self.tmp_max_index:
            raise StopIteration

        if self.max_iter > 0:
            if self.curr_iter > self.max_iter:
                raise StopIteration
    
        return 1.0 * self.array[self.current_index, :]
    
    def get_max_iter(self):
        if self.max_iter != -1:
            return self.max_iter
        else:
            return -1  # TODO
                
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
        # TODO
        raise NotImplementedError()

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
                nrows = self.max_iter + 1
            
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
        # print("I loaded another chunk")
        # i load the next chunk as dataframes
        array = self._get_next_chunk()  # array: load_p
        # i put these dataframes in the right order (columns)
        self._init_attrs(array)  # TODO
        # i don't forget to reset the reading index to 0
        self.current_index = 0
        
    def _get_next_chunk(self):
        array = None
        if self._data_chunk[self.array_name] is not None:
            array = next(self._data_chunk[self.array_name])
            self.tmp_max_index = array.shape[0]
        return array

# handler:
# set_path OK
# done OK
# initialize OK
# get_max_iter OK
# next_chronics
# load_next_maintenance SOON
# load_next_hazard SOON
# load_next OK
# check_validity(backend) SOON

class FromHandlers(GridValue):    
    def __init__(
        self,
        path,  # can be None !
        gen_p_handler,
        gen_v_handler,
        load_p_handler,
        load_q_handler,
        maintenance_handler=None,
        hazards_handler=None,
        gen_p_for_handler=None,
        gen_v_for_handler=None,
        load_p_for_handler=None,
        load_q_for_handler=None,
        time_interval=timedelta(minutes=5),
        sep=";",
        max_iter=-1,
        start_datetime=datetime(year=2019, month=1, day=1),
        chunk_size=None,
    ):
        GridValue.__init__(
            self,
            time_interval=time_interval,
            max_iter=max_iter,
            start_datetime=start_datetime,
            chunk_size=chunk_size,
        )
        
        self.path = path
        if self.path is not None:
            self._init_date_time()
            
        # all my "handlers" (I need to perform a deepcopy otherwise data are kept between episode...)
        self.gen_p_handler = copy.deepcopy(gen_p_handler)
        self.gen_v_handler = copy.deepcopy(gen_v_handler)
        self.load_p_handler = copy.deepcopy(load_p_handler)
        self.load_q_handler = copy.deepcopy(load_q_handler)
        self.maintenance_handler = copy.deepcopy(maintenance_handler)
        self.hazards_handler = copy.deepcopy(hazards_handler)
        self.gen_p_for_handler = copy.deepcopy(gen_p_for_handler)
        self.gen_v_for_handler = copy.deepcopy(gen_v_for_handler)
        self.load_p_for_handler = copy.deepcopy(load_p_for_handler)
        self.load_q_for_handler = copy.deepcopy(load_q_for_handler)
        
        # when there are no maintenance / hazards, build this only once 
        self._no_mh_time = None
        self._no_mh_duration = None
        
        # set the path if needed
        if chunk_size is not None:
            warnings.warn("chunk_size has no effect here, please set it in the handler directly (*eg* `CSVHandler(chunk_size=...)`) instead")
        if max_iter==-1:
            warnings.warn("max_iter has no effect here, please set it in the handler directly (*eg* `CSVHandler(max_iter=...)`) instead")
        
        # define the active handlers
        self._active_handlers = [self.gen_p_handler, self.gen_v_handler, self.load_p_handler, self.load_q_handler]
        if self.maintenance_handler is not None:
            self._active_handlers.append(self.maintenance_handler)
        if self.hazards_handler is not None:
            self._active_handlers.append(self.hazards_handler)
        if self.gen_p_for_handler is not None:
            self._active_handlers.append(self.gen_p_for_handler)
        if self.load_p_for_handler is not None:
            self._active_handlers.append(self.load_p_for_handler)
        if self.gen_v_for_handler is not None:
            self._active_handlers.append(self.gen_v_for_handler)
        if self.load_q_for_handler is not None:
            self._active_handlers.append(self.load_q_for_handler)
            
        # set the current path of the time series
        self._set_path(self.path)
        
    def initialize(
        self,
        order_backend_loads,
        order_backend_prods,
        order_backend_lines,
        order_backend_subs,
        names_chronics_to_backend=None,
    ):
        # set the current path of the time series
        self._set_path(self.path)
        
        self.n_gen = len(order_backend_prods)
        self.n_load = len(order_backend_loads)
        self.n_line = len(order_backend_lines)
        self.curr_iter = 0
        
        self.gen_p_handler.initialize(order_backend_prods, names_chronics_to_backend)
        self.gen_v_handler.initialize(order_backend_prods, names_chronics_to_backend)
        if self.gen_p_for_handler is not None:
            self.gen_p_for_handler.initialize(order_backend_prods, names_chronics_to_backend)
        if self.gen_v_for_handler is not None:
            self.gen_v_for_handler.initialize(order_backend_prods, names_chronics_to_backend)
            
        self.load_p_handler.initialize(order_backend_loads, names_chronics_to_backend)
        self.load_q_handler.initialize(order_backend_loads, names_chronics_to_backend)
        if self.load_p_handler is not None:
            self.load_p_handler.initialize(order_backend_loads, names_chronics_to_backend)
        if self.load_q_handler is not None:
            self.load_q_handler.initialize(order_backend_loads, names_chronics_to_backend)
            
        if self.maintenance_handler is not None:
            self.maintenance_handler.initialize(order_backend_lines, names_chronics_to_backend)
            
        if self.hazards_handler is not None:
            self.hazards_handler.initialize(order_backend_lines, names_chronics_to_backend)
            
        # when there are no maintenance / hazards, build this only once 
        self._no_mh_time = np.full(self.n_line, fill_value=-1, dtype=dt_int)
        self._no_mh_duration = np.full(self.n_line, fill_value=0, dtype=dt_int)
            
    def load_next(self):
        self.current_datetime += self.time_interval
        self.curr_iter += 1
        
        res = {}
        # load the injection
        dict_inj, prod_v = self._load_injection()
        res["injection"] = dict_inj
        
        # load maintenance
        if self.maintenance_handler is not None:
            maintenance_time, maintenance_duration = self.maintenance_handler.load_next_maintenance()
            res["maintenance"] = self.maintenance_handler.load_next(res)
        else:
            maintenance_time = self._no_mh_time
            maintenance_duration = self._no_mh_duration
        
        # load hazards
        if self.hazard_duration is not None:
            hazard_duration = self.hazards_handler.load_next_hazard()
            res["hazards"] = self.hazards_handler.load_next(res)
        else:
            hazard_duration = self._no_mh_duration
            
        return (
            self.current_datetime,
            res,
            maintenance_time,
            maintenance_duration,
            hazard_duration,
            prod_v,
        )
    
    def max_timestep(self):
        return self.max_iter
        
    def next_chronics(self):
        self.current_datetime = self.start_datetime
        self.curr_iter = 0
        
        for el in self._active_handlers:
            el.next_chronics()
            
        self._update_max_iter()
    
    def done(self):
        res = False
        # if self.current_index > self.n_:
        #     res = True
        if self.max_iter > 0:
            if self.curr_iter > self.max_iter:
                res = True
        return res
    
    def check_validity(self, backend):
        for el in self._active_handlers:
            el.check_validity(backend)
        # TODO other things here maybe ???
        return True
    
    def fast_forward(self, nb_timestep):
        # TODO
        super().fast_forward(nb_timestep)

    def forecasts(self):
        # TODO
        return []
            
    def get_kwargs(self, dict_):
        # TODO
        raise NotImplementedError()
    
    def tell_id(self, id_num, previous=False):
        # TODO
        raise NotImplementedError()
    
    def get_id(self) -> str:
        # TODO
        raise NotImplementedError()
    
    def shuffle(self, shuffler=None):
        # TODO
        pass
    
    def sample_next_chronics(self, probabilities=None):
        # TODO
        pass
    
    def set_chunk_size(self, new_chunk_size):
        # TODO
        pass
        
    def _set_path(self, path):
        """tell the handler where this chronics is located"""
        if path is None:
            return
        
        for el in self._active_handlers:
            el.set_path(path)
            
    def _update_max_iter(self):
        max_iters = [el.get_max_iter() for el in self._active_handlers]
        max_iters = [el for el in max_iters if el != -1]
        self.max_iter = np.min(max_iters)
        
    def _load_injection(self):
        dict_ = {}
        prod_v = None
        if self.load_p_handler is not None:
            dict_["load_p"] = 1.0 * self.load_p_handler.load_next(dict_)
        if self.load_q_handler is not None:
            dict_["load_q"] = 1.0 * self.load_q_handler.load_next(dict_)
        if self.gen_p_handler is not None:
            dict_["prod_p"] = 1.0 * self.gen_p_handler.load_next(dict_)
        if self.gen_v_handler is not None:
            prod_v = 1.0 * self.gen_v_handler.load_next(dict_)
        return dict_, prod_v
        
    def _init_date_time(self):  # in csv handler
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
    