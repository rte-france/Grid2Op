# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import copy
from typing import Optional, Union
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta

import grid2op
from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Exceptions import IncorrectNumberOfElements, ChronicsError, ChronicsNotFoundError
from grid2op.Exceptions import IncorrectNumberOfLoads, IncorrectNumberOfGenerators, IncorrectNumberOfLines
from grid2op.Exceptions import EnvError, InsufficientData
from grid2op.Chronics.gridValue import GridValue


class FromNPY(GridValue):
    """
    This class allows to generate some chronics compatible with grid2op if the data are provided in numpy format.

    It also enables the use of the starting the chronics at different time and end if a different time.

    .. warnings::
        It assume the order of the elements are consistent with the powergrid backend ! It will not attempt to reorder the columns of the dataset

    Examples
    --------

    Usage example, for what you don't really have to do:

    .. code-block:: python

        import grid2op
        from grid2op.Chronics import FromNPY

        env_name = ...
        env = grid2op.make(env_name, chronics_class=FromNPY, data_feeding_kwargs={"i_start": 5, "i_end": 18})
        
    Attributes
    ----------
    
    """
    def __init__(self,
                 load_p : np.ndarray,
                 load_q : np.ndarray,
                 prod_p : np.ndarray,
                 prod_v : Optional[np.ndarray]=None,
                 hazards : Optional[np.ndarray]=None,
                 maintenance : Optional[np.ndarray]=None,
                 load_p_forecast : Optional[np.ndarray]=None,  # TODO forecasts !!
                 load_q_forecast : Optional[np.ndarray]=None,
                 prod_p_forecast : Optional[np.ndarray]=None,
                 prod_v_forecast : Optional[np.ndarray]=None,
                 time_interval: timedelta=timedelta(minutes=5),
                 max_iter: int=-1,
                 start_datetime: datetime=datetime(year=2019, month=1, day=1),
                 chunk_size: Optional[int]=None,
                 i_start: int=0,
                 i_end: Optional[int]=None,  # excluded, as always in python
                 **kwargs):
        GridValue.__init__(self, time_interval=time_interval, max_iter=max_iter, start_datetime=start_datetime,
                           chunk_size=chunk_size)
        self._i_start : int = i_start
        self.n_gen : int = prod_p.shape[1]
        self.n_load : int = load_p.shape[1]
        self.n_line : Union[int, None] = None

        self.load_p : np.ndarray = 1.0 * load_p
        self.load_q : np.ndarray = 1.0 * load_q
        self.prod_p : np.ndarray = 1.0 * prod_p
        
        self.prod_v = None
        if prod_v is not None:
            self.prod_v = 1.0 * prod_v
        
        self.__new_load_p : Optional[np.ndarray] = None
        self.__new_prod_p : Optional[np.ndarray] = None
        self.__new_prod_v : Optional[np.ndarray] = None
        self.__new_load_q : Optional[np.ndarray] = None
        
        if i_end is None:
            i_end = load_p.shape[0]
        self._i_end : int = i_end

        self.has_maintenance = False
        self.maintenance = None
        self.maintenance_duration = None
        self.maintenance_time = None
        if maintenance is not None:
            self.has_maintenance = True
            self.n_line = maintenance.shape[1]
            assert load_p.shape[0] == maintenance.shape[0]
            self.maintenance = maintenance  # TODO copy

            self.maintenance_time = np.zeros(shape=(self.maintenance.shape[0], self.n_line), dtype=dt_int) - 1
            self.maintenance_duration = np.zeros(shape=(self.maintenance.shape[0], self.n_line), dtype=dt_int)
            for line_id in range(self.n_line):
                self.maintenance_time[:, line_id] = self.get_maintenance_time_1d(self.maintenance[:, line_id])
                self.maintenance_duration[:, line_id] = self.get_maintenance_duration_1d(self.maintenance[:, line_id])

        self.has_hazards = False
        self.hazards = None
        self.hazard_duration = None
        if hazards is not None:
            self.has_hazards = True
            if self.n_line is None:
                self.n_line = hazards.shape[1]
            else:
                assert self.n_line == hazards.shape[1]
            assert load_p.shape[0] == hazards.shape[0]

            self.hazards = hazards  # TODO copy !
            self.hazard_duration = np.zeros(shape=(self.hazards.shape[0], self.n_line), dtype=dt_int)
            for line_id in range(self.n_line):
                self.hazard_duration[:, line_id] = self.get_hazard_duration_1d(self.hazards[:, line_id])


    def initialize(self, order_backend_loads, order_backend_prods, order_backend_lines, order_backend_subs,
                   names_chronics_to_backend=None):
        assert len(order_backend_prods) == self.n_gen
        assert len(order_backend_loads) == self.n_load
        if self.n_line is None:
            self.n_line = len(order_backend_lines)
        else:
            assert len(order_backend_lines) == self.n_line

        self.maintenance_time_nomaint = np.zeros(shape=(self.n_line, ), dtype=dt_int) - 1
        self.maintenance_duration_nomaint = np.zeros(shape=(self.n_line, ), dtype=dt_int)
        self.hazard_duration_nohaz = np.zeros(shape=(self.n_line, ), dtype=dt_int)

        self.curr_iter = 0
        self.current_index = self._i_start - 1

    def get_id(self) -> str:
        """
        To return a unique ID of the chronics, we use a hash function (black2b), but it outputs a name too big.
        So we hash it again with md5 to get a reasonable length id (32 characters)

        Returns:
            str:  the hash of the arrays (load_p, load_q, etc.) in the chronics
        """
        import hashlib
        # get the "long hash" from blake2b
        hash_ = hashlib.blake2b()  # should be faster than md5 ! (and safer, but we only care about speed here)
        hash_.update(self.load_p.tobytes())
        hash_.update(self.load_q.tobytes())
        hash_.update(self.prod_p.tobytes())
        if self.prod_v is not None:
            hash_.update(self.prod_v.tobytes())
        if self.maintenance is not None:
            hash_.update(self.maintenance.tobytes())
        if self.hazards is not None:
            hash_.update(self.hazards.tobytes())
        # TODO forecast !
        
        # now shorten it with md5
        long_hash_byte = hash_.digest()
        short_hash = hashlib.md5(long_hash_byte)
        return short_hash.hexdigest()
    
    def load_next(self):
        self.current_index += 1

        if self.current_index > self._i_end or self.current_index >= self.load_p.shape[0]:
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

        if self.maintenance is not None and self.has_maintenance:
            res["maintenance"] = self.maintenance[self.current_index, :]
        if self.hazards is not None and self.has_hazards:
            res["hazards"] = self.hazards[self.current_index, :]

        self.current_datetime += self.time_interval
        self.curr_iter += 1

        if self.maintenance_time is not None and self.maintenance_duration is not None and self.has_maintenance:
            maintenance_time = dt_int(1 * self.maintenance_time[self.current_index, :])
            maintenance_duration = dt_int(1 * self.maintenance_duration[self.current_index, :])
        else:
            maintenance_time = self.maintenance_time_nomaint
            maintenance_duration = self.maintenance_duration_nomaint

        if self.hazard_duration is not None and self.has_hazards:
            hazard_duration = 1 * self.hazard_duration[self.current_index, :]
        else:
            hazard_duration = self.hazard_duration_nohaz

        return self.current_datetime, res, maintenance_time, maintenance_duration, hazard_duration, prod_v

    def check_validity(self, backend: Optional["grid2op.Backend.Backend.Backend"]) -> None:
        assert self.load_p.shape[0] == self.load_q.shape[0]
        assert self.load_p.shape[0] == self.prod_p.shape[0]
        if self.prod_v is not None:
            assert self.load_p.shape[0] == self.prod_v.shape[0]
        
        if self.hazards is not None:
            assert self.hazards.shape[1] == self.n_line
        if self.maintenance is not None:
            assert self.maintenance.shape[1] == self.n_line
        if self.maintenance_duration is not None:
            assert self.n_line == self.maintenance_duration.shape[1]
        if self.maintenance_time is not None:
            assert self.n_line == self.maintenance_time.shape[1]

    def next_chronics(self):
        # restart the chronics: read it again !
        self.current_datetime = self.start_datetime
        self.curr_iter = 0
        self.current_index = self._i_start
        
        if self.__new_load_p is not None:
            self.load_p = self.__new_load_p
            self.__new_load_p = None
        if self.__new_load_q is not None:
            self.load_q = self.__new_load_q
            self.__new_load_q = None
        if self.__new_prod_p is not None:
            self.prod_p = self.__new_prod_p
            self.__new_prod_p = None
        if self.__new_prod_v is not None:
            self.prod_v = self.__new_prod_v
            self.__new_prod_v = None

        self.check_validity(backend=None)
        
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
        if self.current_index >= self._i_end or self.current_index >= self.load_p.shape[0]:
            res = True
        elif self.max_iter > 0:
            if self.curr_iter > self.max_iter:
                res = True
        return res

    def forecasts(self):
        # TODO
        return []

    def change_chronics(self,
                        new_load_p: np.ndarray = None,
                        new_load_q: np.ndarray = None, 
                        new_prod_p: np.ndarray = None,
                        new_prod_v: np.ndarray = None):
        """
        Allows to change the data used by this class.
        
        .. warning::
            This has an effect only after "env.reset" has been called !


        Args:
            new_load_p (np.ndarray, optional): change the load_p. Defaults to None (= do not change).
            new_load_q (np.ndarray, optional): change the load_q. Defaults to None (= do not change).
            new_prod_p (np.ndarray, optional): change the prod_p. Defaults to None (= do not change).
            new_prod_v (np.ndarray, optional): change the prod_v. Defaults to None (= do not change).
            
        Examples
        ---------
        
        .. code-block:: python
        
            import grid2op
            
            TODO
        """
        if new_load_p is not None:
            self.__new_load_p = 1.0 * new_load_p
        if new_load_q is not None:
            self.__new_load_q = 1.0 * new_load_q
        if new_prod_p is not None:
            self.__new_prod_p = 1.0 * new_prod_p
        if new_prod_v is not None:
            self.__new_prod_v = 1.0 * new_prod_v
        