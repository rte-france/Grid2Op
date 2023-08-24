# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from datetime import datetime, timedelta
import os
import numpy as np
import copy
import warnings
from typing import Optional, Union, List
from pathlib import Path

from grid2op.Exceptions import (
    ChronicsError, ChronicsNotFoundError
)

from grid2op.Chronics.gridValue import GridValue

from grid2op.dtypes import dt_int, dt_float
from grid2op.Chronics.fromOneEpisodeData import TYPE_EP_DATA_INGESTED, FromOneEpisodeData


class FromMultiEpisodeData(GridValue): 
    MULTI_CHRONICS = True
    def __init__(self,
                 path,  # can be None !
                 li_ep_data: List[TYPE_EP_DATA_INGESTED],
                 time_interval=timedelta(minutes=5),
                 sep=";",  # here for compatibility with grid2op, but not used
                 max_iter=-1,
                 start_datetime=datetime(year=2019, month=1, day=1),
                 chunk_size=None,
                 list_perfect_forecasts=None,  # TODO
                 **kwargs,  # unused
                 ):
        super().__init__(time_interval, max_iter, start_datetime, chunk_size)
        self.li_ep_data = [FromOneEpisodeData(path,
                                              ep_data=el,
                                              time_interval=time_interval,
                                              max_iter=max_iter,
                                              chunk_size=chunk_size,
                                              list_perfect_forecasts=list_perfect_forecasts,
                                              start_datetime=start_datetime)
                           for el in li_ep_data
                           ]
        self._prev_cache_id = 0
        self.data = self.li_ep_data[0]
        
    def next_chronics(self):
        self._prev_cache_id += 1
        # TODO implement the shuffling indeed.
        # if self._prev_cache_id >= len(self._order):
        #     self.space_prng.shuffle(self._order)
        self._prev_cache_id %= len(self.li_ep_data)

    def initialize(
        self,
        order_backend_loads,
        order_backend_prods,
        order_backend_lines,
        order_backend_subs,
        names_chronics_to_backend=None,
    ):

        self.data = self.li_ep_data[self._prev_cache_id]
        self.data.initialize(
            order_backend_loads,
            order_backend_prods,
            order_backend_lines,
            order_backend_subs,
            names_chronics_to_backend=names_chronics_to_backend,
        )
        
    def done(self):
        return self.data.done()

    def load_next(self):
        return self.data.load_next()
    
    def check_validity(self, backend):
        return self.data.check_validity(backend)

    def forecasts(self):
        return self.data.forecasts()
    
    def tell_id(self, id_num, previous=False):
        if not isinstance(id_num, (int, dt_int)):
            raise ChronicsError("FromMultiEpisodeData can only be used with `tell_id` being an integer "
                                "at the moment. Feel free to write a feature request if you want more.")

        self._prev_cache_id = id_num
        self._prev_cache_id %= len(self.li_ep_data)

        if previous:
            self._prev_cache_id -= 1
            self._prev_cache_id %= len(self.li_ep_data)
    
    def get_id(self) -> int:
        return self._prev_cache_id - 1  # TODO check
    
    def max_timestep(self):
        return self.data.max_timestep()
    
    def fast_forward(self, nb_timestep):
        self.data.fast_forward(nb_timestep)
