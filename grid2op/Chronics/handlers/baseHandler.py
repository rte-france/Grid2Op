# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import os
from typing import Optional
from grid2op.Space import RandomObject
from datetime import timedelta, datetime

class BaseHandler(RandomObject):
    def __init__(self, array_name, max_iter=-1, h_forecast=(5, )):
        super().__init__()
        self.max_iter : int = max_iter
        self.init_datetime : Optional[datetime] = None
        self.time_interval : Optional[timedelta] = None
        self.array_name : str = array_name
        self._h_forecast : tuple = copy.deepcopy(h_forecast)
        self.path : Optional[os.PathLike] = None
        self.max_episode_duration : Optional[int] = None
    
    def set_max_iter(self, max_iter):
        if max_iter is not None:
            self.max_iter = int(max_iter)
        else:
            self.max_iter = -1
    def set_max_episode_duration(self, max_episode_duration):
        if max_episode_duration is not None:
            self.max_episode_duration = int(max_episode_duration)
        else:
            self.max_episode_duration = None
        
    def get_max_iter(self):
        return self.max_iter
    
    def set_path(self, path):
        self.path = path
    
    def set_chunk_size(self, chunk_size):
        pass
        
    def set_times(self,
                  init_datetime,
                  time_interval):
        self.init_datetime = init_datetime
        self.time_interval = time_interval
    
    def _clear(self):
        self.init_datetime = None
        self.time_interval = None

    def get_kwargs(self, dict_):
        pass
    
    def set_h_forecast(self, h_forecast):
        self._h_forecast = copy.deepcopy(h_forecast)
    
    def get_available_horizons(self):
        return copy.deepcopy(self._h_forecast)
        
    def initialize(self, order_backend_arrays, names_chronics_to_backend):
        raise NotImplementedError()
    
    def done(self):
        raise NotImplementedError()
    
    def load_next(self, dict_):
        raise NotImplementedError()
    
    def check_validity(self, backend):
        raise NotImplementedError()    
    
    def load_next_maintenance(self):
        # TODO
        raise NotImplementedError()
    
    def load_next_hazard(self):
        # TODO
        raise NotImplementedError()
    
    def forecast(self,
                 forecast_horizon_id,
                 inj_dict_previous_forecast,
                 load_p_handler,
                 load_q_handler,
                 gen_p_handler,
                 gen_v_handler):
        raise NotImplementedError()
        
    