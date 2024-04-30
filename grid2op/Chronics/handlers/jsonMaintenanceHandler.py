# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import json
import os
from grid2op.Chronics.GSFFWFWM import GridStateFromFileWithForecastsWithMaintenance
from grid2op.Chronics.gridValue import GridValue

from grid2op.Chronics.handlers.baseHandler import BaseHandler


class JSONMaintenanceHandler(BaseHandler):
    """This type of handlers will generate maintenance based on some json files.
    
    Maintenance generated with this class will be stochastic: some different 
    maintenance time / duration will be generated for each new episode (of course
    you can seed your environment for a purely deterministic process)
    
    The json file it will read should be called `json_file_name` (by default 
    `"maintenance_meta.json"`)
    
    It should contain the data:
    
    - "line_to_maintenance": the list of the name of the powerline that can be 
      "in maintenance" for this episode
    - "maintenance_starting_hour" : the starting hour for all maintenance
    - "maintenance_ending_hour" : the hour at which each maintenance ends
    - "daily_proba_per_month_maintenance" : it's a list having 12 elements (one 
      for each month of the year) that gives, for each month the probability
      for any given line to be in maintenance. For example if 
      `daily_proba_per_month_maintenance[6] = 0.1` it means that for the 
      6th month of the year (june) there is a 10% for each powerline to be in
      maintenance
    - "max_daily_number_per_month_maintenance": maximum number of powerlines
      allowed in maintenance at the same time.
    
    .. warning::
        Use this class only for the MAINTENANCE and not for environment
        data ("load_p", "load_q", "prod_p" or "prod_v") nor for 
        forecast (in this case use :class:`CSVForecastHandler`) 
        nor for setting the initial state state (in this case use 
        :class:`JSONInitStateHandler`)
        
    """
    def __init__(self,
                 array_name="maintenance",
                 json_file_name="maintenance_meta.json",
                 max_iter=-1,
                 _duration_episode_default=24*12, # if max_iter is not set, then maintenance are computed for a whole day
                 ):
        super().__init__(array_name, max_iter)
        self.json_file_name = json_file_name
        self.dict_meta_data = None
        self.maintenance = None
        self.maintenance_time = None
        self.maintenance_duration = None
        self.n_line = None  # used in one of the GridStateFromFileWithForecastsWithMaintenance functions
        self._duration_episode_default = _duration_episode_default
        self.current_index = 0
    
    def get_maintenance_time_1d(self, maintenance):
        return GridValue.get_maintenance_time_1d(maintenance)
    
    def get_maintenance_duration_1d(self, maintenance):
        return GridValue.get_maintenance_duration_1d(maintenance)
    
    def _create_maintenance_arrays(self, current_datetime):
        # create the self.maintenance, self.maintenance_time and self.maintenance_duration
        self.maintenance = GridStateFromFileWithForecastsWithMaintenance._generate_matenance_static(
            self._order_backend_arrays,
            self.max_episode_duration if self.max_episode_duration is not None else self._duration_episode_default,
            self.dict_meta_data["line_to_maintenance"],
            self.time_interval,
            current_datetime,
            self.dict_meta_data["maintenance_starting_hour"],
            self.dict_meta_data["maintenance_ending_hour"],
            self.dict_meta_data["daily_proba_per_month_maintenance"],
            self.dict_meta_data["max_daily_number_per_month_maintenance"],
            self.space_prng
        )
        GridStateFromFileWithForecastsWithMaintenance._fix_maintenance_format(self)
        
    def initialize(self, order_backend_arrays, names_chronics_to_backend):
        self._order_backend_arrays = copy.deepcopy(order_backend_arrays)
        self.names_chronics_to_backend = copy.deepcopy(names_chronics_to_backend)
        self.n_line = len(self._order_backend_arrays)
        self.current_index = 0
        
        # read the description file
        with open(os.path.join(self.path, self.json_file_name), "r", encoding="utf-8") as f:
            self.dict_meta_data = json.load(f)
        
        # and now sample the maintenance
        self._create_maintenance_arrays(self.init_datetime)
    
    def check_validity(self, backend):
        # TODO
        pass
    
    def load_next_maintenance(self):
        maint_time = 1 * self.maintenance_time[self.current_index, :]
        maint_duration = 1 * self.maintenance_duration[self.current_index, :]
        return maint_time, maint_duration

    def load_next(self, dict_):
        self.current_index += 1
        if self.current_index >= self.maintenance.shape[0]:
            # regenerate some maintenance if needed
            self.current_index = 0
            self.init_datetime += self.maintenance.shape[0] * self.time_interval
            self._create_maintenance_arrays(self.init_datetime)
        return copy.deepcopy(self.maintenance[self.current_index, :])
    
    def _clear(self):
        super()._clear()
        self.dict_meta_data = None
        self.maintenance = None
        self.maintenance_time = None
        self.maintenance_duration = None
        self.n_line = None
        self.current_index = 0
    
    def done(self):
        # maintenance can be generated on the fly so they are never "done"
        return False