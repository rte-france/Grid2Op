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
from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Chronics.gridValue import GridValue
from grid2op.Chronics.handlers.csvHandler import CSVHandler



class CSVMaintenanceHandler(CSVHandler):
    """Read the time series from a csv.
    
    Only for Maintenance data, not for environment nor for forecasts
    
    Allows to read maintenance data in a csv format. Each row of the csv indicates a time step. Each
    column of the table a powerline.
    
    There is a 0 in this csv if the given line is not in maitnenance at the given step, otherwise there 
    should be a 1.
    
    It uses pandas to read files. File can be compressed.
    """
    def __init__(self,
                 array_name="maintenance",
                 sep=";",
                 max_iter=-1) -> None:
        super().__init__(array_name, sep, None, max_iter)
        # None corresponds to "chunk_size" which is not supported for maintenance
    
    def _init_attrs(self, array):
        super()._init_attrs(array)
        n_line = self.array.shape[1]
        
        self.maintenance_time = (
            np.zeros(shape=(self.array.shape[0], n_line), dtype=dt_int)
            - 1
        )
        self.maintenance_duration = np.zeros(
            shape=(self.array.shape[0], n_line), dtype=dt_int
        )

        # test that with chunk size
        for line_id in range(n_line):
            self.maintenance_time[:, line_id] = GridValue.get_maintenance_time_1d(
                self.array[:, line_id]
            )
            self.maintenance_duration[
                :, line_id
            ] = GridValue.get_maintenance_duration_1d(self.array[:, line_id])

        # there are _maintenance and hazards only if the value in the file is not 0.
        self.array = self.array != 0.0
        self.array = self.array.astype(dt_bool)
    
    def load_next_maintenance(self):
        maint_time = 1 * self.maintenance_time[self.current_index, :]
        maint_duration = 1 * self.maintenance_duration[self.current_index, :]
        return maint_time, maint_duration
    
    def set_chunk_size(self, chunk_size):
        # skip the definition in CSVHandler to jump to the level "above"
        return super(CSVHandler, self).set_chunk_size(chunk_size)
