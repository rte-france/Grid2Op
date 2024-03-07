# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from typing import Tuple
import pandas as pd
import numpy as np

from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Chronics.gridValue import GridValue
from grid2op.Chronics.handlers.csvHandler import CSVHandler


class CSVMaintenanceHandler(CSVHandler):
    """Reads and produce time series if given by a csv file (possibly compressed).
    
    The separator used can be specified as input. 
    
    The file name should match the "array_name". If you want to use
    the maintenance file present in the file "my_maintenance_file.csv.gz"
    then you should create a CSVMaintenanceHandler with 
    `array_name="my_maintenance_file"`.
    
    The csv should be structured as follow:
    
    - it should not have any "index" or anything, only data used by 
      grid2op will be used
    - Each element powerline is represented by a `column`.
    - It should have a header with the name of the powerlines that
      should match the one in the environment. For example 
      if "0_1_0" is the name of a powerline in your environment, 
      then a column should be called "0_1_0".
    - each time step is represented as a `row` and in order. For example
      (removing the header), row 1 (first row) will be step 1, row 2 will
      be step 2 etc.
    - only binary data (0 or 1) should be present in the file. No "bool", 
      no string etc.
      
    .. warning::
        Use this class only for the ENVIRONMENT data ("load_p", "load_q",
        "prod_p" or "prod_v") and not for maintenance (in this case
        use :class:`CSVMaintenanceHandler`) nor for 
        forecast (in this case use :class:`CSVForecastHandler`) 
    
    This is the default way to provide data to grid2op and its used for
    most l2rpn environments.
    
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
        self.array = np.abs(self.array) >= 1e-7
        self.array = self.array.astype(dt_bool)
    
    def load_next_maintenance(self) -> Tuple[np.ndarray, np.ndarray]:
        maint_time = 1 * self.maintenance_time[self.current_index, :]
        maint_duration = 1 * self.maintenance_duration[self.current_index, :]
        return maint_time, maint_duration
    
    def set_chunk_size(self, chunk_size):
        # skip the definition in CSVHandler to jump to the level "above"
        return super(CSVHandler, self).set_chunk_size(chunk_size)
