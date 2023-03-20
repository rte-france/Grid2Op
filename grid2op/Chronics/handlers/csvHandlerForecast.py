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
from grid2op.Chronics.handlers.csvHandler import CSVHandler


class CSVHandlerForecast(CSVHandler):
    """Read the time series from a csv.
    
    Only for FORECAST data, not for ENVIRONMENT
    
    TODO heavily uses the fact that forecast made at the same datetime (for different horizons) 
    are contiguous in the data frame.
    """
    def __init__(self,
                 array_name,
                 sep=";",
                 chunk_size=None,
                 max_iter=-1,
                 h_forecast=(5, ),) -> None:
        super().__init__(array_name, sep, chunk_size, max_iter)
        self._h_forecast = copy.deepcopy(h_forecast)
        
    def load_next(self, dict_):
        raise HandlerError("You should only use this class for FORECAST data, and not for ENVIRONMENT data. "
                           "Please consider using `CSVHandler` (`from grid2op.Chronics.handlers import CSVHandler`) "
                           "for your environment data.")
    
    def get_available_horizons(self):
        return copy.deepcopy(self._h_forecast)
    
    def forecast(self,
                 forecast_horizon_id,
                 inj_dict_previous_forecast,
                 load_p_handler,
                 load_q_handler,
                 gen_p_handler,
                 gen_v_handler
                 ):
        return super().load_next(inj_dict_previous_forecast)
