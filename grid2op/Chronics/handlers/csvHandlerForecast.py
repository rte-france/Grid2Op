# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Exceptions import (
    ChronicsError, HandlerError
)

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
                 max_iter=-1) -> None:
        super().__init__(array_name, sep, chunk_size, max_iter)
        self._h_forecast = None
        self._nb_row_per_step = 1
        
    def load_next(self, dict_):
        raise HandlerError("You should only use this class for FORECAST data, and not for ENVIRONMENT data. "
                           "Please consider using `CSVHandler` (`from grid2op.Chronics.handlers import CSVHandler`) "
                           "for your environment data.")
    
    def set_chunk_size(self, chunk_size):
        super().set_chunk_size(self._nb_row_per_step * int(chunk_size))
        
    def set_max_iter(self, max_iter):
        super().set_max_iter(self._nb_row_per_step * int(max_iter))
    
    def set_h_forecast(self, h_forecast):
        super().set_h_forecast(h_forecast)
        self._nb_row_per_step = len(self._h_forecast) 
        
    def get_available_horizons(self):
        # skip the definition in CSVHandler to jump to the level "above"
        return super(CSVHandler, self).get_available_horizons()
        
    def forecast(self,
                 forecast_horizon_id,
                 inj_dict_env,
                 inj_dict_previous_forecast,
                 # eg gen_p_handler if this is set to gen_p_for_handler:
                 env_handler,  
                 # list of the 4 env handlers: (load_p_handler, load_q_handler, gen_p_handler, gen_v_handler)
                 env_handlers):
        res = super().load_next(inj_dict_previous_forecast)
        return res
