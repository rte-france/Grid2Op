# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from typing import Tuple

from grid2op.Exceptions import (
    HandlerError
)
from grid2op.Chronics.handlers.baseHandler import BaseHandler
from grid2op.Chronics.handlers.csvHandler import CSVHandler


class CSVForecastHandler(CSVHandler):
    """Reads and produce time series if given by a csv file (possibly compressed).
    
    The separator used can be specified as input. 
    
    The file name should match the "array_name":
    for example if the data you want to use for "load_p_forecasted" in the environment
    are in the file "my_load_p_forecast.csv.bz2" should name this handler 
    "my_load_p_forecast" and not "load_p" nor "my_load_p_forecast.csv" nor
    "my_load_p_forecast.csv.bz2"
    
    The csv should be structured as follow:
    
    - it should not have any "index" or anything, only data used directly
      by grid2op (so only "active loads" if this handler is responsible 
      for the generation of "load_p")
    - Each element (for example a load) is represented by a `column`.
    - It should have a header with the name of the elements it "handles" and 
      this name should match the one in the environment. For example 
      if "load_1_0" is the name of a load and you read data for "load_p"
      or "load_q" then one column of your csv should be named "load_1_0".
    - only floating point numbers should be present in the data (no bool, string
      and integers will be casted to float)
    
    The structuration of the rows are a bit different than for :class:`CSVHandler`
    because this class can read "multiple steps ahead forecast", provided that
    it knows for how many different horizons forecasts are made.
    
    Let's take the example that forecast are available for h = 5, 10 and 15
    minutes ahead (so for the next, next next and next next next steps). In this case:
    
    - the first row (not counting the header) will be the forecast made
      for h = 5 at the first step: the forecasts available at t=0 for t=5mins
    - the second row will be the forecasts made
      for h = 10 at the first step: the forecasts available at t=0 for t=10mins
    - the third row will be the forecasts made
      for h = 15 at the first step: the forecasts available at t=0 for t=15mins
    - the fourth row will be the forecasts made
      for h = 5 at the second step: the forecasts available at t=5 for t=10mins
    - the fifth row will be the forecasts made
      for h = 10 at the second step: the forecasts available at t=5 for t=15mins
    - etc.
      
    .. warning::
        Use this class only for the FORECAST data ("load_p_forecasted", 
        "load_q_forecasted", "prod_p_forecasted" or "prod_v_forecasted") and 
        not for maintenance (in this case
        use :class:`CSVMaintenanceHandler`) nor for 
        environment data (in this case use :class:`CSVHandler`) 
        nor for setting the initial state state (in this case use 
        :class:`JSONInitStateHandler`)
    
    This is the default way to provide data to grid2op and its used for
    most l2rpn environments when forecasts are available.
    
    .. note::
        The current implementation heavily relies on the fact that the 
        :func:`CSVForecastHandler.forecast` method is called
        exactly once per horizon and per step.
    
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
        
    def _set_max_iter(self, max_iter):
        super()._set_max_iter(self._nb_row_per_step * int(max_iter))
    
    def set_h_forecast(self, h_forecast):
        super().set_h_forecast(h_forecast)
        self._nb_row_per_step = len(self._h_forecast) 
        
    def get_available_horizons(self):
        # skip the definition in CSVHandler to jump to the level "above"
        return super(CSVHandler, self).get_available_horizons()
        
    def forecast(self,
                 forecast_horizon_id : int,
                 inj_dict_env : dict,
                 inj_dict_previous_forecast : dict,
                 # eg gen_p_handler if this is set to gen_p_for_handler:
                 env_handler : "BaseHandler",  
                 # list of the 4 env handlers: (load_p_handler, load_q_handler, gen_p_handler, gen_v_handler)
                 env_handlers : Tuple["BaseHandler", "BaseHandler", "BaseHandler", "BaseHandler"]):
        res = super().load_next(inj_dict_previous_forecast)
        return res
