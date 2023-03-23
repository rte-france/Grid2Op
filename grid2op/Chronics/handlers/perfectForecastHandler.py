# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from typing import Tuple
from grid2op.Exceptions import HandlerError
from grid2op.Chronics.handlers.baseHandler import BaseHandler
from grid2op.Chronics.handlers.persitenceForecastHandler import PersistenceForecastHandler


class PerfectForecastHandler(BaseHandler):    
    def __init__(self, array_name, max_iter=-1):
        super().__init__(array_name, max_iter)
             
    def initialize(self, order_backend_arrays, names_chronics_to_backend):
        pass
    
    def done(self):
        # this handler is never "done", only when the "real data" it depends on is done
        return False
    
    def load_next(self, dict_):
        raise HandlerError("You should only use this class for FORECAST data, and not for ENVIRONMENT data. "
                           "You might want to use the `DoNothingHandler` that will `do_nothing` for "
                           "the environment data (see `from grid2op.Chronics.handlers import DoNothingHandler`)")
        
    def check_validity(self, backend):
        return True
    
    def load_next_maintenance(self):
        raise HandlerError("You should only use this class for FORECAST data and not for MAINTENANCE data")
    
    def load_next_hazard(self):
        raise HandlerError("You should only use this class for FORECAST data and not for HAZARDS data")

    def forecast(self,
                 forecast_horizon_id : int,
                 inj_dict_env : dict,
                 inj_dict_previous_forecast : dict,
                 # eg gen_p_handler if this is set to gen_p_for_handler:
                 env_handler : BaseHandler,  
                 # list of the 4 env handlers: (load_p_handler, load_q_handler, gen_p_handler, gen_v_handler)
                 env_handlers : Tuple[BaseHandler, BaseHandler, BaseHandler, BaseHandler]
                 ):
        return env_handler.get_future_data(self._h_forecast[forecast_horizon_id])
