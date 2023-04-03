# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import re
from typing import Tuple
from grid2op.Exceptions import HandlerError
from grid2op.Chronics.handlers.baseHandler import BaseHandler


class PersistenceForecastHandler(BaseHandler):
    """
    This type of handler will generate the "persitence" type of forecast: basically it will copy
    paste the last known data of the environment.
    
    You should use it only for FORECAST data and not for environment data as the name suggest.
    """
    INJ_KEYS = ("load_p", "load_q", "prod_p", "prod_v")
    def __init__(self, array_name, max_iter=-1):
        super().__init__(array_name, max_iter)
        tmp = re.sub("_for.*$", "", array_name)
        tmp = tmp.replace("gen", "prod")
        if tmp in type(self).INJ_KEYS:
             self._possible_key = tmp
        else:   
            self._possible_key = None
             
    def initialize(self, order_backend_arrays, names_chronics_to_backend):
        # nothing particular to do at initialization
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
    
    def _aux_get_inj_key(self, env_handler):
        # return the key I need to look for in the "inj_dict_env" to get the 
        # real data
        if self._possible_key is not None:
            return self._possible_key
        if env_handler.array_name == "load_p":
            return "load_p"
        if env_handler.array_name == "load_q":
            return "load_q"
        if env_handler.array_name == "prod_p":
            return "prod_p"
        if env_handler.array_name == "prod_v":
            return "prod_v"
        if env_handler.array_name == "gen_p":
            return "prod_p"
        if env_handler.array_name == "gen_v":
            return "prod_v"
        return None

    def forecast(self,
                 forecast_horizon_id : int,
                 inj_dict_env : dict,
                 inj_dict_previous_forecast : dict,
                 # eg gen_p_handler if this is set to gen_p_for_handler:
                 env_handler : "BaseHandler",  
                 # list of the 4 env handlers: (load_p_handler, load_q_handler, gen_p_handler, gen_v_handler)
                 env_handlers : Tuple["BaseHandler", "BaseHandler", "BaseHandler", "BaseHandler"]
                 ):
        key = self._aux_get_inj_key(env_handler)
        if key is None:
            raise HandlerError("Impossible to find which key to look for in the dictionary for "
                               "the real time data. Please change the name of the PersistenceHandler "
                               f"currently named {self.array_name} to something more specific "
                               "such as \"gen_p\" or \"load_p_forecast\"")
        
        if key not in inj_dict_env["injection"]:
            raise HandlerError(f"Please the remove the handler {self.array_name}. Indeed there is no data for "
                               f"{key} in any of the environment handler, so I cannot \"continue to copy\" "
                               "this data in the forecast (NB for \"prod_v\" environment data are handled "
                               "somewhere else, namely in the VoltageControler, so they cannot be 'copied' "
                               "from the time series).")
            
        return inj_dict_env["injection"][key]
