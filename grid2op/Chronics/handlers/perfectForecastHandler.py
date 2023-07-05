# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from typing import Tuple
import warnings

from grid2op.Exceptions import HandlerError
from grid2op.Chronics.handlers.baseHandler import BaseHandler


class PerfectForecastHandler(BaseHandler):    
    """This class is allows to generate "perfect forecast", with this class the agent
    will know what will be the exact production, loads etc for the near future.
    
    This is a strong "assumption" and it is not realistic. 
    
    To have make things more realistic, you can use the :class:`NoisyForecastHandler` but again, 
    this class is far from perfect.
    
    More "research" is needed in this area and any contribution is more than welcome !
    
    As the name suggest, you should use this class only for the FORECAST data and not for environment or maintenance.
    
    .. warning::
        It only works if the handlers of the environments supports the :func:`BaseHandler.get_future_data` is implemented
        for the environment handlers. 
        
    """
    def __init__(self, array_name, max_iter=-1, quiet_warnings : bool=False):
        super().__init__(array_name, max_iter)
        self.quiet_warnings = quiet_warnings
             
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

    def forecast(self,
                 forecast_horizon_id : int,
                 inj_dict_env : dict,
                 inj_dict_previous_forecast : dict,
                 # eg gen_p_handler if this is set to gen_p_for_handler:
                 env_handler : BaseHandler,  
                 # list of the 4 env handlers: (load_p_handler, load_q_handler, gen_p_handler, gen_v_handler)
                 env_handlers : Tuple[BaseHandler, BaseHandler, BaseHandler, BaseHandler]
                 ):
        res = env_handler.get_future_data(self._h_forecast[forecast_horizon_id], self.quiet_warnings)
        return res
