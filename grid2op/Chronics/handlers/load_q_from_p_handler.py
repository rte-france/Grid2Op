# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np

from grid2op.Exceptions import (
    HandlerError
)

from grid2op.dtypes import  dt_float
from grid2op.Chronics.handlers.baseHandler import BaseHandler


class LoadQFromPHandler(BaseHandler):
    """This handler is specific for "load_q" type of data.
    
    You can use it for both "forecast" ("load_q_forecasted")
    and for environment data ("load_q").
    
    It will generate load_q based on a "q over p" ratio provided
    as input. Basically, whenever called, it will
    return (when possible): `load_q = ratio * load_p`

    .. note::
        Its current implementation heavily relies on the fact that
        when the "load_q" / "load_q_forecasted" handlers 
        are called the "load_p" / "load_p_forecasted" data are already
        computed and known.
        
    """
    def __init__(self,
                 array_name="load_q",
                 qp_ratio: float=0.7,
                 max_iter=-1):
        super().__init__(array_name, max_iter)
        if isinstance(qp_ratio, np.ndarray):
            self._qp_ratio = (1.0 * qp_ratio).astype(dt_float)
        else:
            self._qp_ratio = dt_float(qp_ratio)
        
    def done(self):
        # this is never done as long as there is a "load_p"
        return False

    def load_next(self, dict_):
        if "load_p" in dict_ and not "prod_p" in dict_ and not "prod_v" in dict_:
            if dict_["load_p"] is not None:
                return self._qp_ratio * dict_["load_p"]
        return None
    
    def initialize(self, order_backend_prods, names_chronics_to_backend):
        # nothing to do for this particular handler
        pass
    
    def check_validity(self, backend):
        if isinstance(self._qp_ratio, np.ndarray):
            if backend.n_load != self._qp_ratio.shape[0]:
                raise HandlerError(f"{self.array_name}: qp_ratio should either be a single float "
                                    "or a numpy array with as many loads as there are loads on the grid. "
                                    f"You provided {self._qp_ratio.shape[0]} ratios but there are "
                                    f"{backend.n_load} loads on the grid.")
                
    def load_next_maintenance(self):
        raise HandlerError(f"load_next_maintenance {self.array_name}: You should only "
                           "use this class for ENVIRONMENT data, and not for FORECAST data nor MAINTENANCE data. "
                          )
    
    def load_next_hazard(self):
        raise HandlerError(f"load_next_hazard {self.array_name}: You should only use "
                           "this class for ENVIRONMENT data, and not for FORECAST ")
        
    def forecast(self,
                 forecast_horizon_id,
                 inj_dict_env,
                 inj_dict_previous_forecast,
                 # eg gen_p_handler if this is set to gen_p_for_handler:
                 env_handler,  
                 # list of the 4 env handlers: (load_p_handler, load_q_handler, gen_p_handler, gen_v_handler)
                 env_handlers):
        return self.load_next(inj_dict_previous_forecast)
