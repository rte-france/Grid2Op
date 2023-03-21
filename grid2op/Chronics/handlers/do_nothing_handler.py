# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy


class DoNothingHandler:
    """This is the type of Time Series Handler that does nothing. 
    
    The environment will act as if the time series this Handler is reponsible for will never change.
    """
    def __init__(self) -> None:
        self._h_forecast = None
    
    def set_path(self, path):
        pass
    
    def initialize(self, order_backend_prods, names_chronics_to_backend):
        pass
    
    def check_validity(self, backend):
        pass
    
    def load_next(self, dict):
        pass
    
    def set_max_iter(self, max_iter):
        pass
    
    def set_chunk_size(self, chunk_size):
        pass    

    def get_available_horizons(self):
        return copy.deepcopy(self._h_forecast)
    
    def set_h_forecast(self, h_forecast):
        self._h_forecast = h_forecast
        
    def done(self):
        return False
    
    def forecast(self,
                 forecast_horizon_id,
                 inj_dict_previous_forecast,
                 load_p_handler,
                 load_q_handler,
                 gen_p_handler,
                 gen_v_handler
                 ):
        pass
    