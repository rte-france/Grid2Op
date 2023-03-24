# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
from typing import Union, Literal, Callable, Iterable, Tuple

from grid2op.dtypes import dt_float
from grid2op.Exceptions import HandlerError
from grid2op.Chronics.handlers.baseHandler import BaseHandler
from grid2op.Chronics.handlers.perfectForecastHandler import PerfectForecastHandler


class NoisyForecastHandler(PerfectForecastHandler): 
    def __init__(self,
                 array_name,
                 sigma: Union[Callable, Iterable]=None,
                 noise_type : Literal["mult"] = "mult",  # TO BE ADDED LATER
                 max_iter=-1):
        super().__init__(array_name, max_iter)
        self.noise_type = noise_type
        self.sigma = sigma
        self._my_noise = None
    
    @staticmethod
    def _default_noise(horizon: int):
        """horizon in minutes"""
        return np.sqrt(1.0 * horizon) * 0.01  # error of ~8% at 1h
    
    def _get_list(self, sigma: Union[Callable, Iterable]):
        if sigma is None:
            res = [type(self)._default_noise(h) for h in self._h_forecast]
        elif callable(sigma):
            res = [sigma(h) for h in self._h_forecast]
        else:
            try:
                iter(sigma)
            except TypeError as exc_:
                raise HandlerError(f"{type(self)} ({self.array_name}): make "
                                   "sure the sigma_*** are either callable or iterable") from exc_
            res = [float(el) for el in sigma]
            if len(res) < len(self._h_forecast):
                last_el = res[-1]
                import warnings
                warnings.warn(f"{type(self)} ({self.array_name}): a list too short was provided "
                              f"for one of the sigma_*** parameter ({len(res)} elements "
                              f"given but forecasts are made for {len(self._h_forecast)} horizons)")
                for _ in range(len(self._h_forecast) - len(res)):
                    res.append(last_el) 
        return res
        
    def set_h_forecast(self, h_forecast):
        super().set_h_forecast(h_forecast)
        self._my_noise = self._get_list(self.sigma)
        
    def forecast(self,
                 forecast_horizon_id : int,
                 inj_dict_env : dict,
                 inj_dict_previous_forecast : dict,
                 # eg gen_p_handler if this is set to gen_p_for_handler:
                 env_handler : BaseHandler,  
                 # list of the 4 env handlers: (load_p_handler, load_q_handler, gen_p_handler, gen_v_handler)
                 env_handlers : Tuple[BaseHandler, BaseHandler, BaseHandler, BaseHandler]
                 ):
        res = super().forecast(forecast_horizon_id, inj_dict_env, inj_dict_previous_forecast, env_handler, env_handlers)
        if res is not None:
            if self.noise_type == "mult":
                res *= self.space_prng.lognormal(sigma=self._my_noise[forecast_horizon_id])
            else:
                raise HandlerError(f"{self.array_name}: the type of noise {self.noise_type} is not supported. "
                                   f"Only multiplicative noise are supported at the moment")
            if ("load_p" in inj_dict_previous_forecast and 
                "load_q" in inj_dict_previous_forecast and 
                "prod_p" not in inj_dict_previous_forecast
                ):
                # so this handler is for the generation: I scale it to be slightly above the total loads
                if inj_dict_previous_forecast["load_p"] is not None:
                    res *= inj_dict_previous_forecast["load_p"].sum() / res.sum()
                # TODO ramps, pmin, pmax !
        return res.astype(dt_float) if res is not None else None
