# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
from typing import Union, Callable, Iterable, Tuple
try:
    from typing import Literal
except ImportError:
    # Literal not available in python 3.7
    from typing_extensions import Literal

from grid2op.dtypes import dt_float
from grid2op.Exceptions import HandlerError
from grid2op.Chronics.handlers.baseHandler import BaseHandler
from grid2op.Chronics.handlers.perfectForecastHandler import PerfectForecastHandler


class NoisyForecastHandler(PerfectForecastHandler): 
    """This class allows you to generate some noisy multiple steps ahead 
    forecasts for a given environment.
    
    To make "noisy" forecast, this class first retrieve the "perfect forecast"
    (see :class:`PerfectForecastHandler`) and then it "adds" some noise to 
    each individual component of this vector.
     
    Noise is not "added" but "multiply" with the formula: 
    `output = lognormal(0., sigma) * input` with sigma being the
    standard deviation of the noise that depends on the forecast horizons.
    
    .. seealso::
        The class :class:`PerfectForecastHandler`
    
    .. warning::
        This class has the same limitation as the :class:`PerfectForecastHandler`. It only works 
        if the handlers of the environments supports the :func:`BaseHandler.get_future_data` is implemented
        for the environment handlers. 
        
    Notes
    ------
    
    **Independance of the noise:**
    
    The noise is applied independantly for each variable and each
    "horizon" and each "step".
    
    This means that:
    
    - the forecast noise applied at t0=0 for tf=5 for generator 1 is independant
      from the noise applied t0=0 for tf=5 for generator 2
    - the forecast noise applied at t0=0 for tf=5 is independant from the 
      applied made at t0=5 for tf=5
    - the forecast noise applied at t0=0 for tf=5 is independant from the 
      applied made at t0=0 for tf=10, or tf=15 etc.
      
    In other words, there are no "correlation" between the noise of any kind.
    
    If you want better quality forecast, you should use a dedicated tools to
    generate some. Among which is "chronix2grid".
        
        
    **Noise depending on the forecast horizon:**
    
    For now, you can only use "multiplicative noise" that is applied
    by multiplyig the sampling of a LogNormal distribution with the 
    "perfect forecast".
    
    The "standard deviation" of this lognormal can be parametrized: the
    "forecast error" can be dependant on the forecast horizon.
    
    For that, you can input :
    
    - either a callable (=function) that takes as input a forecast horizon 
      (in minutes) and return the std of the noise for this horizon
    - or an iterable that directly contains the std of the noise.
    
    For example:
    
    .. code-block::  python
    
        import grid2op
        from grid2op.Chronics import FromHandlers
        from grid2op.Chronics.handlers import CSVHandler, NoisyForecastHandler
        
        env_name = "l2rpn_case14_sandbox"
        hs_ = [5*(i+1) for i in range(12)]
        
        # uses the default noise: sqrt(horizon) * 0.01 : error of 8% 1h ahead
        env = grid2op.make(env_name,
                           data_feeding_kwargs={"gridvalueClass": FromHandlers,
                                                "gen_p_handler": CSVHandler("prod_p"),
                                                "load_p_handler": CSVHandler("load_p"),
                                                "gen_v_handler": CSVHandler("prod_v"),
                                                "load_q_handler": CSVHandler("load_q"),
                                                "h_forecast": hs_,
                                                "gen_p_for_handler": NoisyForecastHandler("prod_p_forecasted"),
                                                "load_p_for_handler": NoisyForecastHandler("load_p_forecasted"),
                                                "load_q_for_handler": NoisyForecastHandler("load_q_forecasted"),
                                               }
                            )  
                            
        # uses the noise: `horizon -> 0.01 * horizon` : error of 6% 1h ahead
        env = grid2op.make(env_name,
                           data_feeding_kwargs={"gridvalueClass": FromHandlers,
                                                "gen_p_handler": CSVHandler("prod_p"),
                                                "load_p_handler": CSVHandler("load_p"),
                                                "gen_v_handler": CSVHandler("prod_v"),
                                                "load_q_handler": CSVHandler("load_q"),
                                                "h_forecast": hs_,
                                                "gen_p_for_handler": NoisyForecastHandler("prod_p_forecasted",
                                                                                          sigma=lambda x: 0.01 * x),
                                                "load_p_for_handler": NoisyForecastHandler("load_p_forecasted",
                                                                                           sigma=lambda x: 0.01 * x),
                                                "load_q_for_handler": NoisyForecastHandler("load_q_forecasted",
                                                                                           sigma=lambda x: 0.01 * x),
                                               }
                            )  
    
    **Caveats:**
    
    1) Be carefull, the noise for "gen_p" / "prod_p" is not exactly the one given in input.
       This is because of the "first law of power system" (saying that total generation should be equal to
       todal demand and losses). To make sure this "first law" is "more met" we scale the generation to make sure
       that it is roughly 1.02 * total load (when possible) [to be really exhaustive, the ratio 1.02 is, whenever 
       possible modified and computed from the real time data]
    2) There might be some pmin / pmax violation for the generated generators
    3) There will be some max_ramp_down / max_ramp_up violations for the generated generators
    4) The higher the noise, the higher the trouble you will encounter
    
    To "get rid" of all these limitations, you can of course use an "offline" way to generate more realistic forecasts,
    for example using chronix2grid.
                             
    """
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
                                   "sure the sigma are either callable or iterable") from exc_
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
    
    def _env_loss_ratio(self, inj_dict_env):
        res = 1.02
        if ("load_p" in inj_dict_env and
            "prod_p" in inj_dict_env):
            if (inj_dict_env["load_p"] is not None and 
                inj_dict_env["prod_p"] is not None):
                res = inj_dict_env["prod_p"].sum() / inj_dict_env["load_p"].sum()
        return res
    
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
                    res *= self._env_loss_ratio(inj_dict_env)
                # TODO ramps, pmin, pmax !
        return res.astype(dt_float) if res is not None else None
