# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from datetime import datetime, timedelta
import os
import numpy as np
import copy
from typing import Optional, Union
from pathlib import Path

from grid2op.Exceptions import (
    ChronicsError, ChronicsNotFoundError
)

from grid2op.Chronics.gridValue import GridValue

from grid2op.dtypes import dt_int, dt_float
from grid2op.Episode import EpisodeData


class FromOneEpisodeData(GridValue): 
    """This class allows to use the :class:`grid2op.Chronics.handlers.BaseHandler` to read back data 
    stored in :class:`grid2op.Episode.EpisodeData`
    
    .. newinversion:: 1.9.2
    
    TODO make sure to use the opponent from OppFromEpisodeData if your initial environment had an opponent !
    
    TODO there will be "perfect" forecast, as original forecasts are not stored !

    Examples
    ---------
    You can use this class this way:
    
    .. code-block:: python
    
        import grid2op
        from grid2op.Chronics import FromHandlers
        from grid2op.Chronics.handlers import CSVHandler, DoNothingHandler, PerfectForecastHandler
        env_name = "l2rpn_case14_sandbox"
        
        env = grid2op.make(env_name,
                       data_feeding_kwargs={"gridvalueClass": FromEpisodeData,
                                            "ep_data": EpData
                                           }
                      )

        obs = env.reset()
    
        # and now you can use "env" as any grid2op environment.
     
    """
    MULTI_CHRONICS = False
       
    def __init__(
        self,
        path,  # can be None !
        ep_data: Union[str, Path],
        time_interval=timedelta(minutes=5),
        sep=";",  # here for compatibility with grid2op, but not used
        max_iter=-1,
        start_datetime=datetime(year=2019, month=1, day=1),
        chunk_size=None,
        h_forecast=(5,),
    ):
        GridValue.__init__(
            self,
            time_interval=time_interval,
            max_iter=max_iter,
            start_datetime=start_datetime,
            chunk_size=chunk_size,
        )
        
        self.path = path
        if self.path is not None:
            # logger: this has no impact
            pass
        
        if isinstance(ep_data, EpisodeData):
            self._episode_data = ep_data
        elif isinstance(ep_data, (str, Path)):
            try:
                self._episode_data = EpisodeData.from_disk(*os.path.split(ep_data))
            except Exception as exc_:
                raise ChronicsError("Impossible to build the FromOneEpisodeData with the `ep_data` provided.") from exc_
        elif isinstance(ep_data, (tuple, list)):
            if len(ep_data) != 2:
                raise ChronicsError("When you provide a tuple, or a list, FromOneEpisodeData can only be used if this list has length 2. "
                                    f"Length {len(ep_data)} found.")
            try:
                self._episode_data = EpisodeData.from_disk(*ep_data)
            except Exception as exc_:
                raise ChronicsError("Impossible to build the FromOneEpisodeData with the `ep_data` provided.") from exc_
        else:
            raise ChronicsError("FromOneEpisodeData can only read data either directly from an EpisodeData, from a path pointing to one, or from a tuple")
        self.current_inj = None
            
    def initialize(
        self,
        order_backend_loads,
        order_backend_prods,
        order_backend_lines,
        order_backend_subs,
        names_chronics_to_backend=None,
    ):
        # set the current path of the time series
        self._set_path(self.path)
        
        self.n_gen = len(order_backend_prods)
        self.n_load = len(order_backend_loads)
        self.n_line = len(order_backend_lines)
        self.curr_iter = 0
        self.current_inj = None
            
        # TODO check if consistent, and compute the order !
            
        # when there are no maintenance / hazards, build this only once 
        self._no_mh_time = np.full(self.n_line, fill_value=-1, dtype=dt_int)
        self._no_mh_duration = np.full(self.n_line, fill_value=0, dtype=dt_int)
        
    def load_next(self):
        self.current_datetime += self.time_interval
        self.curr_iter += 1
        
        res = {}
        # load the injection
        dict_inj, prod_v = self._load_injection()
        res["injection"] = dict_inj
        
        # load maintenance
        obs = self._episode_data.observations[self.curr_iter]
        res["maintenance"] = obs.time_next_maintenance == 0
        maintenance_time = self._no_mh_time.copy()
        maintenance_duration = self._no_mh_duration.copy()
        # TODO !
        maintenance_time[res["maintenance"] ] = 0
        maintenance_duration[res["maintenance"] ] = 1
        
        self.current_inj = res
        return (
            self.current_datetime,
            res,
            maintenance_time,
            maintenance_duration,
            self._no_mh_duration,
            prod_v,
        )
    
    def max_timestep(self):
        if self.max_iter > 0:
            return min(self.max_iter, len(self._episode_data))
        return len(self._episode_data)
        
    def next_chronics(self):
        self.current_datetime = self.start_datetime
        self.curr_iter = 0
    
    def done(self):
        # I am done if the part I control is "over"
        if self._max_iter > 0 and self.curr_iter > self._max_iter:
            return True
        # TODO
        return False
    
    def check_validity(self, backend):
        # TODO
        return True

    def _aux_forecasts(self, h_id, dict_, key,
                       for_handler, base_handler, handlers):
        if for_handler is not None:
            tmp_ = for_handler.forecast(h_id, self.current_inj, dict_, base_handler, handlers)
            if tmp_ is not None:
                dict_[key] = dt_float(1.0) * tmp_
        
    def forecasts(self):
        res = []
        if not self._forcast_handlers:
            # nothing to handle forecast in this class
            return res
        
        handlers = (self.load_p_handler, self.load_q_handler, self.gen_p_handler, self.gen_v_handler)
        for h_id, h in enumerate(self._forcast_handlers[0].get_available_horizons()):
            dict_ = {}
            self._aux_forecasts(h_id, dict_, "load_p", self.load_p_for_handler, self.load_p_handler, handlers)
            self._aux_forecasts(h_id, dict_, "load_q", self.load_q_for_handler, self.load_q_handler, handlers)
            self._aux_forecasts(h_id, dict_, "prod_p", self.gen_p_for_handler, self.gen_p_handler, handlers)
            self._aux_forecasts(h_id, dict_, "prod_v", self.gen_v_for_handler, self.gen_v_handler, handlers)
            
            res_d = {}
            if dict_:
                res_d["injection"] = dict_

            forecast_datetime = self.current_datetime + timedelta(minutes=h)
            res.append((forecast_datetime, res_d))
        return res
            
    def get_kwargs(self, dict_):
        dict_["ep_data"] = copy.deepcopy(self._episode_data)
        return dict_
    
    def get_id(self) -> str:
        if self.path is not None:
            return self.path
        else:
            # TODO EpisodeData.path !!!
            return ""
            raise NotImplementedError()
    
    def shuffle(self, shuffler=None):
        # TODO
        pass
    
    def sample_next_chronics(self, probabilities=None):
        # TODO
        pass
    
    def seed(self, seed):
        # nothing to do in this case, environment is purely deterministic
        super().seed(seed)
        
    def _load_injection(self):
        dict_ = {}
        obs = self._episode_data.observations[self.curr_iter]
        prod_v = None
        
        tmp_ = obs.load_p
        if tmp_ is not None:
            dict_["load_p"] = dt_float(1.0) * tmp_
        
        tmp_ = obs.load_q
        if tmp_ is not None:
            dict_["load_q"] = dt_float(1.0) * tmp_
        
        tmp_ = obs.gen_p
        if tmp_ is not None:
            dict_["prod_p"] = dt_float(1.0) * tmp_
        
        tmp_ = obs.gen_v
        if tmp_ is not None:
            prod_v = dt_float(1.0) * tmp_
            
        return dict_, prod_v
        
    def _init_date_time(self):  # in csv handler
        if os.path.exists(os.path.join(self.path, "start_datetime.info")):
            with open(os.path.join(self.path, "start_datetime.info"), "r") as f:
                a = f.read().rstrip().lstrip()
            try:
                tmp = datetime.strptime(a, "%Y-%m-%d %H:%M")
            except ValueError:
                tmp = datetime.strptime(a, "%Y-%m-%d")
            except Exception:
                raise ChronicsNotFoundError(
                    'Impossible to understand the content of "start_datetime.info". Make sure '
                    'it\'s composed of only one line with a datetime in the "%Y-%m-%d %H:%M"'
                    "format."
                )
            self.start_datetime = tmp
            self.current_datetime = tmp

        if os.path.exists(os.path.join(self.path, "time_interval.info")):
            with open(os.path.join(self.path, "time_interval.info"), "r") as f:
                a = f.read().rstrip().lstrip()
            try:
                tmp = datetime.strptime(a, "%H:%M")
            except ValueError:
                tmp = datetime.strptime(a, "%M")
            except Exception:
                raise ChronicsNotFoundError(
                    'Impossible to understand the content of "time_interval.info". Make sure '
                    'it\'s composed of only one line with a datetime in the "%H:%M"'
                    "format."
                )
            self.time_interval = timedelta(hours=tmp.hour, minutes=tmp.minute)
            
    def fast_forward(self, nb_timestep):
        for _ in range(nb_timestep):
            self.load_next()
            # for this class I suppose the real data AND the forecast are read each step
            self.forecasts()
