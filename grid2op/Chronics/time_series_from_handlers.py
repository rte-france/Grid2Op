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
from typing import Optional

from grid2op.Exceptions import (
    ChronicsNotFoundError, HandlerError
)

from grid2op.Chronics.gridValue import GridValue
from grid2op.Chronics.handlers import BaseHandler

from grid2op.dtypes import dt_int, dt_float


class FromHandlers(GridValue): 
    """This class allows to use the :class:`grid2op.Chronics.handlers.BaseHandler` to 
    generate the "input time series" of the environment.

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    HandlerError
        _description_
    NotImplementedError
        _description_
    ChronicsNotFoundError
        _description_
    ChronicsNotFoundError
        _description_
    """
    MULTI_CHRONICS = False
       
    def __init__(
        self,
        path,  # can be None !
        gen_p_handler,
        gen_v_handler,
        load_p_handler,
        load_q_handler,
        maintenance_handler=None,
        hazards_handler=None,
        gen_p_for_handler=None,
        gen_v_for_handler=None,
        load_p_for_handler=None,
        load_q_for_handler=None,
        time_interval=timedelta(minutes=5),
        sep=";",
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
            self._init_date_time()
            
        # all my "handlers" (I need to perform a deepcopy otherwise data are kept between episode...)
        self.gen_p_handler : BaseHandler = copy.deepcopy(gen_p_handler)
        self.gen_v_handler : BaseHandler = copy.deepcopy(gen_v_handler)
        self.load_p_handler : BaseHandler = copy.deepcopy(load_p_handler)
        self.load_q_handler : BaseHandler = copy.deepcopy(load_q_handler)
        self.maintenance_handler : Optional[BaseHandler] = copy.deepcopy(maintenance_handler)
        self.hazards_handler : Optional[BaseHandler] = copy.deepcopy(hazards_handler)
        self.gen_p_for_handler : Optional[BaseHandler] = copy.deepcopy(gen_p_for_handler)
        self.gen_v_for_handler : Optional[BaseHandler] = copy.deepcopy(gen_v_for_handler)
        self.load_p_for_handler : Optional[BaseHandler] = copy.deepcopy(load_p_for_handler)
        self.load_q_for_handler : Optional[BaseHandler] = copy.deepcopy(load_q_for_handler)
        
        # when there are no maintenance / hazards, build this only once 
        self._no_mh_time = None
        self._no_mh_duration = None

        # define the active handlers
        self._active_handlers = [self.gen_p_handler, self.gen_v_handler, self.load_p_handler, self.load_q_handler]
        self._forcast_handlers = []
        if self.maintenance_handler is not None:
            self._active_handlers.append(self.maintenance_handler)
        if self.hazards_handler is not None:
            self._active_handlers.append(self.hazards_handler)
        if self.gen_p_for_handler is not None:
            self._active_handlers.append(self.gen_p_for_handler)
            self._forcast_handlers.append(self.gen_p_for_handler)
        if self.gen_v_for_handler is not None:
            self._active_handlers.append(self.gen_v_for_handler)
            self._forcast_handlers.append(self.gen_v_for_handler)
        if self.load_p_for_handler is not None:
            self._active_handlers.append(self.load_p_for_handler)
            self._forcast_handlers.append(self.load_p_for_handler)
        if self.load_q_for_handler is not None:
            self._active_handlers.append(self.load_q_for_handler)
            self._forcast_handlers.append(self.load_q_for_handler)
        self._check_types()
        
        # now synch all handlers
        for handl in self._forcast_handlers:
            handl.set_h_forecast(h_forecast)
            
        # set the current path of the time series
        self._set_path(self.path)
        
        if chunk_size is not None:
            self.set_chunk_size(chunk_size)
        
        if max_iter != -1:
            self.set_max_iter(max_iter)
            
        self.init_datetime()
        self.current_inj = None
    
    def _check_types(self):
        for handl in self._active_handlers:
            if not isinstance(handl, BaseHandler):
                raise HandlerError("One of the \"handler\" used in your time series does not "
                                   "inherit from `BaseHandler`. This is not supported.")
            
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
        
        # give the right date and times to the "handlers"
        self.init_datetime()
        
        self.n_gen = len(order_backend_prods)
        self.n_load = len(order_backend_loads)
        self.n_line = len(order_backend_lines)
        self.curr_iter = 0
        self.current_inj = None
        
        self.gen_p_handler.initialize(order_backend_prods, names_chronics_to_backend)
        self.gen_v_handler.initialize(order_backend_prods, names_chronics_to_backend)
        self.load_p_handler.initialize(order_backend_loads, names_chronics_to_backend)
        self.load_q_handler.initialize(order_backend_loads, names_chronics_to_backend)
        
        self._update_max_iter()  # might be used in the forecast
        if self.gen_p_for_handler is not None:
            self.gen_p_for_handler.initialize(order_backend_prods, names_chronics_to_backend)
        if self.gen_v_for_handler is not None:
            self.gen_v_for_handler.initialize(order_backend_prods, names_chronics_to_backend)
        if self.load_p_for_handler is not None:
            self.load_p_for_handler.initialize(order_backend_loads, names_chronics_to_backend)
        if self.load_q_for_handler is not None:
            self.load_q_for_handler.initialize(order_backend_loads, names_chronics_to_backend)
        
        self._update_max_iter()  # might be used in the maintenance
        if self.maintenance_handler is not None:
            self.maintenance_handler.initialize(order_backend_lines, names_chronics_to_backend)
        if self.hazards_handler is not None:
            self.hazards_handler.initialize(order_backend_lines, names_chronics_to_backend)
            
        # when there are no maintenance / hazards, build this only once 
        self._no_mh_time = np.full(self.n_line, fill_value=-1, dtype=dt_int)
        self._no_mh_duration = np.full(self.n_line, fill_value=0, dtype=dt_int)
        
        self._update_max_iter()
            
    def load_next(self):
        self.current_datetime += self.time_interval
        self.curr_iter += 1
        
        res = {}
        # load the injection
        dict_inj, prod_v = self._load_injection()
        res["injection"] = dict_inj
        
        # load maintenance
        if self.maintenance_handler is not None:
            tmp_ = self.maintenance_handler.load_next(res)
            if tmp_ is not None:
                res["maintenance"] = tmp_
                maintenance_time, maintenance_duration = self.maintenance_handler.load_next_maintenance()
        else:
            maintenance_time = self._no_mh_time
            maintenance_duration = self._no_mh_duration
        
        # load hazards
        if self.hazard_duration is not None:
            res["hazards"] = self.hazards_handler.load_next(res)
            hazard_duration = self.hazards_handler.load_next_hazard()
        else:
            hazard_duration = self._no_mh_duration
        
        self.current_inj = res
        return (
            self.current_datetime,
            res,
            maintenance_time,
            maintenance_duration,
            hazard_duration,
            prod_v,
        )
    
    def max_timestep(self):
        return self.max_iter
        
    def next_chronics(self):
        self.current_datetime = self.start_datetime
        self.curr_iter = 0
        
        for el in self._active_handlers:
            el.next_chronics()
            
        self._update_max_iter()
    
    def done(self):
        # I am done if the part I control is "over"
        if self.max_iter > 0:
            if self.curr_iter > self.max_iter:
                return True
            
        # or if any of the handler is "done"
        for handl in self._active_handlers:
            if handl.done():
                return True
        return False
    
    def check_validity(self, backend):
        for el in self._active_handlers:
            el.check_validity(backend)
        # TODO other things here maybe ???
        return True

    def forecasts(self):
        res = []
        if not self._forcast_handlers:
            # nothing to handle forecast in this class
            return res
        
        for h_id, h in enumerate(self._forcast_handlers[0].get_available_horizons()):
            res_d = {}
            dict_ = {}
            
            if self.load_p_for_handler is not None:
                tmp_ = self.load_p_for_handler.forecast(h_id, self.current_inj, dict_, self.load_p_handler,
                                                        (self.load_p_handler, self.load_q_handler,
                                                         self.gen_p_handler, self.gen_v_handler))
                if tmp_ is not None:
                    dict_["load_p"] = dt_float(1.0) * tmp_
            if self.load_q_for_handler is not None:
                tmp_ = self.load_q_for_handler.forecast(h_id, self.current_inj, dict_, self.load_q_handler,
                                                        (self.load_p_handler, self.load_q_handler,
                                                         self.gen_p_handler, self.gen_v_handler))
                if tmp_ is not None:
                    dict_["load_q"] = dt_float(1.0) * tmp_
            if self.gen_p_for_handler is not None:
                tmp_ = self.gen_p_for_handler.forecast(h_id, self.current_inj, dict_, self.gen_p_handler,
                                                       (self.load_p_handler, self.load_q_handler,
                                                        self.gen_p_handler, self.gen_v_handler))
                if tmp_ is not None:
                    dict_["prod_p"] = dt_float(1.0) * tmp_
            if self.gen_v_for_handler is not None:
                tmp_ = self.gen_v_for_handler.forecast(h_id, self.current_inj, dict_, self.gen_v_handler, 
                                                       (self.load_p_handler, self.load_q_handler,
                                                        self.gen_p_handler, self.gen_v_handler))
                if tmp_ is not None:
                    dict_["prod_v"] = dt_float(1.0) * tmp_
            if dict_:
                res_d["injection"] = dict_

            forecast_datetime = self.current_datetime + timedelta(minutes=h)
            res.append((forecast_datetime, res_d))
        return res
            
    def get_kwargs(self, dict_):
        dict_["gen_p_handler"] = copy.deepcopy(self.gen_p_handler)._clear() if self.gen_p_handler is not None else None
        dict_["gen_v_handler"] = copy.deepcopy(self.gen_v_handler)._clear() if self.gen_v_handler is not None else None
        dict_["load_p_handler"] = copy.deepcopy(self.load_p_handler)._clear() if self.load_p_handler is not None else None
        dict_["load_q_handler"] = copy.deepcopy(self.load_q_handler)._clear() if self.load_q_handler is not None else None
        dict_["maintenance_handler"] = copy.deepcopy(self.maintenance_handler)._clear() if self.maintenance_handler is not None else None
        dict_["hazards_handler"] = copy.deepcopy(self.hazards_handler)._clear() if self.hazards_handler is not None else None
        dict_["gen_p_for_handler"] = copy.deepcopy(self.gen_p_for_handler)._clear() if self.gen_p_for_handler is not None else None
        dict_["gen_v_for_handler"] = copy.deepcopy(self.gen_v_for_handler)._clear() if self.gen_v_for_handler is not None else None
        dict_["load_p_for_handler"] = copy.deepcopy(self.load_p_for_handler)._clear() if self.load_p_for_handler is not None else None
        dict_["load_q_for_handler"] = copy.deepcopy(self.load_q_for_handler)._clear() if self.load_q_for_handler is not None else None
        return dict_
    
    def get_id(self) -> str:
        if self.path is not None:
            return self.path
        else:
            # TODO
            raise NotImplementedError()
    
    def shuffle(self, shuffler=None):
        # TODO
        pass
    
    def sample_next_chronics(self, probabilities=None):
        # TODO
        pass
    
    def set_chunk_size(self, new_chunk_size):
        # TODO
        for el in self._active_handlers:
            el.set_chunk_size(new_chunk_size)
    
    def set_max_iter(self, max_iter):
        self.max_iter = int(max_iter)
        for el in self._active_handlers:
            el.set_max_iter(max_iter)
    
    def init_datetime(self):
        for handl in self._active_handlers:
            handl.set_times(self.start_datetime, self.time_interval)
    
    def seed(self, seed):
        super().seed(seed)
        max_seed = np.iinfo(dt_int).max
        seeds = self.space_prng.randint(max_seed, size=10)
        # this way of doing ensure the same seed given by the environment is
        # used even if some "handlers" are missing
        # (if env.seed(0) is called, then regardless of maintenance_handler or not, 
        # gen_p_for_handler will always be seeded with the same number)
        gp_seed = self.gen_p_handler.seed(seeds[0])
        gv_seed = self.gen_v_handler.seed(seeds[1])
        lp_seed = self.load_p_handler.seed(seeds[2])
        lq_seed = self.load_q_handler.seed(seeds[3])
        maint_seed = None
        if self.maintenance_handler is not None:
            maint_seed = self.maintenance_handler.seed(seeds[4])
        haz_seed = None
        if self.hazards_handler is not None:
            haz_seed = self.hazards_handler.seed(seeds[5])
        gpf_seed = None
        if self.gen_p_for_handler is not None:
            gpf_seed = self.gen_p_for_handler.seed(seeds[6])
        gvf_seed = None
        if self.gen_v_for_handler is not None:
            gvf_seed = self.gen_v_for_handler.seed(seeds[7])
        lpf_seed = None
        if self.load_p_for_handler is not None:
            lpf_seed = self.load_p_for_handler.seed(seeds[8])    
        lqf_seed = None
        if self.load_q_for_handler is not None:
            lqf_seed = self.load_q_for_handler.seed(seeds[9])   
        return (seed, gp_seed, gv_seed, lp_seed, lq_seed, 
                maint_seed, haz_seed, gpf_seed, gvf_seed,
                lpf_seed, lqf_seed) 
        
    def _set_path(self, path):
        """tell the handler where this chronics is located"""
        if path is None:
            return
        
        for el in self._active_handlers:
            el.set_path(path)
    
    def set_max_episode_duration(self, max_ep_dur):
        for handl in self._active_handlers:
            handl.set_max_episode_duration(max_ep_dur)
            
    def _update_max_iter(self):
        # get the max iter from the handlers
        max_iters = [el.get_max_iter() for el in self._active_handlers]
        max_iters = [el for el in max_iters if el != -1]
        # get the max iter from myself
        max_iters.append(self.max_iter)
        # take the minimum
        self.max_iter = np.min(max_iters)
        
        # update everyone with the "new" max iter
        max_ep_dur = [el.max_episode_duration for el in self._active_handlers]
        max_ep_dur = [el for el in max_ep_dur if el is not None]
        if max_ep_dur:
            if self.max_iter == -1:
                self.max_iter = np.min(max_ep_dur)
            else:
                self.max_iter = min(self.max_iter, np.min(max_ep_dur))
            
        if self.max_iter != -1:
            self.set_max_episode_duration(self.max_iter)
        
    def _load_injection(self):
        dict_ = {}
        prod_v = None
        if self.load_p_handler is not None:
            tmp_ = self.load_p_handler.load_next(dict_)
            if tmp_ is not None:
                dict_["load_p"] = dt_float(1.0) * tmp_
        if self.load_q_handler is not None:
            tmp_ = self.load_q_handler.load_next(dict_)
            if tmp_ is not None:
                dict_["load_q"] = dt_float(1.0) * tmp_
        if self.gen_p_handler is not None:
            tmp_ = self.gen_p_handler.load_next(dict_)
            if tmp_ is not None:
                dict_["prod_p"] = dt_float(1.0) * tmp_
        if self.gen_v_handler is not None:
            tmp_ = self.gen_v_handler.load_next(dict_)
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
