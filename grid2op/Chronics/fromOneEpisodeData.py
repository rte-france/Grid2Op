# Copyright (c) 2023, RTE (https://www.rte-france.com)
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
import warnings
from typing import Union, Tuple, Optional, Dict, Literal
from pathlib import Path

import grid2op
from grid2op.Exceptions import (
    ChronicsError, ChronicsNotFoundError
)

from grid2op.Chronics.gridValue import GridValue

from grid2op.dtypes import dt_int, dt_float
from grid2op.Episode import EpisodeData

TYPE_EP_DATA_INGESTED = Union[str, Path, EpisodeData, Tuple[str, str]]


class FromOneEpisodeData(GridValue): 
    """This class allows to use the :class:`grid2op.Chronics.handlers.BaseHandler` to read back data 
    stored in :class:`grid2op.Episode.EpisodeData`
    
    It can be used if you want to loop indefinitely through one episode.
    
    .. versionadded:: 1.9.4
    
    TODO there will be "perfect" forecast, as original forecasts are not stored !

    .. warning::
        Original forecasts are not stored by the runner. This is why you cannot use
        the same information as available in the original "obs.simulate".
        
        However, you can still use PERFECT FORECAST if you want to by providing the extra
        parameters "list_perfect_forecasts=[forecast_horizon_1, forecast_horizon_2, etc.]"
        when you build this class. (see examples below)
         
    .. danger::
        If you want the created environment to be exactly that the original environment, make
        sure to generate data using a "do nothing" agent.
        
        If the agent modified the injections (*eg* with redispatching, curtailment or storage)
        then the resulting time series will "embed" these modifications: they will
        NOT match the orignal implementation
    
    .. danger::
        If you load an episode data with an opponent, make sure also to build your environment with
        :class:`grid2op.Opponent.FromEpisodeDataOpponent` and assign `opponent_attack_cooldown=1`
        (see example below) otherwise you might end up with different time series than what you
        initially had in the EpisodeData.
    
    .. note::
        As this class reads from the hard drive an episode that has been played, we strongly encourage
        you to build this class with a complete episode (and not using an agent that games over after a 
        few steps), for example by using the "RecoPowerlineAgent" and the `NO_OVERFLOW_DISCONNECTION` 
        parameters (see example below)
    
    .. seealso::
        :class:`grid2op.Chronics.FromMultiEpisodeData` if you want to use multiple episode data
        
    Examples
    ---------
    You can use this class this way:
    
    First, you generate some data by running an episode with do nothing or reco powerline agent,
    preferably episode that go until the end of your time series
    
    .. code-block:: python
    
        import grid2op
        from grid2op.Runner import Runner
        from grid2op.Agent import RecoPowerlineAgent
        
        path_agent = ....
        env_name = "l2rpn_case14_sandbox"  # or any other name
        env = grid2op.make(env_name, etc.)
        
        # optional (change the parameters to allow the )
        param = env.parameters
        param.NO_OVERFLOW_DISCONNECTION = True
        env.change_parameters(param)
        env.reset()
        # end optional
        
        runner = Runner(**env.get_params_for_runner(),
                        agentClass=RecoPowerlineAgent)
        runner.run(nb_episode=1,
                   path_save=path_agent)
    
    And then you can load it back and run the exact same environment with the same
    time series, the same attacks etc. with:
    
    .. code-block:: python
    
        import grid2op
        from grid2op.Chronics import FromOneEpisodeData
        from grid2op.Opponent import FromEpisodeDataOpponent
        from grid2op.Episode import EpisodeData
        
        path_agent = ....  # same as above
        env_name = .... # same as above
        
        # path_agent is the path where data coming from a grid2op runner are stored
        # NB it should come from a do nothing agent, or at least
        # an agent that does not modify the injections (no redispatching, curtailment, storage)
        li_episode = EpisodeData.list_episode(path_agent)
        ep_data = li_episode[0]
        
        env = grid2op.make(env_name,
                           chronics_class=FromOneEpisodeData,
                           data_feeding_kwargs={"ep_data": ep_data},
                           opponent_class=FromEpisodeDataOpponent,
                           opponent_attack_cooldown=1,
                      )
        # ep_data can be either a tuple of 2 elements (like above)
        # or a full path to a saved episode
        # or directly an object of type EpisodeData

        obs = env.reset()
    
        # and now you can use "env" as any grid2op environment.
    
    If you want to include perfect forecast (unfortunately you cannot retrieve the original forecasts)
    you can do:
    
    .. code-block:: python
    
        # same as above
        
        env = grid2op.make(env_name,
                    chronics_class=FromOneEpisodeData,
                    data_feeding_kwargs={"ep_data": ep_data,
                                         "list_perfect_forecasts": (5, 10, 15)},
                    opponent_class=FromEpisodeDataOpponent,
                    opponent_attack_cooldown=1,
                )
        # it creates an environment with perfect forecasts available for the next step (5),
        # the step afterwards (10) and again the following one (15)
        
    .. seealso::
        :class:`grid2op.Opponent.FromEpisodeDataOpponent`
        
    """
    MULTI_CHRONICS = False
       
    def __init__(
        self,
        path,  # can be None !
        ep_data: TYPE_EP_DATA_INGESTED,
        time_interval=timedelta(minutes=5),
        sep=";",  # here for compatibility with grid2op, but not used
        max_iter=-1,
        start_datetime=datetime(year=2019, month=1, day=1),
        chunk_size=None,
        list_perfect_forecasts=None,  # TODO
        **kwargs,  # unused
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
            raise ChronicsError("FromOneEpisodeData can only read data either directly from an EpisodeData, "
                                "from a path pointing to one, or from a tuple")
        self.current_inj = None
        
        if list_perfect_forecasts is not None:
            self.list_perfect_forecasts = list_perfect_forecasts
        else:
            self.list_perfect_forecasts = []
        self._check_list_perfect_forecasts()
    
    def _check_list_perfect_forecasts(self):
        if not self.list_perfect_forecasts:
            return
        self.list_perfect_forecasts = [int(el) for el in self.list_perfect_forecasts]
        for horizon in self.list_perfect_forecasts:
            tmp = horizon * 60. / self.time_interval.total_seconds()
            if tmp - int(tmp) != 0:
                raise ChronicsError(f"All forecast horizons should be multiple of self.time_interval (and given in minutes), found {horizon}")

        for h_id, horizon in enumerate(self.list_perfect_forecasts):
            if horizon * 60 != (h_id + 1) * (self.time_interval.total_seconds()):
                raise ChronicsError("For now all horizons should be consecutive, you cannot 'skip' a forecast: (5, 10, 15) " 
                                    "is ok but (5, 15, 30) is NOT.")
        
    def initialize(
        self,
        order_backend_loads,
        order_backend_prods,
        order_backend_lines,
        order_backend_subs,
        names_chronics_to_backend=None,
    ):
        # set the current path of the time series
        
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
        obs = self._episode_data.observations[self.curr_iter]
        self.current_datetime += self.time_interval
        self.curr_iter += 1
         
        res = {}
        # load the injection
        dict_inj, prod_v = self._load_injection(obs)
        res["injection"] = dict_inj
        
        # load maintenance
        res["maintenance"] = obs.time_next_maintenance == 0
        maintenance_time = 1 * obs.time_next_maintenance
        maintenance_duration = 1 * obs.duration_next_maintenance
        
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
        if self.curr_iter > len(self._episode_data):
            return True
        return False
    
    def check_validity(self, backend):
        warning_msg = ("An action modified the injection with {}, resulting data might be "
                       "different from original data used in the generation of the initial EpisodeData.")
        redisp_issued = False
        sto_issued = False
        curt_issued = False
        for act in self._episode_data.actions:
            if act._modif_redispatch:
                if not redisp_issued:
                    warnings.warn(warning_msg.format("redispatching"))
                    redisp_issued = True
            if act._modif_storage:
                if not sto_issued:
                    warnings.warn(warning_msg.format("storage"))
                    sto_issued = True
            if act._modif_curtailment:
                if not curt_issued:
                    warnings.warn(warning_msg.format("curtailment"))
                    curt_issued = True
        return True

    def _aux_forecasts(self, h_id, dict_, key,
                       for_handler, base_handler, handlers):
        if for_handler is not None:
            tmp_ = for_handler.forecast(h_id, self.current_inj, dict_, base_handler, handlers)
            if tmp_ is not None:
                dict_[key] = dt_float(1.0) * tmp_
        
    def forecasts(self):
        """Retrieve PERFECT forecast from this time series generator.
        
        .. danger::
            These are **perfect forecast** and **NOT** the original forecasts.
            
        Notes
        -----
        As in grid2op the forecast information is not stored by the runner, 
        it is NOT POSSIBLE to retrieve the forecast informations used by the 
        "original" env (the one that generated the EpisodeData).
        
        This class however, thanks to the `list_perfect_forecasts` kwarg you
        can set at building time, can generate perfect forecasts: the agent will
        see into the future if using these forecasts.

        """
        if not self.list_perfect_forecasts:
            return []
        res = []
        for h_id, h in enumerate(self.list_perfect_forecasts):
            res_d = {}
            obs = self._episode_data.observations[min(self.curr_iter + h_id, len(self._episode_data) - 1)]
            # load the injection
            dict_inj, prod_v = self._load_injection(obs)
            dict_inj["prod_v"] = prod_v
            res_d["injection"] = dict_inj
            forecast_datetime = self.current_datetime + timedelta(minutes=h)
            res.append((forecast_datetime, res_d))
        return res
            
    def get_kwargs(self, dict_):
        dict_["ep_data"] = copy.deepcopy(self._episode_data)
        # dict_["list_perfect_forecasts"] = copy.deepcopy(self.list_perfect_forecasts)
        return dict_
    
    def get_id(self) -> str:
        if self.path is not None:
            return self.path
        else:
            # TODO EpisodeData.path !!!
            return ""
    
    def shuffle(self, shuffler=None):
        # TODO
        pass
    
    def sample_next_chronics(self, probabilities=None):
        # TODO
        pass
    
    def seed(self, seed):
        # nothing to do in this case, environment is purely deterministic
        super().seed(seed)
        
    def _load_injection(self, obs):
        dict_ = {}
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
        
    def _init_date_time(self):  # from csv handler
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
            
    def get_init_action(self, names_chronics_to_backend: Optional[Dict[Literal["loads", "prods", "lines"], Dict[str, str]]]=None) -> Union["grid2op.Action.playableAction.PlayableAction", None]:
        # names_chronics_to_backend is ignored because it does not really make sense
        # when read from the hard drive
        obs = self._episode_data.observations[0]
        dict_set = {"set_bus": obs.topo_vect}
        if self.action_space.supports_type("redispatch"):
            dict_set["redispatch"] = obs.target_dispatch
        if self.action_space.supports_type("set_storage"):
            dict_set["set_storage"] = obs.storage_power_target
        if self.action_space.supports_type("curtail"):
            dict_set["curtail"] = obs.curtailment_limit
            dict_set["curtail"][~type(obs).gen_renewable] = -1
        # TODO shunts !
        return self.action_space(dict_set, check_legal=False)
