# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import json
from typing import Optional, Union
import numpy as np
import hashlib
from datetime import datetime, timedelta

import grid2op
from grid2op.dtypes import dt_bool
from grid2op.Chronics.GSFFWFWM import GridStateFromFileWithForecastsWithMaintenance
from grid2op.dtypes import dt_int
from grid2op.Chronics import GridValue, ChangeNothing
from grid2op.Exceptions import ChronicsError
            

class FromChronix2grid(GridValue):
    def __init__(self,
                 env_path: os.PathLike,
                 with_maintenance: bool,
                 with_loss: bool = True,
                 time_interval: timedelta = timedelta(minutes=5),
                 max_iter: int = 2016,  # set to one week (default)
                 start_datetime: datetime = datetime(year=2019, month=1, day=1),
                 chunk_size: Optional[int] = None,
                 **kwargs):
        
        # here to prevent circular import
        try:
            from chronix2grid.grid2op_utils import generate_one_episode
        except ImportError as exc_:
            raise ChronicsError(
                f"Chronix2grid package is not installed. Install it with `pip install grid2op[chronix2grid]`"
                f"Please visit https://github.com/bdonnot/chronix2grid#installation "
                f"for further install instructions."
            ) from exc_

        self._generate_one_episode = generate_one_episode
        
        GridValue.__init__(
            self,
            time_interval=time_interval,
            max_iter=max_iter,
            start_datetime=start_datetime,
            chunk_size=chunk_size,
        )
        import grid2op
        self.env = grid2op.make(env_path, _add_to_name="_fromChronix2grid", chronics_class=ChangeNothing)
    
        # required parameters
        with open(os.path.join(self.env.get_path_env(), "scenario_params.json"), "r", encoding="utf-8") as f:
            self.dict_ref = json.load(f)
            
        self.dt = self.dict_ref["dt"]
        self.li_months = self.dict_ref["all_dates"]
        
        self.current_index = 0
        
        self._load_p = None
        self._load_q = None
        self._gen_p = None
        self._gen_v = None
        
        self.with_maintenance = with_maintenance
        if with_maintenance:
            # initialize the parameters from the json
            # TODO copy paste from GridStateFromFileWithForecastWithMaintenance
            with open(
                os.path.join(env_path, "maintenance_meta.json"), "r", encoding="utf-8"
            ) as f:
                dict_ = json.load(f)

            self.maintenance_starting_hour = dict_["maintenance_starting_hour"]
            self.maintenance_ending_hour = dict_["maintenance_ending_hour"]

            self.line_to_maintenance = set(dict_["line_to_maintenance"])

            # frequencies of maintenance
            self.daily_proba_per_month_maintenance = dict_[
                "daily_proba_per_month_maintenance"
            ]

            self.max_daily_number_per_month_maintenance = dict_[
                "max_daily_number_per_month_maintenance"
            ]
            
        self.maintenance = None  # TODO
        self.maintenance_time = None
        self.maintenance_duration = None
        self.maintenance_time_nomaint = None
        self.maintenance_duration_nomaint = None
        
        self.hazards = None  # TODO
        self.has_hazards = False  # TODO
        self.hazard_duration_nohaz = None
        
        self._forecasts = None  # TODO
        
        self._init_datetime = None
        self._seed_used_for_chronix2grid = None
        self._reuse_seed = False
        
        self._with_loss = with_loss
        
    def check_validity(
        self, backend: Optional["grid2op.Backend.backend.Backend"]
    ) -> None:
        pass
        # TODO also do some checks here !
            
    def initialize(
        self,
        order_backend_loads,
        order_backend_prods,
        order_backend_lines,
        order_backend_subs,
        names_chronics_to_backend=None,
    ):
        self.n_line = len(order_backend_lines)
        self.maintenance_time_nomaint = np.zeros(shape=(self.n_line,), dtype=dt_int) - 1
        self.maintenance_duration_nomaint = np.zeros(shape=(self.n_line,), dtype=dt_int)
        self.hazard_duration_nohaz = np.zeros(shape=(self.n_line,), dtype=dt_int)
        self.next_chronics()
        # TODO perform the checks: number of loads, name of the laods etc.
    
    def get_id(self) -> str:
        # get the seed
        return f"{self._seed_used_for_chronix2grid}@{self._init_datetime}"
    
    def tell_id(self, id_, previous=False):
        _seed_used_for_chronix2grid, datetime_ = id_.split("@")
        self._seed_used_for_chronix2grid = int(_seed_used_for_chronix2grid)
        self._init_datetime = datetime_
        self._reuse_seed = True
        
    def load_next(self):# TODO refacto with fromNPY
        self.current_index += 1

        if self.current_index >= self._load_p.shape[0]:
            raise StopIteration

        res = {}
        dict_ = {}
        prod_v = None
        if self._load_p is not None:
            dict_["load_p"] = 1.0 * self._load_p[self.current_index, :]
        if self._load_q is not None:
            dict_["load_q"] = 1.0 * self._load_q[self.current_index, :]
        if self._gen_p is not None:
            dict_["prod_p"] = 1.0 * self._gen_p[self.current_index, :]
        if self._gen_v is not None:
            prod_v = 1.0 * self._gen_v[self.current_index, :]
        if dict_:
            res["injection"] = dict_

        if self.maintenance is not None and self.with_maintenance:
            res["maintenance"] = self.maintenance[self.current_index, :]
        if self.hazards is not None and self.has_hazards:
            res["hazards"] = self.hazards[self.current_index, :]

        self.current_datetime += self.time_interval
        self.curr_iter += 1

        if (
            self.maintenance_time is not None
            and self.maintenance_duration is not None
            and self.with_maintenance
        ):
            maintenance_time = dt_int(1 * self.maintenance_time[self.current_index, :])
            maintenance_duration = dt_int(
                1 * self.maintenance_duration[self.current_index, :]
            )
        else:
            maintenance_time = self.maintenance_time_nomaint
            maintenance_duration = self.maintenance_duration_nomaint

        if self.hazard_duration is not None and self.has_hazards:
            hazard_duration = 1 * self.hazard_duration[self.current_index, :]
        else:
            hazard_duration = self.hazard_duration_nohaz

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
    
    def forecasts(self):
        """
        By default, forecasts are only made 1 step ahead.

        We could change that. Do not hesitate to make a feature request
        (https://github.com/rte-france/Grid2Op/issues/new?assignees=&labels=enhancement&template=feature_request.md&title=) if that is necessary for you.
        """
        # TODO implement that and maybe refacto with fromNPY ?
        if self._forecasts is None:
            return []
        self._forecasts.current_index = self.current_index - 1
        dt, dict_, *rest = self._forecasts.load_next()
        return [(self.current_datetime + self.time_interval, dict_)]
    
    def done(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Compare to :func:`GridValue.done` an episode can be over for 2 main reasons:

          - :attr:`GridValue.max_iter` has been reached
          - There are no data in the numpy array.
          - i_end has been reached

        The episode is done if one of the above condition is met.

        Returns
        -------
        res: ``bool``
            Whether the episode has reached its end or not.

        """
        res = False
        if self.current_index >= self._load_p.shape[0]:
            res = True
        elif self.max_iter > 0:
            if self.curr_iter > self.max_iter:
                res = True
        return res
    
    def next_chronics(self):
        # generate the next possible chronics
        if not self._reuse_seed:
            self._init_datetime = self.space_prng.choice(self.li_months, 1)[0]
            self._seed_used_for_chronix2grid = self.space_prng.randint(2**32-1)
            self._reuse_seed = False
        self.current_datetime = datetime.strptime(self._init_datetime, "%Y-%m-%d")
        
        self.curr_iter = 0
        self.current_index = self.curr_iter
        res_gen = self._generate_one_episode(self.env, self.dict_ref, self.dt, self._init_datetime,
                                             seed=self._seed_used_for_chronix2grid,
                                             with_loss=self._with_loss,
                                             nb_steps=self.max_iter)
        
        self._load_p = res_gen[0].values
        self._load_p_forecasted = res_gen[1].values
        self._load_q = res_gen[2].values
        self._load_q_forecasted = res_gen[3].values
        self._gen_p = res_gen[4].values
        self._gen_p_forecasted = res_gen[5].values
        
        if self.with_maintenance:
            self.maintenance = GridStateFromFileWithForecastsWithMaintenance._generate_matenance_static(
                self.env.name_line,
                self.max_iter,
                self.line_to_maintenance,
                self.time_interval,
                self.current_datetime,
                self.maintenance_starting_hour,
                self.maintenance_ending_hour,
                self.daily_proba_per_month_maintenance,
                self.max_daily_number_per_month_maintenance,
                self.space_prng
            )
            
            ##########
            # same as before in GridStateFromFileWithForecasts
            self.maintenance_time = (
                np.zeros(shape=(self.maintenance.shape[0], self.n_line), dtype=dt_int) - 1
            )
            self.maintenance_duration = np.zeros(
                shape=(self.maintenance.shape[0], self.n_line), dtype=dt_int
            )

            # test that with chunk size
            for line_id in range(self.n_line):
                self.maintenance_time[:, line_id] = self.get_maintenance_time_1d(
                    self.maintenance[:, line_id]
                )
                self.maintenance_duration[:, line_id] = self.get_maintenance_duration_1d(
                    self.maintenance[:, line_id]
                )

            # there are _maintenance and hazards only if the value in the file is not 0.
            self.maintenance = self.maintenance != 0.0
            self.maintenance = self.maintenance.astype(dt_bool)
        self.check_validity(backend=None)
        