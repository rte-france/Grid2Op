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
from grid2op.dtypes import dt_bool, dt_int
from grid2op.Chronics import GridValue, ChangeNothing
from grid2op.Chronics.GSFFWFWM import GridStateFromFileWithForecastsWithMaintenance
from grid2op.Chronics.fromNPY import FromNPY
from grid2op.Exceptions import ChronicsError
            

class FromChronix2grid(GridValue):
    """This class of "chronix" allows to use the `chronix2grid` package to generate data "on the fly" rather
    than having to read it from the hard drive.
    
    .. versionadded:: 1.6.6
    
    
    .. warning::
        It requires the `chronix2grid` package to be installed, please install it with :
    
        `pip install grid2op[chronix2grid]`
        
        And visit https://github.com/bdonnot/chronix2grid#installation for more installation details (in particular
        you need the coinor-cbc software on your machine)
        
    As of writing, this class is really slow compared to reading data from the hard drive. Indeed to generate a week of data
    at the 5 mins time resolution (*ie* to generate the data for a "standard" episode) it takes roughly 40/45 s for
    the `l2rpn_wcci_2022` environment (based on the IEEE 118).
    
    Notes
    ------
    It requires lots of extra metadata to use this class. As of writing, only the  `l2rpn_wcci_2022` is compatible with it.
    
    
    Examples
    ----------
    To use it (though we do not recommend to use it) you can do:
    
    .. code-block:: python
    
        import grid2op
        from grid2op.Chronics import FromChronix2grid
        env_nm = "l2rpn_wcci_2022"  # only compatible environment at time of writing
        
        env = grid2op.make(env_nm,
                           chronics_class=FromChronix2grid,
                           data_feeding_kwargs={"env_path": os.path.join(grid2op.get_current_local_dir(), env_nm),
                                                "with_maintenance": True,  # whether to include maintenance (optional)
                                                "max_iter": 2 * 288,  # duration (in number of steps) of the data generated (optional)
                                                }
                           )
    
    Before using it, please consult the :ref:`generate_data_flow` section of the document, that provides a much faster way
    to do this.
    
    """
    REQUIRED_FILES = ["loads_charac.csv", "params.json", "params_load.json",
                      "params_loss.json", "params_opf.json", "params_res.json", 
                      "prods_charac.csv", "scenario_params.json"]
    MULTI_CHRONICS = False
    def __init__(self,
                 env_path: os.PathLike,
                 with_maintenance: bool,
                 with_loss: bool = True,
                 time_interval: timedelta = timedelta(minutes=5),
                 max_iter: int = 2016,  # set to one week (default)
                 start_datetime: datetime = datetime(year=2019, month=1, day=1),
                 chunk_size: Optional[int] = None,
                 **kwargs):
                
        for el in type(self).REQUIRED_FILES:
            tmp_ = os.path.join(env_path, el)
            if not (os.path.exists(tmp_) and os.path.isfile(tmp_)):
                raise ChronicsError(f"The file \"{el}\" is required but is missing from your environment. "
                                    f"Check data located at \"env_path={env_path}\" and make sure you "
                                    f"can use this environment to generate data.")
                
        GridValue.__init__(
            self,
            time_interval=time_interval,
            max_iter=max_iter,
            start_datetime=start_datetime,
            chunk_size=chunk_size,
        )
        import grid2op
        self.env = grid2op.make(env_path,
                                _add_to_name="_fromChronix2grid",
                                chronics_class=ChangeNothing,
                                data_feeding_kwargs={"max_iter": 5}  # otherwise for some opponent I might run into trouble
                                )
    
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
        
        self.has_maintenance = with_maintenance
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
    
    def _generate_one_episode(self, *args, **kwargs):
        # here to prevent circular import
        try:
            from chronix2grid.grid2op_utils import generate_one_episode
        except ImportError as exc_:
            raise ChronicsError(
                f"Chronix2grid package is not installed. Install it with `pip install grid2op[chronix2grid]`"
                f"Please visit https://github.com/bdonnot/chronix2grid#installation "
                f"for further install instructions."
            ) from exc_

        return generate_one_episode(*args, **kwargs)
    
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
        prod_v = FromNPY._create_dict_inj(res, self)
        maintenance_time, maintenance_duration, hazard_duration = FromNPY._create_dict_maintenance_hazards(res, self)

        self.current_datetime += self.time_interval
        self.curr_iter += 1

        return (
            self.current_datetime,
            res,
            maintenance_time,
            maintenance_duration,
            hazard_duration,
            prod_v,
        )
    
    def max_timestep(self):
        return self._max_iter
    
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
        elif self._max_iter > 0:
            if self.curr_iter > self._max_iter:
                res = True
        return res
    
    def next_chronics(self):
        # generate the next possible chronics
        if not self._reuse_seed:
            self._init_datetime = self.space_prng.choice(self.li_months, 1)[0]
            self._seed_used_for_chronix2grid = self.space_prng.randint(np.iinfo(dt_int).max)
            self._reuse_seed = False
        self.current_datetime = datetime.strptime(self._init_datetime, "%Y-%m-%d")
        
        self.curr_iter = 0
        self.current_index = self.curr_iter
        res_gen = self._generate_one_episode(self.env, self.dict_ref, self.dt, self._init_datetime,
                                             seed=self._seed_used_for_chronix2grid,
                                             with_loss=self._with_loss,
                                             nb_steps=self._max_iter)
        
        self._load_p = res_gen[0].values
        self._load_p_forecasted = res_gen[1].values
        self._load_q = res_gen[2].values
        self._load_q_forecasted = res_gen[3].values
        self._gen_p = res_gen[4].values
        self._gen_p_forecasted = res_gen[5].values
        
        if self.has_maintenance:
            self.maintenance = GridStateFromFileWithForecastsWithMaintenance._generate_matenance_static(
                self.env.name_line,
                self._load_p.shape[0],
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
            GridStateFromFileWithForecastsWithMaintenance._fix_maintenance_format(self)
            
        self.check_validity(backend=None)
        