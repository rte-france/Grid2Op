# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import copy
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta

from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Exceptions import IncorrectNumberOfElements, ChronicsError, ChronicsNotFoundError
from grid2op.Exceptions import IncorrectNumberOfLoads, IncorrectNumberOfGenerators, IncorrectNumberOfLines
from grid2op.Exceptions import EnvError, InsufficientData
from grid2op.Chronics.GridValue import GridValue


class FromNPY(GridValue):
    """
    This class allows to generate some chronics compatible with grid2op if the data are provided in numpy format.

    It also enables the use of the starting the chronics at different time and end if a different time.

    .. warnings::
        It assume the order of the elements are consistent with the powergrid backend ! It will not attempt to reorder the columns of the dataset

    Examples
    --------

    Usage example, for what you don't really have to do:

    .. code-block:: python

        import grid2op
        from grid2op.Chronics import FromNPY

        env_name = ...
        env = grid2op.make(env_name, chronics_class=FromNPY, data_feeding_kwargs={"i_start": 5, "i_end": 18})
    """
    def __init__(self,
                 load_p,
                 load_q,
                 prod_p,
                 prod_v,  # TODO make prod_v optional
                 hazards=None,
                 maintenance=None,
                 load_p_forecast=None,  # TODO forecasts !!
                 load_q_forecast=None,
                 prod_p_forecast=None,
                 prod_v_forecast=None,
                 time_interval=timedelta(minutes=5),
                 max_iter=-1,
                 start_datetime=datetime(year=2019, month=1, day=1),
                 chunk_size=None,
                 i_start=0,
                 i_end=None,  # excluded, as always in python
                 **kwargs):
        GridValue.__init__(self, time_interval=time_interval, max_iter=max_iter, start_datetime=start_datetime,
                           chunk_size=chunk_size)
        self._i_start = i_start
        self._i_end = i_end
        self.n_gen = prod_p.shape[1]
        self.n_load = load_p.shape[1]
        self.n_line = None

        assert load_p.shape[0] == load_q.shape[0]
        assert load_p.shape[0] == prod_p.shape[0]
        assert load_p.shape[0] == prod_v.shape[0]

        self.load_p = 1.0 * load_p
        self.load_q = 1.0 * load_q
        self.prod_p = 1.0 * prod_p
        self.prod_v = 1.0 * prod_v

        if self._i_end is None:
            self._i_end = load_p.shape[0]

        self.has_maintenance = False
        self.maintenance = None
        self.maintenance_duration = None
        self.maintenance_time = None
        if maintenance is not None:
            self.has_maintenance = True
            self.n_line = maintenance.shape[1]
            assert load_p.shape[0] == maintenance.shape[0]
            self.maintenance = maintenance  # TODO copy

            self.maintenance_time = np.zeros(shape=(self.maintenance.shape[0], self.n_line), dtype=dt_int) - 1
            self.maintenance_duration = np.zeros(shape=(self.maintenance.shape[0], self.n_line), dtype=dt_int)
            for line_id in range(self.n_line):
                self.maintenance_time[:, line_id] = self.get_maintenance_time_1d(self.maintenance[:, line_id])
                self.maintenance_duration[:, line_id] = self.get_maintenance_duration_1d(self.maintenance[:, line_id])

        self.has_hazards = False
        self.hazards = None
        self.hazard_duration = None
        if hazards is not None:
            self.has_hazards = True
            if self.n_line is None:
                self.n_line = hazards.shape[1]
            else:
                assert self.n_line == hazards.shape[1]
            assert load_p.shape[0] == hazards.shape[0]

            self.hazards = hazards  # TODO copy !
            self.hazard_duration = np.zeros(shape=(self.hazards.shape[0], self.n_line), dtype=dt_int)
            for line_id in range(self.n_line):
                self.hazard_duration[:, line_id] = self.get_hazard_duration_1d(self.hazards[:, line_id])


    def initialize(self, order_backend_loads, order_backend_prods, order_backend_lines, order_backend_subs,
                   names_chronics_to_backend=None):
        assert len(order_backend_prods) == self.n_gen
        assert len(order_backend_loads) == self.n_load
        if self.n_line is None:
            self.n_line = len(order_backend_lines)
            self.maintenance_time = np.zeros(shape=(self.n_line, ), dtype=dt_int) - 1
            self.maintenance_duration = np.zeros(shape=(self.n_line, ), dtype=dt_int)
            self.hazard_duration = np.zeros(shape=(self.n_line, ), dtype=dt_int)
        else:
            assert len(order_backend_lines) == self.n_line

        self.maintenance_time_nomaint = np.zeros(shape=(self.n_line, ), dtype=dt_int) - 1
        self.maintenance_duration_nomaint = np.zeros(shape=(self.n_line, ), dtype=dt_int)
        self.hazard_duration_nohaz = np.zeros(shape=(self.n_line, ), dtype=dt_int)

        self.curr_iter = 0
        self.current_index = self._i_start - 1

    def load_next(self):
        self.current_index += 1

        if self.current_index > self._i_end or self.current_index >= self.load_p.shape[0]:
            raise StopIteration

        res = {}
        dict_ = {}
        prod_v = None
        if self.load_p is not None:
            dict_["load_p"] = 1.0 * self.load_p[self.current_index, :]
        if self.load_q is not None:
            dict_["load_q"] = 1.0 * self.load_q[self.current_index, :]
        if self.prod_p is not None:
            dict_["prod_p"] = 1.0 * self.prod_p[self.current_index, :]
        if self.prod_v is not None:
            prod_v = 1.0 * self.prod_v[self.current_index, :]
            # dict_["prod_v"] = prod_v
        if dict_:
            res["injection"] = dict_

        if self.maintenance is not None and self.has_maintenance:
            res["maintenance"] = self.maintenance[self.current_index, :]
        if self.hazards is not None and self.has_hazards:
            res["hazards"] = self.hazards[self.current_index, :]

        self.current_datetime += self.time_interval
        self.curr_iter += 1

        if self.maintenance_time is not None and self.has_maintenance:
            maintenance_time = dt_int(1 * self.maintenance_time[self.current_index, :])
            maintenance_duration = dt_int(1 * self.maintenance_duration[self.current_index, :])
        else:
            maintenance_time = np.full(self.n_line, fill_value=-1, dtype=dt_int)
            maintenance_duration = np.full(self.n_line, fill_value=0, dtype=dt_int)

        if self.hazard_duration is not None and self.has_hazards:
            hazard_duration = 1 * self.hazard_duration[self.current_index, :]
        else:
            hazard_duration = np.full(self.n_line, fill_value=-1, dtype=dt_int)

        return self.current_datetime, res, maintenance_time, maintenance_duration, hazard_duration, prod_v

    def check_validity(self, backend):
        return True

    def next_chronics(self):
        # restart the chronics: read it again !
        self.current_datetime = self.start_datetime
        self.curr_iter = 0
        self.current_index = self._i_start

    def done(self):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Compare to :func:`GridValue.done` an episode can be over for 2 main reasons:

          - :attr:`GridValue.max_iter` has been reached
          - There are no data in the csv.

        The episode is done if one of the above condition is met.

        Returns
        -------
        res: ``bool``
            Whether the episode has reached its end or not.

        """
        res = False
        if self.current_index >= self._i_end or self.current_index >= self.load_p.shape[0]:
            res = True
        elif self.max_iter > 0:
            if self.curr_iter > self.max_iter:
                res = True
        return res

    def forecasts(self):
        # TODO
        return []
