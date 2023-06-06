# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


from grid2op.dtypes import dt_bool, dt_int
from grid2op.Exceptions import Grid2OpException
from grid2op.Chronics.gridStateFromFileWithForecasts import (
    GridStateFromFileWithForecasts,
)


class GridStateFromFileWithForecastsWithoutMaintenance(GridStateFromFileWithForecasts):
    """
    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        This class is made mainly for debugging. And it is not well tested.


    Behaves exactly like "GridStateFromFileWithForecasts" but ignore all maintenance and hazards

    Examples
    ---------

    You can use it as follow:

    .. code-block:: python

        import grid2op
        from grid2op.Chronics import GridStateFromFileWithForecastsWithoutMaintenance

        env= make(ENV_NAME,
                  data_feeding_kwargs={"gridvalueClass": GridStateFromFileWithForecastsWithoutMaintenance},
                  )

        # even if there are maintenance in the environment, they will not be used.
    """

    MULTI_CHRONICS = False
    def __init__(
        self,
        path,
        sep=";",
        time_interval=timedelta(minutes=5),
        max_iter=-1,
        chunk_size=None,
        h_forecast=(5, ),
    ):
        GridStateFromFileWithForecasts.__init__(
            self,
            path,
            sep=sep,
            time_interval=time_interval,
            max_iter=max_iter,
            chunk_size=chunk_size,
            h_forecast=h_forecast,
        )

        self.n_gen = None
        self.n_load = None
        self.n_line = None

        self.maintenance_time_no_maint = None
        self.maintenance_duration_no_maint = None
        self.hazard_duration_no_haz = None

    def initialize(
        self,
        order_backend_loads,
        order_backend_prods,
        order_backend_lines,
        order_backend_subs,
        names_chronics_to_backend=None,
    ):
        self.n_gen = len(order_backend_prods)
        self.n_load = len(order_backend_loads)
        self.n_line = len(order_backend_lines)

        super().initialize(
            order_backend_loads,
            order_backend_prods,
            order_backend_lines,
            order_backend_subs,
            names_chronics_to_backend=names_chronics_to_backend,
        )

        self.maintenance_time_no_maint = (
            np.zeros(shape=(self.n_line,), dtype=dt_int) - 1
        )
        self.maintenance_duration_no_maint = np.zeros(
            shape=(self.n_line,), dtype=dt_int
        )
        self.hazard_duration_no_haz = np.zeros(shape=(self.n_line,), dtype=dt_int)

    def load_next(self):
        (
            current_datetime,
            res,
            maintenance_time,
            maintenance_duration,
            hazard_duration,
            prod_v,
        ) = super().load_next()
        if "maintenance" in res:
            del res["maintenance"]
        if "hazards" in res:
            del res["hazards"]
        return (
            current_datetime,
            res,
            self.maintenance_time_no_maint,
            self.maintenance_duration_no_maint,
            self.hazard_duration_no_haz,
            prod_v,
        )
