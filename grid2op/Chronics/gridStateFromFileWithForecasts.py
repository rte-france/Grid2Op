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
from datetime import timedelta

from grid2op.dtypes import dt_float, dt_bool
from grid2op.Exceptions import (
    EnvError,
    IncorrectNumberOfLoads,
    IncorrectNumberOfLines,
    IncorrectNumberOfGenerators,
)
from grid2op.Exceptions import ChronicsError
from grid2op.Chronics.gridStateFromFile import GridStateFromFile


class GridStateFromFileWithForecasts(GridStateFromFile):
    """
    An extension of :class:`GridStateFromFile` that implements the "forecast" functionality.

    Forecast are also read from a file. For this class, only 1 forecast per timestep is read. The "forecast"
    present in the file at row $i$ is the one available at the corresponding time step, so valid for the grid state
    at the next time step.

    To have more advanced forecasts, this class could be overridden.

    Attributes
    ----------
    load_p_forecast: ``numpy.ndarray``, dtype: ``float``
        Array used to store the forecasts of the load active values.

    load_q_forecast: ``numpy.ndarray``, dtype: ``float``
        Array used to store the forecasts of the load reactive values.

    prod_p_forecast: ``numpy.ndarray``, dtype: ``float``
        Array used to store the forecasts of the generator active production setpoint.

    prod_v_forecast: ``numpy.ndarray``, dtype: ``float``
        Array used to store the forecasts of the generator voltage magnitude setpoint.

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

        self.load_p_forecast = None
        self.load_q_forecast = None
        self.prod_p_forecast = None
        self.prod_v_forecast = None

        # for when you read data in chunk
        self._order_load_p_forecasted = None
        self._order_load_q_forecasted = None
        self._order_prod_p_forecasted = None
        self._order_prod_v_forecasted = None
        self._data_already_in_mem = False  # says if the "main" value from the base class had to be reloaded (used for chunk)

        self._nb_forecast = len(h_forecast)
        self._h_forecast = copy.deepcopy(h_forecast)
        self._check_hs_consistent(self._h_forecast, time_interval)
        # init base class
        GridStateFromFile.__init__(
            self,
            path,
            sep=sep,
            time_interval=time_interval,
            max_iter=max_iter,
            chunk_size=chunk_size,
        )        
    
    def _clear(self):
        super()._clear()
        self.load_p_forecast = None
        self.load_q_forecast = None
        self.prod_p_forecast = None
        self.prod_v_forecast = None

        # for when you read data in chunk
        self._order_load_p_forecasted = None
        self._order_load_q_forecasted = None
        self._order_prod_p_forecasted = None
        self._order_prod_v_forecasted = None
        self._data_already_in_mem = False  # says if the "main" value from the base class had to be reloaded (used for chunk)
        
    def _check_hs_consistent(self, h_forecast, time_interval):
        prev = timedelta(minutes=0)
        for i, h in enumerate(h_forecast):
            prev += time_interval
            if prev.total_seconds() // 60 != h:
                raise ChronicsError("For now you cannot build non contiuguous forecast. "
                                    "Forecast should look like [5, 10, 15, 20] "
                                    "but not [10, 15, 20] (missing h=5mins) or [5, 10, 20] "
                                    f"(missing h=15 in this example). Missing h={prev} "
                                    f"at position {i}, found {h}")
        
    def _get_next_chunk_forecasted(self):
        load_p = None
        load_q = None
        prod_p = None
        prod_v = None
        if self._data_chunk["load_p_forecasted"] is not None:
            load_p = next(self._data_chunk["load_p_forecasted"])
        if self._data_chunk["load_q_forecasted"] is not None:
            load_q = next(self._data_chunk["load_q_forecasted"])
        if self._data_chunk["prod_p_forecasted"] is not None:
            prod_p = next(self._data_chunk["prod_p_forecasted"])
        if self._data_chunk["prod_v_forecasted"] is not None:
            prod_v = next(self._data_chunk["prod_v_forecasted"])
        return load_p, load_q, prod_p, prod_v

    def _data_in_memory(self):
        res = super()._data_in_memory()
        self._data_already_in_mem = res
        return res

    def initialize(
        self,
        order_backend_loads,
        order_backend_prods,
        order_backend_lines,
        order_backend_subs,
        names_chronics_to_backend=None,
    ):
        """
        The same condition as :class:`GridStateFromFile.initialize` applies also for
        :attr:`GridStateFromFileWithForecasts.load_p_forecast`,  :attr:`GridStateFromFileWithForecasts.load_q_forecast`,
        :attr:`GridStateFromFileWithForecasts.prod_p_forecast`, and 
        :attr:`GridStateFromFileWithForecasts.prod_v_forecast`.

        Parameters
        ----------
        See help of :func:`GridValue.initialize` for a detailed help about the _parameters.

        """
        super().initialize(
            order_backend_loads,
            order_backend_prods,
            order_backend_lines,
            order_backend_subs,
            names_chronics_to_backend,
        )

        if self.chunk_size is not None:
            chunk_size = self.chunk_size * self._nb_forecast
        else:
            chunk_size = None
            
        if self._max_iter > 0:
            nrows_to_load = (self._max_iter + 1) * self._nb_forecast
            
        load_p_iter = self._get_data("load_p_forecasted",
                                     chunk_size, nrows_to_load)
        load_q_iter = self._get_data("load_q_forecasted",
                                     chunk_size, nrows_to_load)
        prod_p_iter = self._get_data("prod_p_forecasted",
                                     chunk_size, nrows_to_load)
        prod_v_iter = self._get_data("prod_v_forecasted",
                                     chunk_size, nrows_to_load)
        hazards = None  # no hazards in forecast
        maintenance = None  # maintenance are read from the real data and propagated in the chronics

        if self.chunk_size is None:
            load_p = load_p_iter
            load_q = load_q_iter
            prod_p = prod_p_iter
            prod_v = prod_v_iter
        else:
            self._data_chunk["load_p_forecasted"] = load_p_iter
            self._data_chunk["load_q_forecasted"] = load_q_iter
            self._data_chunk["prod_p_forecasted"] = prod_p_iter
            self._data_chunk["prod_v_forecasted"] = prod_v_iter
            load_p, load_q, prod_p, prod_v = self._get_next_chunk_forecasted()

        order_backend_loads = {el: i for i, el in enumerate(order_backend_loads)}
        order_backend_prods = {el: i for i, el in enumerate(order_backend_prods)}
        order_backend_lines = {el: i for i, el in enumerate(order_backend_lines)}

        (
            order_chronics_load_p,
            order_backend_load_q,
            order_backend_prod_p,
            order_backend_prod_v,
            order_backend_hazards,
            order_backend_maintenance,
        ) = self._get_orders(
            load_p,
            load_q,
            prod_p,
            prod_v,
            hazards,
            maintenance,
            order_backend_loads,
            order_backend_prods,
            order_backend_lines,
        )

        self._order_load_p_forecasted = np.argsort(order_chronics_load_p)
        self._order_load_q_forecasted = np.argsort(order_backend_load_q)
        self._order_prod_p_forecasted = np.argsort(order_backend_prod_p)
        self._order_prod_v_forecasted = np.argsort(order_backend_prod_v)

        self._init_attrs_forecast(
            load_p, load_q, prod_p, prod_v
        )

    def _init_attrs_forecast(self, load_p, load_q, prod_p, prod_v):
        # TODO refactor that with _init_attrs from super()
        self.load_p_forecast = None
        self.load_q_forecast = None
        self.prod_p_forecast = None
        self.prod_v_forecast = None

        if load_p is not None:
            self.load_p_forecast = copy.deepcopy(
                load_p.values[:, self._order_load_p_forecasted].astype(dt_float)
            )
        if load_q is not None:
            self.load_q_forecast = copy.deepcopy(
                load_q.values[:, self._order_load_q_forecasted].astype(dt_float)
            )
        if prod_p is not None:
            self.prod_p_forecast = copy.deepcopy(
                prod_p.values[:, self._order_prod_p_forecasted].astype(dt_float)
            )
        if prod_v is not None:
            self.prod_v_forecast = copy.deepcopy(
                prod_v.values[:, self._order_prod_v_forecasted].astype(dt_float)
            )

    def check_validity(self, backend):
        super(GridStateFromFileWithForecasts, self).check_validity(backend)
        at_least_one = False

        if self.load_p_forecast is not None:
            if self.load_p_forecast.shape[1] != backend.n_load:
                raise IncorrectNumberOfLoads(
                    "for the active part. It should be {} but is in fact {}"
                    "".format(backend.n_load, len(self.load_p))
                )
            at_least_one = True

        if self.load_q_forecast is not None:
            if self.load_q_forecast.shape[1] != backend.n_load:
                raise IncorrectNumberOfLoads(
                    "for the reactive part. It should be {} but is in fact {}"
                    "".format(backend.n_load, len(self.load_q))
                )
            at_least_one = True

        if self.prod_p_forecast is not None:
            if self.prod_p_forecast.shape[1] != backend.n_gen:
                raise IncorrectNumberOfGenerators(
                    "for the active part. It should be {} but is in fact {}"
                    "".format(backend.n_gen, len(self.prod_p))
                )
            at_least_one = True

        if self.prod_v_forecast is not None:
            if self.prod_v_forecast.shape[1] != backend.n_gen:
                raise IncorrectNumberOfGenerators(
                    "for the voltage part. It should be {} but is in fact {}"
                    "".format(backend.n_gen, len(self.prod_v))
                )
            at_least_one = True

        if not at_least_one:
            raise ChronicsError(
                "You used a class that read forecasted data, yet there is no forecasted data in"
                '"{}". Please fall back to using class "GridStateFromFile" instead of '
                '"{}"'.format(self.path, type(self))
            )

        for name_arr, arr in zip(
            ["load_q", "load_p", "prod_v", "prod_p"],
            [
                self.load_q_forecast,
                self.load_p_forecast,
                self.prod_v_forecast,
                self.prod_p_forecast
            ],
        ):
            if arr is not None:
                if self.chunk_size is None:
                    if arr.shape[0] < self.n_ * self._nb_forecast:
                        raise EnvError(
                            "Array for forecast {}_forecasted as not the same number of rows of {} x nb_forecast. "
                            "The chronics cannot be loaded properly.".format(name_arr, name_arr)
                        )

    def _load_next_chunk_in_memory_forecast(self):
        # i load the next chunk as dataframes
        load_p, load_q, prod_p, prod_v = self._get_next_chunk_forecasted()
        # i put these dataframes in the right order (columns)
        self._init_attrs_forecast(load_p, load_q, prod_p, prod_v)
        # resetting the index has been done in _load_next_chunk_in_memory, or at least it should have

    def forecasts(self):
        """
        This is the major difference between :class:`GridStateFromFileWithForecasts` and :class:`GridStateFromFile`.
        It returns non empty forecasts.

        As explained in the :func:`GridValue.forecasts`, forecasts are made of list of tuple. Each tuple having
        exactly 2 elements:

          1. Is the time stamp of the forecast
          2. An :class:`grid2op.BaseAction` representing the modification of the powergrid after the forecast.

        For this class, only the forecast of the next time step is given, and only for the injections and maintenance.

        Returns
        -------
        See :func:`GridValue.forecasts` for more information.

        """
        if not self._data_already_in_mem:
            try:
                self._load_next_chunk_in_memory_forecast()
            except StopIteration as exc_:
                raise exc_
        res = []
        for h_id, h in enumerate(self._h_forecast):
            res_d = {}
            dict_ = {}
            indx_to_look = self._nb_forecast * self.current_index + h_id
            if self.load_p_forecast is not None:
                dict_["load_p"] = dt_float(
                    1.0 * self.load_p_forecast[indx_to_look, :]
                )
            if self.load_q_forecast is not None:
                dict_["load_q"] = dt_float(
                    1.0 * self.load_q_forecast[indx_to_look, :]
                )
            if self.prod_p_forecast is not None:
                dict_["prod_p"] = dt_float(
                    1.0 * self.prod_p_forecast[indx_to_look, :]
                )
            if self.prod_v_forecast is not None:
                dict_["prod_v"] = dt_float(
                    1.0 * self.prod_v_forecast[indx_to_look, :]
                )
            if dict_:
                res_d["injection"] = dict_

            forecast_datetime = self.current_datetime + timedelta(minutes=h)
            res.append((forecast_datetime, res_d))
        return res

    def get_id(self) -> str:
        return self.path

    def _init_res_split(self, nb_rows):
        res_load_p_f = None
        res_load_q_f = None
        res_prod_p_f = None
        res_prod_v_f = None
        if self.prod_p_forecast is not None:
            res_prod_p_f = np.zeros((nb_rows, self.n_gen), dtype=dt_float)
        if self.prod_v_forecast is not None:
            res_prod_v_f = np.zeros((nb_rows, self.n_gen), dtype=dt_float)
        if self.load_p_forecast is not None:
            res_load_p_f = np.zeros((nb_rows, self.n_load), dtype=dt_float)
        if self.load_q_forecast is not None:
            res_load_q_f = np.zeros((nb_rows, self.n_load), dtype=dt_float)
            
        res = super()._init_res_split(nb_rows)
        res += tuple(
            [res_prod_p_f, res_prod_v_f, res_load_p_f, res_load_q_f]
        )
        return res

    def _update_res_split(self, i, tmp, *arrays):
        (
            *args_super,
            res_prod_p_f,
            res_prod_v_f,
            res_load_p_f,
            res_load_q_f
        ) = arrays
        super()._update_res_split(i, tmp, *args_super)
        if res_prod_p_f is not None:
            res_prod_p_f[i, :] = tmp._extract_array("prod_p_forecast")
        if res_prod_v_f is not None:
            res_prod_v_f[i, :] = tmp._extract_array("prod_v_forecast")
        if res_load_p_f is not None:
            res_load_p_f[i, :] = tmp._extract_array("load_p_forecast")
        if res_load_q_f is not None:
            res_load_q_f[i, :] = tmp._extract_array("load_q_forecast")

    def _clean_arrays(self, i, *arrays):
        (
            *args_super,
            res_prod_p_f,
            res_prod_v_f,
            res_load_p_f,
            res_load_q_f
        ) = arrays
        res = super()._clean_arrays(i, *args_super)
        if res_prod_p_f is not None:
            res_prod_p_f = res_prod_p_f[:i, :]
        if res_prod_v_f is not None:
            res_prod_v_f = res_prod_v_f[:i, :]
        if res_load_p_f is not None:
            res_load_p_f = res_load_p_f[:i, :]
        if res_load_q_f is not None:
            res_load_q_f = res_load_q_f[:i, :]
            
        res += tuple(
            [res_prod_p_f, res_prod_v_f, res_load_p_f, res_load_q_f]
        )
        return res

    def _get_name_arrays_for_saving(self):
        res = super()._get_name_arrays_for_saving()
        res += [
            "prod_p_forecasted",
            "prod_v_forecasted",
            "load_p_forecasted",
            "load_q_forecasted"
        ]
        return res

    def _get_colorder_arrays_for_saving(self):
        res = super()._get_colorder_arrays_for_saving()
        res += tuple(
            [
                self._order_backend_prods,
                self._order_backend_prods,
                self._order_backend_loads,
                self._order_backend_loads
            ]
        )
        return res
