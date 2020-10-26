# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import copy
import warnings

from datetime import timedelta, datetime
import numpy as np
import pandas as pd

from grid2op.dtypes import dt_int
from grid2op.Chronics import GridStateFromFileWithForecasts
from grid2op.Exceptions import ChronicsError


# Names of the csv were not the same
class ReadPypowNetData(GridStateFromFileWithForecasts):
    def __init__(self, path, sep=";", time_interval=timedelta(minutes=5),
                 max_iter=-1,
                 chunk_size=None):
        GridStateFromFileWithForecasts.__init__(self, path, sep=sep, time_interval=time_interval,
                                                max_iter=max_iter, chunk_size=chunk_size)

    def initialize(self, order_backend_loads, order_backend_prods, order_backend_lines, order_backend_subs,
                   names_chronics_to_backend=None):
        """
        TODO Doc
        """
        self.n_gen = len(order_backend_prods)
        self.n_load = len(order_backend_loads)
        self.n_line = len(order_backend_lines)

        self.names_chronics_to_backend = copy.deepcopy(names_chronics_to_backend)
        if self.names_chronics_to_backend is None:
            self.names_chronics_to_backend = {}
        if not "loads" in self.names_chronics_to_backend:
            self.names_chronics_to_backend["loads"] = {k: k for k in order_backend_loads}
        else:
            self._assert_correct(self.names_chronics_to_backend["loads"], order_backend_loads)
        if not "prods" in self.names_chronics_to_backend:
            self.names_chronics_to_backend["prods"] = {k: k for k in order_backend_prods}
        else:
            self._assert_correct(self.names_chronics_to_backend["prods"], order_backend_prods)
        if not "lines" in self.names_chronics_to_backend:
            self.names_chronics_to_backend["lines"] = {k: k for k in order_backend_lines}
        else:
            self._assert_correct(self.names_chronics_to_backend["lines"], order_backend_lines)
        if not "subs" in self.names_chronics_to_backend:
            self.names_chronics_to_backend["subs"] = {k: k for k in order_backend_subs}
        else:
            self._assert_correct(self.names_chronics_to_backend["subs"], order_backend_subs)

        # print(os.listdir(self.path))
        read_compressed = ".csv"
        if not os.path.exists(os.path.join(self.path, "_N_loads_p.csv")):
            # try to read compressed data
            if os.path.exists(os.path.join(self.path, "_N_loads_p.csv.bz2")):
                read_compressed = ".csv.bz2"
            elif os.path.exists(os.path.join(self.path, "_N_loads_p.zip")):
                read_compressed = ".zip"
            elif os.path.exists(os.path.join(self.path, "_N_loads_p.csv.gzip")):
                read_compressed = ".csv.gzip"
            elif os.path.exists(os.path.join(self.path, "_N_loads_p.csv.xz")):
                read_compressed = ".csv.xz"
            else:
                raise RuntimeError(
                    "GridStateFromFile: unable to locate the data files that should be at \"{}\"".format(self.path))
        load_p = pd.read_csv(os.path.join(self.path, "_N_loads_p{}".format(read_compressed)), sep=self.sep)
        load_q = pd.read_csv(os.path.join(self.path, "_N_loads_q{}".format(read_compressed)), sep=self.sep)
        prod_p = pd.read_csv(os.path.join(self.path, "_N_prods_p{}".format(read_compressed)), sep=self.sep)
        prod_v = pd.read_csv(os.path.join(self.path, "_N_prods_v{}".format(read_compressed)), sep=self.sep)
        hazards = pd.read_csv(os.path.join(self.path, "hazards{}".format(read_compressed)), sep=self.sep)
        maintenance = pd.read_csv(os.path.join(self.path, "maintenance{}".format(read_compressed)), sep=self.sep)

        order_backend_loads = {el: i for i, el in enumerate(order_backend_loads)}
        order_backend_prods = {el: i for i, el in enumerate(order_backend_prods)}
        order_backend_lines = {el: i for i, el in enumerate(order_backend_lines)}

        order_chronics_load_p = np.array([order_backend_loads[self.names_chronics_to_backend["loads"][el]]
                                          for el in load_p.columns]).astype(dt_int)
        order_backend_load_q = np.array([order_backend_loads[self.names_chronics_to_backend["loads"][el]]
                                         for el in load_q.columns]).astype(dt_int)
        order_backend_prod_p = np.array([order_backend_prods[self.names_chronics_to_backend["prods"][el]]
                                         for el in prod_p.columns]).astype(dt_int)
        order_backend_prod_v = np.array([order_backend_prods[self.names_chronics_to_backend["prods"][el]]
                                         for el in prod_v.columns]).astype(dt_int)
        order_backend_hazards = np.array([order_backend_lines[self.names_chronics_to_backend["lines"][el]]
                                          for el in hazards.columns]).astype(dt_int)
        order_backend_maintenance = np.array([order_backend_lines[self.names_chronics_to_backend["lines"][el]]
                                              for el in maintenance.columns]).astype(dt_int)

        self.load_p = copy.deepcopy(load_p.values[:, np.argsort(order_chronics_load_p)])
        self.load_q = copy.deepcopy(load_q.values[:, np.argsort(order_backend_load_q)])
        self.prod_p = copy.deepcopy(prod_p.values[:, np.argsort(order_backend_prod_p)])
        self.prod_v = copy.deepcopy(prod_v.values[:, np.argsort(order_backend_prod_v)])
        self.hazards = copy.deepcopy(hazards.values[:, np.argsort(order_backend_hazards)])
        self.maintenance = copy.deepcopy(maintenance.values[:, np.argsort(order_backend_maintenance)])

        # date and time
        datetimes_ = pd.read_csv(os.path.join(self.path, "_N_datetimes{}".format(read_compressed)), sep=self.sep)
        self.start_datetime = datetime.strptime(datetimes_.iloc[0, 0], "%Y-%b-%d")

        # there are maintenance and hazards only if the value in the file is not 0.
        self.maintenance = self.maintenance != 0.
        self.hazards = self.hazards != 0.

        self.curr_iter = 0
        if self.max_iter == -1:
            # if the number of maximum time step is not set yet, we set it to be the number of
            # data in the chronics (number of rows of the files) -1.
            # the -1 is present because the initial grid state doesn't count as a "time step" but is read
            # from these data.
            self.max_iter = self.load_p.shape[0]-1

        load_p = pd.read_csv(os.path.join(self.path, "_N_loads_p_planned{}".format(read_compressed)), sep=self.sep)
        load_q = pd.read_csv(os.path.join(self.path, "_N_loads_q_planned{}".format(read_compressed)), sep=self.sep)
        prod_p = pd.read_csv(os.path.join(self.path, "_N_prods_p_planned{}".format(read_compressed)), sep=self.sep)
        prod_v = pd.read_csv(os.path.join(self.path, "_N_prods_v_planned{}".format(read_compressed)), sep=self.sep)
        maintenance = pd.read_csv(os.path.join(self.path, "maintenance{}".format(read_compressed)),
                                  sep=self.sep)

        order_backend_loads = {el: i for i, el in enumerate(order_backend_loads)}
        order_backend_prods = {el: i for i, el in enumerate(order_backend_prods)}
        order_backend_lines = {el: i for i, el in enumerate(order_backend_lines)}

        order_chronics_load_p = np.array([order_backend_loads[self.names_chronics_to_backend["loads"][el]]
                                          for el in load_p.columns]).astype(dt_int)
        order_backend_load_q = np.array([order_backend_loads[self.names_chronics_to_backend["loads"][el]]
                                         for el in load_q.columns]).astype(dt_int)
        order_backend_prod_p = np.array([order_backend_prods[self.names_chronics_to_backend["prods"][el]]
                                         for el in prod_p.columns]).astype(dt_int)
        order_backend_prod_v = np.array([order_backend_prods[self.names_chronics_to_backend["prods"][el]]
                                         for el in prod_v.columns]).astype(dt_int)
        order_backend_maintenance = np.array([order_backend_lines[self.names_chronics_to_backend["lines"][el]]
                                              for el in maintenance.columns]).astype(dt_int)

        self.load_p_forecast = copy.deepcopy(load_p.values[:, np.argsort(order_chronics_load_p)])
        self.load_q_forecast = copy.deepcopy(load_q.values[:, np.argsort(order_backend_load_q)])
        self.prod_p_forecast = copy.deepcopy(prod_p.values[:, np.argsort(order_backend_prod_p)])
        self.prod_v_forecast = copy.deepcopy(prod_v.values[:, np.argsort(order_backend_prod_v)])
        self.maintenance_forecast = copy.deepcopy(maintenance.values[:, np.argsort(order_backend_maintenance)])

        # there are maintenance and hazards only if the value in the file is not 0.
        self.maintenance_time = np.zeros(shape=(self.load_p.shape[0], self.n_line), dtype=dt_int) - 1
        self.maintenance_duration = np.zeros(shape=(self.load_p.shape[0], self.n_line), dtype=dt_int)
        self.hazard_duration = np.zeros(shape=(self.load_p.shape[0], self.n_line), dtype=dt_int)
        for line_id in range(self.n_line):
            self.maintenance_time[:, line_id] = self.get_maintenance_time_1d(self.maintenance[:, line_id])
            self.maintenance_duration[:, line_id] = self.get_maintenance_duration_1d(self.maintenance[:, line_id])
            self.hazard_duration[:, line_id] = self.get_maintenance_duration_1d(self.hazards[:, line_id])

        self.maintenance_forecast = self.maintenance != 0.

        self.curr_iter = 0
        if self.maintenance is not None:
            n_ = self.maintenance.shape[0]
        elif self.hazards is not None:
            n_ = self.hazards.shape[0]
        else:
            n_ = None
            for fn in ["prod_p", "load_p", "prod_v", "load_q"]:
                ext_ = self._get_fileext(fn)
                if ext_ is not None:
                    n_ = self._file_len(os.path.join(self.path, "{}{}".format(fn, ext_)), ext_)
                    break
            if n_ is None:
                raise ChronicsError("No files are found in directory \"{}\". If you don't want to load any chronics,"
                                    " use  \"ChangeNothing\" and not \"{}\" to load chronics."
                                    "".format(self.path, type(self)))
        self.n_ = n_  # the -1 is present because the initial grid state doesn't count as a "time step"
        self.tmp_max_index = load_p.shape[0]
