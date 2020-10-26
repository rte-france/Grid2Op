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
from grid2op.Chronics.GridStateFromFileWithForecasts import GridStateFromFileWithForecasts


class GridStateFromFileWithForecastsWithMaintenance(GridStateFromFileWithForecasts):
    """
    An extension of :class:`GridStateFromFileWithForecasts` that implements the maintenance chronic generator
    on the fly (maintenance are not read from files, but are rather generated when the chronics is created).

    Attributes
    ----------
    maintenance_starting_hour: ``int``
        The hour at which every maintenance will start

    maintenance_ending_hour: ``int``
        The hour at which every maintenance will end (we suppose mainteance end on same day for now

    line_to_maintenance: ``array``, dtype: ``string``
        Array used to store the name of the lines that can happen to be in maintenance

    daily_proba_per_month_maintenance: ``array``, dtype: ``float``
        Array used to store probability each line can be in maintenance on a day for a given month

    max_daily_number_per_month_maintenance: ``array``, dtype: ``int``
        Array used to store maximum number of maintenance per day for each month

    """

    def initialize(self, order_backend_loads, order_backend_prods, order_backend_lines, order_backend_subs,
                   names_chronics_to_backend=None):

        self.name_line = order_backend_lines

        # properties of maintenance
        # self.maintenance_duration= 8*(self.time_interval.total_seconds()*60*60)#8h, 9am to 5pm
        # 8h furation, 9am to 5pm
        with open(os.path.join(self.path, "maintenance_meta.json"), "r", encoding="utf-8") as f:
            dict_ = json.load(f)

        self.maintenance_starting_hour = dict_["maintenance_starting_hour"]
        # self.maintenance_duration= 8*(self.time_interval.total_seconds()*60*60) # not used for now, could be used later
        self.maintenance_ending_hour = dict_["maintenance_ending_hour"]

        self.line_to_maintenance = set(dict_["line_to_maintenance"])

        # frequencies of maintenance
        self.daily_proba_per_month_maintenance = dict_["daily_proba_per_month_maintenance"]

        self.max_daily_number_per_month_maintenance = dict_["max_daily_number_per_month_maintenance"]

        super().initialize(order_backend_loads, order_backend_prods, order_backend_lines, order_backend_subs,
                           names_chronics_to_backend)

    def _init_attrs(self, load_p, load_q, prod_p, prod_v, hazards=None, maintenance=None):
        super()._init_attrs(load_p, load_q, prod_p, prod_v, hazards=hazards, maintenance=None)
        # ignore the maitenance but keep hazards

        ########
        # new method to introduce generated maintenance
        self.maintenance = self._generate_maintenance()  #

        ##########
        # same as before in GridStateFromFileWithForecasts
        self.maintenance_time = np.zeros(shape=(self.maintenance.shape[0], self.n_line), dtype=dt_int) - 1
        self.maintenance_duration = np.zeros(shape=(self.maintenance.shape[0], self.n_line), dtype=dt_int)

        # test that with chunk size
        for line_id in range(self.n_line):
            self.maintenance_time[:, line_id] = self.get_maintenance_time_1d(self.maintenance[:, line_id])
            self.maintenance_duration[:, line_id] = self.get_maintenance_duration_1d(self.maintenance[:, line_id])

        # there are _maintenance and hazards only if the value in the file is not 0.
        self.maintenance = self.maintenance != 0.
        self.maintenance = self.maintenance.astype(dt_bool)

    def _generate_maintenance(self):
        # define maintenance dataframe with size (nbtimesteps,nlines)
        columnsNames = self.name_line
        nbTimesteps = self.n_
        res = np.zeros((nbTimesteps, len(self.name_line)))

        # read the maintenance line
        idx_line_maintenance = np.array([el in self.line_to_maintenance for el in columnsNames])
        nb_line_maint = np.sum(idx_line_maintenance)
        if nb_line_maint == 0:
            # TODO log something there !
            return res

        if nb_line_maint != len(self.line_to_maintenance):
            raise Grid2OpException("Lines that are suppose to be in maintenance are:\n{}\nand lines in the grid "
                                   "are\n{}\nCheck that all lines in maintenance are in the grid."
                                   "".format(self.line_to_maintenance, self.name_line))

        # identify the timestamps of the chronics to find out the month and day of the week
        freq = str(
            int(self.time_interval.total_seconds())) + "s"  # should be in the timedelta frequency format in pandas
        datelist = pd.date_range(self.start_datetime, periods=nbTimesteps, freq=freq)

        datelist = np.unique(np.array([el.date() for el in datelist]))
        datelist = datelist[:-1]

        n_lines_maintenance = len(self.line_to_maintenance)

        _24_h = timedelta(seconds=86400)
        nb_rows = int(86400 / self.time_interval.total_seconds())
        selected_rows_beg = int(self.maintenance_starting_hour * 3600 / self.time_interval.total_seconds())
        selected_rows_end = int(self.maintenance_ending_hour * 3600 / self.time_interval.total_seconds())

        # TODO this is INSANELY slow for now. find a way to make it faster
        # HINT: vectorize everything into one single numpy array, everything can be vectorized there...
        month = 0
        maintenance_daily_proba = -1
        maxDailyMaintenance = -1
        for nb_day_since_beg, this_day in enumerate(datelist):
            dayOfWeek = this_day.weekday()
            if dayOfWeek < 5:  # only maintenance starting on working days
                month = this_day.month

                maintenance_me = np.zeros((nb_rows, nb_line_maint))
                # Careful: month start at 1 but inidces start at 0 in python
                maintenance_daily_proba = self.daily_proba_per_month_maintenance[(month - 1)]
                maxDailyMaintenance = self.max_daily_number_per_month_maintenance[(month - 1)]

                # now for each line in self.line_to_maintenance, sample to know if we generate a maintenance
                # for line in self.line_to_maintenance:
                are_lines_in_maintenance = self.space_prng.choice([False, True],
                                                                  p=[(1. - maintenance_daily_proba),
                                                                     maintenance_daily_proba],
                                                                  size=n_lines_maintenance)

                n_Generated_Maintenance = np.sum(are_lines_in_maintenance)
                # check if the number of maintenance is not above the max allowed. otherwise randomly pick up the right
                # number
                if (n_Generated_Maintenance > maxDailyMaintenance):
                    # we pick up only maxDailyMaintenance elements
                    not_chosen = self.space_prng.choice(n_Generated_Maintenance,
                                                        replace=False,
                                                        size=n_Generated_Maintenance - maxDailyMaintenance)
                    are_lines_in_maintenance[np.where(are_lines_in_maintenance)[0][not_chosen]] = False
                maintenance_me[selected_rows_beg:selected_rows_end, are_lines_in_maintenance] = 1.0

                # handle last iteration
                n_max = res[(nb_day_since_beg*nb_rows):((nb_day_since_beg+1) * nb_rows), idx_line_maintenance].shape[0]
                res[(nb_day_since_beg*nb_rows):((nb_day_since_beg+1) * nb_rows), idx_line_maintenance] = \
                    maintenance_me[:n_max, :]
        return res