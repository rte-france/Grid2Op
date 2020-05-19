# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import os
import numpy as np
import pandas as pd
import random

from grid2op.dtypes import dt_bool, dt_int
from grid2op.Chronics.GridStateFromFileWithForecasts import GridStateFromFileWithForecasts


class GridStateFromFileWithForecastsWithMaintenance(GridStateFromFileWithForecasts):
    """
    An extension of :class:`GridStateFromFileWithForecasts` that implements the maintenance chronic generator.

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
        self.maintenance_starting_hour = 9
        # self.maintenance_duration= 8*(self.time_interval.total_seconds()*60*60) # not used for now, could be used later
        self.maintenance_ending_hour = 17

        self.line_to_maintenance = list(
            pd.read_csv(os.path.join(self.path, 'lines_in_maintenance.csv'), squeeze=True,header=None).values)

        # frequencies of maintenance
        self.daily_proba_per_month_maintenance = list(
            pd.read_csv(os.path.join(self.path, 'maintenance_daily_proba_per_month.csv'), squeeze=True,header=None).values)

        self.max_daily_number_per_month_maintenance = list(
            pd.read_csv(os.path.join(self.path, 'max_daily_Number_of_Mainteance_per_month.csv'), squeeze=True,header=None).values)

        super().initialize(order_backend_loads, order_backend_prods, order_backend_lines, order_backend_subs,
                           names_chronics_to_backend)

    def _init_attrs(self, load_p, load_q, prod_p, prod_v, hazards=None, maintenance=None):
        super()._init_attrs(load_p, load_q, prod_p, prod_v, hazards=None, maintenance=None)

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
        columnsNames = self.name_line  # [i for i in range(self.n_line)]
        nbTimesteps = self.load_p.shape[0]

        zero_data = np.zeros(shape=(nbTimesteps, len(columnsNames)))
        maintenances_df = pd.DataFrame(zero_data, columns=columnsNames)

        # identify the timestamps of the chronics to find out the month and day of the week
        freq = str(
            int(self.time_interval.total_seconds())) + "s"  # should be in the timedelta frequency format in pandas
        datelist = pd.date_range(self.start_datetime, periods=nbTimesteps, freq=freq)
        maintenances_df.index = datelist

        # we consider here that a chronic is not more than a year and we loop over days given 'dayofyear'
        daysOfYear = datelist.dayofyear.unique()
        n_lines_maintenance = len(self.line_to_maintenance)

        for idx, day_maintenances in maintenances_df.groupby(maintenances_df.index.date):

            day_datetime = pd.to_datetime(idx)
            dayOfWeek = day_datetime.dayofweek
            month = day_datetime.month

            linesInMaintenance = []

            if (dayOfWeek in range(5)):  # only maintenance starting on working days
                maintenance_daily_proba = self.daily_proba_per_month_maintenance[(month-1)] #Careful: month start at 1 but inidces start at 0 in python
                maxDailyMaintenance = self.max_daily_number_per_month_maintenance[(month-1)]
                
                # now for each line in self.line_to_maintenance, sample to know if we generate a maintenance
                # for line in self.line_to_maintenance:
                are_lines_in_maintenance = np.random.choice([0, 1],
                                                            p=[(1 - maintenance_daily_proba), maintenance_daily_proba],
                                                            size=(n_lines_maintenance))
                
                linesInMaintenance = [self.line_to_maintenance[i] for i in range(n_lines_maintenance) if
                                      are_lines_in_maintenance[i] == 1]

                n_Generated_Maintenance = np.sum(are_lines_in_maintenance)
                # check if the number of maintenance is not above the max allowed. otherwise randomly pick up the right number
                if (n_Generated_Maintenance > maxDailyMaintenance):
                    # we pick up only maxDailyMaintenance elements
                    #linesInMaintenance = self.space_prng.choices(linesCurrentlyInMaintenance, k=maxDailyMaintenance)
                    linesInMaintenance = random.choices(linesInMaintenance, k=maxDailyMaintenance)

                # update maintenance for line in mainteance
                timeIndex = day_maintenances.index[day_maintenances.index.hour.isin(
                    list(range(self.maintenance_starting_hour, self.maintenance_ending_hour)))]
                maintenances_df.at[timeIndex.values, linesInMaintenance] = 1

        return maintenances_df.values