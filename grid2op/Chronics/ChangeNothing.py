# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
from datetime import datetime, timedelta
import pdb

from grid2op.Chronics.GridValue import GridValue


class ChangeNothing(GridValue):
    """
    This class is the most basic class to modify a powergrid values.
    It does nothing exceptie increasing :attr:`GridValue.max_iter` and the :attr:`GridValue.current_datetime`.
    """
    def __init__(self, time_interval=timedelta(minutes=5), max_iter=-1,
                 start_datetime=datetime(year=2019, month=1, day=1),
                 chunk_size=None, **kargs):
        GridValue.__init__(self, time_interval=time_interval, max_iter=max_iter, start_datetime=start_datetime,
                           chunk_size=chunk_size)

    def initialize(self, order_backend_loads, order_backend_prods, order_backend_lines, order_backend_subs,
                   names_chronics_to_backend=None):
        self.n_gen = len(order_backend_prods)
        self.n_load = len(order_backend_loads)
        self.n_line = len(order_backend_lines)
        self.curr_iter = 0

        self.maintenance_time = np.zeros(shape=(self.n_line, ), dtype=np.int) - 1
        self.maintenance_duration = np.zeros(shape=(self.n_line, ), dtype=np.int)
        self.hazard_duration = np.zeros(shape=(self.n_line, ), dtype=np.int)

    def load_next(self):
        """
        This function does nothing but the two requirements of load_next ie:

          - increasing :attr:`GridValue.curr_iter` of 1
          - increasing :attr:`GridValue.current_datetime`

        Returns
        -------
        timestamp: ``datetime.datetime``
            The current timestamp for which the modifications have been generated.

        dict_: ``dict``
            Always empty, indicating i do nothing.

        maintenance_time: ``numpy.ndarray``, dtype:``int``
            Information about the next planned maintenance. See :attr:`GridValue.maintenance_time` for more information.

        maintenance_duration: ``numpy.ndarray``, dtype:``int``
            Information about the duration of next planned maintenance. See :attr:`GridValue.maintenance_duration`
            for more information.

        hazard_duration: ``numpy.ndarray``, dtype:``int``
            Information about the current hazard. See :attr:`GridValue.hazard_duration`
            for more information.

        """
        self.current_datetime += self.time_interval
        self.curr_iter += 1
        return self.current_datetime, {}, self.maintenance_time, self.maintenance_duration, self.hazard_duration, None

    def check_validity(self, backend):
        """

        Parameters
        ----------
        backend: :class:`grid2op.Backend`
            The backend, not used here.

        Returns
        -------
        res: ``bool``
            Always ``True``. As this doesn't change the powergird, there is no way to make invalid changed.
        """
        return True

    def next_chronics(self):
        """
        Restarts:

          - :attr:`GridValue.current_datetime` to its origin value ( 2019 / 01 / 01)
          - :attr:`GridValue.curr_iter` to 0

        Returns
        -------

        """
        self.current_datetime = self.start_datetime
        self.curr_iter = 0
