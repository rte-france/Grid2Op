# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
from datetime import datetime, timedelta

from grid2op.dtypes import dt_int
from grid2op.Chronics.gridValue import GridValue


class ChangeNothing(GridValue):
    """
    INTERNAL

    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Do not attempt to create an object of this class. This is initialized by the environment
        at its creation.

    This set of class is mainly internal.

    We don't recommend you, unless you want to code a custom "chroncis class" to change anything
    on these classes.

    This class is the most basic class to modify a powergrid values.
    It does nothing aside from increasing :attr:`GridValue.max_iter` and the :attr:`GridValue.current_datetime`.

    Examples
    --------

    Usage example, for what you don't really have to do:

    .. code-block:: python

        import grid2op
        from grid2op.Chronics import ChangeNothing

        env_name = "l2rpn_case14_sandbox"  # or any other name
        # env = grid2op.make(env_name, data_feeding_kwargs={"gridvalueClass": ChangeNothing})
        env = grid2op.make(env_name, chronics_class=ChangeNothing)

    It can also be used with the "blank" environment:

    .. code-block:: python

        import grid2op
        from grid2op.Chronics import ChangeNothing
        env = grid2op.make("blank",
                           test=True,
                           grid_path=EXAMPLE_CASEFILE,
                           chronics_class=ChangeNothing,
                           action_class=TopologyAndDispatchAction)
        
    """
    MULTI_CHRONICS = False
    def __init__(
        self,
        time_interval=timedelta(minutes=5),
        max_iter=-1,
        start_datetime=datetime(year=2019, month=1, day=1),
        chunk_size=None,
        **kwargs
    ):
        GridValue.__init__(
            self,
            time_interval=time_interval,
            max_iter=max_iter,
            start_datetime=start_datetime,
            chunk_size=chunk_size,
        )
        self.n_gen = None
        self.n_load = None
        self.n_line = None
        
        self.maintenance_time = None
        self.maintenance_duration = None
        self.hazard_duration = None
        
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
        self.curr_iter = 0

        self.maintenance_time = np.zeros(shape=(self.n_line,), dtype=dt_int) - 1
        self.maintenance_duration = np.zeros(shape=(self.n_line,), dtype=dt_int)
        self.hazard_duration = np.zeros(shape=(self.n_line,), dtype=dt_int)

    def load_next(self):
        self.current_datetime += self.time_interval
        self.curr_iter += 1
        return (
            self.current_datetime,
            {},
            self.maintenance_time,
            self.maintenance_duration,
            self.hazard_duration,
            None,
        )

    def check_validity(self, backend):
        return True

    def next_chronics(self):
        self.current_datetime = self.start_datetime
        self.curr_iter = 0
