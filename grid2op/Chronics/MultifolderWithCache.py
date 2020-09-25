# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import numpy as np
from datetime import timedelta, datetime

from grid2op.dtypes import dt_int
from grid2op.Chronics.MultiFolder import Multifolder
from grid2op.Chronics.GridStateFromFile import GridStateFromFile


class MultifolderWithCache(Multifolder):
    """
    This class is a particular type of :class:`Multifolder` that, instead of reading is all from disk each time
    stores it into memory.

    For now it's only compatible (because it only present some kind of interest) with :class:`GridValue` class
    inheriting from :class:`GridStateFromFile`.

    The function :func:`MultifolderWithCache.reset_cache` will redo the cache from scratch. You can filter which
    type of data will be cached or not with the :func:`MultifolderWithCache.set_filter` function.

    **NB** Efficient use of this class can dramatically increase the speed of the learning algorithm, especially at
    the beginning where lots of data are read from the hard drive and the agent games over after a few time steps (
    typically, data are given by months, so 30*288 >= 8600 time steps, while during exploration an agent usually
    performs less than a few dozen of steps leading to more time spent reading 8600 rows than computing the
    few dozen of steps.

    Examples
    ---------
    This is how this class can be used:

    .. code-block:: python

        import re
        from grid2op import make
        from grid2op.Chronics import MultifolderWithCache
        env = make(...,chronics_class=MultifolderWithCache)

        # set the chronics to limit to one week of data (lower memory footprint)
        env.chronics_handler.set_max_iter(7*288)
        # assign a filter, use only chronics that have "december" in their name
        env.chronics_handler.real_data.set_filter(lambda x: re.match(".*december.*", x) is not None)
        # create the cache
        env.chronics_handler.real_data.reset_cache()

        # and now you can use it as you would do any gym environment:
        my_agent = ...
        obs = env.reset()
        done = False
        reward = env.reward_range[0]
        while not done:
            act = my_agent.act(obs, reward, done)
            obs, reward, done, info = env.step(act)  # and step will NOT load any data from disk.

    """
    def __init__(self, path,
                 time_interval=timedelta(minutes=5),
                 start_datetime=datetime(year=2019, month=1, day=1),
                 gridvalueClass=GridStateFromFile,
                 sep=";",
                 max_iter=-1,
                 chunk_size=None):
        Multifolder.__init__(self,
                             path=path,
                             time_interval=time_interval,
                             start_datetime=start_datetime,
                             gridvalueClass=gridvalueClass,
                             sep=sep,
                             max_iter=max_iter,
                             chunk_size=None)
        self._cached_data = None
        self.cache_size = 0
        if not issubclass(self.gridvalueClass, GridStateFromFile):
            raise RuntimeError("MultifolderWithCache does not work when \"gridvalueClass\" does not inherit from "
                               "\"GridStateFromFile\".")
        self.__i = 0

    def _default_filter(self, x):
        """
        default filter used at the initialization. It keeps only the first data encountered.
        """
        if self.__i > 0:
            return False
        else:
            self.__i += 1
            return True

    def reset(self):
        """
        Rebuilt the cache as if it were built from scratch. This call might take a while to process.
        """
        self._cached_data = [None for _ in self.subpaths]
        self.__i = 0
        # select the right paths, and store their id in "_order"
        super().reset()
        self.cache_size = 0
        for i in self._order:
            # everything in "_order" need to be put in cache
            path = self.subpaths[i]
            data = self.gridvalueClass(time_interval=self.time_interval,
                                       sep=self.sep,
                                       path=path,
                                       max_iter=self.max_iter,
                                       chunk_size=None)
            if self.seed is not None:
                max_int = np.iinfo(dt_int).max
                seed_chronics = self.space_prng.randint(max_int)
                data.seed(seed_chronics)

            data.initialize(self._order_backend_loads,
                            self._order_backend_prods,
                            self._order_backend_lines,
                            self._order_backend_subs,
                            self._names_chronics_to_backend)
            self._cached_data[i] = data
            self.cache_size += 1

        if self.cache_size == 0:
            raise RuntimeError("Impossible to initialize the new cache.")

        return self.subpaths[self._order]

    def initialize(self, order_backend_loads, order_backend_prods, order_backend_lines, order_backend_subs,
                   names_chronics_to_backend=None):
        self._order_backend_loads = order_backend_loads
        self._order_backend_prods = order_backend_prods
        self._order_backend_lines = order_backend_lines
        self._order_backend_subs = order_backend_subs
        self._names_chronics_to_backend = names_chronics_to_backend

        self.n_gen = len(order_backend_prods)
        self.n_load = len(order_backend_loads)
        self.n_line = len(order_backend_lines)
        if self._cached_data is None:
            # initialize the cache
            self.reset()

        id_scenario = self._order[self._prev_cache_id]
        self.data = self._cached_data[id_scenario]
        self.data.next_chronics()
