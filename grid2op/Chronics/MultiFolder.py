# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import json
import warnings
import numpy as np
from datetime import timedelta, datetime

from grid2op.dtypes import dt_int
from grid2op.Exceptions import *
from grid2op.Chronics.GridValue import GridValue
from grid2op.Chronics.GridStateFromFile import GridStateFromFile


class Multifolder(GridValue):
    """
    The classes :class:`GridStateFromFile` and :class:`GridStateFromFileWithForecasts` implemented the reading of a
    single folder representing a single episode.

    This class is here to "loop" between different episode. Each one being stored in a folder readable by
    :class:`GridStateFromFile` or one of its derivate (eg. :class:`GridStateFromFileWithForecasts`).

    Chronics are always read in the alpha-numeric order for this class. This means that if the folder is not modified,
    the data are always loaded in the same order, regardless of the :class:`grid2op.Backend`, :class:`grid2op.BaseAgent` or
    :class:`grid2op.Environment`.

    Attributes
    -----------
    gridvalueClass: ``type``, optional
        Type of class used to read the data from the disk. It defaults to :class:`GridStateFromFile`.

    data: :class:`GridStateFromFile`
        Data that will be loaded and used to produced grid state and forecasted values.


    path: ``str``
        Path where the folders of the episodes are stored.

    sep: ``str``
        Columns separtor, forwarded to :attr:`Multifolder.data` when it's built at the beginning of each episode.

    subpaths: ``list``
        List of all the episode that can be "played". It's a sorted list of all the directory in
        :attr:`Multifolder.path`. Each one should contain data in a format that is readable by
        :attr:`MultiFolder.gridvalueClass`.

    """
    def __init__(self, path,
                 time_interval=timedelta(minutes=5),
                 start_datetime=datetime(year=2019, month=1, day=1),
                 gridvalueClass=GridStateFromFile,
                 sep=";", max_iter=-1,
                 chunk_size=None):
        GridValue.__init__(self, time_interval=time_interval, max_iter=max_iter, chunk_size=chunk_size,
                           start_datetime=start_datetime)
        self.gridvalueClass = gridvalueClass
        self.data = None
        self.path = os.path.abspath(path)
        self.sep = sep
        try:
            self.subpaths = [os.path.join(self.path, el) for el in os.listdir(self.path)
                             if os.path.isdir(os.path.join(self.path, el))]
            self.subpaths.sort()
            self.subpaths = np.array(self.subpaths)
        except FileNotFoundError:
            raise ChronicsError("Path \"{}\" doesn't exists.".format(self.path)) from None

        if len(self.subpaths) == 0:
            raise ChronicsNotFoundError("Not chronics are found in \"{}\". Make sure there are at least "
                                        "1 chronics folder there.".format(self.path))
        # TODO clarify that
        # np.random.shuffle(self.subpaths)
        self.chunk_size = chunk_size

        # for saving
        self._order_backend_loads = None
        self._order_backend_prods = None
        self._order_backend_lines = None
        self._order_backend_subs = None
        self._names_chronics_to_backend = None

        # improving looping strategy
        self._filter = self._default_filter
        self._prev_cache_id = 0
        self._order = None

    def _default_filter(self, x):
        """
        default filter used at the initialization. It keeps only the first data encountered.
        """
        return True

    def set_filter(self, filter_fun):
        """
        Assign a filtering function to remove some chronics from the next time a call to "reset_cache" is called.

        **NB** filter_fun is applied to all element of :attr:`Multifolder.subpaths`. If ``True`` then it will
        be put in cache, if ``False`` this data will NOT be put in the cache.

        **NB** this has no effect until :attr:`Multifolder.reset` is called.

        Examples
        --------
        Let's assume in your chronics, the folder names are "Scenario_august_dummy", and
        "Scenario_february_dummy". For the sake of the example, we want the environment to loop
        only through the month of february, because why not. Then we can do the following:

        .. code-block:: python

            import re
            import grid2op
            env = grid2op.make("l2rpn_neurips_2020_track1", test=True)  # don't add "test=True" if
            # you don't want to perform a test.

            # check at which month will belong each observation
            for i in range(10):
                obs = env.reset()
                print(obs.month)
                # it always alternatively prints "8" (if chronics if from august) or
                # "2" if chronics is from february)

            # to see where the chronics are located
            print(env.chronics_handler.subpaths)

            # keep only the month of february
            env.chronics_handler.set_filter(lambda path: re.match(".*february.*", path) is not None)
            env.chronics_handler.reset()  # if you don't do that it will not have any effect

            for i in range(10):
                obs = env.reset()
                print(obs.month)
                # it always prints "2" (representing february)

        """
        self._filter = filter_fun

    def next_chronics(self):
        self._prev_cache_id += 1
        # TODO implement the shuffling indeed.
        # if self._prev_cache_id >= len(self._order):
        #     self.space_prng.shuffle(self._order)
        self._prev_cache_id %= len(self._order)

    def sample_next_chronics(self, probabilities=None):
        """
        This function should be called before "next_chronics".
        It can be used to sample non uniformly for the next next chronics.

        Parameters
        -----------
        probabilities: ``np.ndarray``
            Array of integer with the same size as the number of chronics in the cache.
            If it does not sum to one, it is rescaled such that it sums to one.

        Returns
        -------
        selected: ``int``
            The integer that was selected.

        Examples
        --------

        Let's assume in your chronics, the folder names are "Scenario_august_dummy", and
        "Scenario_february_dummy". For the sake of the example, we want the environment to loop
        75% of the time to the month of february and 25% of the time to the month of august.

        .. code-block:: python

            import grid2op
            env = grid2op.make("l2rpn_neurips_2020_track1", test=True)  # don't add "test=True" if
            # you don't want to perform a test.

            # check at which month will belong each observation
            for i in range(10):
                obs = env.reset()
                print(obs.month)
                # it always alternatively prints "8" (if chronics if from august) or
                # "2" if chronics is from february) with a probability of 50% / 50%

            env.seed(0)  # for reproducible experiment
            for i in range(10):
                _ = env.chronics_handler.sample_next_chronics([0.25, 0.75])
                obs = env.reset()
                print(obs.month)
                # it prints "2" with probability 0.75 and "8" with probability 0.25

        """
        self._prev_cache_id = -1
        if probabilities is None:
            probabilities = np.ones(self._order.shape[0])

        # make sure it sums to 1
        probabilities /= np.sum(probabilities)
        # take one at "random" among these
        selected = self.space_prng.choice(self._order,  p=probabilities)
        id_sel = np.where(self._order == selected)[0]
        self._prev_cache_id = selected - 1
        return id_sel

    def reset(self):
        """
        Rebuilt the :attr:`Multifolder._order`. This should be called after a call to :func:`Multifolder.set_filter`
        is performed.

        **NB** This "reset" is different from the "env.reset". It should be only called after the function to set
        the filtering function has been called.

        Returns
        -------
        new_order: ``numpy.ndarray``, dtype: str
            The selected chronics paths after a call to this method.

        """
        self._order = []
        self._prev_cache_id = 0
        for i, path in enumerate(self.subpaths):
            if not self._filter(path):
                continue
            self._order.append(i)

        if len(self._order) == 0:
            raise RuntimeError("Impossible to initialize the Multifolder. Your \"filter_fun\" filters out all the "
                               "possible scenarios.")
        self._order = np.array(self._order)
        # TODO this shuffling there
        # self.space_prng.shuffle(self._order)
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

        if self._order is None:
            # initialize the cache
            self.reset()

        id_scenario = self._order[self._prev_cache_id]
        this_path = self.subpaths[id_scenario]
        self.data = self.gridvalueClass(time_interval=self.time_interval,
                                        sep=self.sep,
                                        path=this_path,
                                        max_iter=self.max_iter,
                                        chunk_size=self.chunk_size)
        if self.seed is not None:
            max_int = np.iinfo(dt_int).max
            seed_chronics = self.space_prng.randint(max_int)
            self.data.seed(seed_chronics)

        self.data.initialize(order_backend_loads, order_backend_prods, order_backend_lines, order_backend_subs,
                             names_chronics_to_backend=names_chronics_to_backend)

    def done(self):
        """
        Tells the :class:`grid2op.Environment` if the episode is over.

        Returns
        -------
        res: ``bool``
            Whether or not the episode, represented by :attr:`MultiFolder.data` is over.

        """
        return self.data.done()

    def load_next(self):
        """
        Load the next data from the current episode. It loads the next time step for the current episode.

        Returns
        -------
        See the return type of  :class:`GridStateFromFile.load_next` (or of :attr:`MultiFolder.gridvalueClass` if it
        has been changed) for more information.

        """
        return self.data.load_next()

    def check_validity(self, backend):
        """
        This method check that the data loaded can be properly read and understood by the :class:`grid2op.Backend`.

        Parameters
        ----------
        backend: :class:`grid2op.Backend`
            The backend used for the experiment.

        Returns
        -------
        See the return type of  :class:`GridStateFromFile.check_validity` (or of :attr:`MultiFolder.gridvalueClass` if it
        has been changed) for more information.
        """
        return self.data.check_validity(backend)

    def forecasts(self):
        """
        The representation of the forecasted grid state(s), if any.

        Returns
        -------
        See the return type of  :class:`GridStateFromFile.forecasts` (or of :attr:`MultiFolder.gridvalueClass` if it
        has been changed) for more information.
        """
        return self.data.forecasts()

    def tell_id(self, id_num):
        """
        This tells this chronics to load for the next episode.
        By default, if id_num is greater than the number of episode, it is equivalent at restarting from the first
        one: episode are played indefinitely in the same order.

        Parameters
        ----------
        id_num: ``int``
            Id of the chronics to load.

        Returns
        -------

        """
        self._prev_cache_id = id_num
        self._prev_cache_id %= len(self._order)

    def get_id(self) -> str:
        """
        Full absolute path of the current folder used for the current episode.

        Returns
        -------
        res: ``str``
            Path from which the data are generated for the current episode.

        """
        return self.subpaths[self._order[self._prev_cache_id]]

    def max_timestep(self):
        return self.data.max_timestep()

    def shuffle(self, shuffler=None):
        """
        This method is used to have a better control on the order in which the subfolder containing the episode are
        processed.

        It can focus the evaluation on one specific folder, shuffle the folders, use only a subset of them etc. See the
        examples for more information.

        Parameters
        ----------
        shuffler: ``object``
            Shuffler should be a function that is called on :attr:`MultiFolder.subpaths` that will shuffle them.
            It can also be used to remove some path if needed (see example).

        Returns
        --------
        new_order: ``numpy.ndarray``, dtype: str
            The order in which the chronics will be looped through

        Examples
        ---------
        If you want to simply shuffle the data you can do:

        .. code-block:: python

            import numpy as np
            data = Multifolder(path=".")
            data.shuffle(shuffler=lambda x: x[np.random.choice(len(x), size=len(x), replace=False)])

        If you want to use only a subset of the path, say for example the path with index 1, 5, and 6

        .. code-block:: python

            data = Multifolder(path=".")
            data.shuffle(shuffler=lambda x: x[1, 5, 6])

        """
        if shuffler is None:
            def shuffler(x):
                return x[self.space_prng.choice(len(x), size=len(x), replace=False)]

        self._order = shuffler(self._order)
        return self.subpaths[self._order]

    def set_chunk_size(self, new_chunk_size):
        self.chunk_size = new_chunk_size

    def split_and_save(self, datetime_beg, datetime_end, path_out):
        """
        This function allows you to split the data (keeping only the data between datetime_beg and datetime_end) and to
        save it on your local machine. This is espacially handy if you want to extract only a piece of the dataset we
        provide for example.

        Parameters
        ----------
        datetime_beg: ``dict``
            Keys are the name id of the scenarios you want to save. Values
            are the corresponding starting date and time (in "%Y-%m-ùd %H:%M"
            format). See example for more information.
        datetime_end: ``dict``
            keys must be the same as in the "datetime_beg" argument.

            See example for more information

        path_out: ``str``
            The path were the data will be stored.

        Examples
        ---------

        Here is a short example on how to use it

        .. code-block:: python

            import grid2op
            import os
            env = grid2op.make()

            env.chronics_handler.real_data.split_and_save({"004": "2019-01-08 02:00",
                                                 "005": "2019-01-30 08:00",
                                                 "006": "2019-01-17 00:00",
                                                 "007": "2019-01-17 01:00",
                                                 "008": "2019-01-21 09:00",
                                                 "009": "2019-01-22 12:00",
                                                 "010": "2019-01-27 19:00",
                                                 "011": "2019-01-15 12:00",
                                                 "012": "2019-01-08 13:00",
                                                 "013": "2019-01-22 00:00"},
                                                {"004": "2019-01-11 02:00",
                                                 "005": "2019-02-01 08:00",
                                                 "006": "2019-01-18 00:00",
                                                 "007": "2019-01-18 01:00",
                                                 "008": "2019-01-22 09:00",
                                                 "009": "2019-01-24 12:00",
                                                 "010": "2019-01-29 19:00",
                                                 "011": "2019-01-17 12:00",
                                                 "012": "2019-01-10 13:00",
                                                 "013": "2019-01-24 00:00"},
                                      path_out=os.path.join("/tmp"))

        """
        if not isinstance(datetime_beg, dict):
            datetime_beg_orig = datetime_beg
            datetime_beg = {}
            for subpath in self.subpaths:
                id_this_chron = os.path.split(subpath)[-1]
                datetime_beg[id_this_chron] = datetime_beg_orig
        if not isinstance(datetime_end, dict):
            datetime_end_orig = datetime_end
            datetime_end = {}
            for subpath in self.subpaths:
                id_this_chron = os.path.split(subpath)[-1]
                datetime_end[id_this_chron] = datetime_end_orig
        seed_chronics_all = {}
        for subpath in self.subpaths:
            id_this_chron = os.path.split(subpath)[-1]
            if not id_this_chron in datetime_beg:
                continue
            tmp = self.gridvalueClass(time_interval=self.time_interval,
                                      sep=self.sep,
                                      path=subpath,
                                      max_iter=self.max_iter,
                                      chunk_size=self.chunk_size)
            seed_chronics = None
            if self.seed is not None:
                max_int = np.iinfo(dt_int).max
                seed_chronics = self.space_prng.randint(max_int)
                tmp.seed(seed_chronics)
            seed_chronics_all[subpath] = seed_chronics
            tmp.initialize(self._order_backend_loads,
                           self._order_backend_prods,
                           self._order_backend_lines,
                           self._order_backend_subs,
                           self._names_chronics_to_backend)
            path_out_chron = os.path.join(path_out, id_this_chron)
            tmp.split_and_save(datetime_beg[id_this_chron], datetime_end[id_this_chron], path_out_chron)

            meta_params = {}
            meta_params["datetime_beg"] = datetime_beg
            meta_params["datetime_end"] = datetime_end
            meta_params["path_out"] = path_out
            meta_params["all_seeds"] = seed_chronics_all
            try:
                with open(os.path.join(path_out, "split_and_save_meta_params.json"), "w", encoding="utf-8") as f:
                    json.dump(obj=meta_params, fp=f,
                              sort_keys=True,
                              indent=4
                              )
            except Exception as exc_:
                warnings.warn("Impossible to save the \"metadata\" for the chronics with error:\n\"{}\""
                              "".format(exc_))
