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
from grid2op.Chronics.gridValue import GridValue
from grid2op.Chronics.gridStateFromFile import GridStateFromFile


class Multifolder(GridValue):
    """
    The classes :class:`GridStateFromFile` and :class:`GridStateFromFileWithForecasts` implemented the reading of a
    single folder representing a single episode.

    This class is here to "loop" between different episode. Each one being stored in a folder readable by
    :class:`GridStateFromFile` or one of its derivate (eg. :class:`GridStateFromFileWithForecasts`).

    Chronics are always read in the alpha-numeric order for this class. This means that if the folder is not modified,
    the data are always loaded in the same order, regardless of the :class:`grid2op.Backend`, :class:`grid2op.BaseAgent` or
    :class:`grid2op.Environment`.

    .. note::
        Most grid2op environments, by default, use this type of "chronix", read from the hard drive.
        
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
    MULTI_CHRONICS = True

    def __init__(
        self,
        path,
        time_interval=timedelta(minutes=5),
        start_datetime=datetime(year=2019, month=1, day=1),
        gridvalueClass=GridStateFromFile,
        sep=";",
        max_iter=-1,
        chunk_size=None,
        filter_func=None,
        **kwargs
    ):
        self._kwargs = kwargs
        GridValue.__init__(
            self,
            time_interval=time_interval,
            max_iter=max_iter,
            chunk_size=chunk_size,
            start_datetime=start_datetime,
        )
        self.gridvalueClass = gridvalueClass
        self.data = None
        self.path = os.path.abspath(path)
        self.sep = sep
        self.init_subpath()

        if len(self.subpaths) == 0:
            raise ChronicsNotFoundError(
                'Not chronics are found in "{}". Make sure there are at least '
                "1 chronics folder there.".format(self.path)
            )
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
        if filter_func is None:
            self._filter = self._default_filter
        else:
            if not callable(filter_func):
                raise ChronicsError(
                    "The filtering function you provided ("
                    "kwargs: filter_func) is not callable."
                )
            self._filter = filter_func
        self._prev_cache_id = 0
        self._order = None

    def init_subpath(self):
        """
        Read the content of the main directory and initialize the `subpaths` 
        where the data could be located.
        
        This is usefull, for example, if you generated data and want to be able to use them.
        
        **NB** this has no effect until :attr:`Multifolder.reset` is called.
        
        .. warning::
            By default, it will only consider data that are present at creation time. If you add data after, you need
            to call this function (and do a reset)
            
        Examples
        ---------
        
        A "typical" usage of this function can be the following workflow.
        
        Start a script to train an agent (say "train_agent.py"):
        
        .. code-block:: python
        
            import os
            import grid2op
            from lightsim2grid import LightSimBackend  # highly recommended for speed !
            
            env_name = "l2rpn_wcci_2022"  # only compatible with what comes next (at time of writing)
            env = grid2op.make(env_name, backend=LightSimBackend())
            
            # now train an agent
            # see l2rpn_baselines package for more information, for example
            # l2rpn-baselines.readthedocs.io/
            from l2rpn_baselines.PPO_SB3 import train
            nb_iter = 10000  # train for that many iterations
            agent_name = "WhaetverIWant"  # or any other name
            agent_path = os.path.expand("~")  # or anywhere else on your computer
            trained_agent = train(env,
                                  iterations=nb_iter,
                                  name=agent_name,
                                  save_path=agent_path)
            
        On another script (say "generate_data.py"), you can generate more data:
                
        .. code-block:: python
        
            import grid2op            
            env_name = "l2rpn_wcci_2022"  # only compatible with what comes next (at time of writing)
            env = grid2op.make(env_name)
            env.generate_data(nb_year=50)  # generates 50 years of data 
            # (takes roughly 50s per week, around 45mins per year, in this case 50 * 45 mins = lots of minutes)
            
        Let the script to generate the data run normally (don't interupt it).
        And from time to time, in the script "train_agent.py" you can do:
        
        .. code-block:: python
            
            # reload the generated data
            env.chronics_handler.init_subpath()
            env.chronics_handler.reset()
            
            # retrain the agent taking into account new data
            trained_agent = train(env,
                                  iterations=nb_iter,
                                  name=agent_name,
                                  save_path=agent_path,
                                  load_path=agent_path
                                  )
            
            # the script to generate data is still running, you can reload some data again
            env.chronics_handler.init_subpath()
            env.chronics_handler.reset()
            
            # retrain the agent
            trained_agent = train(env,
                                  iterations=nb_iter,
                                  name=agent_name,
                                  save_path=agent_path,
                                  load_path=agent_path
                                  )
                                  
            # etc.
        
        Both scripts you run "at the same time" for it to work efficiently.
        
        To recap:
        - script "generate_data.py" will... generate data
        - these data will be reloaded from time to time by the script "train_agent.py"
        
        .. warning:: 
            Do not delete data between calls to `env.chronics_handler.init_subpath()` and `env.chronics_handler.reset()`,
            and even less so during training !
            
            If you want to delete data (for example not to overload your hard drive) you should remove them 
            right before calling `env.chronics_handler.init_subpath()`.
            
        """
        try:
            self.subpaths = [
                os.path.join(self.path, el)
                for el in os.listdir(self.path)
                if os.path.isdir(os.path.join(self.path, el)) and (el != "__pycache__") and (not el.startwith("."))
            ]
            self.subpaths.sort()
            self.subpaths = np.array(self.subpaths)
        except FileNotFoundError as exc_:
            raise ChronicsError(
                'Path "{}" doesn\'t exists.'.format(self.path)
            ) from exc_
        self._order = None  # to trigger a "reset" when chronix will next be loaded
        
    def get_kwargs(self, dict_):
        if self._filter != self._default_filter:
            dict_["filter_func"] = self._filter

    def available_chronics(self):
        """return the list of available chronics.

        Examples
        --------

        # TODO
        """
        return self.subpaths[self._order]

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
        selected = self.space_prng.choice(self._order, p=probabilities)
        id_sel = np.where(self._order == selected)[0]
        self._prev_cache_id = selected - 1
        return id_sel

    def reset(self):
        """
        Rebuilt the :attr:`Multifolder._order`. This should be called after a call to :func:`Multifolder.set_filter`
        is performed.

        .. warning:: This "reset" is different from the `env.reset`. It should be only called after the function to set
            the filtering function has been called.

            This "reset" only reset which chronics are used for the environment.

        Returns
        -------
        new_order: ``numpy.ndarray``, dtype: str
            The selected chronics paths after a call to this method.

        Notes
        -----
        Except explicitly mentioned, for example by :func:`Multifolder.set_filter` you should not use this
        function. This will erased every selection of chronics, every shuffle etc.

        """
        self._order = []
        self._prev_cache_id = 0
        for i, path in enumerate(self.subpaths):
            if not self._filter(path):
                continue
            self._order.append(i)

        if len(self._order) == 0:
            raise RuntimeError(
                'Impossible to initialize the Multifolder. Your "filter_fun" filters out all the '
                "possible scenarios."
            )
        self._order = np.array(self._order)
        return self.subpaths[self._order]

    def initialize(
        self,
        order_backend_loads,
        order_backend_prods,
        order_backend_lines,
        order_backend_subs,
        names_chronics_to_backend=None,
    ):

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
        self.data = self.gridvalueClass(
            time_interval=self.time_interval,
            sep=self.sep,
            path=this_path,
            max_iter=self.max_iter,
            chunk_size=self.chunk_size,
            **self._kwargs
        )
        if self.seed is not None:
            max_int = np.iinfo(dt_int).max
            seed_chronics = self.space_prng.randint(max_int)
            self.data.seed(seed_chronics)

        self.data.initialize(
            order_backend_loads,
            order_backend_prods,
            order_backend_lines,
            order_backend_subs,
            names_chronics_to_backend=names_chronics_to_backend,
        )

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

    def tell_id(self, id_num, previous=False):
        """
        This tells this chronics to load for the next episode.
        By default, if id_num is greater than the number of episode, it is equivalent at restarting from the first
        one: episode are played indefinitely in the same order.

        Parameters
        ----------
        id_num: ``int`` | ``str``
            Id of the chronics to load.

        previous:
            Do you want to set to the previous value of this one or not (note that in general you want to set to
            the previous value, as calling this function as an impact only after `env.reset()` is called)
        """
        import pdb

        if isinstance(id_num, str):
            # new accepted behaviour starting 1.6.4
            # new in version 1.6.5: you only need to specify the chronics folder id and not the full path
            found = False
            for internal_id_, number in enumerate(self._order):
                if (
                    self.subpaths[number] == id_num
                    or os.path.join(self.path, id_num) == self.subpaths[number]
                ):
                    self._prev_cache_id = internal_id_
                    found = True

            if not found:
                raise ChronicsError(
                    f'Impossible to find the chronics with id "{id_num}". The call to '
                    f"`env.chronics_handler.tell_id(...)` cannot be performed."
                )
        else:
            # default behaviour prior to 1.6.4
            self._prev_cache_id = id_num
            self._prev_cache_id %= len(self._order)

        if previous:
            self._prev_cache_id -= 1
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

            # create an environment
            import numpy as np
            import grid2op
            env_name = "l2rpn_case14_sandbox"
            env = grid2op.make(env_name)

            # shuffle the chronics (uniformly at random, without duplication)
            env.chronics_handler.shuffle()
            # use the environment as you want, here do 10 episode with the selected data
            for i in range(10):
                obs = env.reset()
                print(f"Path of the chronics used: {env.chronics_handler.data.path}")
                done = False
                while not done:
                    act = ...
                    obs, reward, done, info = env.step(act)

            # re shuffle them (still uniformly at random, without duplication)
            env.chronics_handler.shuffle()

            # use the environment as you want, here do 10 episode with the selected data
            for i in range(10):
                obs = env.reset()
                print(f"Path of the chronics used: {env.chronics_handler.data.path}")
                done = False
                while not done:
                    act = ...
                    obs, reward, done, info = env.step(act)


        If you want to use only a subset of the path, say for example the path with index 1, 5, and 6

        .. code-block:: python

            # create an environment
            import numpy as np
            import grid2op
            env_name = "l2rpn_case14_sandbox"
            env = grid2op.make(env_name)

            # select the chronics (here 5 at random amongst the 10 "last" chronics of the environment)
            nb_chron = len(env.chronics_handler.chronics_used)
            chron_id_to_keep = np.random.choice(np.arange(nb_chron - 10, nb_chron), size=5, replace=True)
            env.chronics_handler.shuffle(lambda x: chron_id_to_keep)

            # use the environment as you want, here do 10 episode with the selected data
            for i in range(10):
                obs = env.reset()
                print(f"Path of the chronics used: {env.chronics_handler.data.path}")
                done = False
                while not done:
                    act = ...
                    obs, reward, done, info = env.step(act)

            # re shuffle them (uniformly at random, without duplication, among the chronics "selected" above.)
            env.chronics_handler.shuffle()

            # use the environment as you want, here do 10 episode with the selected data
            for i in range(10):
                obs = env.reset()
                print(f"Path of the chronics used: {env.chronics_handler.data.path}")
                done = False
                while not done:
                    act = ...
                    obs, reward, done, info = env.step(act)

        .. warning:: Though it is possible to use this "shuffle" function to only use some chronics, we highly
            recommend you to have a look at the sections :ref:`environment-module-chronics-info` or
            :ref:`environment-module-train-val-test`. It is likely that you will find better way to do
            what you want to do there. Use this last example with care then.

        .. warning:: As stated on the :func:`MultiFolder.reset`, any call to `env.chronics_handler.reset`
            will remove anything related to shuffling, including the selection of chronics !

        """
        if shuffler is None:

            def shuffler(x):
                return x[self.space_prng.choice(len(x), size=len(x), replace=False)]

        self._order = 1 * shuffler(self._order)
        return self.subpaths[self._order]

    @property
    def chronics_used(self):
        """return the full path of the chronics currently in use."""
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
            tmp = self.gridvalueClass(
                time_interval=self.time_interval,
                sep=self.sep,
                path=subpath,
                max_iter=self._max_iter,
                chunk_size=self.chunk_size,
            )
            seed_chronics = None
            if self.seed is not None:
                max_int = np.iinfo(dt_int).max
                seed_chronics = self.space_prng.randint(max_int)
                tmp.seed(seed_chronics)
            seed_chronics_all[subpath] = seed_chronics
            tmp.initialize(
                self._order_backend_loads,
                self._order_backend_prods,
                self._order_backend_lines,
                self._order_backend_subs,
                self._names_chronics_to_backend,
            )
            path_out_chron = os.path.join(path_out, id_this_chron)
            tmp.split_and_save(
                datetime_beg[id_this_chron], datetime_end[id_this_chron], path_out_chron
            )

            meta_params = {}
            meta_params["datetime_beg"] = datetime_beg
            meta_params["datetime_end"] = datetime_end
            meta_params["path_out"] = path_out
            meta_params["all_seeds"] = seed_chronics_all
            try:
                with open(
                    os.path.join(path_out, "split_and_save_meta_params.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(obj=meta_params, fp=f, sort_keys=True, indent=4)
            except Exception as exc_:
                warnings.warn(
                    'Impossible to save the "metadata" for the chronics with error:\n"{}"'
                    "".format(exc_)
                )
                
    def fast_forward(self, nb_timestep):
        self.data.fast_forward(nb_timestep)
