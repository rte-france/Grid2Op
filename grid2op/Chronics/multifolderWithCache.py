# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import numpy as np
from datetime import timedelta, datetime
import warnings

from grid2op.dtypes import dt_int
from grid2op.Chronics.multiFolder import Multifolder
from grid2op.Chronics.gridStateFromFile import GridStateFromFile
from grid2op.Chronics.time_series_from_handlers import FromHandlers
from grid2op.Exceptions import ChronicsError


class MultifolderWithCache(Multifolder):
    """
    This class is a particular type of :class:`Multifolder` that, instead of reading is all from disk each time
    stores it into memory.

    For now it's only compatible (because it only present some kind of interest) with :class:`GridValue` class
    inheriting from :class:`GridStateFromFile`.

    The function :func:`MultifolderWithCache.reset` will redo the cache from scratch. You can filter which
    type of data will be cached or not with the :func:`MultifolderWithCache.set_filter` function.

    **NB** Efficient use of this class can dramatically increase the speed of the learning algorithm, especially at
    the beginning where lots of data are read from the hard drive and the agent games over after a few time steps (
    typically, data are given by months, so 30*288 >= 8600 time steps, while during exploration an agent usually
    performs less than a few dozen of steps leading to more time spent reading 8600 rows than computing the
    few dozen of steps.

    .. danger::
        When you create an environment with this chronics class (*eg* by doing 
        `env = make(...,chronics_class=MultifolderWithCache)`), the "cache" is not
        pre loaded, only the first scenario is loaded in memory (to save loading time).
        
        In order to load everything, you NEED to call `env.chronics_handler.reset()`, which,
        by default, will load every scenario into memory. If you want to filter some
        data, for example by reading only the scenario of decembre, you can use the 
        `set_filter` method.
        
        A typical workflow (at the start of your program) when using this class is then:
        
        1) create the environment: `env = make(...,chronics_class=MultifolderWithCache)`
        2) (optional but recommended) select some scenarios: 
           `env.chronics_handler.real_data.set_filter(lambda x: re.match(".*december.*", x) is not None)`
        3) load the data in memory: `env.chronics_handler.reset()`
        4) do whatever you want using `env`

    .. note::
        After creation (anywhere in your code), 
        you can use other scenarios by calling the `set_filter` function again:
        
        1) select other scenarios: 
           `env.chronics_handler.real_data.set_filter(lambda x: re.match(".*january.*", x) is not None)` 
        2) load the data in memory: `env.chronics_handler.reset()`
        3) do whatever you want using `env`
    
    Examples
    ---------
    This is how this class can be used:

    .. code-block:: python

        import re
        from grid2op import make
        from grid2op.Chronics import MultifolderWithCache
        env = make(...,chronics_class=MultifolderWithCache)

        # set the chronics to limit to one week of data (lower memory footprint)
        env.set_max_iter(7*288)
        # assign a filter, use only chronics that have "december" in their name
        env.chronics_handler.real_data.set_filter(lambda x: re.match(".*december.*", x) is not None)
        # create the cache
        env.chronics_handler.reset()

        # and now you can use it as you would do any gym environment:
        my_agent = ...
        obs = env.reset()
        done = False
        reward = env.reward_range[0]
        while not done:
            act = my_agent.act(obs, reward, done)
            obs, reward, done, info = env.step(act)  # and step will NOT load any data from disk.

    """
    MULTI_CHRONICS = True
    ERROR_MSG_NOT_LOADED = ("We detected a misusage of the `MultifolderWithCache` class: the cache "
                            "has not been loaded in memory which will most likely cause issues "
                            "with your environment. Do not forget to call "
                            "`env.chronics_handler.set_filter(...)` to tell which time series "
                            "you want to keep and then `env.chronics_handler.reset()` "
                            "to load them. \nFor more information consult the documentation:\n"
                            "https://grid2op.readthedocs.io/en/latest/chronics.html#grid2op.Chronics.MultifolderWithCache")
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
        **kwargs,
    ):
        
        # below: counter to prevent use without explicit call to `env.chronics.handler.reset()`
        if "_DONTUSE_nb_reset_called" in kwargs:
            self.__nb_reset_called = int(kwargs["_DONTUSE_nb_reset_called"])
            del kwargs["_DONTUSE_nb_reset_called"]
        else:
            self.__nb_reset_called = -1
        if "_DONTUSE_nb_step_called" in kwargs:
            self.__nb_step_called = int(kwargs["_DONTUSE_nb_step_called"])
            del kwargs["_DONTUSE_nb_step_called"]
        else:
            self.__nb_step_called = -1
            
        if "_DONTUSE_nb_init_called" in kwargs:
            self.__nb_init_called = int(kwargs["_DONTUSE_nb_init_called"])
            del kwargs["_DONTUSE_nb_init_called"]
        else:
            self.__nb_init_called = -1
        
        # now init the data
        Multifolder.__init__(
            self,
            path=path,
            time_interval=time_interval,
            start_datetime=start_datetime,
            gridvalueClass=gridvalueClass,
            sep=sep,
            max_iter=max_iter,
            chunk_size=None,
            filter_func=filter_func,
            **kwargs
        )
        self._cached_data = None
        self.cache_size = 0
        if not (issubclass(self.gridvalueClass, GridStateFromFile) or 
                issubclass(self.gridvalueClass, FromHandlers)):
            raise RuntimeError(
                'MultifolderWithCache does not work when "gridvalueClass" does not inherit from '
                '"GridStateFromFile".'
            )
        if issubclass(self.gridvalueClass, FromHandlers):
            warnings.warn("You use caching with handler data. This is possible but "
                          "might be a bit risky especially if your handlers are "
                          "heavily 'random' and you want fully reproducible results.")
        self.__i = 0
        self._cached_seeds = None

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
        Rebuilt the cache as if it were built from scratch. 
        This call might take a while to process.
        
        .. danger::
            You NEED to call this function (with `env.chronics_handler.reset()`)
            if you use the `MultiFolderWithCache` class in your experiments.
            
        .. warning::
            If a seed is set (see :func:`MultiFolderWithCache.seed`) then
            all the data in the cache are also seeded when this 
            method is called.
        """
        self._cached_data = [None for _ in self.subpaths]
        self.__i = 0
        # select the right paths, and store their id in "_order"
        super().reset()
        self.cache_size = 0
        max_int = np.iinfo(dt_int).max
        for i in self._order:
            # everything in "_order" need to be put in cache
            path = self.subpaths[i]
            data = self._get_nex_data(path)
            # if issubclass(self.gridvalueClass, GridStateFromFile):
            #     data = self.gridvalueClass(
            #         time_interval=self.time_interval,
            #         sep=self.sep,
            #         path=path,
            #         max_iter=self.max_iter,
            #         chunk_size=None,
            #     )
            # elif issubclass(self.gridvalueClass, FromHandlers):
            #     data = self.gridvalueClass(
            #         time_interval=self.time_interval,
            # else:
            #     raise ChronicsError("Can only use MultiFolderWithCache with GridStateFromFile "
            #                         f"or FromHandlers and not {self.gridvalueClass}")
            if self.seed_used is not None:
                # seed_chronics = self.space_prng.randint(max_int)
                # self._cached_seeds[i] = seed_chronics
                data.seed(self._cached_seeds[i])
                data.regenerate_with_new_seed()

            data.initialize(
                self._order_backend_loads,
                self._order_backend_prods,
                self._order_backend_lines,
                self._order_backend_subs,
                self._names_chronics_to_backend,
            )
            self._cached_data[i] = data
            self.cache_size += 1
            if self.action_space is not None:
                data.action_space = self.action_space

        if self.cache_size == 0:
            raise RuntimeError("Impossible to initialize the new cache.")
        
        self.__nb_reset_called += 1
        return self.subpaths[self._order]
    
    def initialize(
        self,
        order_backend_loads,
        order_backend_prods,
        order_backend_lines,
        order_backend_subs,
        names_chronics_to_backend=None,
    ):
        self.__nb_init_called += 1
        if self.__nb_reset_called <= 0:
            if self.__nb_init_called != 0:
                # authorize the creation of the environment but nothing more
                raise ChronicsError(type(self).ERROR_MSG_NOT_LOADED)
        
        self._order_backend_loads = order_backend_loads
        self._order_backend_prods = order_backend_prods
        self._order_backend_lines = order_backend_lines
        self._order_backend_subs = order_backend_subs
        self._names_chronics_to_backend = names_chronics_to_backend

        self.n_gen = len(order_backend_prods)
        self.n_load = len(order_backend_loads)
        self.n_line = len(order_backend_lines)
        if self._cached_data is None:
            # initialize the cache of this MultiFolder
            self.reset()

        id_scenario = self._order[self._prev_cache_id]
        self.data = self._cached_data[id_scenario]
        self.data.next_chronics()
        if self.seed_used is not None and self.data.seed_used != self._cached_seeds[id_scenario]:
            self.data.seed(self._cached_seeds[id_scenario])
            self.data.regenerate_with_new_seed()
        self._max_iter = self.data.max_iter
    
    @property
    def max_iter(self):
        return self._max_iter
    
    @max_iter.setter
    def max_iter(self, value : int):
        self._max_iter = int(value)
        for el in self._cached_data:
            if el is None:
                continue
            el.max_iter = value
    
    def max_timestep(self):
        return self.data.max_timestep()
    
    def seed(self, seed : int):
        """This seeds both the MultiFolderWithCache
        (which has an impact for example on :func:`MultiFolder.sample_next_chronics`)
        and each data present in the cache.

        .. warning::
            Before grid2op version 1.10.3 this function did not fully ensured 
            reproducible experiments (the cache was not update with the new seed)
            
            For grid2op 1.10.3 and after, this function might trigger some modification 
            in the cached data (calling :func:`GridValue.seed` and then 
            :func:`GridValue.regenerate_with_new_seed`). It might take a while if the cache
            is large.
            
        Parameters
        ----------
        seed : int
            The seed to use
        """
        res = super().seed(seed)
        max_int = np.iinfo(dt_int).max
        self._cached_seeds = np.empty(shape=self._order.shape, dtype=dt_int)
        for i in self._order:
            data = self._cached_data[i]
            seed_ts = self.space_prng.randint(max_int)
            self._cached_seeds[i] = seed_ts
            if data is None:
                continue
            data.seed(seed_ts)
            data.regenerate_with_new_seed()
        return res
        
    def load_next(self):
        self.__nb_step_called += 1
        if self.__nb_reset_called <= 0:
            if self.__nb_step_called != 0:
                # authorize the creation of the environment but nothing more
                raise ChronicsError(type(self).ERROR_MSG_NOT_LOADED)
        return super().load_next()

    def set_filter(self, filter_fun):
        """
        Assign a filtering function to remove some chronics from the next time a call to "reset_cache" is called.

        **NB** filter_fun is applied to all element of :attr:`Multifolder.subpaths`. If ``True`` then it will
        be put in cache, if ``False`` this data will NOT be put in the cache.

        **NB** this has no effect until :attr:`Multifolder.reset` is called.
        
        
        .. danger::
            Calling this function cancels the previous seed used. If you use `env.seed`
            or `env.chronics_handler.seed` before then you need to 
            call it again after otherwise it has no effect.

        Parameters
        ----------
        filter_fun : _type_
            _description_

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

        Returns
        -------
        _type_
            _description_
        """
        self.__nb_reset_called = 0
        self.__nb_step_called = 0
        self.__nb_init_called = 0
        self._cached_seeds = None
        return super().set_filter(filter_fun)

    def get_kwargs(self, dict_):
        dict_["_DONTUSE_nb_reset_called"] = self.__nb_reset_called
        dict_["_DONTUSE_nb_step_called"] = self.__nb_step_called
        dict_["_DONTUSE_nb_init_called"] = self.__nb_init_called
        return super().get_kwargs(dict_)

    def cleanup_action_space(self):
        super().cleanup_action_space()
        for el in self._cached_data:
            if el is None:
                continue
            el.cleanup_action_space()
