# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import numpy as np
from datetime import timedelta

from grid2op.dtypes import dt_int
from grid2op.Exceptions import Grid2OpException, ChronicsError
from grid2op.Space import RandomObject
from grid2op.Chronics.GridValue import GridValue
from grid2op.Chronics.ChangeNothing import ChangeNothing


class ChronicsHandler(RandomObject):
    """
    Represents a Chronics handler that returns a grid state.

    As stated previously, it is not recommended to make an directly an object from the class :class:`GridValue`. This
    utility will ensure that the creation of such objects are properly made.

    The types of chronics used can be specified in the :attr:`ChronicsHandler.chronicsClass` attribute.

    Attributes
    ----------
    chronicsClass: ``type``, optional
        Type of chronics that will be loaded and generated. Default is :class:`ChangeNothing` (*NB* the class, and not
        an object / instance of the class should be send here.) This should be a derived class from :class:`GridValue`.

    kwargs: ``dict``, optional
        key word arguments that will be used to build new chronics.

    max_iter: ``int``, optional
        Maximum number of iterations per episode.

    real_data: :class:`GridValue`
        An instance of type given by :attr:`ChronicsHandler.chronicsClass`.

    path: ``str`` (or None)
        path where the data are located.

    """
    def __init__(self, chronicsClass=ChangeNothing, time_interval=timedelta(minutes=5), max_iter=-1,
                 **kwargs):
        RandomObject.__init__(self)
        if not isinstance(chronicsClass, type):
            raise Grid2OpException("Parameter \"chronicsClass\" used to build the ChronicsHandler should be a type "
                                   "(a class) and not an object (an instance of a class). It is currently "
                                   "\"{}\"".format(type(chronicsClass)))

        if not issubclass(chronicsClass, GridValue):
            raise ChronicsError("ChronicsHandler: the \"chronicsClass\" argument should be a derivative of the "
                               "\"Grid2Op.GridValue\" type and not {}.".format(type(chronicsClass)))

        self.chronicsClass = chronicsClass
        self.kwargs = kwargs
        self.max_iter = max_iter

        self.path = None
        if "path" in kwargs:
            self.path = kwargs["path"]

        self._real_data = None
        try:
            self._real_data = self.chronicsClass(time_interval=time_interval, max_iter=self.max_iter,
                                                 **self.kwargs)
        except TypeError:
            raise ChronicsError("Impossible to build a chronics of type {} with arguments in "
                                "{}".format(chronicsClass, self.kwargs))

    @property
    def real_data(self):
        return self._real_data

    def next_time_step(self):
        """
        This method returns the modification of the powergrid at the next time step for the same episode.

        See definition of :func:`GridValue.load_next` for more information about this method.

        """
        res = self._real_data.load_next()
        return res

    def get_name(self):
        """
        This method retrieve a unique name that is used to serialize episode data on
        disk. 
        
        See definition of :mod:`EpisodeData` for more information about this method.

        """
        return str(os.path.split(self.get_id())[-1])

    def set_max_iter(self, max_iter):
        """
        This function is used to set the maximum number of iterations possible before the chronics ends.

        Parameters
        ----------
        max_iter: ``int``
            The maximum number of timestep in the chronics.

        Returns
        -------

        """

        if not isinstance(max_iter, int):
            raise Grid2OpException("The maximum number of iterations possible for this chronics, before it ends.")
        self.max_iter = max_iter
        self._real_data.max_iter = max_iter

    def seed(self, seed):
        """
        Seed the chronics handler and the :class:`GridValue` that is used to generate the chronics.

        Parameters
        ----------
        seed: ``int``
            Set the seed for this instance and for the data it holds

        Returns
        -------
        seed: ``int``
            The seed used for this object

        seed_chronics: ``int``
            The seed used for the real data

        """
        super().seed(seed)
        max_int = np.iinfo(dt_int).max
        seed_chronics = self.space_prng.randint(max_int)
        self._real_data.seed(seed_chronics)
        return seed, seed_chronics

    def __getattr__(self, name):
        if name in ['__getstate__', '__setstate__']:
            # otherwise there is a recursion depth exceeded in multiprocessing
            # https://github.com/matplotlib/matplotlib/issues/7852/
            return object.__getattr__(self, name)
        return getattr(self._real_data, name)
