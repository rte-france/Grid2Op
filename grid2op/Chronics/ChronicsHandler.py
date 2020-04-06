# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
from datetime import timedelta
import pdb

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

        self.real_data = None
        try:
            self.real_data = self.chronicsClass(time_interval=time_interval, max_iter=self.max_iter,
                                                **self.kwargs)
        except TypeError:
            raise ChronicsError("Impossible to build a chronics of type {} with arguments in "
                                "{}".format(chronicsClass, self.kwargs))

    def initialize(self, order_backend_loads, order_backend_prods, order_backend_lines, order_backend_subs,
                   names_chronics_to_backend=None):
        """
        After being loaded, this method will initialize the data.

        See definition of :func:`GridValue.initialize` for more information about this method.

        Returns
        -------
        ``None``

        """
        self.real_data.initialize(order_backend_loads, order_backend_prods, order_backend_lines, order_backend_subs,
                                  names_chronics_to_backend)

    def check_validity(self, backend):
        """
        This method ensure the data are valid and compatible with the backend used.

        See definition of :func:`GridValue.check_validity` for more information about this method.

        Returns
        -------
        ``None``

        """
        self.real_data.check_validity(backend)

    def next_time_step(self):
        """
        This method returns the modification of the powergrid at the next time step for the same episode.

        See definition of :func:`GridValue.load_next` for more information about this method.

        """
        res = self.real_data.load_next()
        return res

    def done(self):
        """
        This method returns whether or not the episode is done.

        See definition of :func:`GridValue.done` for more information about this method.

        """
        return self.real_data.done()

    def forecasts(self):
        """
        This method returns the forecasts of the data.

        See definition of :func:`GridValue.forecasts` for more information about this method.

        """
        return self.real_data.forecasts()

    def next_chronics(self):
        """
        This method is called when changing the episode after game over or after it has reached the end.

        See definition of :func:`GridValue.next_chronics` for more information about this method.

        """
        self.real_data.next_chronics()

    def tell_id(self, id_num):
        """
        This method is called when setting a given episode after game over or after it has reached the end.

        See definition of :func:`GridValue.tell_id` for more information about this method.

        """
        self.real_data.tell_id(id_num=id_num)

    def max_timestep(self):
        """
        This method gives the maximum number of time step an episode can last.

        See definition of :func:`GridValue.max_timestep` for more information about this method.

        """
        return self.real_data.max_timestep()

    def get_id(self):
        """
        This method gives a unique identifier for the current episode.

        See definition of :func:`GridValue.get_id` for more information about this method.

        """
        return self.real_data.get_id()

    def get_name(self):
        """
        This method retrieve a unique name that is used to serialize episode data on
        disk. 
        
        See definition of :mod:`EpisodeData` for more information about this method.

        """
        return str(os.path.split(self.get_id())[-1])

    def shuffle(self, shuffler):
        """
        Will attempt to shuffle the underlying data.

        Note that a call to this function might not do anything is the :func:`GridValue.shuffle` is not implemented
        for :attr:`ChronicsHandler.real_data`.

        Parameters
        ----------
        shuffler: ``object``
            Anything that is used to shuffle the data.

        """
        self.real_data.shuffle(shuffler)

    def set_chunk_size(self, new_chunk_size):
        """
        This functions allows to adjust dynamically the chunk size when reading the data.

        **NB** this function is not supported by all data generating process.

        Parameters
        ----------
        new_chunk_size: ``int`` or ``None``
            The new chunk size

        """
        if new_chunk_size is None:
            pass
        elif not isinstance(new_chunk_size, int):
            raise Grid2OpException("Impossible to read data with an non integer chunk size.")
        self.real_data.set_chunk_size(new_chunk_size)

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
        self.real_data.max_iter = max_iter
