# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import numpy as np
from datetime import  timedelta
import pdb

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

    id_chron_folder_current: ``int``
        Id (in :attr:`MultiFolder.subpaths`) for which data are generated in the current episode.
    """
    def __init__(self, path,
                 time_interval=timedelta(minutes=5),
                 gridvalueClass=GridStateFromFile,
                 sep=";", max_iter=-1,
                 chunk_size=None):
        GridValue.__init__(self, time_interval=time_interval, max_iter=max_iter, chunk_size=chunk_size)
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
        # np.random.shuffle(self.subpaths)
        self.id_chron_folder_current = 0
        self.chunk_size = chunk_size

    def initialize(self, order_backend_loads, order_backend_prods, order_backend_lines, order_backend_subs,
                   names_chronics_to_backend=None):

        self.n_gen = len(order_backend_prods)
        self.n_load = len(order_backend_loads)
        self.n_line = len(order_backend_lines)
        self.data = self.gridvalueClass(time_interval=self.time_interval,
                                        sep=self.sep,
                                        path=self.subpaths[self.id_chron_folder_current],
                                        max_iter=self.max_iter,
                                        chunk_size=self.chunk_size)
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

    def next_chronics(self):
        """
        Load the next episode.

        Note that :func:`MultiFolder.initialize` must be called after a call to this method has been performed. This is
        either done by the :class:`grid2op.Environemnt` or by the :class:`grid2op.Runner`.

        Returns
        -------
        ``None``

        """
        self.id_chron_folder_current += 1
        self.id_chron_folder_current %= len(self.subpaths)

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
        self.id_chron_folder_current = id_num
        self.id_chron_folder_current %= len(self.subpaths)
        # print("Chronics handler: going to chronics {}".format(self.id_chron_folder_current))

    def get_id(self) -> str:
        """
        Full absolute path of the current folder used for the current episode.

        Returns
        -------
        res: ``str``
            Path from which the data are generated for the current episode.
        """
        return self.subpaths[self.id_chron_folder_current]

    def max_timestep(self):
        return self.data.max_timestep()

    def shuffle(self, shuffler):
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
        self.subpaths = shuffler(self.subpaths)

    def set_chunk_size(self, new_chunk_size):
        self.chunk_size = new_chunk_size
