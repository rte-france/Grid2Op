# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import os
import numpy as np

from grid2op.Action import BaseAction
from grid2op.Converter.Converters import Converter
from grid2op.dtypes import dt_float, dt_int
import pdb


class ConnectivityConverter(Converter):
    """
    In this converter, you have as many output as pairs of object that can be connected, and your model is asked
    to output 0 if he wants these elements disconnected and 1 if he wants them otherwise.

    This type of modeling is rather hard to "get working" the first time, especially because some "conflict" might
    appear. For example, consider three objects (line for example) on a given substation. You can chose to "connect
    A and B", connect "B and C" but "**not connect** A and C" in this case you need an algorithm to disambuate your
    action.

    **NB** compare to :class:`IdToAct` this converter allows for a smaller size. If you have N elements connected at
    a substation, you end up with `N*(N-1)/2` different action. Compare to IdToAct though, it is expected that your
    algorithm produces more than 1 output.

    **/!\ VERY IMPORTANT /!\** : for this converter to work, it needs to remember the previous state of the grid, so you
    absolutely need to call its method :func:`ConnectivityConverter.convert_obs` a each observation.
    """
    def __init__(self, action_space):
        Converter.__init__(self, action_space)
        self.__class__ = ConnectivityConverter.init_grid(action_space)
        self.subs_ids = np.array([], dtype=dt_int)
        self.n = 1
        self.last_obs = None

    def init_converter(self, all_actions=None, **kwargs):
        # compute all pairs of elements that can be connected together
        for sub_id, nb_element in enumerate(self.sub_info):
            if nb_element <= 4:
                continue
            nb_pairs = int(nb_element * (nb_element - 1)/2)
            self.subs_ids = np.concatenate((self.subs_ids, sub_id * np.ones(nb_pairs, dtype=dt_int)))
        self.n = self.subs_ids.shape[0]

    def convert_obs(self, obs):
        """
        This function is used to convert an observation into something that is easier to manipulate.

        Parameters
        ----------
        obs: :class:`grid2op.Observation.Observation`
            The input observation.

        Returns
        -------

        transformed_obs: ``object``
            An different representation of the input observation, typically represented as a 1d vector that can be
            processed by a neural networks.

        """
        self.last_obs = obs
        return obs