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
from grid2op.dtypes import dt_float
import pdb


class ConnectivityConverter(Converter):
    """
    In this converter, you have as many output as pairs of object that can be connected, and your model is asked
    to output 0 if he wants these elements disconnected and 1 if he wants them otherwise.

    This type of modelisation is rather hard to
    **NB** there is a default behaviour to solve the conflict in this converter.

    """
    def __init__(self, action_space):
        Converter.__init__(self, action_space)
        self.__class__ = ConnectivityConverter.init_grid(action_space)
        self.all_actions = []
        # add the do nothing topology
        self.all_actions.append(super().__call__())
        self.n = 1

    def init_converter(self, all_actions=None, **kwargs):
        pass