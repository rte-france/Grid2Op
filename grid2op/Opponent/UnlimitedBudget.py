# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.dtypes import dt_float
from grid2op.Opponent.BaseActionBudget import BaseActionBudget


class UnlimitedBudget(BaseActionBudget):
    """
    This class define an unlimited budget for the opponent.

    It SHOULD NOT be used if the opponent is allowed to take any actions!
    """
    def __init__(self, action_space):
        BaseActionBudget.__init__(self, action_space)
        self._zero = dt_float(0.)

    def __call__(self, attack):
        return self._zero
