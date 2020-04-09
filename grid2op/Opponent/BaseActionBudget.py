# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

"""
The opponent cannot act indefinitely, this would make the "game" impossible to play for the BaseAgent.

Thus, Opponent have some budget. Budget are computed using this class

TODO
"""
import numpy as np
from grid2op.Exceptions import OpponentError


class BaseActionBudget:
    def __init__(self, action_space):
        self.action_space = action_space

    def __call__(self, attack):
        """
        This function takes an attack as input and compute the cost associated to it.

        Parameters
        ----------
        attack: :class:`ŋrid2op.BaseAction.BaseAction`
            The attack performed by the opponent

        Returns
        -------
        cost:
        """
        if not isinstance(attack, self.action_space.actionClass):
            raise OpponentError("Attempt to use an attack of type \"{}\" which is not a instance of \"{}\", "
                                "the type of action the opponent was supposed to use."
                                "".format(type(attack), self.action_space.actionClass))
        aff_lines, aff_subs = attack.get_topological_impact()
        cost = np.sum(aff_lines) + np.sum(aff_subs)
        return cost
