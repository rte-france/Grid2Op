# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
from grid2op.Rules.BaseRules import BaseRules

import pdb


class PreventReconnection(BaseRules):
    """
    A subclass is used to check that an action will not attempt to reconnect a powerlines disconnected because of
    an overflow, or to check that 2 actions acting on the same powerline are distant from the right number of timesteps
    (see :attr:`grid2op.Parameters.Parameters.NB_TIMESTEP_LINE_STATUS_REMODIF`) or if two topological modification
    of the same substation are too close in time
    (see :attr:`grid2op.Parameters.Parameters.NB_TIMESTEP_TOPOLOGY_REMODIF`)

    """
    def __call__(self, action, env):
        """
        This function check only that the action doesn't attempt to reconnect  a powerline that has been disconnected
        due to an overflow.

        See :func:`BaseRules.__call__` for a definition of the parameters of this function.

        """
        aff_lines, aff_subs = action.get_topological_impact()
        if np.any(env.time_remaining_before_reconnection[aff_lines] > 0):
            # i tried to act on a powerline removed because an overflow
            return False

        if np.any(env.times_before_line_status_actionable[aff_lines] > 0):
            # i tried to act on a powerline too shortly after a previous action
            return False

        if np.any(env.times_before_topology_actionable[aff_subs] > 0):
            # I tried to act on a topology too shortly after a previous action
            return False

        return True
