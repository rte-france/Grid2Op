# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
from grid2op.Exceptions import IllegalAction
from grid2op.Rules.BaseRules import BaseRules


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
        # at first iteration, env.current_obs is None...
        # TODO this is used inside the environment (for step) inside LookParam and here
        # this could be computed only once, and fed to this instead
        powerline_status = env.get_current_line_status()

        aff_lines, aff_subs = action.get_topological_impact(powerline_status)
        if (env._times_before_line_status_actionable[aff_lines] > 0).any():
            # i tried to act on a powerline too shortly after a previous action
            # or shut down due to an overflow or opponent or hazards or maintenance
            ids = np.where((env._times_before_line_status_actionable > 0) & aff_lines)[
                0
            ]
            return False, IllegalAction(
                "Powerline with ids {} have been modified illegally (cooldown of {})".format(
                    ids, env._times_before_line_status_actionable[ids]
                )
            )

        if (env._times_before_topology_actionable[aff_subs] > 0).any():
            # I tried to act on a topology too shortly after a previous action
            ids = np.where((env._times_before_topology_actionable > 0) & aff_subs)[0]
            return False, IllegalAction(
                "Substation with ids {} have been modified illegally (cooldown of {})".format(
                    ids, env._times_before_topology_actionable[ids]
                )
            )

        return True, None
