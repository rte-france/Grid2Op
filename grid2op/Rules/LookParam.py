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


class LookParam(BaseRules):
    """
    This subclass only check that the number of powerlines reconnected / disconnected by the agent.

    This class doesn't require any environment information. The "env" argument is only used to look for the
    game rules implemented in :class:`grid2op.Parameters`.

    See :func:`BaseRules.__call__` for a definition of the parameters of this function.

    """
    def __call__(self, action, env):
        """
        See :func:`BaseRules.__call__` for a definition of the parameters of this function.
        """
        # at first iteration, env.current_obs is None...
        if env.current_obs is not None:
            powerline_status = env.current_obs.line_status
        else:
            powerline_status = None

        aff_lines, aff_subs = action.get_topological_impact(powerline_status)
        if np.sum(aff_lines) > env.parameters.MAX_LINE_STATUS_CHANGED:
            return False
        if np.sum(aff_subs) > env.parameters.MAX_SUB_CHANGED:
            return False
        return True
