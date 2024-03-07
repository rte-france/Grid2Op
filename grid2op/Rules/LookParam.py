# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
from grid2op.Exceptions import (
    IllegalAction,
    SimulateUsedTooMuchThisStep,
    SimulateUsedTooMuchThisEpisode,
)
from grid2op.Rules.BaseRules import BaseRules


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
        powerline_status = env.get_current_line_status()

        aff_lines, aff_subs = action.get_topological_impact(powerline_status)
        if aff_lines.sum() > env._parameters.MAX_LINE_STATUS_CHANGED:
            ids = np.nonzero(aff_lines)[0]
            return False, IllegalAction(
                "More than {} line status affected by the action: {}"
                "".format(env.parameters.MAX_LINE_STATUS_CHANGED, ids)
            )
        if aff_subs.sum() > env._parameters.MAX_SUB_CHANGED:
            ids = np.nonzero(aff_subs)[0]
            return False, IllegalAction(
                "More than {} substation affected by the action: {}"
                "".format(env.parameters.MAX_SUB_CHANGED, ids)
            )
        return True, None

    def can_use_simulate(self, nb_simulate_call_step, nb_simulate_call_episode, param):
        if param.MAX_SIMULATE_PER_STEP >= 0:
            if nb_simulate_call_step > param.MAX_SIMULATE_PER_STEP:
                return SimulateUsedTooMuchThisStep(
                    f"attempt to use {nb_simulate_call_step} times `obs.simulate(...)` while the maximum allowed for this step is {param.MAX_SIMULATE_PER_STEP}"
                )
        if param.MAX_SIMULATE_PER_EPISODE >= 0:
            if nb_simulate_call_episode > param.MAX_SIMULATE_PER_EPISODE:
                return SimulateUsedTooMuchThisEpisode(
                    f"attempt to use {nb_simulate_call_episode} times `obs.simulate(...)` while the maximum allowed for this episode is {param.MAX_SIMULATE_PER_EPISODE}"
                )
