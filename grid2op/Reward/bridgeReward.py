# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
import networkx as nx

from grid2op.Reward.baseReward import BaseReward
from grid2op.dtypes import dt_float


class BridgeReward(BaseReward):
    """
    This reward computes a penalty based on how many bridges are present in the grid network.
    In graph theory, a bridge is an edge that if removed will cause the graph to be disconnected.

    Examples
    ---------
    You can use this reward in any environment with:

    .. code-block:

        import grid2op
        from grid2op.Reward import BridgeReward

        # then you create your environment with it:
        NAME_OF_THE_ENVIRONMENT = "l2rpn_case14_sandbox"
        env = grid2op.make(NAME_OF_THE_ENVIRONMENT,reward_class=BridgeReward)
        # and do a step with a "do nothing" action
        obs = env.reset()
        obs, reward, done, info = env.step(env.action_space())
        # the reward is computed with this class (computing the penalty based on the number of "bridges" in the grid)

    """

    def __init__(self, min_pen_lte=0.0, max_pen_gte=1.0, logger=None):
        BaseReward.__init__(self, logger=logger)
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)
        self.min_pen_lte = dt_float(min_pen_lte)
        self.max_pen_gte = dt_float(max_pen_gte)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min

        n_bus = 2

        # Get info from env
        obs = env.current_obs
        n_sub = obs.n_sub
        n_line = obs.n_line
        topo = obs.topo_vect
        or_topo = obs.line_or_pos_topo_vect
        ex_topo = obs.line_ex_pos_topo_vect
        or_sub = obs.line_or_to_subid
        ex_sub = obs.line_ex_to_subid

        # Create a graph of vertices
        # Use one vertex per substation per bus
        G = nx.Graph()

        # Set lines edges for current bus
        for line_idx in range(n_line):
            # Skip if line is disconnected
            if obs.line_status[line_idx] is False:
                continue
            # Get substation index for current line
            lor_sub = or_sub[line_idx]
            lex_sub = ex_sub[line_idx]
            # Get the buses for current line
            lor_bus = topo[or_topo[line_idx]]
            lex_bus = topo[ex_topo[line_idx]]

            if lor_bus <= 0 or lex_bus <= 0:
                continue

            # Compute edge vertices indices for current graph
            left_v = lor_sub + (lor_bus - 1) * n_sub
            right_v = lex_sub + (lex_bus - 1) * n_sub

            # Register edge in graph
            G.add_edge(left_v, right_v)

        # Find the bridges
        n_bridges = dt_float(len(list(nx.bridges(G))))

        # Clip to min penalty
        n_bridges = max(n_bridges, self.min_pen_lte)
        # Clip to max penalty
        n_bridges = min(n_bridges, self.max_pen_gte)
        r = np.interp(
            n_bridges,
            [self.min_pen_lte, self.max_pen_gte],
            [self.reward_max, self.reward_min],
        )
        return dt_float(r)
