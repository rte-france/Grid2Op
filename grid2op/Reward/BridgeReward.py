# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import networkx as nx

from grid2op.Reward.BaseReward import BaseReward


class BridgeReward(BaseReward):
    """
    This reward computes a penalty based on how many bridges are present in the grid netwrok.
    In graph theory, a bridge is an edge that if removed will cause the graph to be disconnected.
    """
    def __init__(self):
        BaseReward.__init__(self)
        self.reward_min = 0.0
        self.reward_max = 1.0
        
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
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
            left_v = (lor_bus - 1) * n_bus + lor_sub
            right_v = (lex_bus - 1) * n_bus + lex_sub

            # Register edge in graph
            G.add_edge(left_v, right_v)
            
        # Find the bridges
        n_bridges = len(list(nx.bridges(G)))

        if n_bridges == 0:
            return self.reward_max
        elif n_bridges == 1:
            return (self.reward_max + self.reward_min) / 2.0
        else:
            return self.reward_min
