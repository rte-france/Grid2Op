import networkx as nx

from grid2op.Reward.BaseReward import BaseReward

class BridgeReward(BaseReward):
    """
    This reward computes a penalty based on how many bridges are present in the grid netwrok.
    In graph theory, a bridge is an edge that if removed will cause the graph to be disconnected.
    """
    def __init__(self):
        BaseReward.__init__(self)
        self.reward_min = -1000.0
        self.reward_max = 100.0

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        n_bus = 3
        
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
        G.add_nodes_from(range(n_sub * n_bus))
        
        # Set lines edges for current bus
        for line_idx in range(n_line):
            # Skip if line is disconnected
            if obs.line_status[line_idx] is False:
                continue
            
            # Get the buses for current line
            lor_bus = topo[or_topo[line_idx]]
            lex_bus = topo[ex_topo[line_idx]]

            if lor_bus == 0 or lex_bus == 0:
                continue

            # Compute edge vertices indices for current graph
            left_v = (lor_bus - 1) * n_sub
            right_v = (lex_bus - 1) * n_sub

            # Register edge in graph
            G.add_edge(left_v, right_v)
            
        # Find the bridges
        n_bridges = len(list(nx.bridges(G)))

        reward = self.reward_max - (250.0 * n_bridges)
        return reward
