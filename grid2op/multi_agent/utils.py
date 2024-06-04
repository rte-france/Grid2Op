# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from numpy.random import shuffle
import grid2op
from lightsim2grid.lightSimBackend import LightSimBackend
import numpy as np
from sknetwork.clustering import Louvain
from scipy.sparse import csr_matrix

def random_order(agents : list, *args, **kwargs):
    """Returns the random order

    Args:
        agents (list): agents' names in the env

    Returns:
        list [int]: the execution order of agents
    """
    # TODO BEN: remove that and use the space_prng of the Environment
    return shuffle(agents)


class AgentSelector:
    """
    Outputs an agent in the given order whenever agent_select is called. Can reinitialize to a new order
    """
    #TODO
    
    def __init__(self, agents : list, agent_order_fn, *args, **kwargs):
        raise NotImplementedError("This function is not implemented at the moment.")
        self.agents = agents
        if agent_order_fn is None:
            agent_order_fn = lambda x : x
        self.agent_order_fn = agent_order_fn
        self.reinit(*args, **kwargs)

    def reinit(self, *args, **kwargs):
        self.agent_order = self.agent_order_fn(self.agents, *args, **kwargs)
        self._current_agent = 0
        self.selected_agent = self.agent_order[0]

    def reset(self, *args, **kwargs):
        self.reinit(*args, **kwargs)
        return self.next()
    
    def get_order(self, new_order = False, *args, **kwargs):
        order = self.agent_order.copy() 
        if new_order :
            self.reinit(*args, **kwargs)
        return order

    def next(self):
        self._current_agent = (self._current_agent + 1) % len(self.agent_order)
        self.selected_agent = self.agent_order[self._current_agent - 1]
        return self.selected_agent

    def is_last(self):
        """
        Does not work as expected if you change the order
        """
        return self.selected_agent == self.agent_order[-1]

    def is_first(self):
        return self.selected_agent == self.agent_order[0]

    def __eq__(self, other):
        if not isinstance(other, AgentSelector):
            return NotImplemented

        return (
            self.agent_order == other.agent_order
            and self._current_agent == other._current_agent
            and self.selected_agent == other.selected_agent
        )


class ClusterUtils:
    """
    Outputs clustered substation based on the Louvain graph clustering method
    """
    
    # Create connectivity matrix
    @staticmethod
    def create_connectivity_matrix(env):
        """
        Creates a connectivity matrix for the given grid environment.

        The connectivity matrix is a 2D NumPy array where the element at position (i, j) is 1 if there is a direct 
        connection between substation i and substation j, and 0 otherwise. The diagonal elements are set to 1 to indicate 
        self-connections.

        Args:
            env (grid2op.Environment): The grid environment for which the connectivity matrix is to be created.

        Returns:
            connectivity_matrix: A 2D Numpy array of dimension (env.n_sub, env.n_sub) representing the substation connectivity of the grid environment.
        """
        connectivity_matrix = np.zeros((env.n_sub, env.n_sub))
        for line_id in range(env.n_line):
            orig_sub = env.line_or_to_subid[line_id]
            extrem_sub = env.line_ex_to_subid[line_id]
            connectivity_matrix[orig_sub, extrem_sub] = 1
            connectivity_matrix[extrem_sub, orig_sub] = 1
        return connectivity_matrix + np.eye(env.n_sub)

    
       
    # Cluster substations
    @staticmethod
    def cluster_substations(env_name):
        """
        Clusters substations in a power grid environment using the Louvain community detection algorithm.

        This function creates a grid environment based on the specified environment name, generates a connectivity matrix 
        representing the connections between substations, and applies the Louvain algorithm to cluster the substations 
        into communities. The resulting clusters are formatted into a dictionary where each key corresponds to an agent 
        and the value is a list of substations assigned to that agent.

        Args:
            env_name (str): The name of the grid environment to be clustered
            
        Returns:
                (MADict):
                    - keys : agents' names 
                    - values : list of substations' id under the control of the agent.
        """
        # Create the environment
        env = grid2op.make(env_name, backend=LightSimBackend())

        # Generate the connectivity matrix
        matrix = ClusterUtils.create_connectivity_matrix(env)

        # Perform clustering using Louvain algorithm
        louvain = Louvain()
        adjacency = csr_matrix(matrix)
        labels = louvain.fit_predict(adjacency)

        # Group substations into clusters
        clusters = {}
        for node, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(node)

        # Format the clusters
        formatted_clusters = {f'agent_{i}': nodes for i, nodes in enumerate(clusters.values())}
        
        return formatted_clusters