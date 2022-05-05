from numpy.random import shuffle

def random_order(n_agents : int, *args, **kwargs):
    """Returns the random order

    Args:
        n_agents (int): Number of agents in the env

    Returns:
        list [int]: the execution order of agents
    """
    return list(shuffle(range(n_agents)))


class AgentSelector:
    """
    Outputs an agent in the given order whenever agent_select is called. Can reinitialize to a new order
    """
    #TODO
    
    def __init__(self, n_agents, agent_order_fn = random_order):
        self.n_agents = n_agents
        self.agent_order_fn = agent_order_fn
        self.reinit(agent_order_fn)

    def reinit(self, *args, **kwargs):
        self.agent_order = self.agent_order_fn(self.n_agents, *args, **kwargs)
        self._current_agent = 0
        self.selected_agent = 0

    def reset(self, *args, **kwargs):
        self.reinit(*args, **kwargs)
        return self.next()

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
        
        
        
class Domain :
    pass