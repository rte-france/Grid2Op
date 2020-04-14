__all__ = [
    "BaseAgent",
    "DoNothingAgent",
    "OneChangeThenNothing",
    "GreedyAgent",
    "PowerLineSwitch",
    "TopologyGreedy",
    "AgentWithConverter",
    "RandomAgent",
    "MLAgent",
    "Agent"
]

from grid2op.Agent.BaseAgent import BaseAgent
from grid2op.Agent.DoNothing import DoNothingAgent
from grid2op.Agent.OneChangeThenNothing import OneChangeThenNothing
from grid2op.Agent.GreedyAgent import GreedyAgent
from grid2op.Agent.PowerlineSwitch import PowerLineSwitch
from grid2op.Agent.TopologyGreedy import TopologyGreedy
from grid2op.Agent.AgentWithConverter import AgentWithConverter
from grid2op.Agent.RandomAgent import RandomAgent
from grid2op.Agent.MLAgent import MLAgent
import warnings


class Agent(BaseAgent):
    """
    .. deprecated:: 0.7.0
        Use :class:`BaseAgent` instead.

    This class has been renamed for better uniformity in the grid2op framework.
    """
    def __init__(self, *args, **kwargs):
        BaseAgent.__init__(self, *args, **kwargs)
        warnings.warn("Agent class has been renamed \"BaseAgent\". The Agent class will be removed"
                      "in future versions.",
                      category=PendingDeprecationWarning)
