import pdb

from grid2op.Converter import IdToAct
from grid2op.Agent.AgentWithConverter import AgentWithConverter


class RandomAgent(AgentWithConverter):
    """
    This agent acts randomnly on the powergrid. It uses the :class:`grid2op.Converters.IdToAct` to compute all the
    possible actions available for the environment. And then chooses a random one among all these.
    """
    def __init__(self, action_space, action_space_converter=IdToAct, **kwargs_converter):
        AgentWithConverter.__init__(self, action_space, action_space_converter, **kwargs_converter)

    def my_act(self, transformed_observation, reward, done=False):
        return self.action_space.sample()
