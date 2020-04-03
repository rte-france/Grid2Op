import pdb

from grid2op.Agent.GreedyAgent import GreedyAgent


class TopologyGreedy(GreedyAgent):
    """
    This is a :class:`GreedyAgent` example, which will attempt to reconfigure the substations connectivity.

    It will choose among:

      - doing nothing
      - changing the topology of one substation.

    """
    def __init__(self, action_space, action_space_converter=None):
        GreedyAgent.__init__(self, action_space, action_space_converter=action_space_converter)
        self.li_actions = None

    def _get_tested_action(self, observation):
        if self.li_actions is None:
            res = [self.action_space({})]  # add the do nothing
            res += self.action_space.get_all_unitary_topologies_change(self.action_space)
            self.li_actions = res
        return self.li_actions

