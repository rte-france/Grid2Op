from abc import ABC, abstractmethod
import numpy as np
import itertools
import pdb

from grid2op.Exceptions import Grid2OpException
from grid2op.Agent.GreedyAgent import GreedyAgent


class PowerLineSwitch(GreedyAgent):
    """
    This is a :class:`GreedyAgent` example, which will attempt to disconnect powerlines.

    It will choose among:

      - doing nothing
      - disconnecting one powerline

    which action that will maximize the reward. All powerlines are tested.

    """

    def __init__(self, action_space):
        GreedyAgent.__init__(self, action_space)

    def _get_tested_action(self, observation):
        res = [self.action_space({})]  # add the do nothing
        for i in range(self.action_space.n_line):
            tmp = np.full(self.action_space.n_line, fill_value=False, dtype=np.bool)
            tmp[i] = True
            action = self.action_space({"change_line_status": tmp})
            if not observation.line_status[i]:
                # so the action consisted in reconnecting the powerline
                # i need to say on which bus (always on bus 1 for this type of agent)
                action = action.update({"set_bus": {"lines_or_id": [(i, 1)], "lines_ex_id": [(i, 1)]}})
            res.append(action)
        return res
