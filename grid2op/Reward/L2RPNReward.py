import numpy as np
from abc import ABC, abstractmethod

from grid2op.Exceptions import Grid2OpException
from grid2op.Reward.Reward import Reward

class L2RPNReward(Reward):
    """
    This is the historical :class:`Reward` used for the Learning To Run a Power Network competition.

    See `L2RPN <https://l2rpn.chalearn.org/>`_ for more information.

    """
    def __init__(self):
        Reward.__init__(self)

    def initialize(self, env):
        self.reward_min = 0.
        self.reward_max = env.backend.n_line

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if not is_done and not has_error:
            line_cap = self.__get_lines_capacity_usage(env)
            res = np.sum(line_cap)
        else:
            # no more data to consider, no powerflow has been run, reward is what it is
            res = self.reward_min
        return res

    @staticmethod
    def __get_lines_capacity_usage(env):
        ampere_flows = np.abs(env.backend.get_line_flow())
        thermal_limits = np.abs(env.backend.get_thermal_limit())
        relative_flow = np.divide(ampere_flows, thermal_limits)

        x = np.minimum(relative_flow, 1)
        lines_capacity_usage_score = np.maximum(1 - x ** 2, 0.)
        return lines_capacity_usage_score
