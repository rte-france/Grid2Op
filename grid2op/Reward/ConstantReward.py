import numpy as np
from abc import ABC, abstractmethod

from grid2op.Exceptions import Grid2OpException
from grid2op.Reward.Reward import Reward

class ConstantReward(Reward):
    """
    Most basic implementation of reward: everything has the same values.

    Note that this :class:`Reward` subtype is not usefull at all, whether to train an :attr:`Agent` nor to assess its
    performance of course.

    """
    def __init__(self):
        Reward.__init__(self)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        return 0
