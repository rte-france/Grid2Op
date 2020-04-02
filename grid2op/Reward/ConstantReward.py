from grid2op.Reward.BaseReward import BaseReward


class ConstantReward(BaseReward):
    """
    Most basic implementation of reward: everything has the same values.

    Note that this :class:`BaseReward` subtype is not usefull at all, whether to train an :attr:`BaseAgent` nor to assess its
    performance of course.

    """
    def __init__(self):
        BaseReward.__init__(self)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        return 0
