from grid2op.Opponent.BaseActionBudget import BaseActionBudget


class UnlimitedBudget(BaseActionBudget):
    """
    This class define an unlimited budget for the opponent.

    It SHOULD NOT be used if the opponent is allow to take any actions!
    """
    def __init__(self, action_space):
        BaseActionBudget.__init__(self, action_space)

    def __call__(self, attack):
        return 0
