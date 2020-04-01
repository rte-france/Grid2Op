
class Opponent(object):
    """

    Attributes
    ----------
    action_space: :class:`grid2op.`
    """
    def __init__(self, action_space, budget, compute_budget):
        self.action_space = action_space
        self.budget = budget
        self.compute_budget = compute_budget