from grid2op.Rules.BaseRules import BaseRules

import pdb


class AlwaysLegal(BaseRules):
    """
    This subclass doesn't implement any rules regarding the legality of the actions. All actions are legal.

    """
    def __call__(self, action, env):
        """
        All actions being legal, this returns always true.
        See :func:`BaseRules.__call__` for a definition of the parameters of this function.

        """
        return True
