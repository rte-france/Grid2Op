from abc import ABC, abstractmethod
import numpy as np

from grid2op.Exceptions import Grid2OpException
from grid2op.Rules.LegalAction import LegalAction

import pdb

class AlwaysLegal(LegalAction):
    """
    This subclass doesn't implement any rules regarding the legality of the actions. All actions are legal.

    """
    def __call__(self, action, env):
     """
     All actions being legal, this returns always true.
     See :func:`LegalAction.__call__` for a definition of the parameters of this function.

     """
     return True
