from abc import ABC, abstractmethod
import numpy as np

from grid2op.Exceptions import Grid2OpException
from grid2op.Rules.LookParam import LookParam
from grid2op.Rules.PreventReconection import PreventReconection

import pdb

class DefaultRules(LookParam, PreventReconection):
    """
    This subclass combine both :class:`LookParam` and :class:`PreventReconection`.
    An action is declared legal if and only if:

      - It doesn't diconnect / reconnect more power lines than  what stated in the actual game _parameters
        :class:`grid2op.Parameters`
      - It doesn't attempt to act on more substations that what is stated in the actual game _parameters
        :class:`grid2op.Parameters`
      - It doesn't attempt to reconnect a powerline out of service.

    """
    def __call__(self, action, env):
        """
        See :func:`LegalAction.__call__` for a definition of the _parameters of this function.
        """
        if not LookParam.__call__(self, action, env):
            return False
        return PreventReconection.__call__(self, action, env)
