from abc import ABC, abstractmethod
import numpy as np
import itertools
import pdb

from grid2op.Exceptions import Grid2OpException
from grid2op.Agent.Agent import Agent


class OneChangeThenNothing(Agent):
    """
    This is a specific kind of Agent. It does an BaseAction (possibly non empty) at the first time step and then does
    nothing.

    This class is an abstract class and cannot be instanciated (ie no object of this class can be created). It must
    be overridden and the method :func:`OneChangeThenNothing._get_dict_act` be defined. Basically, it must know
    what action to do.

    """
    def __init__(self, action_space, action_space_converter=None):
        Agent.__init__(self, action_space)
        self.has_changed = False

    def act(self, observation, reward, done=False):
        if self.has_changed:
            res = self.action_space({})
            self.has_changed = True
        else:
            res = self.action_space(self._get_dict_act())
        return res

    @abstractmethod
    def _get_dict_act(self):
        """
        Function that need to be overridden to indicate which action to perfom.

        Returns
        -------
        res: ``dict``
            A dictionnary that can be converted into a valid :class:`grid2op.BaseAction.BaseAction`. See the help of
            :func:`grid2op.BaseAction.ActionSpace.__call__` for more information.
        """
        pass
