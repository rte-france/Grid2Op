from abc import ABC, abstractmethod
import numpy as np
import itertools
import pdb

from grid2op.Converter import Converter, IdToAct, ToVect
from grid2op.Exceptions import Grid2OpException
from grid2op.Agent.AgentWithConverter import AgentWithConverter

class MLAgent(AgentWithConverter):
    """
    This agent allows to handle only vectors. The "my_act" function will return "do nothing" action (so it needs
    to be override)

    In this class, the "my_act" is expected to return a vector that can be directly converted into a valid action.
    """
    def __init__(self, action_space, action_space_converter=ToVect, **kwargs_converter):
        AgentWithConverter.__init__(self, action_space, action_space_converter, **kwargs_converter)
        self.do_nothing_vect = action_space({}).to_vect()

    def my_act(self, transformed_observation, reward, done=False):
        return self.do_nothing_vect

    def convert_from_vect(self, act):
        """
        Helper to convert an action, represented as a numpy array as an :class:`grid2op.Action` instance.

        Parameters
        ----------
        act: ``numppy.ndarray``
            An action cast as an :class:`grid2op.Action.Action` instance.

        Returns
        -------
        res: :class:`grid2op.Action.Action`
            The `act` parameters converted into a proper :class:`grid2op.Action.Action` object.
        """
        res = self.action_space({})
        res.from_vect(act)
        return res
