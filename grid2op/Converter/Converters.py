"""
In this module of grid2op, the "converters" are defined.

A converter is a specific class of :class:`grid2op.Action.ActionSpace` (ie of Action Space) that allows the agent to
manipulate this action to have a different representation of it.

For example, suppose we are dealing with TopologyAction (only manipulating the graph of the powergrid). This is a
discrete "action space". Often, it's custom to deal with such action space by enumerating all actions, and then assign
to all valid actions a unique ID.

This can be done easily with the :class:`IdToAct` class.

More concretely, the diagram of an agent is:

i) receive an observation (in a form of an object of class :class:`grid2op.Observation.Observation`)
ii) implement the :func:`grid2op.Agent.Agent.act` taking as input an :class:`grid2op.Observation.Observation` and
    returning an :class:`grid2op.Action.Action`
iii) this :class:`grid2op.Action.Action` is then digested by the environment

Introducing some converters lead to the following:

i) receive an observation (:class:`grid2op.Observation.Observation`)
ii) the transformer automatically (using :func:`Converter.convert_obs`) to a `transformed observation`
iii) implement the function :func:`grid2op.Agent.AgentWithConverter.my_act` that takes as input
     a `transformed observation` and returns an `encoded action`
iv) the transformer automatically transforms back the `encoded action` into a proper :class:`grid2op.Action.Action`
v) this :class:`grid2op.Action.Action` is then digested by the environment

This simple mechanism allows people to focus on iii) above (typically implemented with artificial neural networks)
without having to worry each time about the complex representations of actions and observations.

More details and a concrete example is given in the documentation of the class
:class:`grid2op.Agent.AgentWithConverter`.

Some examples of converters are given in :class:`IdToAct` and :class:`ToVect`.
"""

import numpy as np
import itertools

from grid2op.Action import ActionSpace
from grid2op.Exceptions import Grid2OpException

import pdb

# TODO more exhaustive documentation and tests.


class Converter(ActionSpace):
    """
    This Base class should be use to implement any converter. If for some reasons
    """
    def __init__(self, action_space):
        ActionSpace.__init__(self, action_space, action_space.legal_action, action_space.subtype)
        self.space_prng = action_space.space_prng
        self.seed_used = action_space.seed_used

    def init_converter(self, **kwargs):
        pass

    def convert_obs(self, obs):
        """
        This function is used to convert an observation into something that is easier to manipulate.

        Parameters
        ----------
        obs: :class:`grid2op.Observation.Observation`
            The input observation.

        Returns
        -------

        transformed_obs: ``object``
            An different representation of the input observation, typically represented as a 1d vector that can be
            processed by a neural networks.

        """
        transformed_obs = obs
        return transformed_obs

    def convert_act(self, encoded_act):
        """
        This function will transform the action, encoded somehow (for example identified by an id, represented by
        an integer) to a valid actions that can be processed by the environment.

        Parameters
        ----------
        encoded_act: ``object``
            Representation of an action, as a vector or an integer etc.

        Returns
        -------
        regular_act: :class:`grid2op.Action.Action`
            The action corresponding to the `encoded_action` above converted into a format that can be processed
            by the environment.

        """
        regular_act = encoded_act
        return regular_act
