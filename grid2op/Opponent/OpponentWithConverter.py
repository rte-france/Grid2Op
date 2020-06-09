# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from abc import abstractmethod

from grid2op.Converter import Converter
from grid2op.Exceptions import Grid2OpException
from grid2op.Opponent.BaseOpponent import BaseOpponent


class OpponentWithConverter(BaseOpponent):
    """
    Compared to a regular BaseOpponent, these types of Ogents are able to deal with a different representation of
    :class:`grid2op.Action.BaseAction` and :class:`grid2op.Observation.BaseObservation`.

    As any other Opponents, OpponentWithConverter will implement the :func:`BaseOpponent.attack` method. But for them, it's slightly
    different.

    They receive in this method an observation, as an object (ie an instance of
    :class:`grid2op.Observation.BaseObservation`). This
    object can then be converted to any other object with the method :func:`OpponentWithConverter.convert_obs`.

    Then, this `transformed_observation` is pass to the method :func:`OpponentWithConverter.my_attack` that is supposed
    to be defined for each opponent. This function outputs an `encoded_attack` which can be whatever you want to be.

    Finally, the `encoded_attack` is decoded into a proper action, object of class :class:`grid2op.Action.BaseAction`,
    thanks to the method :func:`OpponentWithConverter.convert_act`.

    This allows, for example, to represent actions as integers to train more easily standard discrete control algorithm
    used to solve atari games for example.

    **NB** It is possible to define :func:`OpponentWithConverter.convert_obs` and :func:`OpponentWithConverter.convert_act`
     or to define a :class:`grid2op.Converters.Converter` and feed it to the `action_space_converter` parameters
     used to initialise the class. The second option is preferred, as the :attr:'OpponentWithConverter.action_space`
     will then directly be this converter. Such an BaseOpponent will really behave as if the actions are encoded the way he
     wants.

    Attributes
    ----------
    action_space_converter: :class:`grid2op.Converters.Converter`
        The converter that is used to represents the BaseOpponent action space. Might be set to ``None`` if not initialized

    init_action_space: :class:`grid2op.Action.ActionSpace`
        The initial action space. This corresponds to the action space of the :class:`grid2op.Environment.Environment`.

    action_space: :class:`grid2op.Converters.ActionSpace`
        If a converter is used, then this action space represents is this converter. The opponent will behave as if
        the action space is directly encoded the way it wants.

    """
    def __init__(self, action_space, action_space_converter=None, **kwargs_converter):
        self.action_space_converter = action_space_converter
        self.init_action_space = action_space

        if action_space_converter is None:
            BaseOpponent.__init__(self, action_space)
        else:
            if isinstance(action_space_converter, type):
                if issubclass(action_space_converter, Converter):
                    action_space_converter_this_env_class = action_space_converter.init_grid(action_space)
                    this_action_space = action_space_converter_this_env_class(action_space)
                    BaseOpponent.__init__(self, this_action_space)
                else:
                    raise Grid2OpException("Impossible to make an BaseOpponent with a converter of type {}. "
                                           "Please use a converter deriving from grid2op.ActionSpaceConverter.Converter."
                                           "".format(action_space_converter))
            elif isinstance(action_space_converter, Converter):
                if isinstance(action_space_converter._template_act, self.init_action_space.actionClass):
                    BaseOpponent.__init__(self, action_space_converter)
                else:
                    raise Grid2OpException("Impossible to make an BaseOpponent with the provided converter of type {}. "
                                           "It doesn't use the same type of action as the BaseOpponent's action space."
                                           "".format(action_space_converter))
            else:
                raise Grid2OpException("You try to initialize and BaseOpponent with an invalid converter \"{}\". It must"
                                       "either be a type deriving from \"Converter\", or an instance of a class"
                                       "deriving from it."
                                       "".format(action_space_converter))

            self.action_space.init_converter(**kwargs_converter)

    def convert_obs(self, observation):
        """
        This function convert the observation, that is an object of class :class:`grid2op.Observation.BaseObservation`
        into a representation understandable by the BaseOpponent.

        For example, an opponent could only want to look at the relative flows
        :attr:`grid2op.Observation.BaseObservation.rho`
        to take his decision. This is possible by  overloading this method.

        This method can also be used to scale the observation such that each compononents has mean 0 and variance 1
        for example.

        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            Initial observation received by the opponent in the :func:`BaseOpponent.attack` method.

        Returns
        -------
        res: ``object``
            Anything that will be used by the BaseOpponent to take decisions.

        """
        return self.action_space.convert_obs(observation)

    def convert_act(self, encoded_attack):
        """
        This function will convert an "ecnoded action" that be of any types, to a valid action that can be ingested
        by the environment.

        Parameters
        ----------
        encoded_attack: ``object``
            Anything that represents an action.

        Returns
        -------
        attack: :grid2op.BaseAction.BaseAction`
            A valid actions, represented as a class, that corresponds to the encoded action given as input.

        """
        return self.action_space.convert_act(encoded_attack)

    def attack(self, observation, *args):
        """
        Standard method of an :class:`BaseOpponent`. There is no need to overload this function.

        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            The current observation of the :class:`grid2op.Environment.Environment`

        reward: ``float``
            The current reward. This is the reward obtained by the previous action

        done: ``bool``
            Whether the episode has ended or not. Used to maintain gym compatibility

        Returns
        -------
        res: :class:`grid2op.Action.Action`
            The action chosen by the bot / controler / opponent.

        """
        transformed_observation = self.convert_obs(observation)
        # Todo : convert the other arguments?
        encoded_attack = self.my_attack(transformed_observation, *args)
        return self.convert_act(encoded_attack)

    @abstractmethod
    def my_attack(self, transformed_observation, agent_action, env_action, budget, previous_fails):
        """
        This method should be overide if this class is used. It is an "abstract" method.

        If someone wants to make a opponent that handles different kinds of actions an observation.

        Parameters
        ----------
        transformed_observation: ``object``
            Anything that will be used to create an action. This is the results to the call of
            :func:`OpponentWithConverter.convert_obs`. This is likely a numpy array.

        reward: ``float``
            The current reward. This is the reward obtained by the previous action

        done: ``bool``
            Whether the episode has ended or not. Used to maintain gym compatibility

        Returns
        -------
        res: ``object``
            A representation of an action in any possible format. This action will then be ingested and formatted into
            a valid action with the :func:`OpponentWithConverter.convert_act` method.

        """
        transformed_action = None
        return transformed_action
