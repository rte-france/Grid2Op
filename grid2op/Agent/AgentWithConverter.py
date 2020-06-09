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
from grid2op.Agent.BaseAgent import BaseAgent


class AgentWithConverter(BaseAgent):
    """
    Compared to a regular BaseAgent, these types of Agents are able to deal with a different representation of
    :class:`grid2op.Action.BaseAction` and :class:`grid2op.Observation.BaseObservation`.

    As any other Agents, AgentWithConverter will implement the :func:`BaseAgent.act` method. But for them, it's slightly
    different.

    They receive in this method an observation, as an object (ie an instance of
    :class:`grid2op.Observation.BaseObservation`). This
    object can then be converted to any other object with the method :func:`AgentWithConverter.convert_obs`.

    Then, this `transformed_observation` is pass to the method :func:`AgentWithConverter.my_act` that is supposed
    to be defined for each agents. This function outputs an `encoded_act` which can be whatever you want to be.

    Finally, the `encoded_act` is decoded into a proper action, object of class :class:`grid2op.Action.BaseAction`,
    thanks to the method :func:`AgentWithConverter.convert_act`.

    This allows, for example, to represent actions as integers to train more easily standard discrete control algorithm
    used to solve atari games for example.

    **NB** It is possible to define :func:`AgentWithConverter.convert_obs` and :func:`AgentWithConverter.convert_act`
     or to define a :class:`grid2op.Converters.Converter` and feed it to the `action_space_converter` parameters
     used to initialise the class. The second option is preferred, as the :attr:`AgentWithConverter.action_space`
     will then directly be this converter. Such an BaseAgent will really behave as if the actions are encoded the way he
     wants.

    Examples
    --------
    For example, imagine an BaseAgent uses a neural networks to take its decision.

    Suppose also that, after some
    features engineering, it's best for the neural network to use only the load active values
    (:attr:`grid2op.Observation.BaseObservation.load_p`) and the sum of the
    relative flows (:attr:`grid2op.Observation.BaseObservation.rho`) with the active flow
    (:attr:`grid2op.Observation.BaseObservation.p_or`) [**NB** that agent would not make sense a priori, but who knows]

    Suppose that this neural network can be accessed with a class `AwesomeNN` (not available...) that can
    predict some actions. It can be loaded with the "load" method and make predictions with the
    "predict" method.

    For the sake of the examples, we will suppose that this agent only predicts powerline status (so 0 or 1) that
    are represented as vector. So we need to take extra care to convert this vector from a numpy array to a valid
    action.

    This is done below:


    .. code-block:: python

        import grid2op
        import AwesomeNN # this does not exists!
        # create a simple environment
        env = grid2op.make()

        # define the class above
        class AgentCustomObservation(AgentWithConverter):
            def __init__(self, action_space, path):
                AgentWithConverter.__init__(self, action_space)
                self.my_neural_network = AwesomeNN()
                self.my_neural_networl.load(path)

            def convert_obs(self, observation):
                # convert the observation
                return np.concatenate((observation.load_p, observation.rho + observation.p_or))

            def convert_act(self, encoded_act):
                # convert back the action, output from the NN "self.my_neural_network"
                # to a valid action
                act = self.action_space({"set_status": encoded_act})

            def my_act(self, transformed_observation, reward, done=False):
                act_predicted = self.my_neural_network(transformed_observation)
                return act_predicted


        # make the agent that behaves as expected.
        my_agent = AgentCustomObservation(action_space=env.action_space, path=".")

        # this agent is perfectly working :-) You can use it as any other agents.


    Attributes
    ----------
    action_space_converter: :class:`grid2op.Converters.Converter`
        The converter that is used to represents the BaseAgent action space. Might be set to ``None`` if not initialized

    init_action_space: :class:`grid2op.Action.ActionSpace`
        The initial action space. This corresponds to the action space of the :class:`grid2op.Environment.Environment`.

    action_space: :class:`grid2op.Converters.ActionSpace`
        If a converter is used, then this action space represents is this converter. The agent will behave as if
        the action space is directly encoded the way it wants.

    """
    def __init__(self, action_space, action_space_converter=None, **kwargs_converter):
        self.action_space_converter = action_space_converter
        self.init_action_space = action_space

        if action_space_converter is None:
            BaseAgent.__init__(self, action_space)
        else:
            if isinstance(action_space_converter, type):
                if issubclass(action_space_converter, Converter):
                    action_space_converter_this_env_class = action_space_converter.init_grid(action_space)
                    this_action_space = action_space_converter_this_env_class(action_space)
                    BaseAgent.__init__(self, this_action_space)
                else:
                    raise Grid2OpException("Impossible to make an BaseAgent with a converter of type {}. "
                                           "Please use a converter deriving from grid2op.ActionSpaceConverter.Converter."
                                           "".format(action_space_converter))
            elif isinstance(action_space_converter, Converter):
                if isinstance(action_space_converter._template_act, self.init_action_space.actionClass):
                    BaseAgent.__init__(self, action_space_converter)
                else:
                    raise Grid2OpException("Impossible to make an BaseAgent with the provided converter of type {}. "
                                           "It doesn't use the same type of action as the BaseAgent's action space."
                                           "".format(action_space_converter))
            else:
                raise Grid2OpException("You try to initialize and BaseAgent with an invalid converter \"{}\". It must"
                                       "either be a type deriving from \"Converter\", or an instance of a class"
                                       "deriving from it."
                                       "".format(action_space_converter))

            self.action_space.init_converter(**kwargs_converter)

    def convert_obs(self, observation):
        """
        This function convert the observation, that is an object of class :class:`grid2op.Observation.BaseObservation`
        into a representation understandable by the BaseAgent.

        For example, and agent could only want to look at the relative flows
        :attr:`grid2op.Observation.BaseObservation.rho`
        to take his decision. This is possible by  overloading this method.

        This method can also be used to scale the observation such that each compononents has mean 0 and variance 1
        for example.

        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            Initial observation received by the agent in the :func:`BaseAgent.act` method.

        Returns
        -------
        res: ``object``
            Anything that will be used by the BaseAgent to take decisions.

        """
        return self.action_space.convert_obs(observation)

    def convert_act(self, encoded_act):
        """
        This function will convert an "ecnoded action" that be of any types, to a valid action that can be ingested
        by the environment.

        Parameters
        ----------
        encoded_act: ``object``
            Anything that represents an action.

        Returns
        -------
        act: :grid2op.BaseAction.BaseAction`
            A valid actions, represented as a class, that corresponds to the encoded action given as input.

        """
        return self.action_space.convert_act(encoded_act)

    def act(self, observation, reward, done=False):
        """
        Standard method of an :class:`BaseAgent`. There is no need to overload this function.

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
            The action chosen by the bot / controler / agent.

        """
        transformed_observation = self.convert_obs(observation)
        encoded_act = self.my_act(transformed_observation, reward, done)
        return self.convert_act(encoded_act)

    def seed(self, seed):
        """
        Seed the agent AND the associated converter if it needs to be seeded.

        See a more detailed explanation in :func:`BaseAgent.seed` for more information about seeding.
        """
        super().seed(seed)
        if self.action_space_converter is not None:
            self.action_space.seed(seed)

    @abstractmethod
    def my_act(self, transformed_observation, reward, done=False):
        """
        This method should be overide if this class is used. It is an "abstract" method.

        If someone wants to make a agent that handles different kinds of actions an observation.

        Parameters
        ----------
        transformed_observation: ``object``
            Anything that will be used to create an action. This is the results to the call of
            :func:`AgentWithConverter.convert_obs`. This is likely a numpy array.

        reward: ``float``
            The current reward. This is the reward obtained by the previous action

        done: ``bool``
            Whether the episode has ended or not. Used to maintain gym compatibility

        Returns
        -------
        res: ``object``
            A representation of an action in any possible format. This action will then be ingested and formatted into
            a valid action with the :func:`AgentWithConverter.convert_act` method.

        """
        transformed_action = None
        return transformed_action
