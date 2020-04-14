# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    This class represents the base class of an BaseAgent. All bot / controller / agent used in the Grid2Op simulator
    should derived from this class.

    To work properly, it is advise to create BaseAgent after the :class:`grid2op.Environment` has been created and reuse
    the :attr:`grid2op.Environment.Environment.action_space` to build the BaseAgent.

    Attributes
    -----------
    action_space: :class:`grid2op.Action.ActionSpace`
        It represent the action space ie a tool that can serve to create valid action. Note that a valid action can
        be illegal or ambiguous, and so lead to a "game over" or to a error. But at least it will have a proper size.

    """
    def __init__(self, action_space):
        self.action_space = action_space

    def reset(self, obs):
        """
        This method is called at the beginning of a new episode.
        It is implemented by agents to reset their internal state if needed.

        Attributes
        -----------
        obs: :class:`grid2op.Observation.BaseObservation`
            The first observation corresponding to the initial state of the environment.
        """
        pass

    @abstractmethod
    def act(self, observation, reward, done=False):
        """
        This is the main method of an BaseAgent. Given the current observation and the current reward (ie the reward that
        the environment send to the agent after the previous action has been implemented).

        Parameters
        ----------
        observation: :class:`grid2op.Observation.BaseObservation`
            The current observation of the :class:`grid2op.Environment.Environment`

        reward: ``float``
            The current reward. This is the reward obtained by the previous action

        done: ``bool``
            Whether the episode has ended or not. Used to maintain gym compatibility

        Returns
        -------
        res: :class:`grid2op.Action.PlaybleAction`
            The action chosen by the bot / controler / agent.

        """
        pass
