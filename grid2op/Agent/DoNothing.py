# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Agent.BaseAgent import BaseAgent


class DoNothingAgent(BaseAgent):
    """
    This is the most basic BaseAgent. It is purely passive, and does absolutely nothing.

    As opposed to most reinforcement learning environments, in grid2op, doing nothing is often
    the best solution.

    """
    def __init__(self, action_space):
        BaseAgent.__init__(self, action_space)

    def act(self, observation, reward, done=False):
        """
        As better explained in the document of :func:`grid2op.BaseAction.update` or
        :func:`grid2op.BaseAction.ActionSpace.__call__`.

        The preferred way to make an object of type action is to call :func:`grid2op.BaseAction.ActionSpace.__call__`
        with the dictionary representing the action. In this case, the action is "do nothing" and it is represented by
        the empty dictionary.

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
            The action chosen by the bot / controller / agent.

        """
        res = self.action_space({})
        return res
