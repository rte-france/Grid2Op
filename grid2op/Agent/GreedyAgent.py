# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from abc import abstractmethod
import numpy as np
from grid2op.Agent.BaseAgent import BaseAgent


class GreedyAgent(BaseAgent):
    """
    This is a class of "Greedy BaseAgent". Greedy agents are all executing the same kind of algorithm to take action:

      1. They :func:`grid2op.Observation.Observation.simulate` all actions in a given set
      2. They take the action that maximise the simulated reward among all these actions

    This class is an abstract class (object of this class cannot be created). To create "GreedyAgent" one must
    override this class. Examples are provided with :class:`PowerLineSwitch` and :class:`TopologyGreedy`.
    """
    def __init__(self, action_space, action_space_converter=None):
        BaseAgent.__init__(self, action_space)
        self.tested_action = None

    def act(self, observation, reward, done=False):
        """
        By definition, all "greedy" agents are acting the same way. The only thing that can differentiate multiple
        agents is the actions that are tested.

        These actions are defined in the method :func:`._get_tested_action`. This :func:`.act` method implements the
        greedy logic: take the actions that maximizes the instantaneous reward on the simulated action.

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
        self.tested_action = self._get_tested_action(observation)
        if len(self.tested_action) > 1:
            all_rewards = np.full(shape=len(self.tested_action), fill_value=np.NaN, dtype=np.float)
            for i, action in enumerate(self.tested_action):
                simul_obs, simul_reward, simul_has_error, simul_info = observation.simulate(action)
                all_rewards[i] = simul_reward

            reward_idx = np.argmax(all_rewards)  # rewards.index(max(rewards))
            best_action = self.tested_action[reward_idx]
            # print("reward_idx: {}".format(reward_idx))
        else:
            best_action = self.tested_action[0]
        return best_action

    @abstractmethod
    def _get_tested_action(self, observation):
        """
        Returns the list of all the candidate actions.

        From this list, the one that achieve the best "simulated reward" is used.


        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            The current observation of the :class:`grid2op.Environment.Environment`

        Returns
        -------
        res: ``list``
            A list of all candidate :class:`grid2op.BaseAction.BaseAction`
        """
        pass
