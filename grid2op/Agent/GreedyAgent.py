from abc import ABC, abstractmethod
import numpy as np
import itertools
import pdb

from grid2op.Exceptions import Grid2OpException
from grid2op.Agent.Agent import Agent

class GreedyAgent(Agent):
    """
    This is a class of "Greedy Agent". Greedy agents are all executing the same kind of algorithm to take action:

      1. They :func:`grid2op.Observation.simulate` all actions in a given set
      2. They take the action that maximise the simulated reward among all these actions

    To make the creation of such Agent, we created this abstract class (object of this class cannot be created). Two
    examples of such greedy agents are provided with :class:`PowerLineSwitch` and :class:`TopologyGreedy`.
    """
    def __init__(self, action_space, action_space_converter=None):
        Agent.__init__(self, action_space)
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
            A list of all candidate :class:`grid2op.Action.Action`
        """
        pass
