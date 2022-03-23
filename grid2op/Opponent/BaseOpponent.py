# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Space import RandomObject


class BaseOpponent(RandomObject):
    def __init__(self, action_space):
        RandomObject.__init__(self)
        self.action_space = action_space
        self._do_nothing = self.action_space()

    def init(self, partial_env, **kwargs):
        """
        Generic function used to initialize the derived classes. For example, if an opponent reads from a file, the
        path where is the file is located should be pass with this method.
        """
        pass

    def reset(self, initial_budget):
        """
        This function is called at the end of an episode, when the episode is over. It aims at resetting the
        self and prepare it for a new episode.

        Parameters
        ----------
        initial_budget: ``float``
            The initial budget the opponent has
        """
        pass

    def attack(self, observation, agent_action, env_action, budget, previous_fails):
        """
        This method is the equivalent of "act" for a regular agent.

        Opponent, in this framework can have more information than a regular agent (in particular it can
        view time step t+1), it has access to its current budget etc.

        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            The last observation (at time t)

        opp_reward: ``float``
            THe opponent "reward" (equivalent to the agent reward, but for the opponent) TODO do i add it back ???

        done: ``bool``
            Whether the game ended or not TODO do i add it back ???

        agent_action: :class:`grid2op.Action.Action`
            The action that the agent took

        env_action: :class:`grid2op.Action.Action`
            The modification that the environment will take.

        budget: ``float``
            The current remaining budget (if an action is above this budget, it will be replaced by a do nothing.

        previous_fails: ``bool``
            Wheter the previous attack failed (due to budget or ambiguous action)

        Returns
        -------
        attack: :class:`grid2op.Action.Action`
            The attack performed by the opponent. In this case, a do nothing, all the time.

        duration: ``int``
            The duration of the attack

        """
        # TODO maybe have a class "GymOpponent" where the observation would include the budget  and all other
        # TODO information, and forward something to the "act" method.
        return None, None

    def tell_attack_continues(self, observation, agent_action, env_action, budget):
        """
        The purpose of this method is to tell the agent that his attack is being continued
        and to indicate the current state of the grid.

        At every time step, either "attack" or "tell_acttack_continues" is called exactly once.

        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            The last observation (at time t)

        agent_action: :class:`grid2op.Action.Action`
            The action that the agent took

        env_action: :class:`grid2op.Action.Action`
            The modification that the environment will take.

        budget: ``float``
            The current remaining budget (if an action is above this budget, it will be replaced by a do nothing.
        """
        pass

    def get_state(self):
        """
        This function should return the internal state of the Opponent.

        This means that after a call to `opponent.set_state(opponent.get_state())` the opponent should do the exact
        same things than without these calls.

        Returns
        -------

        """
        return None

    def set_state(self, my_state):
        """
        This function is used to set the internal state of the Opponent.

        Parameters
        ----------
        my_state

        """
        pass

    def _custom_deepcopy_for_copy(self, new_obj, dict_=None):
        super()._custom_deepcopy_for_copy(new_obj)
        if dict_ is None:
            dict_ = {}
        new_obj.action_space = self.action_space  # const
        new_obj._do_nothing = new_obj.action_space()
        new_obj.set_state(self.get_state())
