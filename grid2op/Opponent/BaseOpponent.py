# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.


class BaseOpponent(object):
    def __init__(self, action_space):
        self.action_space = action_space
        self._do_nothing = self.action_space()

    def init(self, *args, **kwargs):
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
        """
        # TODO maybe have a class "GymOpponent" where the observation would include the budget  and all other
        # TODO information, and forward something to the "act" method.
        attack = self._do_nothing
        return attack
