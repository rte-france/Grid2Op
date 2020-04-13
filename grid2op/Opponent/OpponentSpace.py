# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
from grid2op.Exceptions import OpponentError
from grid2op.Reward import RewardHelper, BaseReward


class OpponentSpace(object):
    """

    Attributes
    ----------
    action_space: :class:`grid2op.Action.ActionSpace`
        The action space defining which action the Opponent are allowed to take

    init_budget: ``float``
        The initial budget of the opponent

    compute_budget: :class:`grid2op.Opponent.ActionBudget`
        The tool used to compute the budget

    opponent: :class:`grid2op.Opponent.BaseOpponent`
        The agent that will take malicious actions.

    previous_fails: ``bool``
        Whether the last attack of the opponent failed or not

    budget_per_timestep: ``float``
        The increase of the opponent budget per time step (if any)
    """
    def __init__(self, compute_budget, init_budget, opponent, budget_per_timestep=0., action_space=None):
        if action_space is not None:
            if not isinstance(action_space, compute_budget.action_space):
                raise OpponentError("BaseAction space provided to build the agent is not a subclass from the"
                                    "action space to compute the cost of each action.")
            self.action_space = action_space
        else:
            self.action_space = compute_budget.action_space
        self.init_budget = init_budget
        self.budget = init_budget
        self.compute_budget = compute_budget
        self.opponent = opponent
        self._do_nothing = self.action_space()
        self.previous_fails = False
        self.budget_per_timestep = budget_per_timestep

        if budget_per_timestep != 0.:
            raise NotImplementedError("You should implement that in Environment!")
        if init_budget < 0.:
            raise OpponentError("An opponent should at least have a positive (or null) budget. If you "
                                "want to deactivate the opponent set its budget to 0 and use the"
                                "DontAct class as the \"opponent_class\"")

        # TODO do i add it back
        # if not isinstance(opponent_reward_class, BaseReward):
        #    raise OpponentError("Impossible to build an opponent reward with a reward of type {}".format(opponent_reward_class))
        # self.opp_reward_helper = RewardHelper(opponent_reward_class)

    def init(self, *args, **kwargs):
        """
        Generic function used to initialize the opponent. For example, if an opponent reads from a file, the
        path where is the file is located should be pass with this method.
        """
        self.opponent.init(*args, **kwargs)

    def reset(self):
        """
        Reset the state of the Opponent to its original state, in particular re assign the proper budget to it.
        """
        self.budget = self.init_budget
        self.previous_fails = False
        self.opponent.reset(self.budget)

    def has_failed(self):
        """
        This signal is sent by the environment and indicated the opponent attack could not be implmented on the
        powergrid, most likely due to the attack to be ambiguous.
        """
        self.previous_fails = True

    def attack(self, observation, agent_action, env_action):
        """
        This function calls the attack from the opponent.

        It check whether the budget is consistent with the attack (budget should be more that the cosst
        associated with the attack). If the attack cost too much, then it is replaced by a "do nothing"
        action. Otherwise, the attack will be implemented by the environment.

        Note that if the attack is "ambiguous" it will fails (the environment will replace it by a
        "do nothing" action), but the budget will still be consumed.

        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            The last observation (at time t)

        agent_action: :class:`grid2op.Action.Action`
            The action that the agent took

        env_action: :class:`grid2op.Action.Action`
            The modification that the environment will take.

        Returns
        -------
        res: :class:`grid2op.Action.Action`
            The attack the opponent wants to perform (or "do nothing" if the attack was too costly)

        """
        self.budget += self.budget_per_timestep
        attack = self.opponent.attack(observation, agent_action, env_action, self.budget, self.previous_fails)
        tmp = self.compute_budget(attack)
        if tmp <= self.budget:
            self.previous_fails = False
            self.budget -= tmp
            res = attack
        else:
            self.previous_fails = True
            res = self._do_nothing
        return res
