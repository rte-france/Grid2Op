# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import os
import numpy as np

from grid2op.Exceptions import OpponentError


class OpponentSpace(object):
    """
    Is similar to the action space, but for the opponent.

    This class is used to express some "constraints" on the opponent attack. The opponent is free to attack whatever
    it wants, for how long it wants and when it wants. This class ensures that the opponent does not break any
    rules.

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

    def __init__(
        self,
        compute_budget,
        init_budget,
        opponent,
        attack_duration,  # maximum duration of an attack
        attack_cooldown,  # minimum duration between two consecutive attack
        budget_per_timestep=0.0,
        action_space=None,
    ):

        if action_space is not None:
            if not isinstance(action_space, compute_budget.action_space):
                raise OpponentError(
                    "BaseAction space provided to build the agent is not a subclass from the"
                    "action space to compute the cost of each action."
                )
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
        self.attack_max_duration = attack_duration
        self.attack_cooldown = attack_cooldown
        self.current_attack_duration = 0
        self.current_attack_cooldown = attack_cooldown
        self.last_attack = None

        if init_budget < 0.0:
            raise OpponentError(
                "An opponent should at least have a positive (or null) budget. If you "
                "want to deactivate the opponent set its budget to 0 and use the"
                'DontAct class as the "opponent_class"'
            )

        # TODO do i add it back
        # if not isinstance(opponent_reward_class, BaseReward):
        #    raise OpponentError("Impossible to build an opponent reward with a reward of type {}".format(opponent_reward_class))
        # self.opp_reward_helper = RewardHelper(opponent_reward_class)

    def init_opponent(self, partial_env, **kwargs):
        """
        Generic function used to initialize the opponent. For example, if an opponent reads from a file, the
        path where is the file is located should be pass with this method.
        """
        self.opponent.init(partial_env=partial_env, **kwargs)

    def reset(self):
        """
        Reset the state of the Opponent to its original state, in particular re assign the proper budget to it.
        """
        self.budget = self.init_budget
        self.previous_fails = False
        self.current_attack_duration = 0
        self.current_attack_cooldown = self.attack_cooldown
        self.last_attack = None
        self.opponent.reset(self.budget)
        self.previous_fails = False

    def _get_state(self):
        # used for simulate
        state_me = (
            self.budget,
            self.previous_fails,
            self.current_attack_duration,
            self.current_attack_cooldown,
            self.last_attack,
        )
        state_opp = self.opponent.get_state()
        return state_me, state_opp

    def _set_state(self, my_state, opp_state=None):
        # used for simulate (and for deep copy)
        if opp_state is not None:
            self.opponent.set_state(opp_state)
        (
            budget,
            previous_fails,
            current_attack_duration,
            current_attack_cooldown,
            last_attack,
        ) = my_state
        self.budget = budget
        self.previous_fails = previous_fails
        self.current_attack_duration = current_attack_duration
        self.current_attack_cooldown = current_attack_cooldown
        self.last_attack = last_attack

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

        **NB** it is expected that this function update the :attr:`OpponentSpace.last_attack`  attribute
        with ``None`` if the opponent choose not to attack, or with the attack of the opponent otherwise.

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
        res: :class:`grid2op.Action.Action` : The attack the opponent wants to perform
                                              (or "do nothing" if the attack was too costly)
              or class:`NoneType` : Returns None if no action is taken

        """

        if observation is None:
            # this is the first time step, which is not a "real" one
            # just here to load the data properly, so opponent do not attack there
            return None, 0

        # Update variables
        self.budget += self.budget_per_timestep
        self.current_attack_duration = max(0, self.current_attack_duration - 1)
        self.current_attack_cooldown = max(0, self.current_attack_cooldown - 1)
        attack_called = False

        if self.current_attack_duration > 0:
            # previous attack is not over
            attack = self.last_attack

        elif self.current_attack_cooldown > self.attack_cooldown:
            # minimum time between two consecutive attack not met
            attack = None

        # If the opponent can attack
        else:
            attack_called = True
            attack, duration = self.opponent.attack(
                observation, agent_action, env_action, self.budget, self.previous_fails
            )
            if duration is None:
                if np.isfinite(self.attack_max_duration):
                    duration = self.attack_max_duration
                else:
                    duration = 1

            self.previous_fails = False

            if duration > self.attack_max_duration:
                # duration chosen by the opponent would exceed the maximum duration allowed
                attack = None
                self.previous_fails = True

            # If the cost is too high
            final_budget = (
                self.budget
            )  # TODO add the: + self.budget_per_timestep * (self.attack_duration - 1)

            # i did not do it in case an attack is ok at the beginning, ok at the end, but at some point in the attack
            # process it is not (but i'm not sure this can happen, and don't have time to think about it right now)
            if duration * self.compute_budget(attack) > final_budget:
                attack = None
                self.previous_fails = True

            # If we can afford the attack
            if attack is not None:
                # even if it's "do nothing", it's sill an attack. Too bad if the opponent chose to do nothing.
                self.current_attack_duration = duration
                self.current_attack_cooldown += self.attack_cooldown

        if not attack_called:
            self.opponent.tell_attack_continues(
                observation, agent_action, env_action, self.budget
            )
            self.previous_fails = False

        self.budget -= self.compute_budget(attack)
        self.last_attack = attack

        attack_duration = self.current_attack_duration
        if attack is None:
            attack_duration = 0
        return attack, attack_duration

    def close(self):
        """if this has a reference to a backend, you need to close it for grid2op to work properly. Do not forget to do it."""
        pass
