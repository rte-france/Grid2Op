# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import warnings
import numpy as np

from grid2op.Opponent import BaseOpponent
from grid2op.Exceptions import OpponentError


class NeuripsOpponent(BaseOpponent):
    def __init__(self, action_space):
        BaseOpponent.__init__(self, action_space)
        self._do_nothing = None
        self._attacks = None
        self._lines_ids = None
        self._next_attack_time = None
        self._attack_frequency = None

        # this is the constructor:
        # it should have the exact same signature as here

    def init(self, lines_attacked=[], **kwargs):
        # this if the function used to properly set the object.
        # It has the generic signature above,
        # and it's way more flexible that the other one.

        if len(lines_attacked) == 0:
            warnings.warn(f'The opponent is deactivated as there is no information as to which line to attack. '
                          f'You can set the argument "kwargs_opponent" to the list of the line names you want '
                          f' the opponent to attack in the "make" function.')

        # Store attackable lines IDs
        self._lines_ids = []
        for l_name in lines_attacked:
            l_id = np.where(self.action_space.name_line == l_name)
            if len(l_id) and len(l_id[0]):
                self._lines_ids.append(l_id[0][0])
            else:
                raise OpponentError("Unable to find the powerline named \"{}\" on the grid. For "
                                    "information, powerlines on the grid are : {}"
                                    "".format(l_name, sorted(self.action_space.name_line)))

        # Pre-build attacks actions
        self._do_nothing = self.action_space({})
        self._attacks = []
        for l_id in self._lines_ids:
            a = self.action_space({
                'set_line_status': [(l_id, -1)]
            })
            self._attacks.append(a)
        self._attacks = np.array(self._attacks)

        # Opponent's attack frequency
        self._attack_frequency = kwargs["attack_frequency"] if "attack_frequency" in kwargs else 12*24

    def tell_attack_continues(self, observation, agent_action, env_action, budget):
        self._next_attack_time = None

    def attack(self, observation, agent_action, env_action,
               budget, previous_fails):
        """
        This method is the equivalent of "attack" for a regular agent.

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

        # During creation of the environment, do not attack
        if observation is None:
            return None

        # Decide the time of the next attack
        if self._next_attack_time is None:
            self._next_attack_time = self.space_prng.randint(self._attack_frequency)

        # If the attack time has not come yet, do not attack
        if self._next_attack_time > 0:
            return None

        # If all attackable lines are disconnected, do not attack
        status = observation.line_status[self._lines_ids]
        if np.all(status == False):
            return None

        available_attacks = self._attacks[status]
        rho = observation.rho[self._lines_ids][status]
        attack = self.space_prng.choice(available_attacks, p=rho / rho.sum())
        self._next_attack_time -= 1

        return attack
