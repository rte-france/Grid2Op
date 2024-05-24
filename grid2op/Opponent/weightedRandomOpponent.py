# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import warnings
import numpy as np
import copy

from grid2op.Opponent.baseOpponent  import BaseOpponent
from grid2op.Exceptions import OpponentError


class WeightedRandomOpponent(BaseOpponent):
    """
    This opponent will disconnect lines randomly among the attackable lines `lines_attacked`.
    The sampling is weighted by the lines current usage rate divided by some factor `rho_normalization`
    (see init for more details).

    When an attack becomes possible, the time of the attack will be sampled uniformly
    in the next `attack_period` steps (see init).
    """

    def __init__(self, action_space):
        BaseOpponent.__init__(self, action_space)
        self._do_nothing = None
        self._attacks = None
        self._lines_ids = None
        self._next_attack_time = None
        self._attack_period = None
        self._rho_normalization = None

        # this is the constructor:
        # it should have the exact same signature as here

    def init(
        self,
        partial_env,
        lines_attacked=[],
        rho_normalization=[],
        attack_period=12 * 24,
        **kwargs,
    ):
        """
        Generic function used to initialize the derived classes. For example, if an opponent reads from a file, the
        path where is the file is located should be pass with this method.

        Parameters
        ----------
        lines_attacked: ``list``
            The list of lines that the WeightedRandomOpponent should be able to disconnect

        rho_normalization: ``list``
            The list of mean usage rates for the attackable lines. Should have
            the same length as lines_attacked. If no value is given, no normalization will be performed.
            The weights for sampling the attacked line are rho / rho_normalization.

        attack_period: ``int``
            The number of steps among which the attack may happen.
            If attack_period=10, then whenever an attack can be made, it will happen in the 10
            next steps.
        """

        if len(lines_attacked) == 0:
            warnings.warn(
                f"The opponent is deactivated as there is no information as to which line to attack. "
                f'You can set the argument "kwargs_opponent" to the list of the line names you want '
                f' the opponent to attack in the "make" function.'
            )

        # Store attackable lines IDs
        self._lines_ids = []
        for l_name in lines_attacked:
            l_id = (self.action_space.name_line == l_name).nonzero()
            if len(l_id) and len(l_id[0]):
                self._lines_ids.append(l_id[0][0])
            else:
                raise OpponentError(
                    'Unable to find the powerline named "{}" on the grid. For '
                    "information, powerlines on the grid are : {}"
                    "".format(l_name, sorted(self.action_space.name_line))
                )

        # Pre-build attacks actions
        self._do_nothing = self.action_space({})
        self._attacks = []
        for l_id in self._lines_ids:
            a = self.action_space({"set_line_status": [(l_id, -1)]})
            self._attacks.append(a)
        self._attacks = np.array(self._attacks)

        # Usage rates normalization
        self._rho_normalization = np.ones_like(lines_attacked)
        if len(rho_normalization) == 0:
            warnings.warn(
                "The usage rate normalization is not specified. No normalization will be performed."
            )
        elif len(rho_normalization) != len(lines_attacked):
            raise Warning(
                f"The usage rate normalization must have the same length as the number "
                f"of attacked lines. No normalization will be performed."
            )
        else:
            self._rho_normalization = np.array(rho_normalization)

        # Opponent's attack period
        self._attack_period = attack_period
        if self._attack_period <= 0:
            raise OpponentError("Opponent attack cooldown need to be > 0")

    def reset(self, initial_budget):
        self._next_attack_time = None

    def tell_attack_continues(self, observation, agent_action, env_action, budget):
        self._next_attack_time = None

    def attack(self, observation, agent_action, env_action, budget, previous_fails):
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

        duration: ``int``
            The duration of the attack
        """
        # TODO maybe have a class "GymOpponent" where the observation would include the budget  and all other
        # TODO information, and forward something to the "act" method.

        # During creation of the environment, do not attack
        if observation is None:
            return None, 0

        # Decide the time of the next attack
        if self._next_attack_time is None:
            self._next_attack_time = 1 + self.space_prng.randint(self._attack_period)
        self._next_attack_time -= 1

        # If the attack time has not come yet, do not attack
        if self._next_attack_time > 0:
            return None, 0

        # If all attackable lines are disconnected, do not attack
        status = observation.line_status[self._lines_ids]
        if not status.sum():
            return None, 0

        available_attacks = self._attacks[status]
        rho = observation.rho[self._lines_ids][status] / self._rho_normalization[status]
        rho_sum = rho.sum()
        if rho_sum <= 0.0:
            # this case can happen if a powerline has a flow of 0.0 but is connected, and it's the only one
            # that can be attacked... Pretty rare hey !
            return None, 0
        attack = self.space_prng.choice(available_attacks, p=rho / rho_sum)
        return attack, None

    def _custom_deepcopy_for_copy(self, new_obj, dict_=None):
        super()._custom_deepcopy_for_copy(new_obj, dict_)
        if dict_ is None:
            dict_ = {}

        new_obj._attacks = copy.deepcopy(self._attacks)
        new_obj._lines_ids = copy.deepcopy(self._lines_ids)
        new_obj._next_attack_time = copy.deepcopy(self._next_attack_time)
        new_obj._attack_period = copy.deepcopy(self._attack_period)
        new_obj._rho_normalization = copy.deepcopy(self._rho_normalization)
