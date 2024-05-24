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


class RandomLineOpponent(BaseOpponent):
    """
    An opponent that disconnect at random any powerlines among a specified list given
    at the initialization.

    """

    def __init__(self, action_space):
        BaseOpponent.__init__(self, action_space)
        self._do_nothing = None
        self._attacks = None
        self._lines_ids = None

    def init(self, partial_env, lines_attacked=[], **kwargs):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            Used when the opponent is created.

        Parameters
        ----------
        partial_env
        lines_attacked
        kwargs

        Returns
        -------

        """
        # this if the function used to properly set the object.
        # It has the generic signature above,
        # and it's way more flexible that the other one.

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
        self._attacks = []
        for l_id in self._lines_ids:
            att = self.action_space({"set_line_status": [(l_id, -1)]})
            self._attacks.append(att)
        self._attacks = np.array(self._attacks)

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
            The duration of the attack (if ``None`` then the attack will be made for the longest allowed time)
        """
        # TODO maybe have a class "GymOpponent" where the observation would include the budget  and all other
        # TODO information, and forward something to the "act" method.

        if observation is None:  # during creation of the environment
            return None, 0  # i choose not to attack in this case

        # Status of attackable lines
        status = observation.line_status[self._lines_ids]

        # If all attackable lines are disconnected
        if np.all(~status):
            return None, 0  # i choose not to attack in this case

        # Pick a line among the connected lines
        attack = self.space_prng.choice(self._attacks[status])
        return attack, None

    def _custom_deepcopy_for_copy(self, new_obj, dict_=None):
        super()._custom_deepcopy_for_copy(new_obj, dict_)
        if dict_ is None:
            dict_ = {}

        new_obj._attacks = copy.deepcopy(self._attacks)
        new_obj._lines_ids = copy.deepcopy(self._lines_ids)
