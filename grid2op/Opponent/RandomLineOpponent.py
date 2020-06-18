# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
from grid2op.Opponent import BaseOpponent
from grid2op.Converter import LineDisconnection


class RandomLineOpponent(BaseOpponent):
    def __init__(self, action_space):
        # Apply converter
        converter_action_space = LineDisconnection(action_space)
        BaseOpponent.__init__(self, converter_action_space)
        self.action_space.init_converter()

        # Filter lines
        if action_space.n_line == 59: # WCCI
            lines_maintenance = ["26_30_56", "30_31_45", "16_18_23", "16_21_27", "9_16_18", "7_9_9",
                                 "11_12_13", "12_13_14", "2_3_0", "22_26_39" ]
        elif action_space.n_line == 20: # case 14
            lines_maintenance = ["1_3_3", "1_4_4", "3_6_15", "9_10_12", "11_12_13", "12_13_14"]
        else:
            raise Warning(f'Unknown environment found with {action_space.n_line} lines')
        self.action_space.filter_lines(lines_maintenance)

        self._do_nothing = self.action_space.actions[0]
        self._attacks = self.action_space.actions[1:]

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
        """
        # TODO maybe have a class "GymOpponent" where the observation would include the budget  and all other
        # TODO information, and forward something to the "act" method.

        if observation is None: # during creation of the environment
            return self._do_nothing # do nothing

        action_line_ids = [a.as_dict()['set_line_status']['disconnected_id'][0]
                           for a in self._attacks]
        status = observation.line_status[action_line_ids]

        # If all lines are disconnected
        if not any(status):
            return self._do_nothing

        # Pick a line among the connected lines
        return np.random.choice(self._attacks)
