# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
from grid2op.Opponent import OpponentWithConverter
from grid2op.Converter import LineDisconnection

np.random.seed(0)


class RandomLineOpponent(OpponentWithConverter):
    def __init__(self, action_space, uptime=12, downtime=12*24):
        OpponentWithConverter.__init__(self, action_space,
                                       action_space_converter=LineDisconnection)
        if action_space.n_line == 59: # WCCI
            lines_maintenance = ["26_30_56", "30_31_45", "16_18_23", "16_21_27", "9_16_18", "7_9_9",
                                 "11_12_13", "12_13_14", "2_3_0", "22_26_39" ]
        elif action_space.n_line == 20: # case 14
            lines_maintenance = ["1_3_3", "1_4_4", "3_6_15", "9_10_12", "11_12_13", "12_13_14"]
        else:
            raise Warning(f'Unknown environment found with {action_space.n_line} lines')
        self.action_space.filter_lines(lines_maintenance)
        self._do_nothing = 0
        self.uptime = uptime
        self.current_uptime = 0
        self.downtime = downtime
        self.current_downtime = downtime
        self.current_attack = None

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
        self.current_uptime = 0
        self.current_downtime = self.downtime
        self.current_attack = None

    def my_attack(self, observation, env, opp_space, agent_action, env_action, budget, previous_fails, update=True):
        """
        This method is the equivalent of "attack" for a regular agent.

        Opponent, in this framework can have more information than a regular agent (in particular it can
        view time step t+1), it has access to its current budget etc.

        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            The last observation (at time t)

        env: :class:`grid2op.Environment.Environment`
            The environment

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

        # Decrement counters
        new_uptime = max(0, self.current_uptime - 1)
        new_downtime = max(0, self.current_downtime - 1)

        # If currently attacking
        if new_uptime > 0:
            attack = self.current_attack
            # If the cost is too high
            if not opp_space.compute_budget(self.convert_act(attack)) <= opp_space.budget:
                raise Warning('The attack is too expensive to be fully completed')

        # If the opponent has already attacked today
        elif new_downtime > self.downtime:
            attack = self._do_nothing

        # If the opponent can attack  
        else:      
            attack = self._my_attack_raw(observation)
            # If the cost is too high
            if not opp_space.compute_budget(self.convert_act(attack)) <= opp_space.budget:
                attack = self._do_nothing
            # If we can afford the attack
            elif attack != self._do_nothing:
                new_uptime = min(self.uptime,
                                 int(opp_space.budget / opp_space.compute_budget(self.convert_act(attack))))
                new_downtime += self.downtime

        # If this is launched from env.step (not obs.simulate)
        from grid2op.Observation import _ObsEnv
        if update and not isinstance(env, _ObsEnv): # update the opponent and the environment
            self.current_uptime = new_uptime
            self.current_downtime = new_downtime
            self.current_attack = attack
            # Todo : Check that the value is correct and must only be set when attacking (=check that the default value is 0 ; need Benjamin's advice)
            if attack != self._do_nothing:
                line_attacked = self.convert_act(attack).as_dict()['set_line_status']['disconnected_id'][0]
                env.times_before_line_status_actionable[line_attacked] = new_uptime

        return attack

    def _my_attack_raw(self, observation):
        if observation is None: # during creation of the environment
            return self._do_nothing # do nothing

        action_line_ids = [a.as_dict()['set_line_status']['disconnected_id'][0] for a in self.action_space.actions[1:]]
        status = observation.line_status[action_line_ids]

        # If all lines are disconnected
        if not any(status):
            return self._do_nothing

        # Pick a line among the connected lines
        picked = 1 + np.random.choice(np.argwhere(status).ravel())
        return picked
