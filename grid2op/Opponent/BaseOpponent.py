# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
import grid2op
from grid2op.Space import RandomObject
from grid2op.Parameters import Parameters
from grid2op.Action import TopologyAction
from grid2op.Opponent.BaseActionBudget import BaseActionBudget


class BaseOpponent(RandomObject):
    def __init__(self, action_space):
        RandomObject.__init__(self)
        self.action_space = action_space
        self._do_nothing = self.action_space()

    def init(self, **kwargs):
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
        return None

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

    @classmethod
    def evaluate(cls, opponent_attack_duration=40, opponent_attack_cooldown=100):
        """
        This function is used to evaluate the Opponent and returns the according metrics.
        """
        lines_attacked = ["26_30_56", "30_31_45", "16_18_23", "16_21_27", "9_16_18", "7_9_9",
                          "11_12_13", "12_13_14", "2_3_0", "22_26_39" ]
        opponent_init_budget = 1000
        opponent_budget_per_ts = 0
        agent_line_cooldown = 10
        max_iter = 500

        kwargs = {'test': False,
                  'opponent_init_budget': opponent_init_budget,
                  'opponent_budget_per_ts': opponent_budget_per_ts,
                  'opponent_action_class': TopologyAction,
                  'opponent_budget_class': BaseActionBudget,
                  'opponent_attack_duration': opponent_attack_duration,
                  'opponent_attack_cooldown': opponent_attack_cooldown,
                  'kwargs_opponent': {"lines_attacked": lines_attacked}}

        def get_env(no_overflow_disconnection=False, with_opponent=False):
            param = Parameters()
            param.NO_OVERFLOW_DISCONNECTION = no_overflow_disconnection
            param.NB_TIMESTEP_COOLDOWN_LINE = agent_line_cooldown
            if with_opponent:
                return grid2op.make('l2rpn_wcci_2020',
                                    param=param,
                                    opponent_class=cls,
                                    **kwargs)
            else:
                return grid2op.make('l2rpn_wcci_2020',
                                    param=param,
                                    **kwargs)

        # Measure the number of overflows (no agent, no disconnections)
        # Without opponent
        env = get_env(no_overflow_disconnection=True, with_opponent=False)
        done = False
        step = 0
        overflows_no_opp = 0
        while not done and step < max_iter:
            obs, reward, done, info = env.step(env.action_space())
            step += 1
            overflows_no_opp += np.sum(obs.rho > 1)

        # With opponent
        env = get_env(no_overflow_disconnection=True, with_opponent=True)
        done = False
        step = 0
        overflows_opp = 0
        while not done and step < max_iter:
            obs, reward, done, info = env.step(env.action_space())
            step += 1
            overflows_opp += np.sum(obs.rho > 1)

        # Measure the number of steps (no agent, disconnections enabled)
        # Without opponent
        env = get_env(no_overflow_disconnection=False, with_opponent=False)
        done = False
        step = 0
        while not done and step < max_iter:
            obs, reward, done, info = env.step(env.action_space())
            step += 1
        steps_no_opp = step

        # With opponent
        env = get_env(no_overflow_disconnection=False, with_opponent=True)
        done = False
        step = 0
        while not done and step < max_iter:
            obs, reward, done, info = env.step(env.action_space())
            step += 1
        steps_opp = step

        metrics = {'overflow_delta': overflows_opp - overflows_no_opp,
                   'overflow_delta_relative': (overflows_opp - overflows_no_opp) / overflows_no_opp,
                   'step_delta': steps_opp - steps_no_opp,
                   'step_delta_relative': (steps_opp - steps_no_opp) / steps_no_opp}
        return metrics
