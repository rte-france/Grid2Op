# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
import copy

from grid2op.Exceptions import Grid2OpException
from grid2op.Reward import AlarmReward
from grid2op.dtypes import dt_float


class _AlarmScore(AlarmReward):
    """
        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            It **must not** serve as a reward. This scored needs to be minimized,
            and a reward needs to be maximized! Also, this "reward" is not scaled or anything. Use it as your
            own risk.

    Implemented as a reward to make it easier to use in the context of the L2RPN competitions, this "reward"
    computed the "grid operation cost". It should not be used to train an agent.

    The "reward" the closest to this score is given by the :class:`AlarmReward` class.

    This reward is based on the "alarm feature" where the agent is asked to send information about potential issue
    on the grid.
    On this case, when the environment is in a "game over" state (eg it's the end) then the reward is computed
    the following way:
    - if the environment has been successfully manage until the end of the chronics, then 1.0 is returned
    - if no alarm has been raised, then -2.0 is return
    - points for pointing to the right zones are computed based on the lines disconnected either in a short window
    before game over or otherwise at the time of game over

    Examples
    ---------
    You can use this reward in any environment with:

    .. code-block:

        import grid2op
        from grid2op.Reward import AlarmReward
        # then you create your environment with it:
        NAME_OF_THE_ENVIRONMENT = "l2rpn_case14_sandbox"
        env = grid2op.make(NAME_OF_THE_ENVIRONMENT,reward_class=AlarmScore)
        # and do a step with a "do nothing" action
        obs = env.reset()
        obs, reward, done, info = env.step(env.action_space())
        # the reward is computed with the AlarmScore class

    """

    def __init__(self, logger=None):
        AlarmReward.__init__(self, logger=logger)
        # required if you want to design a custom reward taking into account the
        # alarm feature
        self.reward_min = dt_float(-2.0)
        # we keep other parameters values from AlarmReward as is

        self.mult_for_right_zone = 1.5
        self.window_disconnection = 4

        self.disc_lines_all_before_cascade = []
        self.n_line = None
        self._i_am_simulate = True

    def initialize(self, env):
        if not env._has_attention_budget:
            raise Grid2OpException(
                'Impossible to use the "_AlarmScore" with an environment for which this feature '
                'is disabled. Please make sure "env._has_attention_budget" is set to ``True`` or '
                "change the reward class with `grid2op.make(..., reward_class=AnyOtherReward)`"
            )
        self.n_line = env.n_line
        self.reset(env)

    def reset(self, env):
        super().reset(env)
        self.window_disconnection = max(self.best_time - self.window_size, 4)
        self.disc_lines_all_before_cascade = []
        self._i_am_simulate = self.is_simulated_env(env)

    def _lines_disconnected_first(self, disc_lines_at_cascading_time):
        """
        here we detect the disconnected lines that we will consider to compute the `mult_for_zone` multiplying factor.
        Either the lines that were disconnected in a short period before final failure. Otherwise the first lines
        disconnected at the time of failure

        :param disc_lines_at_cascading_time: lines that are disconnected first at time of failure
        :return:
        """

        disc_lines_to_consider_for_score = np.zeros(self.n_line, dtype=bool)

        nb_obs = len(self.disc_lines_all_before_cascade)
        for step in range(nb_obs - self.window_disconnection, nb_obs):
            disc_lines_to_consider_for_score[
                self.disc_lines_all_before_cascade[step] >= 0
            ] = True

        if disc_lines_to_consider_for_score.sum() == 0:
            disc_lines_to_consider_for_score = disc_lines_at_cascading_time == 0

        # if we are there, it is because we have identified before that the failure is due to disconnected powerlines
        assert (disc_lines_to_consider_for_score).any()

        # we transform the vector so that disconnected lines have a zero, to be coherent with env._disc_lines
        return 1 - disc_lines_to_consider_for_score

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if self._i_am_simulate:
            return self.reward_no_game_over

        disc_lines_now = env._disc_lines

        if is_done:
            if not has_error:
                # agent went until the end
                return self.reward_max
            if np.all(env._disc_lines == -1):
                # game over is not caused by the tripping of a powerline
                return self.reward_min

            if len(env._attention_budget._all_successful_alarms) == 0:
                # no alarm have been sent, so it's the minimum
                return self.reward_min

            successfull_alarms = env._attention_budget._all_successful_alarms
            step_game_over = env.nb_time_step

            disc_lines_to_consider_for_score = self._lines_disconnected_first(
                disc_lines_now
            )

            # so now i can consider the alarms.
            best_score, is_alarm_used = self.reward_min, False
            for alarm in successfull_alarms:
                tmp_sc, tmp_is = self._points_for_alarm(
                    *alarm,
                    step_game_over=step_game_over,
                    disc_lines=disc_lines_to_consider_for_score,
                    env=env
                )
                if tmp_sc > best_score:
                    best_score = tmp_sc
                    is_alarm_used = tmp_is

            self.is_alarm_used = is_alarm_used
            return best_score
        else:
            # make sure to deepcopy, otherwise it gets updated with the last timestep value for every previous timesteps
            # we log the line disconnected over time
            # TODO have a cache there and store only the last few states, most of what is stored here is not used
            self.disc_lines_all_before_cascade.append(copy.deepcopy(disc_lines_now))
            res = self.reward_no_game_over
        return res
