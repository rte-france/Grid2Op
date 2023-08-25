# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np

from grid2op.Exceptions import Grid2OpException
from grid2op.Reward.baseReward import BaseReward
from grid2op.dtypes import dt_float


class AlarmReward(BaseReward):
    """
    This reward is based on the "alarm feature" where the agent is asked to send information about potential issue
    on the grid.

    On this case, when the environment is in a "game over" state (eg it's the end) then the reward is computed
    the following way:

    - if the environment has been successfully manage until the end of the chronics, then 1.0 is returned
    - if no alarm has been raised, then -1.0 is return


    Examples
    ---------
    You can use this reward in any environment with:

    .. code-block:

        import grid2op
        from grid2op.Reward import AlarmReward

        # then you create your environment with it:
        NAME_OF_THE_ENVIRONMENT = "l2rpn_case14_sandbox"
        env = grid2op.make(NAME_OF_THE_ENVIRONMENT,reward_class=AlarmReward)
        # and do a step with a "do nothing" action
        obs = env.reset()
        obs, reward, done, info = env.step(env.action_space())
        # the reward is computed with the AlarmReward class


    """

    def __init__(self, logger=None):
        BaseReward.__init__(self, logger=logger)
        # required if you want to design a custom reward taking into account the
        # alarm feature
        self.has_alarm_component = True
        self.is_alarm_used = False  # required to update it in __call__ !!

        self.total_time_steps = dt_float(0.0)
        self.reward_min = dt_float(-1.0)
        self.reward_max = dt_float(1.0)
        self.reward_no_game_over = dt_float(0.0)

        self.window_size = None
        self.best_time = None

        self.mult_for_right_zone = 2

    def initialize(self, env):
        if not env._has_attention_budget:
            raise Grid2OpException(
                'Impossible to use the "AlarmReward" with an environment for which this feature '
                'is disabled. Please make sure "env._has_attention_budget" is set to ``True`` or '
                "change the reward class with `grid2op.make(..., reward_class=AnyOtherReward)`"
            )
        self.reset(env)

    def reset(self, env):
        self.total_time_steps = env.max_episode_duration()
        self.best_time = env.parameters.ALARM_BEST_TIME
        self.window_size = env.parameters.ALARM_WINDOW_SIZE

    def _tmp_score_time(self, step_alarm, step_game_over):
        """
        compute the "temporal" score.

        Should give a number between 0 and 1
        """
        if step_game_over - step_alarm > self.best_time + self.window_size:
            # alarm too soon
            res = 0
        elif step_game_over - step_alarm < self.best_time - self.window_size:
            # alarm too late
            res = 0
        else:
            # square function such that: it gives 1 if step_game_over - step_alarm equals self.best_time
            # and 0 if  step_game_over - step_alarm = self.best_time + self.window_size or
            # if step_game_over - step_alarm self.best_time - self.window_size
            dist_to_game_over = step_game_over - step_alarm
            dist_to_best = dist_to_game_over - self.best_time

            # set it to 0 for the extreme case
            polynom = (dist_to_best - self.window_size) * (
                dist_to_best + self.window_size
            )
            # scale it such that it is 1 for dist_to_best == 0 (ie step_game_over - step_alarm == self.best_time)
            res = -polynom / self.window_size**2
        return res

    def _mult_for_zone(self, alarm, disc_lines, env):
        """compute the multiplicative factor that increases the score if the right zone is predicted"""
        res = 1.0
        # extract the lines that have been disconnected due to cascading failures
        lines_disconnected_first = np.where(disc_lines == 0)[0]

        if (
            alarm.sum() > 1
        ):  # if we have more than one zone in the alarm, we cannot discrtiminate, no bonus points
            return res

        # extract the zones they belong too
        zones_these_lines = set()
        zone_for_each_lines = env.alarms_lines_area
        for line_id in lines_disconnected_first:
            line_name = env.name_line[line_id]
            for zone_name in zone_for_each_lines[line_name]:
                zones_these_lines.add(zone_name)

        # now retrieve the id of the zones in which a powerline has been disconnected
        list_zone_names = list(zones_these_lines)
        list_zone_ids = np.where(np.isin(env.alarms_area_names, list_zone_names))[0]
        # and finally, award some extra points if one of the zone, containing one of the powerline disconnected
        # by protection is in the alarm
        if alarm[list_zone_ids].any():
            res *= self.mult_for_right_zone
        return res

    def _points_for_alarm(self, step_alarm, alarm, step_game_over, disc_lines, env):
        """how much points are given for this specific alarm"""
        is_alarm_used = False
        score = self.reward_min
        score_for_time = self._tmp_score_time(step_alarm, step_game_over)
        if score_for_time != 0:
            is_alarm_used = True  # alarm is in the right time window
            score = score_for_time
            score *= (
                self._mult_for_zone(alarm, disc_lines, env) / self.mult_for_right_zone
            )
        return score, is_alarm_used

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
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
            disc_lines = env._disc_lines

            # so now i can consider the alarms.
            best_score, is_alarm_used = self.reward_min, False
            for alarm in successfull_alarms:
                tmp_sc, tmp_is = self._points_for_alarm(
                    *alarm,
                    step_game_over=step_game_over,
                    disc_lines=disc_lines,
                    env=env
                )
                if tmp_sc > best_score:
                    best_score = tmp_sc
                    is_alarm_used = tmp_is

            self.is_alarm_used = is_alarm_used
            return best_score
        else:
            res = self.reward_no_game_over
        return res
