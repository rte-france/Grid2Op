# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import numpy as np
import unittest
import os
import tempfile
from grid2op.tests.helper_path_test import *

from grid2op.Parameters import Parameters
from grid2op.operator_attention import LinearAttentionBudget
from grid2op import make
from grid2op.Reward import RedispReward, _AlarmScore
from grid2op.Exceptions import Grid2OpException
from grid2op.Runner import Runner
from grid2op.Environment import Environment
from grid2op.Episode import EpisodeData


class TestAlarmScore(unittest.TestCase):
    """test the basic bahavior of the alarm feature"""

    def setUp(self) -> None:
        self.env_nm = os.path.join(
            PATH_DATA_TEST, "l2rpn_neurips_2020_track1_with_alarm"
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = make(
                self.env_nm, test=True, reward_class=_AlarmScore
            )  # ,param=param)

        self.env.seed(0)
        self.env.reset()
        self.env.reset()
        self.max_iter = 10
        self.default_kwargs_att_budget = {
            "max_budget": 3.0,
            "budget_per_ts": 1.0 / (12.0 * 16),
            "alarm_cost": 1.0,
            "init_budget": 2.0,
        }

    def _aux_trigger_cascading_failure(self):
        act_ko1 = self.env.action_space()
        act_ko1.line_set_status = [(56, -1)]
        obs, reward, done, info = self.env.step(act_ko1)
        assert not done
        assert reward == 0
        act_ko2 = self.env.action_space()
        act_ko2.line_set_status = [(41, -1)]
        obs, reward, done, info = self.env.step(act_ko2)
        assert not done
        assert reward == 0
        act_ko3 = self.env.action_space()
        act_ko3.line_set_status = [(40, -1)]
        obs, reward, done, info = self.env.step(act_ko3)
        assert not done
        assert reward == 0

        act_ko4 = self.env.action_space()
        act_ko4.line_set_status = [(39, -1)]
        obs, reward, done, info = self.env.step(act_ko4)
        assert not done
        assert reward == 0

        act_ko5 = self.env.action_space()
        act_ko5.line_set_status = [(13, -1)]
        obs, reward, done, info = self.env.step(act_ko5)
        assert not done
        assert reward == 0

    def test_alarm_reward_simple(self):
        """very basic test for the reward and"""
        # normal step, no game over => 0
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert reward == 0
        self.env.fast_forward_chronics(861)
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert not done
        assert reward == 0

        # end of an episode, no game over: +1
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert done
        assert reward == +1
        assert not obs.was_alarm_used_after_game_over

    def test_reward_game_over_connex(self):
        """test i don't get any points if there is a game over for non connex grid"""
        # game over not due to line disconnection, no points
        obs = self.env.reset()
        act_ko = self.env.action_space()
        act_ko.gen_set_bus = [(0, -1)]
        obs, reward, done, info = self.env.step(act_ko)
        assert done
        assert reward == -2
        assert not obs.was_alarm_used_after_game_over

    def test_reward_no_alarm(self):
        """test that i don't get any points if i don't send any alarm"""
        # FYI parrallel lines:
        # 48, 49  || 18, 19  || 27, 28 || 37, 38

        # game not due to line disconnection, but no alarm => no points
        self._aux_trigger_cascading_failure()
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert done
        assert reward == -2
        assert not obs.was_alarm_used_after_game_over

    def test_reward_wrong_area_wrong_time(self):
        """test that i got a few point for the wrong area, but at the wrong time"""
        # now i raise an alarm, and after i do a cascading failure (but i send a wrong alarm)
        act = self.env.action_space()
        act.raise_alarm = [0]
        obs, reward, done, info = self.env.step(act)
        self._aux_trigger_cascading_failure()
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert done
        assert (
            reward == 0.5
        )  # different from AlarmReward because mult_for_right_zone is lower here, 1.5 instead of 2
        assert obs.was_alarm_used_after_game_over

    def test_reward_right_area_not_best_time(self):
        """test that i got some point for the right area, but at the wrong time"""
        # now i raise an alarm, and after i do a cascading failure (and i send a right alarm)
        act = self.env.action_space()
        act.raise_alarm = [1]
        obs, reward, done, info = self.env.step(act)
        self._aux_trigger_cascading_failure()
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert done
        assert reward == 0.75
        assert obs.was_alarm_used_after_game_over

    def test_reward_right_time_wrong_area(self):
        """test that the alarm has half "value" if taken exactly at the right time but for the wrong area"""
        # now i raise an alarm just at the right time, and after i do a cascading failure (wrong zone)
        act = self.env.action_space()
        act.raise_alarm = [0]
        obs, reward, done, info = self.env.step(act)
        for _ in range(6):
            obs, reward, done, info = self.env.step(self.env.action_space())
        self._aux_trigger_cascading_failure()
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert done
        assert (
            reward == 2 / 3
        )  # different from AlarmReward because mult_for_right_zone is lower here, 1.5 instead of 2
        assert obs.was_alarm_used_after_game_over

    def test_reward_right_time_right_area(self):
        """test that the alarm has perfect "value" if taken exactly at the right time and for the right area"""
        # now i raise an alarm just at the right time, and after i do a cascading failure (right zone)
        act = self.env.action_space()
        act.raise_alarm = [1]
        obs, reward, done, info = self.env.step(act)
        for _ in range(6):
            obs, reward, done, info = self.env.step(self.env.action_space())
        self._aux_trigger_cascading_failure()
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert done
        assert reward == 1
        assert obs.was_alarm_used_after_game_over

    def test_reward_right_area_too_early(self):
        """test that the alarm is not taken into account if send too early"""
        # now i raise an alarm but too early, i don't get any points (even if right zone)
        act = self.env.action_space()
        act.raise_alarm = [1]
        obs, reward, done, info = self.env.step(act)
        for _ in range(6):
            obs, reward, done, info = self.env.step(self.env.action_space())
        for _ in range(12):
            obs, reward, done, info = self.env.step(self.env.action_space())
        self._aux_trigger_cascading_failure()
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert done
        assert reward == -2
        assert not obs.was_alarm_used_after_game_over

    def test_reward_correct_alarmused_right_early(self):
        """test that the maximum is taken, when an alarm is send at the right time, and another one too early"""
        # now i raise two alarms: one at just the right time, another one a bit earlier, and i check the correct
        # one is used
        act = self.env.action_space()
        act.raise_alarm = [1]
        obs, reward, done, info = self.env.step(act)  # a bit too early
        for _ in range(3):
            obs, reward, done, info = self.env.step(self.env.action_space())
        obs, reward, done, info = self.env.step(act)
        for _ in range(6):
            obs, reward, done, info = self.env.step(
                self.env.action_space()
            )  # just at the right time
        self._aux_trigger_cascading_failure()
        obs, reward, done, info = self.env.step(self.env.action_space())

        assert done
        assert reward == 1.0  # it should count this one
        assert obs.was_alarm_used_after_game_over

    def test_reward_correct_alarmused_right_toolate(self):
        """test that the maximum is taken, when an alarm is send at the right time, and another one too late"""
        # now i raise two alarms: one at just the right time, another one a bit later, and i check the correct
        # one is used
        act = self.env.action_space()
        act.raise_alarm = [1]
        obs, reward, done, info = self.env.step(act)  # just at the right time
        for _ in range(3):
            obs, reward, done, info = self.env.step(self.env.action_space())
        obs, reward, done, info = self.env.step(act)  # a bit too early
        for _ in range(2):
            obs, reward, done, info = self.env.step(self.env.action_space())
        self._aux_trigger_cascading_failure()
        obs, reward, done, info = self.env.step(self.env.action_space())

        assert done
        assert reward == 1.0  # it should count this one
        assert obs.was_alarm_used_after_game_over

    def _is_cascade_in_zone_alarm(self, disc_lines_in_cascade, zone_alarm):
        is_cascade_in_zone_alarm = False
        for line_id in disc_lines_in_cascade:
            zones_line = self.env.alarms_lines_area[self.env.name_line[line_id]]
            if zone_alarm in zones_line:
                is_cascade_in_zone_alarm = True
                break
        return is_cascade_in_zone_alarm

    def test_reward_correct_alarm_only_cascade_right_zone(self):
        """test that the maximum is taken, when an alarm is send at the right time with only lines got diconnected at
        the time of cascading failure"""

        # changing parameters to allow for multiple line actions in order to create a cascading failure in one time_step
        param = Parameters()
        param.init_from_dict({"MAX_LINE_STATUS_CHANGED": 999})
        self.env.change_parameters(param)
        self.env.reset()  # env.reset() to reset the parameters. Note that we also jump to a new scenario by doing
        # env.reset(), which is desired here

        act = self.env.action_space()
        alarm_zone_id = 1  # right zone
        zone_alarm = self.env.alarms_area_names[alarm_zone_id]
        act.raise_alarm = [alarm_zone_id]
        disc_lines_before_cascade = []

        obs, reward, done, info = self.env.step(act)  # just at the right time
        assert (
            len(list(np.where(info["disc_lines"] == 0)[0])) == 0
        )  # no line disconnected before cascade
        for _ in range(11):
            obs, reward, done, info = self.env.step(self.env.action_space())
            assert len(list(np.where(info["disc_lines"] == 0)[0])) == 0

        # cascading failure in one step
        act = self.env.action_space(
            {"set_line_status": [(56, -1), (41, -1), (40, -1), (39, -1), (13, -1)]}
        )
        obs, reward, done, info = self.env.step(act)

        disc_lines_in_cascade = list(np.where(info["disc_lines"] == 0)[0])

        assert self._is_cascade_in_zone_alarm(disc_lines_in_cascade, zone_alarm)
        assert done
        assert reward == 1.0  # get the max points because cascade linees in right zone
        assert obs.was_alarm_used_after_game_over

    def test_reward_correct_alarm_only_cascade_bad_zone(self):
        """test that the geo points are not taken, when an alarm is send at the right time, but lines only got
        diconnected at the time of cascading failure in a bad zone"""

        param = Parameters()
        param.init_from_dict({"MAX_LINE_STATUS_CHANGED": 999})
        self.env.change_parameters(param)
        self.env.reset()  # env.reset() to reset the parameters. Note that we also jump to a new scenario by doing
        # env.reset(), which is desired here

        act = self.env.action_space()
        alarm_zone_id = 0  # wrong zone
        zone_alarm = self.env.alarms_area_names[alarm_zone_id]
        act.raise_alarm = [alarm_zone_id]

        obs, reward, done, info = self.env.step(act)  # just at the right time
        assert (
            len(list(np.where(info["disc_lines"] == 0)[0])) == 0
        )  # no line disconnected before cascade
        for _ in range(11):
            obs, reward, done, info = self.env.step(self.env.action_space())
            assert len(list(np.where(info["disc_lines"] == 0)[0])) == 0

        # cascading failure in one step
        act = self.env.action_space(
            {"set_line_status": [(56, -1), (41, -1), (40, -1), (39, -1), (13, -1)]}
        )
        obs, reward, done, info = self.env.step(act)

        disc_lines_in_cascade = list(np.where(info["disc_lines"] == 0)[0])

        assert not self._is_cascade_in_zone_alarm(disc_lines_in_cascade, zone_alarm)
        assert done
        assert reward == 1 / 1.5  # get the max time points but not geo points
        assert obs.was_alarm_used_after_game_over

    def test_alarm_reward_linedisconnection_in_window_right_time_bad_zone(self):
        """test that the geo points are not taken, when an alarm is send at the right time, and lines get
        diconnected in the disconnection window before the time of cascading failure, but in a bad zone"""

        # will fail at timestep 13, perfect time.
        # line disconnected before cascading failure are in zone 2. During cascade we also get line disconnected in the alarm zone 1
        # we should not get geo_points in the end
        # dis_lines_before_cascade=[[], [], [45], [], [], [41], [], [], [40], [], []] => only line 40 should be consider for points here
        # disc_lines_in_cascade= [13, 22, 23, 31, 32, 33, 34, 35, 39]

        # changing thermal limits to create an overload since the beginning and a slow cascading failure
        new_thermal_limit = self.env.get_thermal_limit()
        line_id = 45
        new_thermal_limit[line_id] = new_thermal_limit[line_id] / 3
        self.env.set_thermal_limit(new_thermal_limit)

        act = self.env.action_space()
        alarm_zone_id = 1
        zone_alarm = self.env.alarms_area_names[alarm_zone_id]
        act.raise_alarm = [alarm_zone_id]
        obs, reward, done, info = self.env.step(
            act
        )  # first step and an overload occur on our line 45

        t = 1
        disc_lines_before_cascade = []
        while not done:  # will fail at timestep 13
            act = self.env.action_space()
            obs, reward, done, info = self.env.step(act)
            if not done:
                disc_lines_before_cascade.append(
                    list(np.where(info["disc_lines"] == 0)[0])
                )
            else:
                disc_lines_in_cascade = list(np.where(info["disc_lines"] == 0)[0])
            t += 1

        assert self._is_cascade_in_zone_alarm(
            disc_lines_in_cascade, zone_alarm
        )  # a line was during the cascade disconnected in the alamr zone.
        # But it is not the zone in which the first line we considered disconnected. So will not give points
        assert reward == 1 / 1.5  # get points for perfect time but not for zone

    def test_alarm_reward_linedisconnection_in_window_right_time_good_zone(self):
        """test that the geo points are taken, when an alarm is send at the right time, and lines get
        diconnected in the disconnection window before the time of cascading failure, but in a good zone"""

        # will fail at timestep 13, perfect time.
        # line disconnected before cascading failure are in zone 2.
        # we should get geo_points in the end
        # dis_lines_before_cascade=[[], [], [45], [], [], [41], [], [], [40], [], []] => only line 40 should be consider for points here
        # disc_lines_in_cascade= [13, 22, 23, 31, 32, 33, 34, 35, 39]

        # changing thermal limits to create an overload since the beginning and a slow cascading failure
        new_thermal_limit = self.env.get_thermal_limit()
        line_id = 45
        new_thermal_limit[line_id] = new_thermal_limit[line_id] / 3
        self.env.set_thermal_limit(new_thermal_limit)

        act = self.env.action_space()
        alarm_zone_id = 2
        zone_alarm = self.env.alarms_area_names[alarm_zone_id]
        act.raise_alarm = [alarm_zone_id]
        obs, reward, done, info = self.env.step(
            act
        )  # first step and an overload occur on our line 45

        t = 1
        disc_lines_before_cascade = []
        while not done:  # will fail at timestep 13
            act = self.env.action_space()
            obs, reward, done, info = self.env.step(act)

            if not done:
                disc_lines_before_cascade.append(
                    list(np.where(info["disc_lines"] == 0)[0])
                )
            else:
                disc_lines_in_cascade = list(np.where(info["disc_lines"] == 0)[0])
            t += 1

        assert reward == 1  # get points for perfect time and zone

    def test_alarm_reward_linedisconnection_before_window_right_time_good_zone(self):
        """test that the geo points are taken, when an alarm is send at the right time, and lines get
        diconnected before the disconnection window + lines disconnected at cascading failure are in a good zone"""

        # will fail at timestep 13, perfect time.
        # line disconnected before cascading failure are in zone 2. During cascade we also don't get line disconnected in the alarm zone 1
        # we should not get geo_points in the end
        # dis_lines_before_cascade=[[], [], [45], [], [], [41], [], [], [], [], []] => only line 40 should be consider for points here
        # disc_lines_in_cascade= [13, 22, 23, 31, 32, 33, 34, 35, 39]

        # changing thermal limits to create an overload since the beginning and a slow cascading failure
        new_thermal_limit = self.env.get_thermal_limit()
        line_id = 45
        new_thermal_limit[line_id] = new_thermal_limit[line_id] / 3
        self.env.set_thermal_limit(new_thermal_limit)

        act = self.env.action_space()
        alarm_zone_id = 1
        zone_alarm = self.env.alarms_area_names[alarm_zone_id]
        act.raise_alarm = [alarm_zone_id]
        obs, reward, done, info = self.env.step(
            act
        )  # first step and an overload occur on our line 45

        t = 1
        disc_lines_before_cascade = []
        while not done:  # will fail at timestep 13
            act = self.env.action_space()
            if (
                t == 9
            ):  # disconnecting the line ourself so that it does not appear in disc_lines
                # in the window_disconnection
                act = self.env.action_space({"set_line_status": [(40, -1)]})
            obs, reward, done, info = self.env.step(act)
            if not done:
                disc_lines_before_cascade.append(
                    list(np.where(info["disc_lines"] == 0)[0])
                )
            else:
                disc_lines_in_cascade = list(np.where(info["disc_lines"] == 0)[0])
            t += 1

        assert self._is_cascade_in_zone_alarm(
            disc_lines_in_cascade, zone_alarm
        )  # a line was during the cascade disconnected in the alamr zone.
        assert reward == 1  # get points for perfect time but not for zone

    def test_alarm_reward_linedisconnection_in_window_right_time_bad_zone(self):
        """test that the geo points are taken, when an alarm is send at the right time, and lines get
        diconnected before the disconnection window + lines disconnected at cascading failure are in a good zone"""

        # line disconnected before cascading failure are in zone 2. But during cascade we get line disconnected in zone 1
        # we should not get geo_points
        # dis_lines_before_cascade=[[], [], [45], [], [], [41], [], [], [40], [], []] => only line 40 should be consider for points here
        # disc_lines_in_cascade= [13, 22, 23, 31, 32, 33, 34, 35, 39]

        # changing thermal limits to create an overload since the beginning and a slow cascading failure
        new_thermal_limit = self.env.get_thermal_limit()
        line_id = 45
        new_thermal_limit[line_id] = new_thermal_limit[line_id] / 3
        self.env.set_thermal_limit(new_thermal_limit)

        act = self.env.action_space()
        act.raise_alarm = [1]
        obs, reward, done, info = self.env.step(act)  # just at the right time
        t = 1
        while not done:
            act = self.env.action_space()
            obs, reward, done, info = self.env.step(act)
            t += 1

        assert reward == 1 / 1.5  # get points for perfect time but not for zone

    def test_alarm_reward_no_point_factor_multi_zone(self):
        """test that the geo points are not taken, when an alarm is send on multi zones"""

        # changing thermal limits to create an overload since the beginning and a slow cascading failure
        new_thermal_limit = self.env.get_thermal_limit()
        line_id = 45
        new_thermal_limit[line_id] = new_thermal_limit[line_id] / 3
        self.env.set_thermal_limit(new_thermal_limit)

        act = self.env.action_space()
        act.raise_alarm = [0, 1]

        obs, reward, done, info = self.env.step(
            act
        )  # first step and an overload occur on our line 45

        t = 1
        disc_lines_before_cascade = []
        while not done:  # will fail at timestep 13
            act = self.env.action_space()
            obs, reward, done, info = self.env.step(act)
            t += 1

        # But it is not the zone in which the first line we considered disconnected. So will not give points
        assert reward == 1 / 1.5  # get points for perfect time but not for zone

    # zero if dics
    def test_alarm_after_line_diconnection_score_low(self):
        """test that the score is low when an alarm is sent at the time of an observed line disconnection
        at the beginning of disconnection window"""

        # changing thermal limits to create an overload since the beginning and a slow cascading failure
        new_thermal_limit = self.env.get_thermal_limit()
        line_id = 45
        new_thermal_limit[line_id] = new_thermal_limit[line_id] / 3
        self.env.set_thermal_limit(new_thermal_limit)

        act = self.env.action_space()
        alarm_zone_id = 1
        zone_alarm = self.env.alarms_area_names[alarm_zone_id]

        obs, reward, done, info = self.env.step(
            act
        )  # first step and an overload occur on our line 45

        t = 1
        disc_lines_before_cascade = []
        while not done:  # will fail at timestep 13
            act = self.env.action_space()
            if (
                t == 9
            ):  # right at the time of line disconnection in window_disconnection
                act.raise_alarm = [alarm_zone_id]
            obs, reward, done, info = self.env.step(act)
            t += 1

        assert np.round(reward, 2) == 0.29  # get points for perfect zone but not time

    def test_alarm_after_line_diconnection_score_low_2(self):
        """test that the score is low for another alarm_best_time and alarm_window_size
         when an alarm is sent at the time of an observed line disconnection
        at the beginning of disconnection window"""

        # changing alarm time parameters
        param = self.env.parameters
        param.init_from_dict({"ALARM_WINDOW_SIZE": 5, "ALARM_BEST_TIME": 7})
        self.env.change_parameters(param)
        self.env.reset()  # env.reset() to reset the parameters. Note that this also jump to the next scenario.
        self.env.reset()  # we do one more env.reset() to jump back to our initial scenario among the two we have
        ###

        # changing thermal limits to create an overload since the beginning and a slow cascading failure
        new_thermal_limit = self.env.get_thermal_limit()
        line_id = 45
        new_thermal_limit[line_id] = new_thermal_limit[line_id] / 3
        self.env.set_thermal_limit(new_thermal_limit)

        act = self.env.action_space()
        alarm_zone_id = 1

        obs, reward, done, info = self.env.step(
            act
        )  # first step and an overload occur on our line 45

        t = 1
        disc_lines_before_cascade = []
        while not done:  # will fail at timestep 13
            act = self.env.action_space()
            if (
                t == 9
            ):  # right at the time of line disconnection in window_disconnection
                act.raise_alarm = [alarm_zone_id]
            obs, reward, done, info = self.env.step(act)
            if not done:
                disc_lines_before_cascade.append(
                    list(np.where(info["disc_lines"] == 0)[0])
                )
            else:
                disc_lines_in_cascade = list(np.where(info["disc_lines"] == 0)[0])
            t += 1

        assert np.round(reward, 2) == 0.24
