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

from grid2op.operator_attention import LinearAttentionBudget
from grid2op import make
from grid2op.Reward import RedispReward, _AlarmScore
from grid2op.Exceptions import Grid2OpException
from grid2op.Runner import Runner
from grid2op.Environment import Environment
from grid2op.Episode import EpisodeData


class TestAlarmFeature(unittest.TestCase):
    """test the basic bahavior of the alarm feature"""

    def setUp(self) -> None:
        self.env_nm = os.path.join(
            PATH_DATA_TEST, "l2rpn_neurips_2020_track1_with_alarm"
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = make(self.env_nm, test=True)
        self.env.seed(0)
        self.env.reset()
        self.env.reset()
        self.max_iter = 10
        self.default_kwargs_att_budget = {
            "max_budget": 5.0,
            "budget_per_ts": 1.0 / (12.0 * 8),
            "alarm_cost": 1.0,
            "init_budget": 3.0,
        }

    def tearDown(self) -> None:
        self.env.close()

    def test_create_ok(self):
        """TestAlarmFeature.test_create_ok test that the stuff is created with the right parameters"""
        assert self.env._has_attention_budget
        assert self.env._attention_budget is not None
        assert isinstance(self.env._attention_budget, LinearAttentionBudget)
        assert abs(self.env._attention_budget._budget_per_ts - 1.0 / (12.0 * 8)) <= 1e-6
        assert abs(self.env._attention_budget._max_budget - 5) <= 1e-6
        assert abs(self.env._attention_budget._alarm_cost - 1) <= 1e-6
        assert abs(self.env._attention_budget._current_budget - 3.0) <= 1e-6

        with self.assertRaises(Grid2OpException):
            # it raises because the default reward: AlarmReward can only be used
            # if there is an alarm budget
            with make(self.env_nm, has_attention_budget=False, test=True) as env:
                assert env._has_attention_budget is False
                assert env._attention_budget is None

        with make(
            self.env_nm,
            has_attention_budget=False,
            reward_class=RedispReward,
            test=True,
        ) as env:
            assert env._has_attention_budget is False
            assert env._attention_budget is None

        with make(
            self.env_nm,
            test=True,
            kwargs_attention_budget={
                "max_budget": 15,
                "budget_per_ts": 1,
                "init_budget": 0,
                "alarm_cost": 12,
            },
        ) as env:
            assert env._has_attention_budget
            assert env._attention_budget is not None
            assert isinstance(env._attention_budget, LinearAttentionBudget)
            assert abs(env._attention_budget._budget_per_ts - 1.0) <= 1e-6
            assert abs(env._attention_budget._max_budget - 15) <= 1e-6
            assert abs(env._attention_budget._alarm_cost - 12) <= 1e-6
            assert abs(env._attention_budget._current_budget - 0.0) <= 1e-6

    def test_budget_increases_ok(self):
        """test the attention budget properly increases when no alarm are raised
        and that it does not exceed the maximum value"""
        # check increaes ok normally
        self.env.step(self.env.action_space())
        assert (
            abs(self.env._attention_budget._current_budget - (3 + 1.0 / (12.0 * 8.0)))
            <= 1e-6
        )
        self.env.step(self.env.action_space())
        assert (
            abs(self.env._attention_budget._current_budget - (3 + 2.0 / (12.0 * 8.0)))
            <= 1e-6
        )

        # check that it does not "overflow"
        with make(
            self.env_nm,
            kwargs_attention_budget={
                "max_budget": 5,
                "budget_per_ts": 1,
                "alarm_cost": 12,
                "init_budget": 0,
            },
            test=True,
        ) as env:
            env.step(self.env.action_space())
            assert abs(env._attention_budget._current_budget - 1) <= 1e-6
            env.step(self.env.action_space())
            assert abs(env._attention_budget._current_budget - 2) <= 1e-6
            env.step(self.env.action_space())
            assert abs(env._attention_budget._current_budget - 3) <= 1e-6
            env.step(self.env.action_space())
            assert abs(env._attention_budget._current_budget - 4) <= 1e-6
            env.step(self.env.action_space())
            assert abs(env._attention_budget._current_budget - 5) <= 1e-6
            env.step(self.env.action_space())
            assert abs(env._attention_budget._current_budget - 5) <= 1e-6

    def test_alarm_in_legal_action_ok(self):
        """I test the budget is properly updated when the action is legal and non ambiguous"""
        act = self.env.action_space()
        act.raise_alarm = [0]
        self.env.step(act)
        assert abs(self.env._attention_budget._current_budget - 2) <= 1e-6

    def test_reset_ok(self):
        self.env.step(self.env.action_space())
        assert (
            abs(self.env._attention_budget._current_budget - (3 + 1.0 / (12.0 * 8.0)))
            <= 1e-6
        )
        self.env.reset()
        assert abs(self.env._attention_budget._current_budget - 3) <= 1e-6

    def test_illegal_action(self):
        """illegal action should not modify the alarm budget"""
        th_budget = 3.0
        act = self.env.action_space()
        arr = 1 * act.set_bus
        arr[:12] = 1
        act.set_bus = arr
        obs, reward, done, info = self.env.step(act)
        assert info["is_illegal"]
        assert abs(self.env._attention_budget._current_budget - th_budget) <= 1e-6
        assert abs(self.env._attention_budget._current_budget - th_budget) <= 1e-6

        act = self.env.action_space()
        arr = 1 * act.set_bus
        arr[:12] = 1
        act.set_bus = arr
        act.raise_alarm = [0]
        obs, reward, done, info = self.env.step(act)
        assert info["is_illegal"]
        assert abs(self.env._attention_budget._current_budget - th_budget) <= 1e-6
        assert abs(self.env._attention_budget._current_budget - th_budget) <= 1e-6

    def test_ambiguous_action(self):
        """ambiguous action should not modify the alarm budget"""
        th_budget = 3.0
        act = self.env.action_space()
        act.set_bus = [(0, 1)]
        act.change_bus = [0]
        obs, reward, done, info = self.env.step(act)
        assert info["is_ambiguous"]
        assert abs(self.env._attention_budget._current_budget - th_budget) <= 1e-6

        act = self.env.action_space()
        act.set_bus = [(0, 1)]
        act.change_bus = [0]
        act.raise_alarm = [0]
        obs, reward, done, info = self.env.step(act)
        assert info["is_ambiguous"]
        assert abs(self.env._attention_budget._current_budget - th_budget) <= 1e-6

    def test_alarm_obs_noalarm(self):
        """test the observation is behaving correctly concerning the alarm part, when i don't send alarms"""
        obs = self.env.reset()
        assert abs(self.env._attention_budget._current_budget - 3.0) <= 1e-6
        assert abs(obs.attention_budget - 3.0) <= 1e-6
        obs, reward, done, info = self.env.step(self.env.action_space())
        nb_th = 3 + 1.0 / (12.0 * 8.0)
        assert abs(self.env._attention_budget._current_budget - nb_th) <= 1e-6
        assert abs(obs.attention_budget - nb_th) <= 1e-6
        assert obs.time_since_last_alarm == -1

    def test_alarm_obs_whenalarm(self):
        """test the observation is behaving correctly concerning the alarm part, when i send alarms"""
        act = self.env.action_space()
        act.raise_alarm = [0]
        obs, reward, done, info = self.env.step(act)
        nb_th = 2
        assert abs(self.env._attention_budget._current_budget - nb_th) <= 1e-6
        assert abs(obs.attention_budget - nb_th) <= 1e-6
        assert obs.time_since_last_alarm == 0
        assert np.all(obs.last_alarm == [1, -1, -1])

        obs, reward, done, info = self.env.step(self.env.action_space())
        nb_th += 1.0 / (12.0 * 8.0)
        assert abs(self.env._attention_budget._current_budget - nb_th) <= 1e-6
        assert abs(obs.attention_budget - nb_th) <= 1e-6
        assert obs.time_since_last_alarm == 1
        assert np.all(obs.last_alarm == [1, -1, -1])

        obs = self.env.reset()
        nb_th = 3
        assert abs(self.env._attention_budget._current_budget - nb_th) <= 1e-6
        assert abs(obs.attention_budget - nb_th) <= 1e-6
        assert obs.time_since_last_alarm == -1
        assert np.all(obs.last_alarm == [-1, -1, -1])

    def test_simulate_act_ok(self):
        """test the attention budget when simulating an ok action"""
        obs = self.env.reset()
        act = self.env.action_space()
        act.raise_alarm = [0]
        act2 = self.env.action_space()
        act2.raise_alarm = [1]

        # i simulate no action
        sim_obs, *_ = obs.simulate(self.env.action_space())
        nb_th = 3 + 1.0 / (12.0 * 8.0)
        assert abs(sim_obs.attention_budget - nb_th) <= 1e-6
        assert sim_obs.time_since_last_alarm == -1
        assert np.all(sim_obs.last_alarm == [-1, -1, -1])

        # i simulate an action, this should work as for step, if i do no actions
        sim_obs, *_ = obs.simulate(act)
        nb_th = 2
        assert abs(sim_obs.attention_budget - nb_th) <= 1e-6
        assert sim_obs.time_since_last_alarm == 0
        assert np.all(sim_obs.last_alarm == [1, -1, -1])

        # i simulate no action, this should remove the previous stuff and work
        sim_obs, *_ = obs.simulate(self.env.action_space())
        nb_th = 3 + 1.0 / (12.0 * 8.0)
        assert abs(sim_obs.attention_budget - nb_th) <= 1e-6
        assert sim_obs.time_since_last_alarm == -1
        assert np.all(sim_obs.last_alarm == [-1, -1, -1])

        # i do a step and check now
        obs, *_ = self.env.step(act)

        sim_obs, *_ = obs.simulate(self.env.action_space())
        nb_th = 2 + 1.0 / (12.0 * 8.0)
        assert abs(sim_obs.attention_budget - nb_th) <= 1e-6
        assert sim_obs.time_since_last_alarm == 1
        assert np.all(sim_obs.last_alarm == [1, -1, -1])

        # i simulate an action, this should work as for step, if i do no actions
        sim_obs, *_ = obs.simulate(act2)
        nb_th = 1
        assert abs(sim_obs.attention_budget - nb_th) <= 1e-6
        assert sim_obs.time_since_last_alarm == 0
        assert np.all(sim_obs.last_alarm == [1, 2, -1])

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
        assert reward == -1
        assert not obs.was_alarm_used_after_game_over

    def test_reward_no_alarm(self):
        """test that i don't get any points if i don't send any alarm"""
        # FYI parrallel lines:
        # 48, 49  || 18, 19  || 27, 28 || 37, 38

        # game not due to line disconnection, but no alarm => no points
        self._aux_trigger_cascading_failure()
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert done
        assert reward == -1
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
        assert reward == 0.375
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
        assert reward == 0.5
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
        assert reward == -1
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

    def test_runner(self):
        """test i can create properly a runner"""
        runner = Runner(**self.env.get_params_for_runner())

        # normal run
        res = runner.run(nb_episode=1, nb_process=1, max_iter=self.max_iter)
        assert res[0][-1] == 10
        assert res[0][-2] == 10
        assert res[0][-3] == 1.0

        # run + episode data
        with tempfile.TemporaryDirectory() as f:
            res = runner.run(
                nb_episode=1, nb_process=1, max_iter=self.max_iter, path_save=f
            )
            ep_dat = EpisodeData.from_disk(agent_path=f, name=res[0][1])
            assert len(ep_dat) == 10
            assert ep_dat.observations[0].attention_budget == 3
            assert ep_dat.observations[1].attention_budget == 3 + 1.0 / (12.0 * 8.0)

    def test_kwargs(self):
        """test the get_kwargs function properly foward the attention budget"""
        env2 = Environment(**self.env.get_kwargs())
        assert env2._has_attention_budget
        assert env2._kwargs_attention_budget == self.default_kwargs_att_budget
        assert env2._attention_budget_cls == LinearAttentionBudget
        obs = env2.reset()
        assert obs.attention_budget == 3
        obs, reward, done, info = env2.step(env2.action_space())
        assert obs.attention_budget == 3 + 1.0 / (12.0 * 8.0)

    def test_simulate(self):
        """issue reported during icaps 2021"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = make("l2rpn_icaps_2021", test=True, reward_class=_AlarmScore)
        env.set_thermal_limit(
            [
                20,
                70,
                36.049267,
                43.361996,
                407.20905,
                42.96296,
                23.125486,
                7.005345,
                61.224003,
                18.283638,
                20.992632,
                89.384026,
                117.01148,
                62.883495,
                44.568665,
                29.756845,
                14.604381,
                28.99635,
                124.59952,
                124.59952,
                38.46957,
                48.00529,
                112.23501,
                139.56854,
                57.25149,
                35.785202,
                31.468952,
                98.922386,
                97.78254,
                10.58541,
                7.2501163,
                34.89438,
                66.21333,
                89.454895,
                40.088715,
                59.50673,
                54.07072,
                47.005745,
                49.29639,
                60.19898,
                98.318146,
                110.93459,
                178.60854,
                48.504723,
                9.022086,
                197.42432,
                174.3434,
                295.6653,
                149.95523,
                149.95523,
                50.128273,
                31.93147,
                74.32939,
                54.26264,
                41.730865,
                238.96637,
                197.42432,
                113.98372,
                413.98587,
            ]
        )
        env.seed(0)
        _ = env.reset()
        # it crashed
        obs, *_ = env.step(env.action_space())
        obs, *_ = env.step(env.action_space())
        alarm_act = env.action_space()
        alarm_act.raise_alarm = [0]
        obs, reward, done, info = env.step(alarm_act)
        assert not done
        # next step there is a game over due to
        sim_obs, sim_r, sim_done, sim_info = obs.simulate(env.action_space())
        assert sim_done


if __name__ == "__main__":
    unittest.main()
