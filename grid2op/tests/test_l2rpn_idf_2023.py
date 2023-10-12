# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.


import grid2op
from grid2op.gym_compat import GymEnv, BoxGymActSpace, BoxGymObsSpace, DiscreteActSpace, MultiDiscreteActSpace
from grid2op.l2rpn_utils import ActionIDF2023, ObservationIDF2023
from grid2op.Opponent import GeometricOpponentMultiArea
from grid2op.Reward import AlertReward
import unittest
import warnings
import numpy as np

import pdb

class TestL2RPNIDF2023Tester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # this needs to be tested with pandapower backend
            self.env = grid2op.make("l2rpn_idf_2023", test=True, _add_to_name=type(self).__name__)
        self.env.seed(0)
        self.env.set_id(0)
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def legal_action_2subs(self):
        act12 = self.env.action_space({"set_bus": {"substations_id": [(3, (1, 2, 1, 2, 1)), (33, (1, 2, 1, 2, 1, 2))]}})
        act23 = self.env.action_space({"set_bus": {"substations_id": [(33, (1, 2, 1, 2, 1, 2)), (67, (1, 2, 1, 2))]}})
        act13 = self.env.action_space({"set_bus": {"substations_id": [(3, (1, 2, 1, 2, 1)), (67, (1, 2, 1, 2))]}})
        obs, reward, done, info = self.env.step(act12)
        assert not info["is_illegal"]
        self.env.reset()
        
        obs, reward, done, info = self.env.step(act13)
        assert not info["is_illegal"]
        self.env.reset()
        
        obs, reward, done, info = self.env.step(act23)
        assert not info["is_illegal"]
        self.env.reset()

    def test_illegal_action_2subs(self):
        # illegal actions
        act11 = self.env.action_space({"set_bus": {"substations_id": [(3, (1, 2, 1, 2, 1)), (4, (1, 2, 1, 2, 1))]}})
        act22 = self.env.action_space({"set_bus": {"substations_id": [(33, (1, 2, 1, 2, 1, 2)), (36, (1, 2, 1, 2, 1, 2))]}})
        act33 = self.env.action_space({"set_bus": {"substations_id": [(67, (1, 2, 1, 2)), (68, (1, 2, 1, 2, 1, 2, 1)) ]}})
        obs, reward, done, info = self.env.step(act11)
        assert info["is_illegal"]
        self.env.reset()
        
        obs, reward, done, info = self.env.step(act22)
        assert info["is_illegal"]
        self.env.reset()
        
        obs, reward, done, info = self.env.step(act33)
        assert info["is_illegal"]
        self.env.reset()

    def test_legal_action_2lines(self):
        # legal actions
        act12 = self.env.action_space({"set_line_status": [(0, -1), (110, -1)]})
        act23 = self.env.action_space({"set_line_status": [(110, -1), (3, -1)]})
        act13 = self.env.action_space({"set_line_status": [(0, -1), (3, -1)]})
        obs, reward, done, info = self.env.step(act12)
        assert not info["is_illegal"]
        self.env.reset()
        
        obs, reward, done, info = self.env.step(act13)
        assert not info["is_illegal"]
        self.env.reset()
        
        obs, reward, done, info = self.env.step(act23)
        assert not info["is_illegal"]
        self.env.reset()
    
    def test_other_rewards(self):
        assert "alert" in self.env.other_rewards
        assert isinstance(self.env.other_rewards["alert"].template_reward, AlertReward)
        
    def test_illegal_action_2lines(self):
        # illegal actions
        act11 = self.env.action_space({"set_line_status": [(0, -1), (1, -1)]})
        act22 = self.env.action_space({"set_line_status": [(110, -1), (111, -1)]})
        act33 = self.env.action_space({"set_line_status": [(3, -1), (7, -1)]})
        obs, reward, done, info = self.env.step(act11)
        assert info["is_illegal"]
        self.env.reset()
        
        obs, reward, done, info = self.env.step(act22)
        assert info["is_illegal"]
        self.env.reset()
        
        obs, reward, done, info = self.env.step(act33)
        assert info["is_illegal"]
        self.env.reset()
            
    def test_to_gym(self):
        env_gym = GymEnv(self.env)
        for k in ["active_alert",
                  "attack_under_alert",
                  "time_since_last_alert",
                  "alert_duration",
                  "total_number_of_alert",
                  "time_since_last_attack",
                  "was_alert_used_after_attack"]:
            assert k in env_gym.observation_space.spaces, f"missing key {k} in obs space"
        assert "raise_alert" in env_gym.action_space.spaces, f"missing key raise_alert in act space"
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            box_act = BoxGymActSpace(self.env.action_space,
                                     attr_to_keep=(
                                         "set_line_status",
                                         "change_line_status",
                                         "set_bus",
                                         "change_bus",
                                         "redispatch",
                                         "set_storage",
                                         "curtail",
                                         "raise_alert",
                                         ))
            assert box_act.shape[0] == 1543, f'{box_act.shape[0]} vs 1543'
            box_act2 = BoxGymActSpace(self.env.action_space)
            assert box_act2.shape[0] == 69, f'{box_act2.shape[0]} vs 69'
            
            box_obs = BoxGymObsSpace(self.env.observation_space)
            assert box_obs.shape[0] == 5125, f'{box_obs.shape[0]} vs 5125'
            disc_act = DiscreteActSpace(self.env.action_space)
            assert disc_act.n == 147878, f'{disc_act.n} vs 147878'
            
            multidisc_0 = MultiDiscreteActSpace(self.env.action_space)
            assert multidisc_0.shape[0] == 1543, f'{multidisc_0.shape[0]} vs 1543'
            multidisc_1 = MultiDiscreteActSpace(self.env.action_space, attr_to_keep=["raise_alert"])
            assert multidisc_1.shape[0] == 22, f'{multidisc_1.shape[0]} vs 22'
            multidisc_2 = MultiDiscreteActSpace(self.env.action_space, attr_to_keep=["sub_set_bus"])
            assert multidisc_2.shape[0] == 118, f'{multidisc_2.shape[0]} vs 118'
            assert np.array_equal(multidisc_2.nvec, [    4,     4,     8,    10,    17,     4,     4,    14,     3,
                                                         1,    58,   254,     4,     4,   242,     4,    64,     6,
                                                        30,     4,     4,     4,    30,     8,     8,     4,    58,
                                                         4,     4,     9,     8,    32,     4,    30,     4,     4,
                                                        33,     5,     4,    30,     4,   114,     4,     4,    14,
                                                        14,     8,     4, 65506,     4,     8,     4,     4,   126,
                                                        14,   498,     4,     4,   506,    14,    16,    58,     3,
                                                         5,    16,    62,     4,     9,    64,   122,     5,     4,
                                                         1,     4,    32,     6,  1010,     4,     4,  1018,     3,
                                                         8,    14,     4,    62,     4,     1,     4,    64,    26,
                                                         4,   254,     4,    32,     4,    62,     4,     4,     4,
                                                      2034,     4,     4,    16,    14,    62,     8,     6,     4,
                                                         4,    16,     1,     1,    10,     4,     4,     1,     1,
                                                         4])

    
    def test_forecast_env(self):
        obs = self.env.reset()
        for_env = obs.get_forecast_env()
        assert for_env.max_episode_duration() == 13  # 12 + 1
    
    def test_correct_action_observation(self):
        """test the observation and action class"""
        obs = self.env.reset()
        act = self.env.action_space()
        assert isinstance(obs, ObservationIDF2023)
        assert isinstance(act, ActionIDF2023)
        
        assert obs.dim_alerts == 22
        assert act.dim_alerts == 22
        
        assert np.all(act.alertable_line_ids == [106,  93,  88, 162,  68, 117, 180, 160, 136, 141, 131, 121, 125,
                                                 126, 110, 154,  81,  43,  33,  37,  62,  61])

    def test_maintenance_attack(self):
        # test the attacks
        assert isinstance(self.env._oppSpace.opponent, GeometricOpponentMultiArea)
        opp = self.env._oppSpace.opponent
        assert len(opp.list_opponents) == 3
        
        line_attacked = []
        for sub_opp in opp.list_opponents:
            line_attacked += sub_opp._lines_ids
        assert np.all(line_attacked == [106,  93,  88, 162,  68, 117, 180, 160, 136, 141, 131, 121, 125,
                                        126, 110, 154,  81,  43,  33,  37,  62,  61])    
        # test the maintenance
        time_series = self.env.chronics_handler.real_data.data    
        time_series.line_to_maintenance
        assert time_series.line_to_maintenance == {'21_22_93', '93_95_43', '80_79_175', '88_91_33', '41_48_131', '62_58_180',
                                                   '26_31_106', '62_63_160', '44_45_126', '48_53_141', '34_35_110',
                                                   '74_117_81', '12_14_68', '39_41_121', '54_58_154', '17_18_88',
                                                   '91_92_37', '4_10_162', '43_44_125', '48_50_136', '29_37_117'}
    
    def test_was_alert_used_after_attack(self):
        self.env.seed(0)
        obs = self.env.reset()
        for i in range(13):
            obs, reward, done, info = self.env.step(self.env.action_space())
        act = self.env.action_space()
        obs, reward, done, info = self.env.step(act)  # an attack at this step
        assert info["opponent_attack_line"] is not None
        
        # count 12 steps
        for i in range(12):
            obs, reward, done, info = self.env.step(self.env.action_space())
        assert obs.was_alert_used_after_attack[0] == 1
    
    def test_alertreward_counted_only_once_per_attack(self):
        self.env.seed(0)
        obs = self.env.reset()
        for i in range(13):
            obs, reward, done, info = self.env.step(self.env.action_space())
        act = self.env.action_space()
        obs, reward, done, info = self.env.step(act)  # an attack at this step
        assert info["opponent_attack_line"] is not None
        
        for i in range(11):
            obs, reward, done, info = self.env.step(self.env.action_space())
            assert info["rewards"]["alert"] == 0, f"error for step {i}"
            assert obs.was_alert_used_after_attack[0] == 0
        obs, reward, done, info = self.env.step(self.env.action_space())  # end of the time window
        assert obs.was_alert_used_after_attack[0] == 1
        assert info["rewards"]["alert"] != 0
        
        for i in range(15):
            obs, reward, done, info = self.env.step(self.env.action_space())
            assert info["rewards"]["alert"] == 0, f"error for step {i}"
            assert obs.was_alert_used_after_attack[0] == 0, f"error for step {i}"
        
    def do_not_run_oom_error_test_act_space_alert(self):
        # this crashed
        all_act = self.env.action_space.get_all_unitary_alert(self.env.action_space)
        # bug is fixed but OOM error !
    
    
if __name__ == '__main__':
    unittest.main()
