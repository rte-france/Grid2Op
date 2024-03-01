# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import unittest
import warnings
import numpy as np

import grid2op
from grid2op.Backend import PandaPowerBackend
from grid2op.Action import CompleteAction
from grid2op.gym_compat import MultiDiscreteActSpace, GymEnv


class TestMultiDiscreteActSpaceOneLineChangeSet(unittest.TestCase):    
    def get_env_nm(self):
        return "educ_case14_storage"
    
    def get_reset_kwargs(self) -> dict:
        # seed has been tuned for the tests to pass
        return dict(seed=self.seed, options={"time serie id": 0})
    
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(self.get_env_nm(),
                                    backend=PandaPowerBackend(),
                                    action_class=CompleteAction,
                                    test=True,
                                    _add_to_name=type(self).__name__)
        self.seed = 0
        self.gym_env = GymEnv(self.env)
    
    def tearDown(self) -> None:
        self.env.close()
        self.gym_env.close()
        return super().tearDown()
    
    def test_kwargs_ok(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            act_space = MultiDiscreteActSpace(self.env.action_space, attr_to_keep=["one_line_set"])
        assert act_space.nvec[0] == 1 + 2 * type(self.env).n_line
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            act_space = MultiDiscreteActSpace(self.env.action_space, attr_to_keep=["one_line_change"])
        assert act_space.nvec[0] == 1 + type(self.env).n_line

    def _aux_assert_flags(self, glop_act):
        assert not glop_act._modif_alarm
        assert not glop_act._modif_alert
        assert not glop_act._modif_curtailment
        assert not glop_act._modif_storage
        assert not glop_act._modif_redispatch
        assert not glop_act._modif_set_bus
        assert not glop_act._modif_change_bus
        
    def test_action_ok_set(self):
        act_space = MultiDiscreteActSpace(self.env.action_space, attr_to_keep=["one_line_set"])
        act_space.seed(self.seed)
        for _ in range(10):
            act = act_space.sample()
            glop_act = act_space.from_gym(act)
            self._aux_assert_flags(glop_act)
            assert not glop_act._modif_change_status
            lines_, subs_ = glop_act.get_topological_impact()
            assert (~subs_).all()
            if act[0] >= 1:  # 0 is for do nothing
                # 1 is connect line 0, 2 is disconnect line 0
                # 3 is connect line 1, etc.
                assert glop_act._modif_set_status
                assert lines_[(act[0]- 1)  // 2 ]
            else:
                assert not glop_act._modif_set_status
                assert (~lines_).all()
        
        glop_act = act_space.from_gym(np.array([0]))
        lines_, subs_ = glop_act.get_topological_impact()
        assert (~subs_).all()
        assert (~lines_).all()
        self._aux_assert_flags(glop_act)
        assert not glop_act._modif_change_status
        assert not glop_act._modif_set_status
            
        for i in range(1, 2 * type(self.env).n_line + 1):
            glop_act = act_space.from_gym(np.array([i]))
            lines_, subs_ = glop_act.get_topological_impact()
            assert (~subs_).all()
            self._aux_assert_flags(glop_act)
            assert not glop_act._modif_change_status
            assert glop_act._modif_set_status
            l_id = (i- 1)  // 2 
            assert lines_[l_id]
            assert glop_act._set_line_status[l_id] == ((i-1) % 2 == 0) * 2 - 1, f"error for {i}"
                
    def test_action_ok_change(self):
        act_space = MultiDiscreteActSpace(self.env.action_space, attr_to_keep=["one_line_change"])
        act_space.seed(self.seed)
        for _ in range(10):
            act = act_space.sample()
            glop_act = act_space.from_gym(act)
            self._aux_assert_flags(glop_act)
            assert not glop_act._modif_set_status
            lines_, subs_ = glop_act.get_topological_impact()
            assert (~subs_).all()
            if act[0] >= 1:  # 0 is for do nothing
                assert glop_act._modif_change_status
                assert lines_[(act[0]- 1)]
            else:
                assert (~lines_).all()
                assert not glop_act._modif_change_status

        glop_act = act_space.from_gym(np.array([0]))
        lines_, subs_ = glop_act.get_topological_impact()
        assert (~subs_).all()
        assert (~lines_).all()
        self._aux_assert_flags(glop_act)
        assert not glop_act._modif_change_status
        assert not glop_act._modif_set_status
            
        for i in range(1, type(self.env).n_line + 1):
            glop_act = act_space.from_gym(np.array([i]))
            lines_, subs_ = glop_act.get_topological_impact()
            assert (~subs_).all()
            self._aux_assert_flags(glop_act)
            assert glop_act._modif_change_status
            assert not glop_act._modif_set_status
            l_id = (i- 1)
            assert lines_[l_id]
            assert glop_act._switch_line_status[l_id], f"error for {i}"        
        
    def test_can_combine_topo_line_set(self):
        act_space = MultiDiscreteActSpace(self.env.action_space,
                                          attr_to_keep=["one_line_set", "one_sub_set"])
        act_space.seed(self.seed)
        for _ in range(10):
            act = act_space.sample()
            glop_act = act_space.from_gym(act)
            lines_, subs_ = glop_act.get_topological_impact()
            if act[0]:
                assert lines_.sum() == 1
            if act[1]:
                assert subs_.sum() == 1     
        
    def test_can_combine_topo_line_change(self):
        act_space = MultiDiscreteActSpace(self.env.action_space,
                                          attr_to_keep=["one_line_change", "one_sub_change"])
        act_space.seed(self.seed)
        for _ in range(10):
            act = act_space.sample()
            glop_act = act_space.from_gym(act)
            lines_, subs_ = glop_act.get_topological_impact()
            if act[0]:
                assert lines_.sum() == 1
            if act[1]:
                assert subs_.sum() == 1     
            
            
if __name__ == "__main__":
    unittest.main()
