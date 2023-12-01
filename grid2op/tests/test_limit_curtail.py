# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import grid2op
import numpy as np
from lightsim2grid import LightSimBackend
import pdb
import unittest


class TestLimitAction(unittest.TestCase):
    
    def _aux_reset_env(self):
        self.env.seed(self.seed_)
        self.env.set_id(self.scen_nm)
        return self.env.reset()
    
    def setUp(self) -> None:
        self.seed_ = 0
        self.scen_nm = "2050-02-14_0"

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_wcci_2022_dev",
                                    test=True,
                                    backend=LightSimBackend(),
                                    _add_to_name=type(self).__name__)
        
        self.act = self.env.action_space()
        tmp_ = np.zeros(self.env.n_gen, dtype=float) -1
        tmp_[self.env.gen_renewable] = 0.
        self.act.curtail = tmp_

        self.act_stor = self.env.action_space()
        self.act_stor.storage_p = self.act_stor.storage_max_p_absorb
        tmp_ = np.zeros(self.env.n_gen, dtype=float) -1
        tmp_[self.env.gen_renewable] = 0.17
        self.act_stor.curtail = tmp_

    def tearDown(self) -> None:
        self.env.close()
        
    def test_curtailment_limitup(self):
        """test the action is indeed "capped" when there is too much curtailment, 
        eg when the available generators could not increase their power too much 
        to compensate the fall of renewable energy.
        """
        # for curtailment:
        self._aux_reset_env()
        obs, reward, done, info = self.env.step(self.act)
        assert done
        assert info["exception"]

        obs = self._aux_reset_env()
        act2, *_ = self.act.limit_curtail_storage(obs, margin=0., do_copy=True)  # not enough "margin"
        obs, reward, done, info = self.env.step(act2)
        assert done
        assert info["exception"]

        obs = self._aux_reset_env()
        act3, *_ = self.act.limit_curtail_storage(obs, margin=15., do_copy=True)  # not enough "margin"
        obs, reward, done, info = self.env.step(act3)
        assert not done
        assert not info["exception"]

    def test_storage_limitup(self):
        """test when the storage consumption is too much for the generator to compensate"""
        # for storage (I need to add curtailment otherwise i don't have enough "juice")
        obs = self._aux_reset_env()
        obs, reward, done, info = self.env.step(self.act_stor)
        assert done
        assert info["exception"]

        obs = self._aux_reset_env()
        act4, *_ = self.act_stor.limit_curtail_storage(obs, margin=20., do_copy=True)  # not enough "margin"
        obs, reward, done, info = self.env.step(act4)
        assert done
        assert info["exception"]
    
        obs = self._aux_reset_env()
        act5, *_ = self.act_stor.limit_curtail_storage(obs, margin=25., do_copy=True)
        obs, reward, done, info = self.env.step(act5)
        assert not done
        assert not info["exception"]
    
    def aux_test_margin_increase_cut(self, action):
        obs = self._aux_reset_env()
        act5, add_curtailed_5, add_storage_5 = action.limit_curtail_storage(obs, margin=5., do_copy=True)
        act10, add_curtailed_10, add_storage_10 = action.limit_curtail_storage(obs, margin=10., do_copy=True)
        act15, add_curtailed_15, add_storage_15 = action.limit_curtail_storage(obs, margin=15., do_copy=True)
        act20, add_curtailed_20, add_storage_20 = action.limit_curtail_storage(obs, margin=20., do_copy=True)
        act25, add_curtailed_25, add_storage_25 = action.limit_curtail_storage(obs, margin=25., do_copy=True)
        act30, add_curtailed_30, add_storage_30 = action.limit_curtail_storage(obs, margin=30., do_copy=True)
        assert np.all(add_curtailed_30 >= add_curtailed_25)
        assert np.any(add_curtailed_30 > add_curtailed_25)
        assert np.all(add_curtailed_25 >= add_curtailed_20)
        assert np.any(add_curtailed_25 > add_curtailed_20)
        assert np.all(add_curtailed_20 >= add_curtailed_15)
        assert np.any(add_curtailed_20 > add_curtailed_15)
        assert np.all(add_curtailed_15 >= add_curtailed_10)
        assert np.any(add_curtailed_15 > add_curtailed_10)
        assert np.all(add_curtailed_10 >= add_curtailed_5)
        assert np.any(add_curtailed_10 > add_curtailed_5)
        if np.any(action._storage_power != 0.):
            assert np.all(-add_storage_30 >= -add_storage_25)
            assert np.any(-add_storage_30 > -add_storage_25)
            assert np.all(-add_storage_25 >= -add_storage_20)
            assert np.any(-add_storage_25 > -add_storage_20)
            assert np.all(-add_storage_20 >= -add_storage_15)
            assert np.any(-add_storage_20 > -add_storage_15)
            assert np.all(-add_storage_15 >= -add_storage_10)
            assert np.any(-add_storage_15 > -add_storage_10)
            assert np.all(-add_storage_10 >= -add_storage_5)
            assert np.any(-add_storage_10 > -add_storage_5)
    
    def test_margin_increase_cut(self):
        """test that if I increase the "margin=..." it does increase the amount of MW removed"""
        self.aux_test_margin_increase_cut(self.act)
        self.aux_test_margin_increase_cut(self.act_stor)
        
    def _aux_prep_env_for_tests_down(self):
        act0 = self.env.action_space()
        tmp_ = np.zeros(self.env.n_gen, dtype=float) -1
        tmp_[self.env.gen_renewable] = 0.15
        act0.curtail = tmp_
        
        act1 = self.env.action_space()
        tmp_ = np.zeros(self.env.n_gen, dtype=float) -1
        tmp_[self.env.gen_renewable] = 0.09
        act1.curtail = tmp_
        
        act2 = self.env.action_space()
        tmp_ = np.zeros(self.env.n_gen, dtype=float) -1
        tmp_[self.env.gen_renewable] = 0.04
        act2.curtail = tmp_
        
        self._aux_reset_env()
        obs, reward, done, info = self.env.step(act0)
        assert not done
        assert not info["exception"]
        obs, reward, done, info = self.env.step(act1)
        assert not done
        assert not info["exception"]
        obs, reward, done, info = self.env.step(act2)
        assert not done
        assert not info["exception"]
        return obs
    
    def test_curtailment_limitdown(self):
        """test the action is indeed "capped" when there is not enough curtailment, 
        eg when the available generators could not decrease their power too much 
        to compensate the increase of renewable energy.
        """
        act_too_much = self.env.action_space()
        tmp_ = np.zeros(self.env.n_gen, dtype=float) -1
        tmp_[self.env.gen_renewable] = 1.
        act_too_much.curtail = tmp_

        # for curtailment:
        self._aux_prep_env_for_tests_down()
        obs, reward, done, info0 = self.env.step(act_too_much)  # If i do this it crashes
        assert done
        assert info0["exception"]
        
        obs = self._aux_prep_env_for_tests_down()
        act5, *_ = act_too_much.limit_curtail_storage(obs, margin=15., do_copy=True)  # not enough "margin" => it crashes
        obs, reward, done, info = self.env.step(act5)
        assert done
        assert info["exception"]
        
        obs = self._aux_prep_env_for_tests_down()
        act6, *_ = act_too_much.limit_curtail_storage(obs, margin=20., do_copy=True)  # "just enough" "margin" => it passes
        obs, reward, done, info = self.env.step(act6)
        assert not done
        assert not info["exception"]
    
    def test_storage_limitdown(self):
        """test the action is indeed "capped" when there is not enough storage
        
        
        eg when the available generators could not decrease their power too much 
        to compensate the increase of storage power (and curtailment because storage unit is too weak on its own).
        """
        act_too_much = self.env.action_space()
        tmp_ = np.zeros(self.env.n_gen, dtype=float) -1
        tmp_[self.env.gen_renewable] = 0.06
        act_too_much.storage_p = - act_too_much.storage_max_p_prod
        act_too_much.curtail = tmp_

        # for storage:
        self._aux_prep_env_for_tests_down()
        obs, reward, done, info0 = self.env.step(act_too_much)  # If i do this it crashes
        assert done
        assert info0["exception"]
        
        obs = self._aux_prep_env_for_tests_down()
        act7, *_ = act_too_much.limit_curtail_storage(obs, margin=5., do_copy=True)  # not enough "margin" => it crashes
        obs, reward, done, info = self.env.step(act7)
        assert done
        assert info["exception"]
        
        obs = self._aux_prep_env_for_tests_down()
        act8, *_ = act_too_much.limit_curtail_storage(obs, margin=10., do_copy=True)  # "just enough" "margin" => it passes
        obs, reward, done, info = self.env.step(act8)
        assert not done
        assert not info["exception"]
        
        
if __name__ == "__main__":
    unittest.main()
