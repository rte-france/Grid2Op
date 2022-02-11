# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from importlib_metadata import warnings
import numpy as np
import grid2op
from grid2op.Parameters import Parameters
import unittest


import pdb


class TestExtremeCurtail(unittest.TestCase):
    def setUp(self) -> None:
        self.env_name = "l2rpn_icaps_2021_small"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(self.env_name,
                                    data_feeding_kwargs={"max_iter": 10})

        # retrieve the reference values, without curtailment
        self.env.seed(0)
        self.env.set_id(0)
        self.obs_ref = self.env.reset()
        self.obs1_ref , *_ = self.env.step(self.env.action_space())
        self.obs2_ref , *_ = self.env.step(self.env.action_space())
        self.obs3_ref , *_ = self.env.step(self.env.action_space())
        self.obs4_ref , *_ = self.env.step(self.env.action_space())
        self.gen_th = [self.obs_ref.gen_p, self.obs1_ref.gen_p, self.obs2_ref.gen_p, self.obs3_ref.gen_p]
        
        self.curtail_ok = self.env.action_space({"curtail": [(el, 0.64) for el in np.where(self.env.gen_renewable)[0]]})
        self.curtail_ok_if_all_on = self.env.action_space({"curtail": [(el, 0.32) for el in np.where(self.env.gen_renewable)[0]]})
        self.curtail_ko = self.env.action_space({"curtail": [(el, 0.16) for el in np.where(self.env.gen_renewable)[0]]})
        self.all_zero = self.env.action_space({"curtail": [(el, 0.0) for el in np.where(self.env.gen_renewable)[0]]})

    def test_curtail_ok(self):
        """test that the env can automatically turn on all generators to prevent issues if curtailment is too strong
        new in grid2Op version 1.6.6"""
        self.env.seed(0)
        self.env.set_id(0)
        obs = self.env.reset()
        act = self.curtail_ok
        obs, reward, done, info = self.env.step(act)
        assert not done
        assert np.any(obs.gen_p[obs.gen_redispatchable] == 0.)
        
    def test_fix_curtail(self):
        """test that the env can automatically turn on all generators to prevent issues if curtailment is too strong
        new in grid2Op version 1.6.6"""
        self.env.seed(0)
        self.env.set_id(0)
        obs = self.env.reset()
        act = self.curtail_ok_if_all_on
        obs, reward, done, info = self.env.step(act)
        assert not done
        assert np.all(obs.gen_p[obs.gen_redispatchable] > 0.)
        
    def test_curtail_fail(self):
        """test that the env fails if the parameters is set to LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = False"
        default behaviour and only possible behaviour is grid2op <= 1.6.5"""
        self.env.seed(0)
        self.env.set_id(0)
        obs = self.env.reset()
        assert not self.env.parameters.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION
        act = self.curtail_ko
        obs, reward, done, info = self.env.step(act)
        assert done
    
    def test_curtail_dont_fail(self):
        """when setting the parameters to LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = True, 
        the env does not faile anymore (as opposed to test_curtail_fail)"""
        param = self.env.parameters
        param.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = True
        self.env.change_parameters(param)
        self.env.seed(0)
        self.env.set_id(0)
        obs = self.env.reset()
        act = self.curtail_ko
        obs, reward, done, info = self.env.step(act)
        assert not done
        assert np.all(obs.gen_p[obs.gen_redispatchable] > 0.)
        # the curtailment should be limited (so higher that originally)
        gen_part = self.env.gen_renewable & (obs.gen_p > 0.)
        assert np.all(obs.gen_p[gen_part] / obs.gen_pmax[gen_part] > act.curtail[gen_part])
        
    def test_set_back_to_normal(self):
        """test that the curtailment setpoint, once enough time has passed is achieved"""
        param = self.env.parameters
        param.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = True
        self.env.change_parameters(param)
        self.env.seed(0)
        self.env.set_id(0)
        obs = self.env.reset()
        act = self.curtail_ko
        obs, reward, done, info = self.env.step(act)
        assert not done
        gen_part = self.env.gen_renewable & (obs.gen_p > 0.)
        assert np.all(obs.gen_p[gen_part] / obs.gen_pmax[gen_part] > act.curtail[gen_part])
        obs1, reward, done, info = self.env.step(self.env.action_space())
        assert not done
        gen_part = self.env.gen_renewable & (obs1.gen_p > 0.)
        assert np.all(obs1.curtailment_limit[gen_part] == obs1.gen_p[gen_part] / obs1.gen_pmax[gen_part])
        
        # TODO compare with the ref observations !
        
    def test_set_back_to_normal_2(self):
        """test that the curtailment setpoint, once enough time has passed is achieved
        enough time should be 3 steps here"""
        param = self.env.parameters
        param.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = True
        self.env.change_parameters(param)
        self.env.seed(0)
        self.env.set_id(0)
        obs = self.env.reset()
        act = self.all_zero
        print("first act")
        obs, reward, done, info = self.env.step(act)
        assert not done, "env should not have diverge at first acction"
        print("second act (do nothing)")
        obs1, reward, done, info = self.env.step(self.env.action_space())
        pdb.set_trace()
        info["exception"]
        assert not done, "env should not have diverge after first do nothing"
        # gen_part = self.env.gen_renewable & (obs.gen_p > 0.)
        # assert np.all(obs.gen_p[gen_part] / obs.gen_pmax[gen_part] > act.curtail[gen_part])
        # gen_part = self.env.gen_renewable & (obs1.gen_p > 0.)
        # assert np.all(obs1.curtailment_limit[gen_part] == obs1.gen_p[gen_part] / obs1.gen_pmax[gen_part])
        
if __name__ == "__main__":
    unittest.main()
    
# # print(info["exception"])

# # first test: continue to decrease
# param = Parameters()
# param.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = True
# env.change_parameters(param)
# env.set_id(0)
# obs = env.reset()
# act = env.action_space({"curtail": [(el, 0.16) for el in np.where(env.gen_renewable)[0]]})
# obs0, reward, done, info = env.step(act)
# assert not done
# print("doing nothing 1")
# obs1, reward, done, info = env.step(env.action_space())
# assert not done
# print("doing nothing 2")
# obs2, reward, done, info = env.step(env.action_space())
# assert not done
# gen_part = env.gen_renewable & (obs2.gen_p > 0.)
# assert np.all(obs2.curtailment_limit[gen_part] == obs2.curtailment[gen_part])


# # second test: decrease rapidly (too much), then increase

# # TODO : next : symmetric! !