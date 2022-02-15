# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.
import warnings
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
        self.all_one = self.env.action_space({"curtail": [(el, 1.0) for el in np.where(self.env.gen_renewable)[0]]})

    def _aux_test_gen(self, obsbefore, obsafter, tol=1e-4):
        assert np.all(obsbefore.gen_p <= obsbefore.gen_pmax + tol)
        assert np.all(obsbefore.gen_p >= obsbefore.gen_pmin - tol)
        assert np.all(obsafter.gen_p <= obsafter.gen_pmax + tol)
        assert np.all(obsafter.gen_p >= obsafter.gen_pmin - tol)
        dispatchable = obsbefore.gen_redispatchable
        dispatchable[-1] = False  # we remove the slack... !
        assert np.all((obsafter.gen_p[dispatchable] - obsbefore.gen_p[dispatchable]) <= obsbefore.gen_max_ramp_up[dispatchable] + tol)
        assert np.all((obsafter.gen_p[dispatchable] - obsbefore.gen_p[dispatchable]) >= -obsbefore.gen_max_ramp_down[dispatchable] - tol)
        
        # check the slack does not violate too much the constraints (this would indicate an error in the 
        # amount of power that needs to be redispatched)
        slack = -1
        slack_variation = (obsafter.gen_p[slack] - obsbefore.gen_p[slack])
        assert slack_variation <= obsbefore.gen_max_ramp_up[slack] + 1, f"{slack_variation = :.2f} above the ramp up"
        assert slack_variation >= -obsbefore.gen_max_ramp_down[slack] - 1, f"{slack_variation = :.2f} below the ramp down"
        
    def test_curtail_ok(self):
        """test that the env can automatically turn on all generators to prevent issues if curtailment is too strong
        new in grid2Op version 1.6.6"""
        self.env.seed(0)
        self.env.set_id(0)
        obs = self.env.reset()
        act = self.curtail_ok
        obs1, reward, done, info = self.env.step(act)
        assert not done
        # not too much losses (which would indicate errors in the computation of the total amount to dispatch)
        assert np.all(np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)) <= 1
        assert np.any(obs1.gen_p[obs.gen_redispatchable] == 0.)
        self._aux_test_gen(obs, obs1)
    
    def _aux_compare_with_ref(self, env, obs, obs_ref, tol=1e-4):
        slack_id = -1
        # slack does not absorb too much
        assert np.all(np.abs(env._gen_activeprod_t_redisp[:slack_id] - obs.gen_p[:slack_id]) <= tol)
        # power for each generator is the same (when curtailment taken into account)
        assert np.all(np.abs(obs.gen_p[:slack_id] + obs.curtailment_mw[:slack_id] - obs.actual_dispatch[:slack_id] - obs_ref.gen_p[:slack_id]) <= tol)
        # check the slack
        assert abs(obs.gen_p[slack_id] - obs.actual_dispatch[slack_id] -  obs_ref.gen_p[slack_id]) <= 1.
        
    def test_fix_curtail(self):
        """test that the env can automatically turn on all generators to prevent issues if curtailment is too strong
        new in grid2Op version 1.6.6"""
        self.env.seed(0)
        self.env.set_id(0)
        obs = self.env.reset()
        act = self.curtail_ok_if_all_on
        obs1, reward, done, info = self.env.step(act)
        assert not done
        # not too much losses (which would indicate errors in the computation of the total amount to dispatch)
        assert np.all(np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)) <= 1
        assert np.all(obs1.gen_p[obs.gen_redispatchable] > 0.)
        self._aux_test_gen(obs, obs1)
        self._aux_compare_with_ref(self.env, obs1, self.obs1_ref)
        
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
        obs1, reward, done, info = self.env.step(act)
        assert not done
        # not too much losses (which would indicate errors in the computation of the total amount to dispatch)
        assert np.all(np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)) <= 1
        assert np.all(obs1.gen_p[obs1.gen_redispatchable] > 0.)
        # the curtailment should be limited (so higher that originally)
        gen_part = self.env.gen_renewable & (obs1.gen_p > 0.)
        assert np.all(obs1.gen_p[gen_part] / obs1.gen_pmax[gen_part] > act.curtail[gen_part])
        self._aux_test_gen(obs, obs1)
        self._aux_compare_with_ref(self.env, obs1, self.obs1_ref)
        
    def test_set_back_to_normal(self):
        """test that the curtailment setpoint, once enough time has passed is achieved"""
        param = self.env.parameters
        param.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = True
        self.env.change_parameters(param)
        self.env.seed(0)
        self.env.set_id(0)
        obs = self.env.reset()
        act = self.curtail_ko
        
        # first action would break the grid, it is limited
        obs0, reward, done, info = self.env.step(act)
        assert not done
        # not too much losses (which would indicate errors in the computation of the total amount to dispatch)
        assert np.all(np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)) <= 1
        gen_part = self.env.gen_renewable & (obs0.gen_p > 0.)
        assert np.all(obs0.gen_p[gen_part] / obs0.gen_pmax[gen_part] > act.curtail[gen_part])
        assert np.all(obs0.gen_p >= 0.)
        assert np.all(obs0.gen_p_before_curtail[self.env.gen_renewable] == self.obs1_ref.gen_p[self.env.gen_renewable])
        self._aux_test_gen(obs, obs0)
        self._aux_compare_with_ref(self.env, obs0, self.obs1_ref)
        
        # next step = the action can be completely made, it does it
        obs1, reward, done, info = self.env.step(self.env.action_space())
        assert not done
        # not too much losses (which would indicate errors in the computation of the total amount to dispatch)
        assert np.all(np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)) <= 1
        gen_part = self.env.gen_renewable & (obs1.gen_p > 0.)
        assert np.all(obs1.gen_p >= 0.)
        assert np.all(obs1.curtailment_limit[gen_part] == obs1.gen_p[gen_part] / obs1.gen_pmax[gen_part])
        assert np.all(obs1.gen_p_before_curtail[self.env.gen_renewable] == self.obs2_ref.gen_p[self.env.gen_renewable])
        self._aux_test_gen(obs0, obs1)
        self._aux_compare_with_ref(self.env, obs1, self.obs2_ref)
        
        # make sure it stays at the sepoint
        obs2, reward, done, info = self.env.step(self.env.action_space())
        assert not done
        # not too much losses (which would indicate errors in the computation of the total amount to dispatch)
        assert np.all(np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)) <= 1
        gen_part = self.env.gen_renewable & (obs2.gen_p > 0.)
        assert np.all(obs2.gen_p >= 0.)
        assert np.all(obs2.curtailment_limit[gen_part] == obs2.gen_p[gen_part] / obs2.gen_pmax[gen_part])
        assert np.all(obs2.gen_p_before_curtail[self.env.gen_renewable] == self.obs3_ref.gen_p[self.env.gen_renewable])
        self._aux_test_gen(obs1, obs2)
        self._aux_compare_with_ref(self.env, obs2, self.obs3_ref)
        
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
        
        # first action would break the grid, it is limited
        obs0, reward, done, info = self.env.step(act)
        assert not done, "env should not have diverge at first acction"
        # not too much losses (which would indicate errors in the computation of the total amount to dispatch)
        assert np.all(np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)) <= 1
        gen_part = self.env.gen_renewable & (obs0.gen_p > 0.)
        assert np.all(obs0.gen_p[gen_part] / obs0.gen_pmax[gen_part] > act.curtail[gen_part])
        assert np.all(obs0.gen_p_before_curtail[self.env.gen_renewable] == self.obs1_ref.gen_p[self.env.gen_renewable])
        self._aux_test_gen(obs, obs0)
        self._aux_compare_with_ref(self.env, obs0, self.obs1_ref)
        
        # next step = we got close to the setpoint, but still not there yet
        obs1, reward, done, info = self.env.step(self.env.action_space())
        assert not done, "env should not have diverge after first do nothing"
        # not too much losses (which would indicate errors in the computation of the total amount to dispatch)
        assert np.all(np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)) <= 1
        # I got close to the setpoint
        assert np.all(obs1.gen_p[gen_part] / obs1.gen_pmax[gen_part] < obs.gen_p[gen_part] / obs.gen_pmax[gen_part])
        # I am still not at the setpoint
        gen_part = self.env.gen_renewable & (obs1.gen_p > 0.)
        assert np.all(obs1.gen_p[gen_part] / obs1.gen_pmax[gen_part] > act.curtail[gen_part])
        assert np.all(obs1.gen_p_before_curtail[self.env.gen_renewable] == self.obs2_ref.gen_p[self.env.gen_renewable])
        self._aux_test_gen(obs0, obs1)
        self._aux_compare_with_ref(self.env, obs1, self.obs2_ref)
        
        # next step = the action can be completely made, it does it
        obs2, reward, done, info = self.env.step(self.env.action_space())
        assert not done, "env should not have diverge after second do nothing"
        # not too much losses (which would indicate errors in the computation of the total amount to dispatch)
        assert np.all(np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)) <= 1
        gen_part = self.env.gen_renewable & (obs2.gen_p > 0.)
        assert np.all(obs2.gen_p >= 0.)
        assert np.all(obs2.gen_p[gen_part] / obs2.gen_pmax[gen_part] < obs1.gen_p[gen_part] / obs1.gen_pmax[gen_part])
        assert np.all(obs2.gen_p[gen_part] / obs2.gen_pmax[gen_part] == act.curtail[gen_part])
        assert np.all(obs2.gen_p_before_curtail[self.env.gen_renewable] == self.obs3_ref.gen_p[self.env.gen_renewable])
        self._aux_test_gen(obs1, obs2)
        self._aux_compare_with_ref(self.env, obs2, self.obs3_ref)
        
        # make sure it stays at the sepoint
        obs3, reward, done, info = self.env.step(self.env.action_space())
        assert not done,  "env should not have diverge after third do nothing"
        # not too much losses (which would indicate errors in the computation of the total amount to dispatch)
        assert np.all(np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)) <= 1
        gen_part = self.env.gen_renewable & (obs3.gen_p > 0.)
        assert np.all(obs3.gen_p >= 0.)
        assert np.all(obs3.curtailment_limit[gen_part] == obs3.gen_p[gen_part] / obs3.gen_pmax[gen_part])
        assert np.all(obs3.gen_p_before_curtail[self.env.gen_renewable] == self.obs4_ref.gen_p[self.env.gen_renewable])
        self._aux_test_gen(obs2, obs3)
        self._aux_compare_with_ref(self.env, obs3,self.obs4_ref)
        
    def test_down_then_up(self):
        """test that i can curtail down to the setpoint, then up again until the curtailment is canceled
        """
        param = self.env.parameters
        param.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = True
        self.env.change_parameters(param)
        self.env.seed(0)
        self.env.set_id(0)
        obs = self.env.reset()
        act = self.curtail_ko
        # we first do as in test_set_back_to_normal
        obs0, reward, done, info = self.env.step(act)
        assert not done
        obs1, reward, done, info = self.env.step(self.env.action_space())
        assert not done
        
        # now the setpoint is reached, let's increase "at once" (it is possible without violating anything)
        obs2, reward, done, info = self.env.step(self.all_one)
        assert not done
        # not too much losses (which would indicate errors in the computation of the total amount to dispatch)
        assert np.all(np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)) <= 1
        assert np.all(obs2.gen_p >= 0.)
        assert np.all(obs2.gen_p[self.env.gen_renewable] >= obs1.gen_p[self.env.gen_renewable])
        self._aux_test_gen(obs1, obs2)
        self._aux_compare_with_ref(self.env, obs2,self.obs3_ref)
        
        # re increase to check that the setpoint is correct
        obs3, reward, done, info = self.env.step(self.env.action_space())
        assert not done
        # not too much losses (which would indicate errors in the computation of the total amount to dispatch)
        assert np.all(np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)) <= 1
        assert np.all(obs2.gen_p >= 0.)
        assert np.all(obs2.gen_p[self.env.gen_renewable] >= obs1.gen_p[self.env.gen_renewable])
        self._aux_test_gen(obs2, obs3)
        self._aux_compare_with_ref(self.env, obs3,self.obs4_ref)
        gen_part = self.env.gen_renewable & (obs3.gen_p > 0.)
        # generator produce less than pmax
        assert np.all(obs3.curtailment_limit[gen_part] <= obs3.gen_pmax[gen_part])
        # no more curtailment, so productions increase
        assert np.all(obs3.gen_p[self.env.gen_renewable] >= obs2.gen_p[self.env.gen_renewable])
        # information of generation without curtailment is correct
        assert np.all(obs3.gen_p_before_curtail[self.env.gen_renewable] == self.obs4_ref.gen_p[self.env.gen_renewable])
        # setpoint is matched
        assert np.all(obs3.gen_p_before_curtail[self.env.gen_renewable] == obs3.gen_p[self.env.gen_renewable])
        
    def test_down_then_up_2(self):
        """test that i can curtail down to the setpoint, then up again until the curtailment is canceled
        but for a more complex case
        """
        param = self.env.parameters
        param.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = True
        self.env.change_parameters(param)
        self.env.seed(0)
        self.env.set_id(0)
        obs = self.env.reset()
        act = self.all_zero
        # we first do as in test_set_back_to_normal_2
        obs0, reward, done, info = self.env.step(act)
        assert not done
        obs1, reward, done, info = self.env.step(self.env.action_space())
        assert not done
        obs2, reward, done, info = self.env.step(self.env.action_space())
        assert not done
        
        # now the setpoint is reached, let's increase "at once" (it should break a limit => curtailment will be limited)
        obs3, reward, done, info = self.env.step(self.all_one)
        assert not done
        # not too much losses (which would indicate errors in the computation of the total amount to dispatch)
        assert np.all(np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)) <= 1
        obs4, reward, done, info = self.env.step(self.all_one)
        assert not done
        self._aux_test_gen(obs3, obs4)
        
# TODO test with simulate !!!!
     
if __name__ == "__main__":
    unittest.main()
