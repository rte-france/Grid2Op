# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import os
import numpy as np
import grid2op
from grid2op.Action.playableAction import PlayableAction
from grid2op.tests.helper_path_test import *
import unittest

import pdb

"""snippet for the "debug" stuff

if hasattr(self, "_debug") and self._debug:
    import pdb
    pdb.set_trace()
"""


class TestExtremeCurtail(unittest.TestCase):
    def setUp(self) -> None:
        self.env_name = os.path.join(PATH_DATA_TEST, "l2rpn_icaps_2021_small_test")
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                self.env_name,
                test=True,
                _add_to_name=type(self).__name__
            )

        # retrieve the reference values, without curtailment
        self.env.seed(0)
        self.env.set_id(0)
        self.obs_ref = self.env.reset()
        self.obs1_ref, *_ = self.env.step(self.env.action_space())
        self.obs2_ref, *_ = self.env.step(self.env.action_space())
        self.obs3_ref, *_ = self.env.step(self.env.action_space())
        self.obs4_ref, *_ = self.env.step(self.env.action_space())
        self.obs5_ref, *_ = self.env.step(self.env.action_space())
        self.obs6_ref, *_ = self.env.step(self.env.action_space())

        self.curtail_ok = self.env.action_space(
            {"curtail": [(el, 0.64) for el in np.where(self.env.gen_renewable)[0]]}
        )
        self.curtail_ok_if_all_on = self.env.action_space(
            {"curtail": [(el, 0.32) for el in np.where(self.env.gen_renewable)[0]]}
        )
        self.curtail_ko = self.env.action_space(
            {"curtail": [(el, 0.16) for el in np.where(self.env.gen_renewable)[0]]}
        )
        self.all_zero = self.env.action_space(
            {"curtail": [(el, 0.0) for el in np.where(self.env.gen_renewable)[0]]}
        )
        self.all_one = self.env.action_space(
            {"curtail": [(el, 1.0) for el in np.where(self.env.gen_renewable)[0]]}
        )

    @staticmethod
    def _aux_test_gen(obsbefore, obsafter, tol=1e-4, min_loss_slack=0.2):
        assert np.all(obsbefore.gen_p <= obsbefore.gen_pmax + tol)
        assert np.all(obsbefore.gen_p >= obsbefore.gen_pmin - tol)
        assert np.all(obsafter.gen_p <= obsafter.gen_pmax + tol)
        assert np.all(obsafter.gen_p >= obsafter.gen_pmin - tol)
        dispatchable = obsbefore.gen_redispatchable
        dispatchable[-1] = False  # we remove the slack... !
        assert np.all(
            (obsafter.gen_p[dispatchable] - obsbefore.gen_p[dispatchable])
            <= obsbefore.gen_max_ramp_up[dispatchable] + tol
        )
        assert np.all(
            (obsafter.gen_p[dispatchable] - obsbefore.gen_p[dispatchable])
            >= -obsbefore.gen_max_ramp_down[dispatchable] - tol
        )

        # check the slack does not violate too much the constraints (this would indicate an error in the
        # amount of power that needs to be redispatched)
        slack = -1
        slack_variation = obsafter.gen_p[slack] - obsbefore.gen_p[slack]
        loss_after = TestExtremeCurtail.aux_obs_loss(obsafter)
        loss_before = TestExtremeCurtail.aux_obs_loss(obsbefore)
        slack_tol = max(2.0 * abs(loss_after - loss_before), min_loss_slack)
        assert (
            slack_variation <= obsbefore.gen_max_ramp_up[slack] + slack_tol
        ), f"{slack_variation = :.2f}MW, way above the ramp up: {obsbefore.gen_max_ramp_up[slack]:.2f}"
        assert (
            slack_variation >= -obsbefore.gen_max_ramp_down[slack] - slack_tol
        ), f"{slack_variation = :.2f}MW, way below the ramp down: {-obsbefore.gen_max_ramp_down[slack]:.2f}"

    @staticmethod
    def _aux_compare_with_ref(env, obs, obs_ref, tol=1e-4, min_loss_slack=0.2):
        slack_id = -1
        # slack does not absorb too much
        assert np.all(
            np.abs(env._gen_activeprod_t_redisp[:slack_id] - obs.gen_p[:slack_id])
            <= tol
        )
        # power for each generator is the same (when curtailment taken into account)
        assert np.all(
            np.abs(
                obs.gen_p[:slack_id]
                + obs.curtailment_mw[:slack_id]
                - obs.actual_dispatch[:slack_id]
                - obs_ref.gen_p[:slack_id]
            )
            <= tol
        )

        # check the slack
        loss = TestExtremeCurtail.aux_obs_loss(obs)
        loss_ref = TestExtremeCurtail.aux_obs_loss(obs_ref)
        slack_tol = max(2.0 * abs(loss_ref - loss), min_loss_slack)
        assert (
            abs(
                obs.gen_p[slack_id]
                - obs.actual_dispatch[slack_id]
                - obs_ref.gen_p[slack_id]
            )
            <= slack_tol
        )

    @staticmethod
    def aux_obs_loss(obs):
        loss = np.sum(obs.gen_p) - np.sum(obs.storage_power) - np.sum(obs.load_p)
        return loss

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
        assert (
            np.all(
                np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)
            )
            <= 1
        )
        assert np.any(obs1.gen_p[obs.gen_redispatchable] == 0.0)
        self._aux_test_gen(obs, obs1)

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
        assert (
            np.all(
                np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)
            )
            <= 1
        )
        assert np.all(obs1.gen_p[obs.gen_redispatchable] > 0.0)
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
        assert (
            np.all(
                np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)
            )
            <= 1
        )
        assert np.all(obs1.gen_p[obs1.gen_redispatchable] > 0.0)
        # the curtailment should be limited (so higher that originally)
        gen_part = self.env.gen_renewable & (obs1.gen_p > 0.0)
        assert np.all(
            obs1.gen_p[gen_part] / obs1.gen_pmax[gen_part] > act.curtail[gen_part]
        )
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
        assert (
            np.all(
                np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)
            )
            <= 1
        )
        gen_part = self.env.gen_renewable & (obs0.gen_p > 0.0)
        assert np.all(
            obs0.gen_p[gen_part] / obs0.gen_pmax[gen_part] > act.curtail[gen_part]
        )
        assert np.all(obs0.gen_p >= -self.env._tol_poly)
        assert np.all(
            obs0.gen_p_before_curtail[self.env.gen_renewable]
            == self.obs1_ref.gen_p[self.env.gen_renewable]
        )
        self._aux_test_gen(obs, obs0)
        self._aux_compare_with_ref(self.env, obs0, self.obs1_ref)

        # next step = the action can be completely made, it does it
        obs1, reward, done, info = self.env.step(self.env.action_space())
        assert not done
        # not too much losses (which would indicate errors in the computation of the total amount to dispatch)
        assert (
            np.all(
                np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)
            )
            <= 1
        )
        gen_part = self.env.gen_renewable & (obs1.gen_p > 0.0)
        assert np.all(obs1.gen_p >= -self.env._tol_poly)
        assert np.all(
            obs1.curtailment_limit[gen_part]
            == obs1.gen_p[gen_part] / obs1.gen_pmax[gen_part]
        )
        assert np.all(
            obs1.gen_p_before_curtail[self.env.gen_renewable]
            == self.obs2_ref.gen_p[self.env.gen_renewable]
        )
        self._aux_test_gen(obs0, obs1)
        self._aux_compare_with_ref(self.env, obs1, self.obs2_ref)

        # make sure it stays at the sepoint
        obs2, reward, done, info = self.env.step(self.env.action_space())
        assert not done
        # not too much losses (which would indicate errors in the computation of the total amount to dispatch)
        assert (
            np.all(
                np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)
            )
            <= 1
        )
        gen_part = self.env.gen_renewable & (obs2.gen_p > 0.0)
        assert np.all(obs2.gen_p >= -self.env._tol_poly)
        assert np.all(
            obs2.curtailment_limit[gen_part]
            == obs2.gen_p[gen_part] / obs2.gen_pmax[gen_part]
        )
        assert np.all(
            obs2.gen_p_before_curtail[self.env.gen_renewable]
            == self.obs3_ref.gen_p[self.env.gen_renewable]
        )
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
        assert (
            np.all(
                np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)
            )
            <= 1
        )
        gen_part = self.env.gen_renewable & (obs0.gen_p > 0.0)
        assert np.all(
            obs0.gen_p[gen_part] / obs0.gen_pmax[gen_part] > act.curtail[gen_part]
        )
        assert np.all(
            obs0.gen_p_before_curtail[self.env.gen_renewable]
            == self.obs1_ref.gen_p[self.env.gen_renewable]
        )
        self._aux_test_gen(obs, obs0)
        self._aux_compare_with_ref(self.env, obs0, self.obs1_ref)

        # next step = we got close to the setpoint, but still not there yet
        obs1, reward, done, info = self.env.step(self.env.action_space())
        assert not done, "env should not have diverge after first do nothing"
        # not too much losses (which would indicate errors in the computation of the total amount to dispatch)
        assert (
            np.all(
                np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)
            )
            <= 1
        )
        # I got close to the setpoint
        assert np.all(
            obs1.gen_p[gen_part] / obs1.gen_pmax[gen_part]
            < obs.gen_p[gen_part] / obs.gen_pmax[gen_part]
        )
        # I am still not at the setpoint
        gen_part = self.env.gen_renewable & (obs1.gen_p > 0.0)
        assert np.all(
            obs1.gen_p[gen_part] / obs1.gen_pmax[gen_part] > act.curtail[gen_part]
        )
        assert np.all(
            obs1.gen_p_before_curtail[self.env.gen_renewable]
            == self.obs2_ref.gen_p[self.env.gen_renewable]
        )
        self._aux_test_gen(obs0, obs1)
        self._aux_compare_with_ref(self.env, obs1, self.obs2_ref)

        # next step = the action can be completely made, it does it
        obs2, reward, done, info = self.env.step(self.env.action_space())
        assert not done, "env should not have diverge after second do nothing"
        # not too much losses (which would indicate errors in the computation of the total amount to dispatch)
        assert (
            np.all(
                np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)
            )
            <= 1
        )
        gen_part = self.env.gen_renewable & (obs2.gen_p > 0.0)
        assert np.all(obs2.gen_p >= -self.env._tol_poly)
        assert np.all(
            obs2.gen_p[gen_part] / obs2.gen_pmax[gen_part]
            < obs1.gen_p[gen_part] / obs1.gen_pmax[gen_part]
        )
        assert np.all(
            obs2.gen_p[gen_part] / obs2.gen_pmax[gen_part] == act.curtail[gen_part]
        )
        assert np.all(
            obs2.gen_p_before_curtail[self.env.gen_renewable]
            == self.obs3_ref.gen_p[self.env.gen_renewable]
        )
        self._aux_test_gen(obs1, obs2)
        self._aux_compare_with_ref(self.env, obs2, self.obs3_ref)

        # make sure it stays at the sepoint
        obs3, reward, done, info = self.env.step(self.env.action_space())
        assert not done, "env should not have diverge after third do nothing"
        # not too much losses (which would indicate errors in the computation of the total amount to dispatch)
        assert (
            np.all(
                np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)
            )
            <= 1
        )
        gen_part = self.env.gen_renewable & (obs3.gen_p > 0.0)
        assert np.all(obs3.gen_p >= -self.env._tol_poly)
        assert np.all(
            obs3.curtailment_limit[gen_part]
            == obs3.gen_p[gen_part] / obs3.gen_pmax[gen_part]
        )
        assert np.all(
            obs3.gen_p_before_curtail[self.env.gen_renewable]
            == self.obs4_ref.gen_p[self.env.gen_renewable]
        )
        self._aux_test_gen(obs2, obs3)
        self._aux_compare_with_ref(self.env, obs3, self.obs4_ref)

    def test_down_then_up(self):
        """test that i can curtail down to the setpoint, then up again until the curtailment is canceled"""
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
        assert (
            np.all(
                np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)
            )
            <= 1
        )
        assert np.all(obs2.gen_p >= 0.0)
        assert np.all(
            obs2.gen_p[self.env.gen_renewable] >= obs1.gen_p[self.env.gen_renewable]
        )
        self._aux_test_gen(obs1, obs2)
        self._aux_compare_with_ref(self.env, obs2, self.obs3_ref)

        # re increase to check that the setpoint is correct
        obs3, reward, done, info = self.env.step(self.env.action_space())
        assert not done
        # not too much losses (which would indicate errors in the computation of the total amount to dispatch)
        assert (
            np.all(
                np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)
            )
            <= 1
        )
        assert np.all(obs3.gen_p >= -self.env._tol_poly)
        assert np.all(
            obs3.gen_p[self.env.gen_renewable] >= obs2.gen_p[self.env.gen_renewable]
        )
        self._aux_test_gen(obs2, obs3)
        self._aux_compare_with_ref(self.env, obs3, self.obs4_ref)
        gen_part = self.env.gen_renewable & (obs3.gen_p > self.env._tol_poly)
        # generator produce less than pmax
        assert np.all(obs3.curtailment_limit[gen_part] <= obs3.gen_pmax[gen_part])
        # no more curtailment, so productions increase
        assert np.all(
            obs3.gen_p[self.env.gen_renewable] >= obs2.gen_p[self.env.gen_renewable]
        )
        # information of generation without curtailment is correct
        assert np.all(
            obs3.gen_p_before_curtail[self.env.gen_renewable]
            == self.obs4_ref.gen_p[self.env.gen_renewable]
        )
        # setpoint is matched
        assert np.all(
            obs3.gen_p_before_curtail[self.env.gen_renewable]
            == obs3.gen_p[self.env.gen_renewable]
        )

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

        # now the setpoint is reached, let's increase "at once" (it is possible without violating anything)
        obs3, reward, done, info = self.env.step(self.all_one)
        assert not done
        # not too much losses (which would indicate errors in the computation of the total amount to dispatch)
        assert (
            np.all(
                np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)
            )
            <= 1
        )
        assert np.all(obs3.gen_p >= -self.env._tol_poly)
        assert np.all(
            obs3.gen_p[self.env.gen_renewable] >= obs2.gen_p[self.env.gen_renewable]
        )
        self._aux_test_gen(obs2, obs3)
        self._aux_compare_with_ref(self.env, obs3, self.obs4_ref)

        # another do nothing (setpoint still not reached)
        obs4, reward, done, info = self.env.step(self.env.action_space())
        assert not done
        # not too much losses (which would indicate errors in the computation of the total amount to dispatch)
        assert (
            np.all(
                np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)
            )
            <= 1
        )
        assert np.all(obs4.gen_p >= -self.env._tol_poly)
        assert np.all(
            obs4.gen_p[self.env.gen_renewable] >= obs3.gen_p[self.env.gen_renewable]
        )
        self._aux_test_gen(obs3, obs4)
        self._aux_compare_with_ref(self.env, obs4, self.obs5_ref)

        # setpoint should be correct now
        obs5, reward, done, info = self.env.step(self.env.action_space())
        assert not done
        # not too much losses (which would indicate errors in the computation of the total amount to dispatch)
        assert (
            np.all(
                np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)
            )
            <= 1
        )
        assert np.all(obs5.gen_p >= -self.env._tol_poly)
        assert np.all(
            obs5.gen_p[self.env.gen_renewable] >= obs1.gen_p[self.env.gen_renewable]
        )
        self._aux_test_gen(obs4, obs5)
        self._aux_compare_with_ref(self.env, obs5, self.obs6_ref)
        gen_part = self.env.gen_renewable & (obs3.gen_p > 0.0)
        # generator produce less than pmax
        assert np.all(obs5.curtailment_limit[gen_part] <= obs5.gen_pmax[gen_part])
        # no more curtailment, so productions increase
        assert np.all(
            obs5.gen_p[self.env.gen_renewable] >= obs4.gen_p[self.env.gen_renewable]
        )
        # information of generation without curtailment is correct
        assert np.all(
            obs5.gen_p_before_curtail[self.env.gen_renewable]
            == self.obs6_ref.gen_p[self.env.gen_renewable]
        )
        # setpoint is matched
        assert np.all(
            obs5.gen_p_before_curtail[self.env.gen_renewable]
            == obs5.gen_p[self.env.gen_renewable]
        )


class TestExtremeStorage(unittest.TestCase):
    def setUp(self) -> None:
        self.env_name = "educ_case14_storage"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                self.env_name,
                test=True,
                data_feeding_kwargs={"max_iter": 10},
                _add_to_name="TestExtremeStorage",
                action_class=PlayableAction,
                _add_to_name=type(self).__name__,
            )
        # increase the storage capacity
        increase_storage = np.array([15.0, 30.0])
        type(self.env).storage_max_p_absorb[:] = increase_storage
        type(self.env).storage_max_p_prod[:] = increase_storage
        type(self.env.action_space).storage_max_p_absorb[:] = increase_storage
        type(self.env.action_space).storage_max_p_prod[:] = increase_storage
        self.env.action_space.actionClass.storage_max_p_absorb[:] = increase_storage
        self.env.action_space.actionClass.storage_max_p_prod[:] = increase_storage
        self.env.observation_space.observationClass.storage_max_p_absorb[
            :
        ] = increase_storage
        self.env.observation_space.observationClass.storage_max_p_prod[
            :
        ] = increase_storage

        # retrieve the reference values, without curtailment
        self.env.seed(0)
        self.env.set_id(0)
        self.obs_ref = self.env.reset()
        self.obs1_ref, *_ = self.env.step(self.env.action_space())
        self.obs2_ref, *_ = self.env.step(self.env.action_space())
        self.obs3_ref, *_ = self.env.step(self.env.action_space())
        self.obs4_ref, *_ = self.env.step(self.env.action_space())
        self.obs5_ref, *_ = self.env.step(self.env.action_space())
        self.obs6_ref, *_ = self.env.step(self.env.action_space())

        self.storage_ko_down = self.env.action_space(
            {"set_storage": -self.env.storage_max_p_absorb}
        )
        self.storage_ko_up = self.env.action_space(
            {"set_storage": +self.env.storage_max_p_absorb}
        )
        self.storage_ok_down = self.env.action_space(
            {"set_storage": -0.5 * self.env.storage_max_p_absorb}
        )

        self.storage_curtail = self.env.action_space(
            {
                "set_storage": 0.8 * self.env.storage_max_p_absorb,
                "curtail": [(el, 0.0) for el in np.where(self.env.gen_renewable)[0]],
            }
        )

    @staticmethod
    def _aux_test_storage(obsbefore, obsafter, tol=1.1e-2):
        prod_ = obsafter.storage_power < 0.0
        consume_ = obsafter.storage_power > 0.0
        assert np.all(
            obsbefore.storage_power[prod_] >= -obsbefore.storage_max_p_prod[prod_]
        )
        assert np.all(
            obsbefore.storage_power[consume_]
            <= obsbefore.storage_max_p_absorb[consume_]
        )

        prod_ = obsafter.storage_power < 0.0
        consume_ = obsafter.storage_power > 0.0
        assert np.all(
            obsafter.storage_power[prod_] >= -type(obsafter).storage_max_p_prod[prod_]
        )
        assert np.all(
            obsafter.storage_power[consume_]
            <= type(obsafter).storage_max_p_absorb[consume_]
        )

        assert np.all(obsbefore.storage_charge <= type(obsbefore).storage_Emax)
        assert np.all(obsbefore.storage_charge >= type(obsbefore).storage_Emin)
        assert np.all(obsafter.storage_charge <= type(obsafter).storage_Emax)
        assert np.all(obsafter.storage_charge >= type(obsafter).storage_Emin)

        # check links between storage and energy
        delta_t = obsafter.delta_time * 60.0
        energy_to_power = 3600.0 / delta_t
        storage_power = 1.0 * obsafter.storage_power
        delta_energy = obsafter.storage_charge - obsbefore.storage_charge
        delta_energy[delta_energy < 0.0] *= obsbefore.storage_discharging_efficiency[
            delta_energy < 0.0
        ]
        delta_energy[delta_energy > 0.0] /= obsbefore.storage_charging_efficiency[
            delta_energy > 0.0
        ]
        assert np.all(
            np.abs(
                delta_energy * energy_to_power + obsbefore.storage_loss - storage_power
            )
            <= tol
        )

    def test_do_break(self):
        self.env.seed(0)
        self.env.set_id(0)
        obs0 = self.env.reset()
        obs1, reward, done, info = self.env.step(self.storage_ko_down)
        # there is not enough ramp down to "absorb" what the storage units produces
        assert done

        self.env.seed(0)
        self.env.set_id(0)
        obs0 = self.env.reset()
        obs1, reward, done, info = self.env.step(self.storage_ko_up)
        # there is not enough ramp up to "produce" what the storage units absorbs
        assert done
        assert done

        self.env.seed(0)
        self.env.set_id(0)
        obs0 = self.env.reset()
        obs1, reward, done, info = self.env.step(self.storage_curtail)
        # there is not enough ramp up to "produce" what the storage units absorbs
        assert done

    def test_storage_limit_gen_down(self):
        """
        test that the storage action that would lead to a game over (see test_do_break)
        do not when the parameters is properly set
        """
        param = self.env.parameters
        param.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = True
        self.env.change_parameters(param)
        self.env.seed(0)
        self.env.set_id(0)
        obs = self.env.reset()
        obs1, reward, done, info = self.env.step(self.storage_ko_down)
        assert not done
        amount_storage_first_step = 1.0 * self.env._amount_storage
        # not too much losses (which would indicate errors in the computation of the total amount to dispatch)
        assert (
            np.all(
                np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)
            )
            <= 1
        )
        # test the storage is "limited"
        assert np.all(obs1.storage_power > self.storage_ko_down.storage_p)
        # test the energy / power is properly converted
        self._aux_test_storage(obs, obs1)
        # test the generators are ok
        TestExtremeCurtail._aux_test_gen(
            obs, obs1, min_loss_slack=4
        )  # I generate ~40 MW on this grid with storage, losses changes a lot !
        TestExtremeCurtail._aux_compare_with_ref(
            self.env, obs1, self.obs1_ref, min_loss_slack=4
        )

        obs2, reward, done, info = self.env.step(self.env.action_space())
        assert not done
        # assert np.all(obs2.storage_power == 0.)  # this is no more true because i did not get enough "ramp"
        obs2_power_storage = np.sum(obs2.storage_power)
        assert (
            self.env._amount_storage == -amount_storage_first_step + obs2_power_storage
        )
        self._aux_test_storage(obs1, obs2)
        TestExtremeCurtail._aux_test_gen(
            obs1, obs2, min_loss_slack=4
        )  # I generate ~40 MW on this grid with storage, losses changes a lot !
        TestExtremeCurtail._aux_compare_with_ref(
            self.env, obs2, self.obs2_ref, min_loss_slack=4
        )

        obs3, reward, done, info = self.env.step(self.env.action_space())
        assert np.all(obs3.storage_power == 0.0)
        assert abs(self.env._amount_storage - (-obs2_power_storage)) <= 1e-4
        self._aux_test_storage(obs2, obs3)
        TestExtremeCurtail._aux_test_gen(
            obs2, obs3, min_loss_slack=4
        )  # I generate ~40 MW on this grid with storage, losses changes a lot !
        TestExtremeCurtail._aux_compare_with_ref(
            self.env, obs3, self.obs3_ref, min_loss_slack=4
        )

        obs4, reward, done, info = self.env.step(self.env.action_space())
        assert np.all(obs4.storage_power == 0.0)
        assert self.env._amount_storage == 0.0
        self._aux_test_storage(obs3, obs4)
        TestExtremeCurtail._aux_test_gen(
            obs3, obs4, min_loss_slack=4
        )  # I generate ~40 MW on this grid with storage, losses changes a lot !
        TestExtremeCurtail._aux_compare_with_ref(
            self.env, obs4, self.obs4_ref, min_loss_slack=4
        )

    def test_tests_down(self):
        """in this test i do not test the new feature, i test that the tests performed are working
        in a standard grid2op fashion
        """
        self.env.seed(0)
        self.env.set_id(0)
        obs = self.env.reset()
        obs1, reward, done, info = self.env.step(self.storage_ok_down)
        assert not done
        amount_storage_first_step = 1.0 * self.env._amount_storage
        # not too much losses (which would indicate errors in the computation of the total amount to dispatch)
        assert (
            np.all(
                np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)
            )
            <= 1
        )
        # test the storage is "limited"
        assert np.all(obs1.storage_power > self.storage_ko_down.storage_p)
        # test the energy / power is properly converted
        self._aux_test_storage(obs, obs1)
        # test the generators are ok
        TestExtremeCurtail._aux_test_gen(
            obs, obs1, min_loss_slack=4
        )  # I generate ~40 MW on this grid with storage, losses changes a lot !
        TestExtremeCurtail._aux_compare_with_ref(
            self.env, obs1, self.obs1_ref, min_loss_slack=4
        )

        obs2, reward, done, info = self.env.step(self.env.action_space())
        assert np.all(obs2.storage_power == 0.0)
        assert self.env._amount_storage == -amount_storage_first_step
        self._aux_test_storage(obs1, obs2)
        TestExtremeCurtail._aux_test_gen(
            obs1, obs2, min_loss_slack=4
        )  # I generate ~40 MW on this grid with storage, losses changes a lot !
        TestExtremeCurtail._aux_compare_with_ref(
            self.env, obs2, self.obs2_ref, min_loss_slack=4
        )

        obs3, reward, done, info = self.env.step(self.env.action_space())
        assert np.all(obs3.storage_power == 0.0)
        assert self.env._amount_storage == 0.0
        self._aux_test_storage(obs2, obs3)
        TestExtremeCurtail._aux_test_gen(
            obs3, obs3, min_loss_slack=4
        )  # I generate ~40 MW on this grid with storage, losses changes a lot !
        TestExtremeCurtail._aux_compare_with_ref(
            self.env, obs3, self.obs3_ref, min_loss_slack=4
        )

    def test_storage_limit_gen_up(self):
        """
        test that the storage action that would lead to a game over (see test_do_break)
        do not when the parameters is properly set
        """
        param = self.env.parameters
        param.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = True
        self.env.change_parameters(param)
        self.env.seed(0)
        self.env.set_id(0)
        obs = self.env.reset()
        obs1, reward, done, info = self.env.step(self.storage_ko_up)
        assert not done
        amount_storage_first_step = 1.0 * self.env._amount_storage
        # not too much losses (which would indicate errors in the computation of the total amount to dispatch)
        assert (
            np.all(
                np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)
            )
            <= 1
        )
        # test the storage is "limited"
        assert np.all(obs1.storage_power < self.storage_ko_up.storage_p)
        # test the energy / power is properly converted
        self._aux_test_storage(obs, obs1)
        # test the generators are ok
        TestExtremeCurtail._aux_test_gen(
            obs, obs1, min_loss_slack=4
        )  # I generate ~40 MW on this grid with storage, losses changes a lot !
        TestExtremeCurtail._aux_compare_with_ref(
            self.env, obs1, self.obs1_ref, min_loss_slack=4
        )

        obs2, reward, done, info = self.env.step(self.env.action_space())
        assert np.all(obs2.storage_power == 0.0)
        assert self.env._amount_storage == -amount_storage_first_step
        self._aux_test_storage(obs1, obs2)
        TestExtremeCurtail._aux_test_gen(
            obs1, obs2, min_loss_slack=4
        )  # I generate ~40 MW on this grid with storage, losses changes a lot !
        TestExtremeCurtail._aux_compare_with_ref(
            self.env, obs2, self.obs2_ref, min_loss_slack=4
        )

        obs3, reward, done, info = self.env.step(self.env.action_space())
        assert np.all(obs3.storage_power == 0.0)
        assert self.env._amount_storage == 0.0
        self._aux_test_storage(obs2, obs3)
        TestExtremeCurtail._aux_test_gen(
            obs3, obs3, min_loss_slack=4
        )  # I generate ~40 MW on this grid with storage, losses changes a lot !
        TestExtremeCurtail._aux_compare_with_ref(
            self.env, obs3, self.obs3_ref, min_loss_slack=4
        )

    def test_storage_curtail(self):
        param = self.env.parameters
        param.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = True
        self.env.change_parameters(param)
        self.env.seed(0)
        self.env.set_id(0)
        obs = self.env.reset()
        obs1, reward, done, info = self.env.step(self.storage_curtail)
        assert not done
        # not too much losses (which would indicate errors in the computation of the total amount to dispatch)
        assert (
            np.all(
                np.abs(self.env._gen_activeprod_t_redisp - self.env._gen_activeprod_t)
            )
            <= 1
        )
        # test the storage is "limited"
        assert np.all(obs1.storage_power < self.storage_curtail.storage_p)
        gen_curt = obs1.gen_renewable & (obs1.gen_p > 0.0)
        assert np.all(
            obs1.gen_p[gen_curt] / obs1.gen_pmax[gen_curt]
            > self.storage_curtail.curtail[gen_curt]
        )
        # test the energy / power is properly converted
        self._aux_test_storage(obs, obs1)
        # test the generators are ok
        TestExtremeCurtail._aux_test_gen(
            obs, obs1, min_loss_slack=4
        )  # I generate ~40 MW on this grid with storage, losses changes a lot !
        TestExtremeCurtail._aux_compare_with_ref(
            self.env, obs1, self.obs1_ref, min_loss_slack=4
        )

        obs2, reward, done, info = self.env.step(self.env.action_space())
        assert np.all(
            obs2.gen_p[obs2.gen_renewable] >= 0.0
        ), "some curtailment make for a negative production !"
        assert np.all(
            obs2.gen_p[obs2.gen_renewable] == 0.0
        )  # everything is set to 0. now !
        self._aux_test_storage(obs1, obs2)
        # test the generators are ok
        TestExtremeCurtail._aux_test_gen(
            obs1, obs2, min_loss_slack=4
        )  # I generate ~40 MW on this grid with storage, losses changes a lot !
        TestExtremeCurtail._aux_compare_with_ref(
            self.env, obs2, self.obs2_ref, min_loss_slack=4
        )

        obs3, reward, done, info = self.env.step(self.env.action_space())
        assert np.all(
            obs3.gen_p[obs2.gen_renewable] >= 0.0
        ), "some curtailment make for a negative production !"
        assert np.all(
            obs3.gen_p[obs2.gen_renewable] == 0.0
        )  # everything is set to 0. now !
        self._aux_test_storage(obs2, obs3)
        # test the generators are ok
        TestExtremeCurtail._aux_test_gen(
            obs2, obs3, min_loss_slack=4
        )  # I generate ~40 MW on this grid with storage, losses changes a lot !
        TestExtremeCurtail._aux_compare_with_ref(
            self.env, obs3, self.obs3_ref, min_loss_slack=4
        )


# TODO test with simulate !!!!

if __name__ == "__main__":
    unittest.main()
