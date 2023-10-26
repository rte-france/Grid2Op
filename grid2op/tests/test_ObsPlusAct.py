# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
from grid2op.tests.helper_path_test import *

import unittest
import grid2op

from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Exceptions import *
from grid2op.Action import *
from grid2op.Parameters import Parameters
from grid2op.Rules import  AlwaysLegal

import pdb


class BaseHelper:
    """Base class to test the method __add__ of an observation that is able to emulate the "adding" of
    an action from an observation"""

    def reset_without_pp_futurewarnings(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.obs = self.env.reset()
            
    def setUp(self) -> None:
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                "educ_case14_storage",
                test=True,
                action_class=self.get_act_cls(),
                param=param,
                gamerules_class=AlwaysLegal,
                _add_to_name=type(self).__name__,
            )
        self.reset_without_pp_futurewarnings()
        self.act = self.env.action_space()

    def tearDown(self) -> None:
        self.env.close()

    def get_act_cls(self):
        raise NotImplementedError()

    def check_all_other_as_if_game_over(self, tested_obs):
        obs = type(self.obs)()
        obs.set_game_over()
        obs.set_game_over()
        for el in obs._attr_eq:
            if el == "line_status":
                continue
            if el == "topo_vect":
                continue
            if getattr(obs, el).dtype == dt_float:
                # TODO equal_nan throw an error now !
                assert np.array_equal(
                    getattr(obs, el), getattr(tested_obs, el)
                ), f"error for {el}"
            else:
                assert np.array_equal(
                    getattr(obs, el), getattr(tested_obs, el)
                ), f"error for {el}"

    def aux_test_action(
        self,
        res_topo_vect_1,
        res_topo_vect_2,
        res_topo_vect_3,
        res_ls_1,
        res_ls_2,
        res_ls_3,
    ):
        self.reset_without_pp_futurewarnings()
        res = self.obs + self.act
        assert np.all(res.topo_vect == res_topo_vect_1)
        assert np.all(res.line_status == res_ls_1)
        self.check_all_other_as_if_game_over(res)

        self.reset_without_pp_futurewarnings()
        # try to deconnect a powerline
        if "set_line_status" in self.act.authorized_keys:
            obs, reward, done, info = self.env.step(
                self.env.action_space({"set_line_status": [(1, -1)]})
            )
            assert not done
            res2 = obs + self.act
            assert np.all(res2.topo_vect == res_topo_vect_2)
            assert np.all(res2.line_status == res_ls_2)
            self.check_all_other_as_if_game_over(res2)
        elif "change_line_status" in self.act.authorized_keys:
            obs, reward, done, info = self.env.step(
                self.env.action_space({"change_line_status": [1]})
            )
            assert not done
            res2 = obs + self.act
            assert np.all(res2.topo_vect == res_topo_vect_2)
            assert np.all(res2.line_status == res_ls_2)
            self.check_all_other_as_if_game_over(res2)

        self.reset_without_pp_futurewarnings()
        # try to change a substation configuration
        if "set_bus" in self.act.authorized_keys:
            act_step = self.env.action_space(
                {"set_bus": {"substations_id": [(5, (1, 2, 1, 2, 1, 2, 1, 0))]}}
            )
            obs, reward, done, info = self.env.step(act_step)
            assert not done
            res3 = obs + self.act
            assert np.all(res3.topo_vect == res_topo_vect_3)
            assert np.all(res3.line_status == res_ls_3)
            self.check_all_other_as_if_game_over(res3)
        elif "change_bus" in self.act.authorized_keys:
            obs, reward, done, info = self.env.step(
                self.env.action_space(
                    {
                        "change_bus": {
                            "substations_id": [
                                (
                                    5,
                                    (
                                        False,
                                        True,
                                        False,
                                        True,
                                        False,
                                        True,
                                        False,
                                        False,
                                    ),
                                )
                            ]
                        }
                    }
                )
            )
            assert not done
            res3 = obs + self.act
            assert np.all(res3.topo_vect == res_topo_vect_3)
            assert np.all(res3.line_status == res_ls_3)
            self.check_all_other_as_if_game_over(res3)

    def test_dn_action(self):
        """test add do nothing action is properly implemented"""
        # nothing is done, just a step
        (
            res_topo_vect_1,
            res_topo_vect_2,
            res_topo_vect_3,
            res_ls_1,
            res_ls_2,
            res_ls_3,
        ) = self._aux_normal_impact()
        self.aux_test_action(
            res_topo_vect_1=res_topo_vect_1,
            res_ls_1=res_ls_1,
            res_topo_vect_2=res_topo_vect_2,
            res_ls_2=res_ls_2,
            res_topo_vect_3=res_topo_vect_3,
            res_ls_3=res_ls_3,
        )

    def _aux_normal_impact(self):
        # nothing is done, just a step
        res_topo_vect_1 = self.obs.topo_vect
        res_ls_1 = self.obs.line_status

        # the action "disconnect powerline 1" is perfomed to get the obs
        res_topo_vect_2 = np.ones(59, dtype=dt_int)
        res_topo_vect_2[1] = -1
        res_topo_vect_2[19] = -1
        res_ls_2 = np.ones(20, dtype=dt_bool)
        res_ls_2[1] = False

        # the action "change topo of sub 5 with (1, 2, 1, 2, 1, 2, 1, 2)" is performed to get the obs
        res_topo_vect_3 = np.ones(59, dtype=dt_int)
        res_topo_vect_3[24:32] = (1, 2, 1, 2, 1, 2, 1, 1)
        res_ls_3 = np.ones(20, dtype=dt_bool)
        return (
            res_topo_vect_1,
            res_topo_vect_2,
            res_topo_vect_3,
            res_ls_1,
            res_ls_2,
            res_ls_3,
        )

    def test_topo_set_action(self):
        """test i can add an action that do not impact the modification of the observation"""
        if "set_bus" not in self.act.authorized_keys:
            return

        (
            res_topo_vect_1,
            res_topo_vect_2,
            res_topo_vect_3,
            res_ls_1,
            res_ls_2,
            res_ls_3,
        ) = self._aux_normal_impact()

        self.act.set_bus = [(4, 2), (6, 2)]
        # modification implied by the action
        res_topo_vect_1[4] = 2
        res_topo_vect_1[6] = 2
        res_topo_vect_2[4] = 2
        res_topo_vect_2[6] = 2
        res_topo_vect_3[4] = 2
        res_topo_vect_3[6] = 2

        self.aux_test_action(
            res_topo_vect_1=res_topo_vect_1,
            res_ls_1=res_ls_1,
            res_topo_vect_2=res_topo_vect_2,
            res_ls_2=res_ls_2,
            res_topo_vect_3=res_topo_vect_3,
            res_ls_3=res_ls_3,
        )

    def test_topo_set_action2(self):
        """test i can add an action that not impact the modification of the observation (set bus should...
        set the bus)"""
        if "set_bus" not in self.act.authorized_keys:
            return

        (
            res_topo_vect_1,
            res_topo_vect_2,
            res_topo_vect_3,
            res_ls_1,
            res_ls_2,
            res_ls_3,
        ) = self._aux_normal_impact()

        self.act.set_bus = [(24, 2), (25, 2)]
        # modification implied by the action
        res_topo_vect_1[24] = 2
        res_topo_vect_1[25] = 2
        res_topo_vect_2[24] = 2
        res_topo_vect_2[25] = 2
        res_topo_vect_3[24] = 2
        res_topo_vect_3[25] = 2

        self.aux_test_action(
            res_topo_vect_1=res_topo_vect_1,
            res_ls_1=res_ls_1,
            res_topo_vect_2=res_topo_vect_2,
            res_ls_2=res_ls_2,
            res_topo_vect_3=res_topo_vect_3,
            res_ls_3=res_ls_3,
        )

    def test_topo_set_action3(self):
        """test i can add an action that not impact the modification of the observation (set line status should
        reconnect)
        """
        if "set_bus" not in self.act.authorized_keys:
            return

        (
            res_topo_vect_1,
            res_topo_vect_2,
            res_topo_vect_3,
            res_ls_1,
            res_ls_2,
            res_ls_3,
        ) = self._aux_normal_impact()

        self.act.line_or_set_bus = [(1, 2)]
        self.act.line_ex_set_bus = [(1, 2)]
        # modification implied by the action
        res_topo_vect_1[1] = 2
        res_topo_vect_1[19] = 2
        res_topo_vect_2[1] = 2
        res_topo_vect_2[19] = 2
        res_topo_vect_3[1] = 2
        res_topo_vect_3[19] = 2

        res_ls_2[1] = True

        self.aux_test_action(
            res_topo_vect_1=res_topo_vect_1,
            res_ls_1=res_ls_1,
            res_topo_vect_2=res_topo_vect_2,
            res_ls_2=res_ls_2,
            res_topo_vect_3=res_topo_vect_3,
            res_ls_3=res_ls_3,
        )

    def test_topo_change_action(self):
        """test i can add an action that do not impact the modification in the observation"""
        if "change_bus" not in self.act.authorized_keys:
            return

        (
            res_topo_vect_1,
            res_topo_vect_2,
            res_topo_vect_3,
            res_ls_1,
            res_ls_2,
            res_ls_3,
        ) = self._aux_normal_impact()

        self.act.change_bus = [4, 6]
        # modification implied by the action
        res_topo_vect_1[4] = 2
        res_topo_vect_1[6] = 2
        res_topo_vect_2[4] = 2
        res_topo_vect_2[6] = 2
        res_topo_vect_3[4] = 2
        res_topo_vect_3[6] = 2

        self.aux_test_action(
            res_topo_vect_1=res_topo_vect_1,
            res_ls_1=res_ls_1,
            res_topo_vect_2=res_topo_vect_2,
            res_ls_2=res_ls_2,
            res_topo_vect_3=res_topo_vect_3,
            res_ls_3=res_ls_3,
        )

    def test_topo_change_action2(self):
        """
        test i can add an action that do impact the modification in the observation
        (change substation reconfiguration)
        """
        if "change_bus" not in self.act.authorized_keys:
            return

        (
            res_topo_vect_1,
            res_topo_vect_2,
            res_topo_vect_3,
            res_ls_1,
            res_ls_2,
            res_ls_3,
        ) = self._aux_normal_impact()

        self.act.change_bus = [24, 25]
        # modification implied by the action
        res_topo_vect_1[24] = 2
        res_topo_vect_1[25] = 2
        res_topo_vect_2[24] = 2
        res_topo_vect_2[25] = 2
        res_topo_vect_3[24] = 2
        res_topo_vect_3[25] = 1

        self.aux_test_action(
            res_topo_vect_1=res_topo_vect_1,
            res_ls_1=res_ls_1,
            res_topo_vect_2=res_topo_vect_2,
            res_ls_2=res_ls_2,
            res_topo_vect_3=res_topo_vect_3,
            res_ls_3=res_ls_3,
        )

    def test_topo_change_action3(self):
        """
        test i can add an action that do impact the modification in the observation
        (change a line status)
        """
        if "change_bus" not in self.act.authorized_keys:
            return

        (
            res_topo_vect_1,
            res_topo_vect_2,
            res_topo_vect_3,
            res_ls_1,
            res_ls_2,
            res_ls_3,
        ) = self._aux_normal_impact()

        self.act.line_or_change_bus = [1]
        self.act.line_ex_change_bus = [1]
        # should have not impact on disconnected lines (this is why i don't modify the res_ls*)

        # modification implied by the action
        res_topo_vect_1[1] = 2
        res_topo_vect_1[19] = 2
        # for res_topo_vect2 the powerline is disconnected, changing its bus has no effect.
        res_topo_vect_3[1] = 2
        res_topo_vect_3[19] = 2

        self.aux_test_action(
            res_topo_vect_1=res_topo_vect_1,
            res_ls_1=res_ls_1,
            res_topo_vect_2=res_topo_vect_2,
            res_ls_2=res_ls_2,
            res_topo_vect_3=res_topo_vect_3,
            res_ls_3=res_ls_3,
        )

    def test_status_set_action_disco(self):
        """
        test i can properly add an action were i set a powerline status
        by disconnecting it
        """

        if "set_line_status" not in self.act.authorized_keys:
            return

        (
            res_topo_vect_1,
            res_topo_vect_2,
            res_topo_vect_3,
            res_ls_1,
            res_ls_2,
            res_ls_3,
        ) = self._aux_normal_impact()

        # disconnection
        l_id = 2
        self.act.line_set_status = [(l_id, -1)]

        # modification implied by the action
        id_topo_or = self.env.line_or_pos_topo_vect[l_id]
        id_topo_ex = self.env.line_ex_pos_topo_vect[l_id]
        res_topo_vect_1[id_topo_or] = -1
        res_topo_vect_1[id_topo_ex] = -1
        res_topo_vect_2[id_topo_or] = -1
        res_topo_vect_2[id_topo_ex] = -1
        res_topo_vect_3[id_topo_or] = -1
        res_topo_vect_3[id_topo_ex] = -1
        res_ls_1[l_id] = False
        res_ls_2[l_id] = False
        res_ls_3[l_id] = False

        self.aux_test_action(
            res_topo_vect_1=res_topo_vect_1,
            res_ls_1=res_ls_1,
            res_topo_vect_2=res_topo_vect_2,
            res_ls_2=res_ls_2,
            res_topo_vect_3=res_topo_vect_3,
            res_ls_3=res_ls_3,
        )

    def test_status_set_action_reco(self):
        """
        test i can properly add an action were i set a powerline status
        I try to add a "reconnect" status of a disconnected powerline without specifying the bus to
        which i reconnect it
        """

        if "set_line_status" not in self.act.authorized_keys:
            return

        (
            res_topo_vect_1,
            res_topo_vect_2,
            res_topo_vect_3,
            res_ls_1,
            res_ls_2,
            res_ls_3,
        ) = self._aux_normal_impact()

        # disconnection
        l_id = 1
        self.act.line_set_status = [(l_id, +1)]

        # modification implied by the action
        id_topo_or = self.env.line_or_pos_topo_vect[l_id]
        id_topo_ex = self.env.line_ex_pos_topo_vect[l_id]
        # res_topo_vect_1[id_topo_or] = 1  # has no effect, line already connected
        # res_topo_vect_1[id_topo_ex] = 1  # has no effect, line already connected
        res_topo_vect_2[id_topo_or] = 1
        res_topo_vect_2[id_topo_ex] = 1
        # res_topo_vect_3[id_topo_or] = -1  # has no effect, line already connected
        # res_topo_vect_3[id_topo_ex] = -1  # has no effect, line already connected
        res_ls_2[l_id] = True  # it has reconnected it

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # warning because i reconnect a powerline without specifying the bus
            # i test here the test goes till the end
            self.aux_test_action(
                res_topo_vect_1=res_topo_vect_1,
                res_ls_1=res_ls_1,
                res_topo_vect_2=res_topo_vect_2,
                res_ls_2=res_ls_2,
                res_topo_vect_3=res_topo_vect_3,
                res_ls_3=res_ls_3,
            )

        # now i test it properly sends the warning
        with self.assertRaises(UserWarning):
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                self.aux_test_action(
                    res_topo_vect_1=res_topo_vect_1,
                    res_ls_1=res_ls_1,
                    res_topo_vect_2=res_topo_vect_2,
                    res_ls_2=res_ls_2,
                    res_topo_vect_3=res_topo_vect_3,
                    res_ls_3=res_ls_3,
                )

    def test_status_set_action_reco2(self):
        """
        test i can properly add an action were i set a powerline status
        I try to add a "reconnect" status of a disconnected powerline by specifying the bus to
        which i reconnect it
        """

        if "set_line_status" not in self.act.authorized_keys:
            # i need to be able to change the status of powerlines
            return
        if "set_bus" not in self.act.authorized_keys:
            # i need to be able to reconnect something to bus 2
            return

        (
            res_topo_vect_1,
            res_topo_vect_2,
            res_topo_vect_3,
            res_ls_1,
            res_ls_2,
            res_ls_3,
        ) = self._aux_normal_impact()

        # disconnection
        l_id = 1
        self.act.line_set_status = [(l_id, +1)]
        self.act.line_or_set_bus = [(l_id, +1)]
        self.act.line_ex_set_bus = [(l_id, 2)]

        # modification implied by the action
        id_topo_or = self.env.line_or_pos_topo_vect[l_id]
        id_topo_ex = self.env.line_ex_pos_topo_vect[l_id]
        res_topo_vect_1[id_topo_or] = 1
        res_topo_vect_1[id_topo_ex] = 2
        res_topo_vect_2[id_topo_or] = 1
        res_topo_vect_2[id_topo_ex] = 2
        res_topo_vect_3[id_topo_or] = 1
        res_topo_vect_3[id_topo_ex] = 2
        res_ls_2[l_id] = True  # it has reconnected it

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            # it should not raise because i specified the bus to which i reconnect it
            self.aux_test_action(
                res_topo_vect_1=res_topo_vect_1,
                res_ls_1=res_ls_1,
                res_topo_vect_2=res_topo_vect_2,
                res_ls_2=res_ls_2,
                res_topo_vect_3=res_topo_vect_3,
                res_ls_3=res_ls_3,
            )

    def test_status_change_status_action(self):
        """test i can properly add an action were i change a powerline status"""
        # TODO change the "no warning issued when set_bus is available"
        # TODO change regular powerline (eg not powerline with id 1)
        if "change_line_status" not in self.act.authorized_keys:
            return

        (
            res_topo_vect_1,
            res_topo_vect_2,
            res_topo_vect_3,
            res_ls_1,
            res_ls_2,
            res_ls_3,
        ) = self._aux_normal_impact()

        # disconnection
        l_id = 1
        self.act.line_change_status = [l_id]

        # modification implied by the action
        id_topo_or = self.env.line_or_pos_topo_vect[l_id]
        id_topo_ex = self.env.line_ex_pos_topo_vect[l_id]
        res_topo_vect_1[id_topo_or] = -1  # has no effect, line already connected
        res_topo_vect_1[id_topo_ex] = -1  # has no effect, line already connected
        res_topo_vect_2[id_topo_or] = 1
        res_topo_vect_2[id_topo_ex] = 1
        res_topo_vect_3[id_topo_or] = -1  # has no effect, line already connected
        res_topo_vect_3[id_topo_ex] = -1  # has no effect, line already connected
        res_ls_1[l_id] = False
        res_ls_2[l_id] = True  # it has reconnected it
        res_ls_3[l_id] = False

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # warning because i reconnect a powerline without specifying the bus
            # i test here the test goes till the end
            self.aux_test_action(
                res_topo_vect_1=res_topo_vect_1,
                res_ls_1=res_ls_1,
                res_topo_vect_2=res_topo_vect_2,
                res_ls_2=res_ls_2,
                res_topo_vect_3=res_topo_vect_3,
                res_ls_3=res_ls_3,
            )

        # now i test it properly sends the warning
        with self.assertRaises(UserWarning):
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                self.aux_test_action(
                    res_topo_vect_1=res_topo_vect_1,
                    res_ls_1=res_ls_1,
                    res_topo_vect_2=res_topo_vect_2,
                    res_ls_2=res_ls_2,
                    res_topo_vect_3=res_topo_vect_3,
                    res_ls_3=res_ls_3,
                )


class TestCompleteAction(BaseHelper, HelperTests, unittest.TestCase):
    def get_act_cls(self):
        return CompleteAction


class TestDispatchAction(BaseHelper, HelperTests, unittest.TestCase):
    def get_act_cls(self):
        return DispatchAction


class TestDontAct(BaseHelper, HelperTests, unittest.TestCase):
    def get_act_cls(self):
        return DontAct


class TestPlayableAction(BaseHelper, HelperTests, unittest.TestCase):
    def get_act_cls(self):
        return PlayableAction


class TestPowerlineChangeAction(BaseHelper, HelperTests, unittest.TestCase):
    def get_act_cls(self):
        return PowerlineChangeAction


class TestPowerlineChangeAndDispatchAction(BaseHelper, HelperTests, unittest.TestCase):
    def get_act_cls(self):
        return PowerlineChangeAndDispatchAction


class TestPowerlineChangeDispatchAndStorageAction(BaseHelper, HelperTests, unittest.TestCase):
    def get_act_cls(self):
        return PowerlineChangeDispatchAndStorageAction


class TestPowerlineSetAction(BaseHelper, HelperTests, unittest.TestCase):
    def get_act_cls(self):
        return PowerlineSetAction


class TestPowerlineSetAndDispatchAction(BaseHelper, HelperTests, unittest.TestCase):
    def get_act_cls(self):
        return PowerlineSetAndDispatchAction


class TestTopologyAction(BaseHelper, HelperTests, unittest.TestCase):
    def get_act_cls(self):
        return TopologyAction


class TestTopologyAndDispatchAction(BaseHelper, HelperTests, unittest.TestCase):
    def get_act_cls(self):
        return TopologyAndDispatchAction


class TestTopologyChangeAction(BaseHelper, HelperTests, unittest.TestCase):
    def get_act_cls(self):
        return TopologyChangeAction


class TestTopologyChangeAndDispatchAction(BaseHelper, HelperTests, unittest.TestCase):
    def get_act_cls(self):
        return TopologyChangeAndDispatchAction


class TestTopologySetAction(BaseHelper, HelperTests, unittest.TestCase):
    def get_act_cls(self):
        return TopologySetAction


class TestTopologySetAndDispatchAction(BaseHelper, HelperTests, unittest.TestCase):
    def get_act_cls(self):
        return TopologySetAndDispatchAction


if __name__ == "__main__":
    unittest.main()
