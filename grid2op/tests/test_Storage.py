# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import unittest

from grid2op.tests.helper_path_test import *

import grid2op
from grid2op.Parameters import Parameters
from grid2op.dtypes import dt_float
from grid2op.Action import CompleteAction

# TODO check when there is also redispatching


class TestStorageEnv(HelperTests, unittest.TestCase):
    """test the env part of the storage functionality"""

    def setUp(self) -> None:
        super().setUp()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("educ_case14_storage", test=True)

    def tearDown(self) -> None:
        self.env.close()

    def test_init_storage_ok(self):
        """
        test the right storage is given at the init of an environment
        it tests the param.INIT_STORAGE_CAPACITY parameter
        """
        obs = self.env.get_obs()
        assert np.all(
            np.abs(obs.storage_charge - 0.5 * obs.storage_Emax) <= self.tol_one
        )

        obs = self.env.reset()
        assert np.all(
            np.abs(obs.storage_charge - 0.5 * obs.storage_Emax) <= self.tol_one
        )

        param = Parameters()
        param.INIT_STORAGE_CAPACITY = 0.0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("educ_case14_storage", test=True, param=param)
        obs = env.reset()
        assert np.all(np.abs(obs.storage_charge) <= self.tol_one)

        param = Parameters()
        param.INIT_STORAGE_CAPACITY = 1.0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("educ_case14_storage", test=True, param=param)
        obs = env.reset()
        assert np.all(np.abs(obs.storage_charge - obs.storage_Emax) <= self.tol_one)

        # now test the reset works
        act = self.env.action_space({"set_storage": [(0, 1), (1, 5)]})
        obs, reward, done, info = self.env.step(act)
        assert np.any(
            np.abs(obs.storage_charge - 0.5 * obs.storage_Emax) > self.tol_one
        )
        obs = self.env.reset()
        assert np.all(
            np.abs(obs.storage_charge - 0.5 * obs.storage_Emax) <= self.tol_one
        )

    def test_action_ok(self):
        """test a storage action is supported (basic test)"""
        act = self.env.action_space({"set_storage": [(0, 1)]})
        str_ = act.__str__()
        real_str = (
            "This action will:\n"
            "\t - NOT change anything to the injections\n"
            "\t - NOT perform any redispatching action\n"
            "\t - Modify the storage units in the following way:\n"
            '\t \t - Ask unit "storage_5_0" to absorb 1.00 MW (setpoint: 1.00 MW)\n'
            "\t - NOT perform any curtailment\n"
            "\t - NOT force any line status\n"
            "\t - NOT switch any line status\n\t - NOT switch anything in the topology\n"
            "\t - NOT force any particular bus configuration"
        )
        assert str_ == real_str

    def test_env_storage_ok(self):
        """
        test i can perform normal storage actions (no trick here) just normal actions

        this test the proper computing of the charge of the storage units, the proper computing of the
        redispatching etc.
        """

        # first test the storage loss
        act = self.env.action_space()
        loss = 1.0 * self.env.storage_loss
        loss /= 12.0  # i have 12 steps per hour (ts = (mins), losses are given in MW and capacity in MWh
        for nb_ts in range(5):
            obs, reward, done, info = self.env.step(act)
            assert np.all(
                np.abs(
                    obs.storage_charge - (0.5 * obs.storage_Emax - (nb_ts + 1) * loss)
                )
                <= self.tol_one
            ), f"wrong value computed for time step {nb_ts}"

        # now modify the storage capacity (charge a battery, with efficiency != 1.)
        # storage value is [7.4583335, 3.4583333]
        act = self.env.action_space(
            {"set_storage": [(0, 3)]}
        )  # ask the first storage to absorb 3MW (during 5 mins)
        obs, reward, done, info = self.env.step(act)
        assert not info["exception"]  # there should be no exception here
        # without modif it's [7.4583335, 3.4583333] - loss
        # modif adds [3 / 12, 0.]  # 3/12 is because i ask to absorb 3 MW during 5 mins
        # but storage efficiency of battery 0 is 0.95, so i don't store 3/12. but 0.95 * 3/12
        # result is [7.70000017, 3.44999997]
        assert np.all(
            np.abs(obs.storage_charge - [7.68750017, 3.44999997]) <= self.tol_one
        )
        assert np.all(
            np.abs(obs.target_dispatch) <= self.tol_one
        )  # i did not do any dispatch
        # I asked to absorb 3MW, so dispatch should produce 3MW more
        assert np.abs(np.sum(obs.actual_dispatch) - (+3.0)) <= self.tol_one
        assert np.all(np.abs(obs.storage_power_target - [3.0, 0.0]) <= self.tol_one)
        assert np.all(np.abs(obs.storage_power - [3.0, 0.0]) <= self.tol_one)

        # second action (battery produces something, with efficiency != 1.)
        act = self.env.action_space(
            {"set_storage": [(1, -5)]}
        )  # ask the second storage to produce 3MW (during 5 mins)
        obs, reward, done, info = self.env.step(act)
        assert not info["exception"]  # there should be no exception here
        # without modif it's [7.68750017, 3.44999997] - loss
        # modif adds [0., -5/12.]  # -5/12 is because i ask to produce 5 MW during 5 mins
        # but second battery discharging efficiency is 0.9 so the actual charge will decrease of -5/12 * (1/0.9)
        # final result is [7.67916684, 2.97870367]
        assert np.all(
            np.abs(obs.storage_charge - [7.67916684, 2.97870367]) <= self.tol_one
        )
        assert np.all(
            np.abs(obs.target_dispatch) <= self.tol_one
        )  # i did not do any dispatch
        # I asked to produce 5MW, so dispatch should produce 5MW more
        assert np.abs(np.sum(obs.actual_dispatch) - (-5.0)) <= self.tol_one
        assert np.all(np.abs(obs.storage_power_target - [0.0, -5.0]) <= self.tol_one)
        assert np.all(np.abs(obs.storage_power - [0.0, -5.0]) <= self.tol_one)

        # third modify the storage capacity (charge a battery, with efficiency == 1.)
        act = self.env.action_space({"set_storage": [(0, -1)]})
        obs, reward, done, info = self.env.step(act)
        assert not info["exception"]  # there should be no exception here
        # without modif it's [7.67916684, 2.97870367] - loss
        # modif adds [-1/12., 0.]  # -1/12 is because i ask to produce 1 MW during 5 mins
        # final result is [7.58750017, 2.97037034]
        assert np.all(
            np.abs(obs.storage_charge - [7.58750017, 2.97037034]) <= self.tol_one
        )
        assert np.all(
            np.abs(obs.target_dispatch) <= self.tol_one
        )  # i did not do any dispatch
        # I asked to produce 5MW, so dispatch should produce 5MW more
        assert np.abs(np.sum(obs.actual_dispatch) - (-1.0)) <= self.tol_one
        assert np.all(np.abs(obs.storage_power_target - [-1.0, 0.0]) <= self.tol_one)
        assert np.all(np.abs(obs.storage_power - [-1, 0.0]) <= self.tol_one)

        # fourth modify the storage capacity (discharge a battery, with efficiency == 1.)
        act = self.env.action_space({"set_storage": [(1, 2)]})
        obs, reward, done, info = self.env.step(act)
        assert not info["exception"]  # there should be no exception here
        # without modif it's [7.58750017, 2.97037034] - loss
        # modif adds [0., 2/12.]  # 2/12 is because i ask to produce 2 MW during 5 mins
        # final result is [7.57916684, 3.12870367] => rounded to [7.579168 , 3.1287026]
        assert np.all(
            np.abs(obs.storage_charge - [7.57916684, 3.12870367]) <= self.tol_one
        )
        assert np.all(
            np.abs(obs.target_dispatch) <= self.tol_one
        )  # i did not do any dispatch
        # I asked to produce 5MW, so dispatch should produce 5MW more
        assert np.abs(np.sum(obs.actual_dispatch) - (2.0)) <= self.tol_one
        assert np.all(np.abs(obs.storage_power_target - [0.0, 2.0]) <= self.tol_one)
        assert np.all(np.abs(obs.storage_power - [0.0, 2.0]) <= self.tol_one)

        # fifth i do nothing and make sure everything is reset
        act = (
            self.env.action_space()
        )  # ask the second storage to produce 3MW (during 5 mins)
        obs, reward, done, info = self.env.step(act)
        assert not info["exception"]  # there should be no exception here
        # without modif it's [7.579168 , 3.1287026]- loss
        assert np.all(
            np.abs(obs.storage_charge - ([7.57083467, 3.12036927])) <= self.tol_one
        )
        assert np.all(
            np.abs(obs.target_dispatch) <= self.tol_one
        )  # i did not do any dispatch
        # I did not modify the battery, so i should not modify the dispatch
        assert np.abs(np.sum(obs.actual_dispatch) - (0.0)) <= self.tol_one
        assert np.all(np.abs(obs.storage_power_target - [0.0, 0.0]) <= self.tol_one)
        assert np.all(np.abs(obs.storage_power - [0.0, 0.0]) <= self.tol_one)

        # sixth i do nothing and make sure everything is reset
        act = (
            self.env.action_space()
        )  # ask the second storage to produce 3MW (during 5 mins)
        obs, reward, done, info = self.env.step(act)
        assert not info["exception"]  # there should be no exception here
        # without modif it's [7.57083467, 3.12036927] - loss
        assert np.all(
            np.abs(obs.storage_charge - ([7.56250134, 3.11203594])) <= self.tol_one
        )
        assert np.all(
            np.abs(obs.target_dispatch) <= self.tol_one
        )  # i did not do any dispatch
        # I did not modify the battery, so i should not modify the dispatch
        assert np.abs(np.sum(obs.actual_dispatch) - (0.0)) <= self.tol_one
        assert np.all(np.abs(obs.storage_power_target - [0.0, 0.0]) <= self.tol_one)
        assert np.all(np.abs(obs.storage_power - [0.0, 0.0]) <= self.tol_one)

    def test_activate_storage_loss(self):
        """
        test that the parameters param.ACTIVATE_STORAGE_LOSS properly deactivate the loss in the storage
        units
        """
        # first test the storage loss
        act = self.env.action_space()
        loss = 1.0 * self.env.storage_loss
        loss /= 12.0  # i have 12 steps per hour (ts = (mins), losses are given in MW and capacity in MWh
        for nb_ts in range(5):
            obs, reward, done, info = self.env.step(act)
            assert np.all(
                np.abs(
                    obs.storage_charge - (0.5 * obs.storage_Emax - (nb_ts + 1) * loss)
                )
                <= self.tol_one
            ), f"wrong value computed for time step {nb_ts}"

        param = Parameters()
        param.ACTIVATE_STORAGE_LOSS = False
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("educ_case14_storage", test=True, param=param)
        obs = env.get_obs()
        assert np.all(
            np.abs(obs.storage_charge - 0.5 * obs.storage_Emax) <= self.tol_one
        ), "wrong initial capacity"
        for nb_ts in range(5):
            obs, reward, done, info = env.step(act)
            assert np.all(
                np.abs(obs.storage_charge - 0.5 * obs.storage_Emax) <= self.tol_one
            ), f"wrong value computed for time step {nb_ts} (no loss in storage)"

    def test_storage_loss_dont_make_negative(self):
        """
        test that the storage loss dont make negative capacity
        or in other words that loss don't apply when storage are empty
        """
        init_coeff = 0.01
        param = Parameters()
        param.ACTIVATE_STORAGE_LOSS = True
        param.INIT_STORAGE_CAPACITY = init_coeff
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("educ_case14_storage", test=True, param=param)
        obs = env.get_obs()
        init_charge = init_coeff * obs.storage_Emax
        loss = 1.0 * env.storage_loss
        loss /= 12.0  # i have 12 steps per hour (ts =  5mins, losses are given in MW and capacity in MWh
        act = env.action_space()

        assert np.all(
            np.abs(obs.storage_charge - init_charge) <= self.tol_one
        ), "wrong initial capacity"
        for nb_ts in range(8):
            obs, reward, done, info = env.step(act)
            assert np.all(
                np.abs(obs.storage_charge - (init_charge - (nb_ts + 1) * loss))
                <= self.tol_one
            ), f"wrong value computed for time step {nb_ts} (with loss in storage)"

        # now a loss should 'cap' the second battery
        obs, reward, done, info = env.step(act)
        th_storage = init_charge - (nb_ts + 1) * loss
        th_storage[0] -= loss[0]
        th_storage[1] = 0.0
        assert np.all(np.abs(obs.storage_charge - th_storage) <= self.tol_one)
        for nb_ts in range(9):
            obs, reward, done, info = env.step(act)
            th_storage[0] -= loss[0]
            assert np.all(
                np.abs(obs.storage_charge - th_storage) <= self.tol_one
            ), f"capacity error for time step {nb_ts}"

        # all storages are empty
        obs, reward, done, info = env.step(act)
        assert np.all(
            np.abs(obs.storage_charge) <= self.tol_one
        ), "error battery should be empty - 0"
        obs, reward, done, info = env.step(act)
        assert np.all(
            np.abs(obs.storage_charge) <= self.tol_one
        ), "error, battery should be empty - 1"
        obs, reward, done, info = env.step(act)
        assert np.all(
            np.abs(obs.storage_charge) <= self.tol_one
        ), "error, battery should be empty - 2"

    def test_env_storage_ambiguous(self):
        """test i can detect ambiguous storage actions"""

        # above the maximum flow you can absorb (max is 5)
        act = self.env.action_space({"set_storage": [(0, 5.1)]})
        obs, reward, done, info = self.env.step(act)
        assert info["exception"]
        assert np.abs(np.sum(obs.actual_dispatch) - (0.0)) <= self.tol_one
        assert np.all(np.abs(obs.storage_power_target - [0.0, 0.0]) <= self.tol_one)
        assert np.all(np.abs(obs.storage_power - [0.0, 0.0]) <= self.tol_one)

        # lower the maximum flow you can produce (max is 10)
        act = self.env.action_space({"set_storage": [(1, -10.1)]})
        obs, reward, done, info = self.env.step(act)
        assert info["exception"]
        assert np.abs(np.sum(obs.actual_dispatch) - (0.0)) <= self.tol_one
        assert np.all(np.abs(obs.storage_power_target - [0.0, 0.0]) <= self.tol_one)
        assert np.all(np.abs(obs.storage_power - [0.0, 0.0]) <= self.tol_one)

        # wrong number of storage
        with self.assertRaises(Exception):
            act = self.env.action_space({"set_storage": np.zeros(3)})
            obs, reward, done, info = self.env.step(act)
            if info["exception"]:
                raise info["exception"][
                    0
                ]  # test should pass: if action can be done, it should be ambiguous
            # if info["exception"] is [] then i don't raise anything, and the test fails
        assert np.abs(np.sum(obs.actual_dispatch) - (0.0)) <= self.tol_one
        assert np.all(np.abs(obs.storage_power_target - [0.0, 0.0]) <= self.tol_one)
        assert np.all(np.abs(obs.storage_power - [0.0, 0.0]) <= self.tol_one)

        with self.assertRaises(Exception):
            act = self.env.action_space({"set_storage": np.zeros(1)})
            obs, reward, done, info = self.env.step(act)
            if info["exception"]:
                raise info["exception"][
                    0
                ]  # test should pass: if action can be done, it should be ambiguous
            # if info["exception"] is [] then i don't raise anything, and the test fails
        assert np.abs(np.sum(obs.actual_dispatch) - (0.0)) <= self.tol_one
        assert np.all(np.abs(obs.storage_power_target - [0.0, 0.0]) <= self.tol_one)
        assert np.all(np.abs(obs.storage_power - [0.0, 0.0]) <= self.tol_one)

    def test_env_storage_cut_because_too_high_noloss(self):
        """
        test the correct behaviour is met when storage energy would be too high (need to cut DOWN the action)
        and we don't take into account the loss and inefficiencies
        """
        init_coeff = 0.99
        param = Parameters()
        param.ACTIVATE_STORAGE_LOSS = (
            False  # to simplify the computation, in this first test
        )
        param.INIT_STORAGE_CAPACITY = init_coeff
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("educ_case14_storage", test=True, param=param)
        self.env.close()
        self.env = env

        obs = self.env.reset()
        init_storage = np.array([14.85, 6.9300003], dtype=dt_float)
        assert np.all(np.abs(obs.storage_charge - init_storage) <= self.tol_one)

        # too high in second battery, ok for first step
        array_modif = np.array([1.5, 10.0], dtype=dt_float)
        act = self.env.action_space({"set_storage": array_modif})
        obs, reward, done, info = self.env.step(act)
        assert not info["exception"]
        assert np.all(np.abs(obs.storage_power_target - array_modif) <= self.tol_one)
        assert (
            np.abs(np.sum(obs.actual_dispatch) - np.sum(obs.storage_power))
            <= self.tol_one
        )
        assert np.all(obs.storage_charge <= self.env.storage_Emax)
        bat_energy_added = (
            obs.storage_power / 12.0
        )  # amount of energy if the power is maintained for 5mins
        assert np.all(
            np.abs(obs.storage_charge - (bat_energy_added + init_storage))
            <= self.tol_one
        )
        state_of_charge = 1.0 * obs.storage_charge

        # now both batteries are capped
        obs, reward, done, info = self.env.step(act)
        assert not info["exception"]
        assert np.all(np.abs(obs.storage_power_target - array_modif) <= self.tol_one)
        assert (
            np.abs(np.sum(obs.actual_dispatch) - np.sum(obs.storage_power))
            <= self.tol_one
        )
        assert np.all(obs.storage_charge <= self.env.storage_Emax)
        assert (
            np.abs(obs.storage_power[1]) <= self.tol_one
        )  # second battery cannot charge anymore
        bat_energy_added = (
            obs.storage_power / 12.0
        )  # amount of energy if the power is maintained for 5mins
        assert np.all(
            np.abs(obs.storage_charge - (bat_energy_added + state_of_charge))
            <= self.tol_one
        )
        state_of_charge = 1.0 * obs.storage_charge

        # now both batteries are capped
        obs, reward, done, info = self.env.step(act)
        assert not info["exception"]
        assert np.all(np.abs(obs.storage_power_target - array_modif) <= self.tol_one)
        assert (
            np.abs(np.sum(obs.actual_dispatch) - np.sum(obs.storage_power))
            <= self.tol_one
        )
        assert np.all(obs.storage_charge <= self.env.storage_Emax)
        assert np.all(
            np.abs(obs.storage_power) <= self.tol_one
        )  # all batteries cannot charge anymore
        bat_energy_added = (
            obs.storage_power / 12.0
        )  # amount of energy if the power is maintained for 5mins
        assert np.all(
            np.abs(obs.storage_charge - (bat_energy_added + state_of_charge))
            <= self.tol_one
        )

    def test_env_storage_cut_because_too_high_withloss(self):
        """test the correct behaviour is met when storage energy would be too high (need to cut DOWN the action)"""
        init_coeff = 0.99
        param = Parameters()
        param.ACTIVATE_STORAGE_LOSS = True
        param.INIT_STORAGE_CAPACITY = init_coeff
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("educ_case14_storage", test=True, param=param)
        self.env.close()
        self.env = env

        obs = self.env.reset()
        init_storage = np.array([14.85, 6.9300003], dtype=dt_float)
        assert np.all(np.abs(obs.storage_charge - init_storage) <= self.tol_one)

        loss = 1.0 * env.storage_loss
        loss /= 12.0  # i have 12 steps per hour (ts =  5mins, losses are given in MW and capacity in MWh)

        # too high in second battery, ok for first step
        array_modif = np.array([1.5, 10.0], dtype=dt_float)
        act = self.env.action_space({"set_storage": array_modif})
        obs, reward, done, info = self.env.step(act)
        assert not info["exception"]
        assert np.all(np.abs(obs.storage_power_target - array_modif) <= self.tol_one)
        assert (
            np.abs(np.sum(obs.actual_dispatch) - np.sum(obs.storage_power))
            <= self.tol_one
        )
        assert np.all(obs.storage_charge <= self.env.storage_Emax)
        bat_energy_added = (
            obs.storage_power / 12.0
        )  # amount of energy if the power is maintained for 5mins
        bat_energy_added *= obs.storage_charging_efficiency  # there are inefficiencies
        assert np.all(
            np.abs(obs.storage_charge - (bat_energy_added + init_storage - loss))
            <= self.tol_one
        )
        # only the loss makes the second storage unit not full
        assert (
            np.abs(obs.storage_charge[1] - (self.env.storage_Emax[1] - loss[1]))
            <= self.tol_one
        )
        state_of_charge = 1.0 * obs.storage_charge

        # after this action both batteries are capped
        obs, reward, done, info = self.env.step(act)
        assert not info["exception"]
        assert np.all(np.abs(obs.storage_power_target - array_modif) <= self.tol_one)
        assert (
            np.abs(np.sum(obs.actual_dispatch) - np.sum(obs.storage_power))
            <= self.tol_one
        )
        assert np.all(obs.storage_charge <= self.env.storage_Emax)
        # second battery cannot charge more than the loss
        val = env.storage_loss[1] / self.env.storage_charging_efficiency[1]
        assert np.abs(obs.storage_power[1] - val) <= self.tol_one
        # all batteries are charged at maximum now
        assert np.all(
            np.abs(obs.storage_charge - (self.env.storage_Emax - loss)) <= self.tol_one
        )
        bat_energy_added = (
            1.0 * obs.storage_power / 12.0
        )  # amount of energy if the power is maintained for 5mins
        bat_energy_added *= obs.storage_charging_efficiency  # there are inefficiencies
        assert np.all(
            np.abs(obs.storage_charge - (bat_energy_added + state_of_charge - loss))
            <= self.tol_one
        )
        state_of_charge = 1.0 * obs.storage_charge

        # both batteries are at maximum, i can only charge them of the losses
        obs, reward, done, info = self.env.step(act)
        assert not info["exception"]
        assert np.all(np.abs(obs.storage_power_target - array_modif) <= self.tol_one)
        assert (
            np.abs(np.sum(obs.actual_dispatch) - np.sum(obs.storage_power))
            <= self.tol_one
        )
        assert np.all(obs.storage_charge <= self.env.storage_Emax)
        # second battery cannot charge more than the loss
        val = env.storage_loss / self.env.storage_charging_efficiency
        assert np.all(np.abs(obs.storage_power - val) <= self.tol_one)
        # all batteries are charged at maximum now
        assert np.all(
            np.abs(obs.storage_charge - (self.env.storage_Emax - loss)) <= self.tol_one
        )
        bat_energy_added = (
            1.0 * obs.storage_power / 12.0
        )  # amount of energy if the power is maintained for 5mins
        bat_energy_added *= obs.storage_charging_efficiency  # there are inefficiencies
        assert np.all(
            np.abs(obs.storage_charge - (bat_energy_added + state_of_charge - loss))
            <= self.tol_one
        )

    def test_env_storage_cut_because_too_low_noloss(self):
        """
        test the correct behaviour is met when storage energy would be too low (need to cut the action)
        and we don't take into account the loss and inefficiencies
        """
        init_coeff = 0.01
        param = Parameters()
        param.ACTIVATE_STORAGE_LOSS = (
            False  # to simplify the computation, in this first test
        )
        param.INIT_STORAGE_CAPACITY = init_coeff
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("educ_case14_storage", test=True, param=param)
        self.env.close()
        self.env = env

        obs = self.env.reset()
        init_storage = np.array([0.14999999, 0.07], dtype=dt_float)
        assert np.all(np.abs(obs.storage_charge - init_storage) <= self.tol_one)

        # too high in second battery, ok for first step
        array_modif = np.array([-1.5, -10.0], dtype=dt_float)
        act = self.env.action_space({"set_storage": array_modif})
        obs, reward, done, info = self.env.step(act)
        assert not info["exception"]
        assert np.all(np.abs(obs.storage_power_target - array_modif) <= self.tol_one)
        assert (
            np.abs(np.sum(obs.actual_dispatch) - np.sum(obs.storage_power))
            <= self.tol_one
        )
        assert np.all(obs.storage_charge >= self.env.storage_Emin)
        bat_energy_added = (
            obs.storage_power / 12.0
        )  # amount of energy if the power is maintained for 5mins
        assert np.all(
            np.abs(obs.storage_charge - (bat_energy_added + init_storage))
            <= self.tol_one
        )
        state_of_charge = 1.0 * obs.storage_charge

        # now both batteries are capped
        obs, reward, done, info = self.env.step(act)
        assert not info["exception"]
        assert np.all(np.abs(obs.storage_power_target - array_modif) <= self.tol_one)
        assert (
            np.abs(np.sum(obs.actual_dispatch) - np.sum(obs.storage_power))
            <= self.tol_one
        )
        assert np.all(obs.storage_charge >= self.env.storage_Emin)
        assert (
            np.abs(obs.storage_power[1]) <= self.tol_one
        )  # second battery cannot charge anymore
        bat_energy_added = (
            obs.storage_power / 12.0
        )  # amount of energy if the power is maintained for 5mins
        assert np.all(
            np.abs(obs.storage_charge - (bat_energy_added + state_of_charge))
            <= self.tol_one
        )
        state_of_charge = 1.0 * obs.storage_charge

        # now both batteries are capped
        obs, reward, done, info = self.env.step(act)
        assert not info["exception"]
        assert np.all(np.abs(obs.storage_power_target - array_modif) <= self.tol_one)
        assert (
            np.abs(np.sum(obs.actual_dispatch) - np.sum(obs.storage_power))
            <= self.tol_one
        )
        assert np.all(obs.storage_charge >= self.env.storage_Emin)
        assert np.all(
            np.abs(obs.storage_power) <= self.tol_one
        )  # all batteries cannot charge anymore
        bat_energy_added = (
            obs.storage_power / 12.0
        )  # amount of energy if the power is maintained for 5mins
        assert np.all(
            np.abs(obs.storage_charge - (bat_energy_added + state_of_charge))
            <= self.tol_one
        )

    def test_env_storage_cut_because_too_low_withloss(self):
        """test the correct behaviour is met when storage energy would be too low (need to cut the action)"""
        init_coeff = 0.01
        param = Parameters()
        param.ACTIVATE_STORAGE_LOSS = True
        param.INIT_STORAGE_CAPACITY = init_coeff
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("educ_case14_storage", test=True, param=param)
        self.env.close()
        self.env = env

        obs = self.env.reset()
        init_storage = np.array([0.14999999, 0.07], dtype=dt_float)
        assert np.all(np.abs(obs.storage_charge - init_storage) <= self.tol_one)

        loss = 1.0 * env.storage_loss
        loss /= 12.0  # i have 12 steps per hour (ts =  5mins, losses are given in MW and capacity in MWh)

        # too high in second battery, ok for first step
        array_modif = np.array([-1.5, -10.0], dtype=dt_float)
        act = self.env.action_space({"set_storage": array_modif})
        obs, reward, done, info = self.env.step(act)
        assert not info["exception"]
        assert np.all(np.abs(obs.storage_power_target - array_modif) <= self.tol_one)
        assert (
            np.abs(np.sum(obs.actual_dispatch) - np.sum(obs.storage_power))
            <= self.tol_one
        )
        assert np.all(obs.storage_charge >= self.env.storage_Emin)
        assert np.all(obs.storage_power <= 0.0)  # I emptied the battery
        bat_energy_added = (
            obs.storage_power / 12.0
        )  # amount of energy if the power is maintained for 5mins
        # there are inefficiencies (I remove MORE energy from the battery than what i get in the grid)
        bat_energy_added /= obs.storage_discharging_efficiency
        # below i said [loss[0], 0.] because i don't have loss on an empty battery
        assert np.all(
            np.abs(
                obs.storage_charge - (bat_energy_added + init_storage - [loss[0], 0.0])
            )
            <= self.tol_one
        )
        # only the loss makes the second storage unit not full
        assert (
            np.abs(obs.storage_charge[1] - (self.env.storage_Emin[1])) <= self.tol_one
        )
        state_of_charge = 1.0 * obs.storage_charge

        # after this action both batteries are capped
        obs, reward, done, info = self.env.step(act)
        assert not info["exception"]
        assert np.all(np.abs(obs.storage_power_target - array_modif) <= self.tol_one)
        assert (
            np.abs(np.sum(obs.actual_dispatch) - np.sum(obs.storage_power))
            <= self.tol_one
        )
        assert np.all(obs.storage_charge >= self.env.storage_Emin)
        assert np.all(obs.storage_power <= 0.0)  # I emptied the battery
        # second battery cannot be discharged more
        assert np.abs(obs.storage_power[1] - 0.0) <= self.tol_one
        # all batteries are charged at minimum now (only because Emin is 0.)
        assert np.all(
            np.abs(obs.storage_charge - self.env.storage_Emin) <= self.tol_one
        )
        bat_energy_added = (
            1.0 * obs.storage_power / 12.0
        )  # amount of energy if the power is maintained for 5mins
        # there are inefficiencies (I remove MORE energy from the battery than what i get in the grid)
        bat_energy_added /= (
            obs.storage_discharging_efficiency
        )  # there are inefficiencies
        # below i said [0., 0.] because i don't have loss on an empty battery
        assert np.all(
            np.abs(
                obs.storage_charge - (bat_energy_added + state_of_charge - [0.0, 0.0])
            )
            <= self.tol_one
        )
        state_of_charge = 1.0 * obs.storage_charge

        # both batteries are at maximum, i can only charge them of the losses
        obs, reward, done, info = self.env.step(act)
        assert not info["exception"]
        assert np.all(np.abs(obs.storage_power_target - array_modif) <= self.tol_one)
        assert (
            np.abs(np.sum(obs.actual_dispatch) - np.sum(obs.storage_power))
            <= self.tol_one
        )
        assert np.all(obs.storage_charge >= self.env.storage_Emin)
        assert np.all(obs.storage_power <= 0.0)  # I emptied the battery
        # second battery cannot be discharged more than it is
        assert np.all(np.abs(obs.storage_power - 0.0) <= self.tol_one)
        # all batteries are charged at maximum now
        assert np.all(
            np.abs(obs.storage_charge - self.env.storage_Emin) <= self.tol_one
        )
        bat_energy_added = (
            1.0 * obs.storage_power / 12.0
        )  # amount of energy if the power is maintained for 5mins
        # there are inefficiencies (I remove MORE energy from the battery than what i get in the grid)
        bat_energy_added /= (
            obs.storage_discharging_efficiency
        )  # there are inefficiencies
        # below i said [0., 0.] because i don't have loss on an empty battery
        assert np.all(
            np.abs(
                obs.storage_charge - (bat_energy_added + state_of_charge - [0.0, 0.0])
            )
            <= self.tol_one
        )

    def _aux_test_kirchoff(self):
        p_subs, q_subs, p_bus, q_bus, diff_v_bus = self.env.backend.check_kirchoff()
        assert np.all(
            np.abs(p_subs) <= self.tol_one
        ), "error with active value at some substations"
        assert np.all(
            np.abs(q_subs) <= self.tol_one
        ), "error with reactive value at some substations"
        assert np.all(
            np.abs(p_bus) <= self.tol_one
        ), "error with active value at some bus"
        assert np.all(
            np.abs(q_bus) <= self.tol_one
        ), "error with reactive value at some bus"
        assert np.all(diff_v_bus <= self.tol_one), "error with voltage discrepency"

    def test_storage_action_mw(self):
        """test the actions are properly implemented in the backend"""
        array_modif = np.array([-1.5, -10.0], dtype=dt_float)
        act = self.env.action_space({"set_storage": array_modif})
        obs, reward, done, info = self.env.step(act)
        assert not info["exception"]
        storage_p, storage_q, storage_v = self.env.backend.storages_info()
        assert np.all(np.abs(storage_p - array_modif) <= self.tol_one)
        assert np.all(np.abs(storage_q - 0.0) <= self.tol_one)
        self._aux_test_kirchoff()

        array_modif = np.array([2, 8], dtype=dt_float)
        act = self.env.action_space({"set_storage": array_modif})
        obs, reward, done, info = self.env.step(act)
        assert not info["exception"]
        storage_p, storage_q, storage_v = self.env.backend.storages_info()
        assert np.all(np.abs(storage_p - array_modif) <= self.tol_one)
        assert np.all(np.abs(storage_q - 0.0) <= self.tol_one)
        self._aux_test_kirchoff()

        # illegal action
        array_modif = np.array([2, 12], dtype=dt_float)
        act = self.env.action_space({"set_storage": array_modif})
        obs, reward, done, info = self.env.step(act)
        assert info["exception"]
        storage_p, storage_q, storage_v = self.env.backend.storages_info()
        assert np.all(np.abs(storage_p - [0.0, 0.0]) <= self.tol_one)
        assert np.all(np.abs(storage_q - 0.0) <= self.tol_one)
        self._aux_test_kirchoff()

        # full discharge now
        array_modif = np.array([-1.5, -10.0], dtype=dt_float)
        for nb_ts in range(3):
            act = self.env.action_space({"set_storage": array_modif})
            obs, reward, done, info = self.env.step(act)
            assert not info["exception"]
            storage_p, storage_q, storage_v = self.env.backend.storages_info()
            assert np.all(
                np.abs(storage_p - array_modif) <= self.tol_one
            ), f"error for P for time step {nb_ts}"
            assert np.all(
                np.abs(storage_q - 0.0) <= self.tol_one
            ), f"error for Q for time step {nb_ts}"
            self._aux_test_kirchoff()

        obs, reward, done, info = self.env.step(act)
        assert not info["exception"]
        # i have emptied second battery
        assert np.all(
            np.abs(self.env.backend._grid.storage["p_mw"].values - [-1.5, -4.4599934])
            <= self.tol_one
        )
        assert np.all(np.abs(obs.storage_charge[1] - 0.0) <= self.tol_one)
        self._aux_test_kirchoff()

        obs, reward, done, info = self.env.step(act)
        assert not info["exception"]
        # i have emptied second battery
        assert np.all(
            np.abs(self.env.backend._grid.storage["p_mw"].values - [-1.5, 0.0])
            <= self.tol_one
        )
        assert np.all(np.abs(obs.storage_charge[1] - 0.0) <= self.tol_one)
        self._aux_test_kirchoff()

    def test_storage_action_topo(self):
        """test the modification of the bus of a storage unit"""
        param = Parameters()
        param.NB_TIMESTEP_COOLDOWN_SUB = 0
        param.NB_TIMESTEP_COOLDOWN_LINE = 0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make(
                "educ_case14_storage",
                test=True,
                action_class=CompleteAction,
                param=param,
            )
        self.env.close()
        self.env = env
        obs = self.env.reset()

        # first case, standard modification
        array_modif = np.array([-1.5, -10.0], dtype=dt_float)
        act = self.env.action_space(
            {
                "set_storage": array_modif,
                "set_bus": {
                    "storages_id": [(0, 2)],
                    "lines_or_id": [(8, 2)],
                    "generators_id": [(3, 2)],
                },
            }
        )
        obs, reward, done, info = self.env.step(act)

        assert not info["exception"]
        storage_p, storage_q, storage_v = self.env.backend.storages_info()
        assert np.all(np.abs(storage_p - array_modif) <= self.tol_one)
        assert np.all(np.abs(storage_q - 0.0) <= self.tol_one)
        assert obs.storage_bus[0] == 2
        assert obs.line_or_bus[8] == 2
        assert obs.gen_bus[3] == 2
        self._aux_test_kirchoff()

        # second case, still standard modification (set to orig)
        array_modif = np.array([1.5, 10.0], dtype=dt_float)
        act = self.env.action_space(
            {
                "set_storage": array_modif,
                "set_bus": {
                    "storages_id": [(0, 1)],
                    "lines_or_id": [(8, 1)],
                    "generators_id": [(3, 1)],
                },
            }
        )
        obs, reward, done, info = self.env.step(act)
        assert not info["exception"]
        storage_p, storage_q, storage_v = self.env.backend.storages_info()
        assert np.all(np.abs(storage_p - array_modif) <= self.tol_one)
        assert np.all(np.abs(storage_q - 0.0) <= self.tol_one)
        assert obs.storage_bus[0] == 1
        assert obs.line_or_bus[8] == 1
        assert obs.gen_bus[3] == 1
        self._aux_test_kirchoff()

        # THIS IS EXPECTED THAT IT DOES NOT PASS FROM GRID2OP 1.9.6 !
        # fourth case: isolated storage on a busbar (so it is disconnected, but with 0. production => so thats fine)
        # array_modif = np.array([0.0, 7.0], dtype=dt_float)
        # act = self.env.action_space(
        #     {
        #         "set_storage": array_modif,
        #         "set_bus": {
        #             "storages_id": [(0, 2)],
        #             "lines_or_id": [(8, 1)],
        #             "generators_id": [(3, 1)],
        #         },
        #     }
        # )
        # obs, reward, done, info = self.env.step(act)
        # assert not info["exception"]
        # storage_p, storage_q, storage_v = self.env.backend.storages_info()
        # assert np.all(
        #     np.abs(storage_p - [0.0, array_modif[1]]) <= self.tol_one
        # ), "storage is not disconnected, yet alone on its busbar"
        # assert np.all(np.abs(storage_q - 0.0) <= self.tol_one)
        # assert obs.storage_bus[0] == -1, "storage should be disconnected"
        # assert storage_v[0] == 0.0, "storage 0 should be disconnected"
        # assert obs.line_or_bus[8] == 1
        # assert obs.gen_bus[3] == 1
        # self._aux_test_kirchoff()

        # # check that if i don't touch it it's set to 0
        # act = self.env.action_space()
        # obs, reward, done, info = self.env.step(act)
        # assert not info["exception"]
        # storage_p, storage_q, storage_v = self.env.backend.storages_info()
        # assert np.all(
        #     np.abs(storage_p - 0.0) <= self.tol_one
        # ), "storage should produce 0"
        # assert np.all(
        #     np.abs(storage_q - 0.0) <= self.tol_one
        # ), "storage should produce 0"
        # assert obs.storage_bus[0] == -1, "storage should be disconnected"
        # assert storage_v[0] == 0.0, "storage 0 should be disconnected"
        # assert obs.line_or_bus[8] == 1
        # assert obs.gen_bus[3] == 1
        # self._aux_test_kirchoff()

        # # trying to act on a disconnected storage => illegal)
        # array_modif = np.array([2.0, 7.0], dtype=dt_float)
        # act = self.env.action_space({"set_storage": array_modif})
        # obs, reward, done, info = self.env.step(act)
        # assert info["exception"]  # action should be illegal
        # assert not done  # this is fine, as it's illegal it's replaced by do nothing
        # self._aux_test_kirchoff()

        # # trying to reconnect a storage alone on a bus => game over, not connected bus
        # array_modif = np.array([1.0, 7.0], dtype=dt_float)
        # act = self.env.action_space(
        #     {
        #         "set_storage": array_modif,
        #         "set_bus": {
        #             "storages_id": [(0, 2)],
        #             "lines_or_id": [(8, 1)],
        #             "generators_id": [(3, 1)],
        #         },
        #     }
        # )
        # obs, reward, done, info = self.env.step(act)
        # assert info["exception"]  # this is a game over
        # assert done

if __name__ == "__main__":
    unittest.main()
