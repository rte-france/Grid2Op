# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import pdb

from grid2op.tests.helper_path_test import *

from grid2op.Exceptions import *
from grid2op.Observation import CompleteObservation
from grid2op.Chronics import (
    ChronicsHandler,
    GridStateFromFile,
    GridStateFromFileWithForecasts,
)
from grid2op.Rules import RulesChecker, DefaultRules
from grid2op.Reward import L2RPNReward
from grid2op.Parameters import Parameters
from grid2op.Backend import PandaPowerBackend
from grid2op.Environment import Environment

# TODO add unit test for the proper update the backend in the observation [for now there is a "data leakage" as
# the real backend is copied when the observation is built, but i need to make a test to check that's it's properly
# copied]

# temporary deactivation of all the failing test until simulate is fixed
DEACTIVATE_FAILING_TEST = False

import warnings

warnings.simplefilter("error")


class TestObservationHazard(unittest.TestCase):
    def setUp(self):
        """
        The case file is a representation of the case14 as found in the ieee14 powergrid.
        :return:
        """
        # from ADNBackend import ADNBackend
        # self.backend = ADNBackend()
        # self.path_matpower = "/home/donnotben/Documents/RL4Grid/RL4Grid/data"
        # self.case_file = "ieee14_ADN.xml"
        # self.backend.load_grid(self.path_matpower, self.case_file)
        self.tolvect = 1e-2
        self.tol_one = 1e-5
        self.game_rules = RulesChecker()
        # pdb.set_trace()
        self.rewardClass = L2RPNReward
        self.reward_helper = self.rewardClass()
        self.obsClass = CompleteObservation
        self.parameters = Parameters()

        # powergrid
        self.backend = PandaPowerBackend()
        self.path_matpower = PATH_DATA_TEST_PP
        self.case_file = "test_case14.json"

        # chronics
        self.path_chron = os.path.join(PATH_CHRONICS, "chronics_with_hazards")
        self.chronics_handler = ChronicsHandler(
            chronicsClass=GridStateFromFile, path=self.path_chron
        )

        self.tolvect = 1e-2
        self.tol_one = 1e-5
        self.id_chron_to_back_load = np.array([0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9])

        # force the verbose backend
        self.backend.detailed_infos_for_cascading_failures = True

        self.names_chronics_to_backend = {
            "loads": {
                "2_C-10.61": "load_1_0",
                "3_C151.15": "load_2_1",
                "14_C63.6": "load_13_2",
                "4_C-9.47": "load_3_3",
                "5_C201.84": "load_4_4",
                "6_C-6.27": "load_5_5",
                "9_C130.49": "load_8_6",
                "10_C228.66": "load_9_7",
                "11_C-138.89": "load_10_8",
                "12_C-27.88": "load_11_9",
                "13_C-13.33": "load_12_10",
            },
            "lines": {
                "1_2_1": "0_1_0",
                "1_5_2": "0_4_1",
                "9_10_16": "8_9_2",
                "9_14_17": "8_13_3",
                "10_11_18": "9_10_4",
                "12_13_19": "11_12_5",
                "13_14_20": "12_13_6",
                "2_3_3": "1_2_7",
                "2_4_4": "1_3_8",
                "2_5_5": "1_4_9",
                "3_4_6": "2_3_10",
                "4_5_7": "3_4_11",
                "6_11_11": "5_10_12",
                "6_12_12": "5_11_13",
                "6_13_13": "5_12_14",
                "4_7_8": "3_6_15",
                "4_9_9": "3_8_16",
                "5_6_10": "4_5_17",
                "7_8_14": "6_7_18",
                "7_9_15": "6_8_19",
            },
            "prods": {
                "1_G137.1": "gen_0_4",
                "3_G36.31": "gen_2_1",
                "6_G63.29": "gen_5_2",
                "2_G-56.47": "gen_1_0",
                "8_G40.43": "gen_7_3",
            },
        }

        # _parameters for the environment
        self.env_params = Parameters()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = Environment(
                init_grid_path=os.path.join(self.path_matpower, self.case_file),
                backend=self.backend,
                init_env_path=os.path.join(self.path_matpower, self.case_file),
                chronics_handler=self.chronics_handler,
                parameters=self.env_params,
                names_chronics_to_backend=self.names_chronics_to_backend,
                rewardClass=self.rewardClass,
                name="test_obs_env1",
            )

    def tearDown(self) -> None:
        self.env.close()

    def test_1_generating_obs_withhazard(self):
        # test that helper_obs is abl to generate a valid observation
        obs = self.env.get_obs()
        assert np.all(
            obs.time_before_cooldown_line
            == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )
        action = self.env.action_space({})
        _ = self.env.step(action)
        obs = self.env.get_obs()
        assert np.all(
            obs.time_before_cooldown_line
            == [0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )
        _ = self.env.step(action)
        obs = self.env.get_obs()
        assert np.all(
            obs.time_before_cooldown_line
            == [0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )


class TestObservationMaintenance(unittest.TestCase):
    def setUp(self):
        """
        The case file is a representation of the case14 as found in the ieee14 powergrid.
        :return:
        """
        self.tolvect = 1e-2
        self.tol_one = 1e-5
        self.game_rules = RulesChecker()
        self.rewardClass = L2RPNReward
        self.reward_helper = self.rewardClass()
        self.obsClass = CompleteObservation
        self.parameters = Parameters()

        # powergrid
        self.backend = PandaPowerBackend()
        self.path_matpower = PATH_DATA_TEST_PP
        self.case_file = "test_case14.json"

        # chronics
        self.path_chron = os.path.join(PATH_CHRONICS, "chronics_with_maintenance")
        self.chronics_handler = ChronicsHandler(
            chronicsClass=GridStateFromFileWithForecasts, path=self.path_chron
        )

        self.tolvect = 1e-2
        self.tol_one = 1e-5
        self.id_chron_to_back_load = np.array([0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9])

        # force the verbose backend
        self.backend.detailed_infos_for_cascading_failures = True

        self.names_chronics_to_backend = {
            "loads": {
                "2_C-10.61": "load_1_0",
                "3_C151.15": "load_2_1",
                "14_C63.6": "load_13_2",
                "4_C-9.47": "load_3_3",
                "5_C201.84": "load_4_4",
                "6_C-6.27": "load_5_5",
                "9_C130.49": "load_8_6",
                "10_C228.66": "load_9_7",
                "11_C-138.89": "load_10_8",
                "12_C-27.88": "load_11_9",
                "13_C-13.33": "load_12_10",
            },
            "lines": {
                "1_2_1": "0_1_0",
                "1_5_2": "0_4_1",
                "9_10_16": "8_9_2",
                "9_14_17": "8_13_3",
                "10_11_18": "9_10_4",
                "12_13_19": "11_12_5",
                "13_14_20": "12_13_6",
                "2_3_3": "1_2_7",
                "2_4_4": "1_3_8",
                "2_5_5": "1_4_9",
                "3_4_6": "2_3_10",
                "4_5_7": "3_4_11",
                "6_11_11": "5_10_12",
                "6_12_12": "5_11_13",
                "6_13_13": "5_12_14",
                "4_7_8": "3_6_15",
                "4_9_9": "3_8_16",
                "5_6_10": "4_5_17",
                "7_8_14": "6_7_18",
                "7_9_15": "6_8_19",
            },
            "prods": {
                "1_G137.1": "gen_0_4",
                "3_G36.31": "gen_2_1",
                "6_G63.29": "gen_5_2",
                "2_G-56.47": "gen_1_0",
                "8_G40.43": "gen_7_3",
            },
        }

        # _parameters for the environment
        self.env_params = Parameters()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = Environment(
                init_grid_path=os.path.join(self.path_matpower, self.case_file),
                backend=self.backend,
                init_env_path=os.path.join(self.path_matpower, self.case_file),
                chronics_handler=self.chronics_handler,
                parameters=self.env_params,
                names_chronics_to_backend=self.names_chronics_to_backend,
                rewardClass=self.rewardClass,
                name="test_obs_env2",
                legalActClass=DefaultRules,
            )

    def tearDown(self) -> None:
        self.env.close()

    def test_1_generating_obs_withmaintenance(self):
        # test that helper_obs is abl to generate a valid observation
        obs = self.env.get_obs()
        assert np.all(
            obs.time_next_maintenance
            == np.array(
                [
                    -1,
                    -1,
                    -1,
                    -1,
                    1,
                    -1,
                    276,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            )
        )
        assert np.all(
            obs.duration_next_maintenance
            == np.array([0, 0, 0, 0, 12, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        )
        action = self.env.action_space({})
        _ = self.env.step(action)
        obs = self.env.get_obs()
        assert np.all(
            obs.time_next_maintenance
            == np.array(
                [
                    -1,
                    -1,
                    -1,
                    -1,
                    0,
                    -1,
                    275,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            )
        )
        assert np.all(
            obs.duration_next_maintenance
            == np.array([0, 0, 0, 0, 12, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        )
        _ = self.env.step(action)
        obs = self.env.get_obs()
        assert np.all(
            obs.time_next_maintenance
            == np.array(
                [
                    -1,
                    -1,
                    -1,
                    -1,
                    0,
                    -1,
                    274,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            )
        )
        assert np.all(
            obs.duration_next_maintenance
            == np.array([0, 0, 0, 0, 11, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        )

    def test_simulate_disco_planned_maintenance(self):
        
        reco_line = self.env.action_space()
        reco_line.line_set_status = [(4, +1)]
        
        obs = self.env.get_obs()
        assert obs.line_status[4]
        assert obs.time_next_maintenance[4] == 1
        assert obs.duration_next_maintenance[4] == 12

        # line will be disconnected next time step
        sim_obs, *_ = obs.simulate(self.env.action_space(), time_step=1)
        assert not sim_obs.line_status[4]
        assert sim_obs.time_next_maintenance[4] == 0
        assert sim_obs.duration_next_maintenance[4] == 12
        
        # simulation at current step
        sim_obs, *_ = obs.simulate(self.env.action_space(), time_step=0)
        assert sim_obs.line_status[4]
        assert sim_obs.time_next_maintenance[4] == 1
        assert sim_obs.duration_next_maintenance[4] == 12
        
        # line will be disconnected next time step
        sim_obs, *_ = obs.simulate(self.env.action_space(), time_step=1)
        assert not sim_obs.line_status[4]
        assert sim_obs.time_next_maintenance[4] == 0
        assert sim_obs.duration_next_maintenance[4] == 12

        for ts in range(11):
            obs, reward, done, info = self.env.step(self.env.action_space())
            assert obs.time_next_maintenance[4] == 0
            assert obs.duration_next_maintenance[4] == 12-ts, f"should be {12-ts} but is {obs.duration_next_maintenance[4]} (step ts)"

        # at this step if I attempt a reco it fails
        obs, reward, done, info = self.env.step(self.env.action_space())
        # maintenance will be over next time step
        assert not obs.line_status[4]
        assert obs.time_next_maintenance[4] == 0
        assert obs.duration_next_maintenance[4] == 1

        # if i don't do anything, it's updated properly
        sim_obs, *_ = obs.simulate(self.env.action_space(), time_step=1)
        assert not sim_obs.line_status[4]
        assert sim_obs.time_next_maintenance[4] == -1
        assert sim_obs.duration_next_maintenance[4] == 0
        
        # i don't have the right to reconnect it if i don't simulate in the future
        sim_obs, reward, done, info = obs.simulate(reco_line, time_step=0)
        assert info["is_illegal"]
        assert not sim_obs.line_status[4]
        assert sim_obs.time_next_maintenance[4] == 0
        assert sim_obs.duration_next_maintenance[4] == 1
        
        # I still have to wait 1 step before reconnection, so this raises
        sim_obs, reward, done, info = obs.simulate(reco_line, time_step=1)
        assert info["is_illegal"], f"there should be no error, but action is illegal"
        assert not sim_obs.line_status[4]
        assert sim_obs.time_next_maintenance[4] == -1
        assert sim_obs.duration_next_maintenance[4] == 0
        
        # at this step if I attempt a reco it fails
        obs, reward, done, info = self.env.step(reco_line)
        # maintenance will be over next time step
        assert info["is_illegal"]
        assert not obs.line_status[4]
        assert obs.time_next_maintenance[4] == -1
        assert obs.duration_next_maintenance[4] == 0
        
        # I can reco the line next step
        sim_obs, reward, done, info = obs.simulate(reco_line, time_step=1)
        assert not info["is_illegal"], f"there should be no error, but action is illegal"
        assert sim_obs.line_status[4]
        assert sim_obs.time_next_maintenance[4] == -1
        assert sim_obs.duration_next_maintenance[4] == 0
        
        # TODO be careful here, if the rules allows for reconnection, then the
        # TODO action becomes legal, and the powerline is reconnected
        # TODO => this is because the "_obs_env" do not attempt to force the disconnection
        # TODO of maintenance / hazards powerline


if __name__ == "__main__":
    unittest.main()
