# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import pdb
import warnings
from grid2op.tests.helper_path_test import *

from grid2op.dtypes import dt_int, dt_bool, dt_float
from grid2op.Exceptions import *
from grid2op.Environment import Environment
from grid2op.Backend import PandaPowerBackend
from grid2op.Parameters import Parameters
from grid2op.Chronics import ChronicsHandler, GridStateFromFile
from grid2op.Rules import *
from grid2op.MakeEnv import make

import warnings

warnings.simplefilter("error")


class TestLoadingBackendFunc(unittest.TestCase):
    def setUp(self):
        # powergrid
        self.adn_backend = PandaPowerBackend()
        self.path_matpower = PATH_DATA_TEST_PP
        self.case_file = "test_case14.json"

        # data
        self.path_chron = os.path.join(PATH_CHRONICS, "chronics")
        self.chronics_handler = ChronicsHandler(
            chronicsClass=GridStateFromFile, path=self.path_chron
        )

        self.tolvect = 1e-2
        self.tol_one = 1e-5

        # force the verbose backend
        self.adn_backend.detailed_infos_for_cascading_failures = True

        # _parameters for the environment
        self.env_params = Parameters()

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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = Environment(
                init_grid_path=os.path.join(self.path_matpower, self.case_file),
                backend=self.adn_backend,
                init_env_path=os.path.join(self.path_matpower, self.case_file),
                chronics_handler=self.chronics_handler,
                parameters=self.env_params,
                names_chronics_to_backend=self.names_chronics_to_backend,
                name="test_rules_env1",
            )

        self.helper_action = self.env._helper_action_env

    def test_AlwaysLegal(self):
        # build a random action acting on everything
        new_vect = np.random.randn(self.helper_action.n_load)
        new_vect2 = np.random.randn(self.helper_action.n_load)

        change_status_orig = np.random.randint(0, 2, self.helper_action.n_line).astype(
            dt_bool
        )
        set_status_orig = np.random.randint(-1, 2, self.helper_action.n_line)
        set_status_orig[change_status_orig] = 0

        change_topo_vect_orig = np.random.randint(
            0, 2, self.helper_action.dim_topo
        ).astype(dt_bool)
        # powerline that are set to be reconnected, can't be moved to another bus
        change_topo_vect_orig[
            self.helper_action.line_or_pos_topo_vect[set_status_orig == 1]
        ] = False
        change_topo_vect_orig[
            self.helper_action.line_ex_pos_topo_vect[set_status_orig == 1]
        ] = False
        # powerline that are disconnected, can't be moved to the other bus
        change_topo_vect_orig[
            self.helper_action.line_or_pos_topo_vect[set_status_orig == -1]
        ] = False
        change_topo_vect_orig[
            self.helper_action.line_ex_pos_topo_vect[set_status_orig == -1]
        ] = False

        set_topo_vect_orig = np.random.randint(0, 3, self.helper_action.dim_topo)
        set_topo_vect_orig[change_topo_vect_orig] = 0  # don't both change and set
        # I need to make sure powerlines that are reconnected are indeed reconnected to a bus
        set_topo_vect_orig[
            self.helper_action.line_or_pos_topo_vect[set_status_orig == 1]
        ] = 1
        set_topo_vect_orig[
            self.helper_action.line_ex_pos_topo_vect[set_status_orig == 1]
        ] = 1
        # I need to make sure powerlines that are disconnected are not assigned to a bus
        set_topo_vect_orig[
            self.helper_action.line_or_pos_topo_vect[set_status_orig == -1]
        ] = 0
        set_topo_vect_orig[
            self.helper_action.line_ex_pos_topo_vect[set_status_orig == -1]
        ] = 0

        action = self.helper_action(
            {
                "change_bus": change_topo_vect_orig,
                "set_bus": set_topo_vect_orig,
                "injection": {"load_p": new_vect, "load_q": new_vect2},
                "change_line_status": change_status_orig,
                "set_line_status": set_status_orig,
            }
        )

        # game rules
        gr = RulesChecker()
        assert gr.legal_action(action, self.env)

    def test_LookParam(self):
        id_1 = 1
        id_2 = 12
        id_line = 17
        id_line2 = 15

        arr1 = np.array([False, False, False, True, True, True], dtype=dt_bool)
        arr2 = np.array([1, 1, 2, 2], dtype=dt_int)
        arr_line1 = np.full(self.helper_action.n_line, fill_value=False, dtype=dt_bool)
        arr_line1[id_line] = True
        arr_line2 = np.full(self.helper_action.n_line, fill_value=0, dtype=dt_int)
        arr_line2[id_line2] = 1

        self.helper_action.legal_action = RulesChecker(
            legalActClass=LookParam
        ).legal_action

        self.env._parameters.MAX_SUB_CHANGED = 2
        self.env._parameters.MAX_LINE_STATUS_CHANGED = 2
        _ = self.helper_action(
            {
                "change_bus": {"substations_id": [(id_1, arr1)]},
                "set_bus": {"substations_id": [(id_2, arr2)]},
                "change_line_status": arr_line1,
                "set_line_status": arr_line2,
            },
            env=self.env,
            check_legal=True,
        )

        try:
            self.env._parameters.MAX_SUB_CHANGED = 1
            self.env._parameters.MAX_LINE_STATUS_CHANGED = 2
            _ = self.helper_action(
                {
                    "change_bus": {"substations_id": [(id_1, arr1)]},
                    "set_bus": {"substations_id": [(id_2, arr2)]},
                    "change_line_status": arr_line1,
                    "set_line_status": arr_line2,
                },
                env=self.env,
                check_legal=True,
            )
            raise RuntimeError("This should have thrown an IllegalException")
        except IllegalAction:
            pass

        try:
            self.env._parameters.MAX_SUB_CHANGED = 2
            self.env._parameters.MAX_LINE_STATUS_CHANGED = 1
            _ = self.helper_action(
                {
                    "change_bus": {"substations_id": [(id_1, arr1)]},
                    "set_bus": {"substations_id": [(id_2, arr2)]},
                    "change_line_status": arr_line1,
                    "set_line_status": arr_line2,
                },
                env=self.env,
                check_legal=True,
            )
            raise RuntimeError("This should have thrown an IllegalException")
        except IllegalAction:
            pass

        self.env._parameters.MAX_SUB_CHANGED = 1
        self.env._parameters.MAX_LINE_STATUS_CHANGED = 1
        _ = self.helper_action(
            {
                "change_bus": {"substations_id": [(id_1, arr1)]},
                "set_line_status": arr_line2,
            },
            env=self.env,
            check_legal=True,
        )

    def test_PreventReconection(self):
        id_1 = 1
        id_2 = 12
        id_line = 17
        id_line2 = 15

        arr1 = np.array([False, False, False, True, True, True], dtype=dt_bool)
        arr2 = np.array([1, 1, 2, 2], dtype=dt_int)
        arr_line1 = np.full(self.helper_action.n_line, fill_value=False, dtype=dt_bool)
        arr_line1[id_line] = True
        arr_line2 = np.full(self.helper_action.n_line, fill_value=0, dtype=dt_int)
        arr_line2[id_line2] = 1

        self.helper_action.legal_action = RulesChecker(
            legalActClass=PreventReconnection
        ).legal_action
        self.env._parameters.MAX_SUB_CHANGED = 1
        self.env._parameters.MAX_LINE_STATUS_CHANGED = 2
        act = self.helper_action(
            {
                "change_bus": {"substations_id": [(id_1, arr1)]},
                "set_bus": {"substations_id": [(id_2, arr2)]},
                "change_line_status": arr_line1,
                "set_line_status": arr_line2,
            },
            env=self.env,
            check_legal=True,
        )
        _ = self.env.step(act)

        try:
            self.env._parameters.MAX_SUB_CHANGED = 2
            self.env._parameters.MAX_LINE_STATUS_CHANGED = 1
            self.env._times_before_line_status_actionable[id_line] = 1
            _ = self.helper_action(
                {
                    "change_bus": {"substations_id": [(id_1, arr1)]},
                    "set_bus": {"substations_id": [(id_2, arr2)]},
                    "change_line_status": arr_line1,
                    "set_line_status": arr_line2,
                },
                env=self.env,
                check_legal=True,
            )
            raise RuntimeError("This should have thrown an IllegalException")
        except IllegalAction:
            pass

        self.env._times_before_line_status_actionable[:] = 0
        self.env._parameters.MAX_SUB_CHANGED = 2
        self.env._parameters.MAX_LINE_STATUS_CHANGED = 1
        self.env._times_before_line_status_actionable[1] = 1
        _ = self.helper_action(
            {
                "change_bus": {"substations_id": [(id_1, arr1)]},
                "set_bus": {"substations_id": [(id_2, arr2)]},
                "change_line_status": arr_line1,
                "set_line_status": arr_line2,
            },
            env=self.env,
            check_legal=True,
        )

    def test_linereactionnable_throw(self):
        id_1 = 1
        id_2 = 12
        id_line = 17
        id_line2 = 15

        arr1 = np.array([False, False, False, True, True, True], dtype=dt_bool)
        arr2 = np.array([1, 1, 2, 2], dtype=dt_int)
        arr_line1 = np.full(self.helper_action.n_line, fill_value=False, dtype=dt_bool)
        arr_line1[id_line] = True

        arr_line2 = np.full(self.helper_action.n_line, fill_value=0, dtype=dt_int)
        arr_line2[id_line2] = -1

        self.env._max_timestep_line_status_deactivated = 1
        self.helper_action.legal_action = RulesChecker(
            legalActClass=PreventReconnection
        ).legal_action

        # i act a first time on powerline 15
        act = self.helper_action(
            {"set_line_status": arr_line2}, env=self.env, check_legal=True
        )
        self.env.step(act)
        try:
            # i try to react on it, it should throw an IllegalAction exception.
            act = self.helper_action(
                {"set_line_status": arr_line2}, env=self.env, check_legal=True
            )
            raise RuntimeError("This should have thrown an IllegalException")
        except IllegalAction:
            pass

    def test_linereactionnable_nothrow(self):
        id_1 = 1
        id_2 = 12
        id_line = 17
        id_line2 = 15

        arr1 = np.array([False, False, False, True, True, True], dtype=dt_bool)
        arr2 = np.array([1, 1, 2, 2], dtype=dt_int)
        arr_line1 = np.full(self.helper_action.n_line, fill_value=False, dtype=dt_bool)
        arr_line1[id_line] = True

        arr_line2 = np.full(self.helper_action.n_line, fill_value=0, dtype=dt_int)
        arr_line2[id_line2] = -1

        self.env._max_timestep_line_status_deactivated = 1
        self.helper_action.legal_action = RulesChecker(
            legalActClass=PreventReconnection
        ).legal_action

        # i act a first time on powerline 15
        act = self.helper_action(
            {"set_line_status": arr_line2}, env=self.env, check_legal=True
        )
        self.env.step(act)
        # i compute another time step without doing anything
        self.env.step(self.helper_action({}))

        # i try to react on it, it should NOT throw an IllegalAction exception, but
        act = self.helper_action(
            {"set_line_status": arr_line2}, env=self.env, check_legal=True
        )

    def test_linereactionnable_throw_longerperiod(self):
        id_1 = 1
        id_2 = 12
        id_line = 17
        id_line2 = 15

        arr1 = np.array([False, False, False, True, True, True], dtype=dt_bool)
        arr2 = np.array([1, 1, 2, 2], dtype=dt_int)
        arr_line1 = np.full(self.helper_action.n_line, fill_value=False, dtype=dt_bool)
        arr_line1[id_line] = True

        arr_line2 = np.full(self.helper_action.n_line, fill_value=0, dtype=dt_int)
        arr_line2[id_line2] = -1

        self.env._max_timestep_line_status_deactivated = 2
        self.env._parameters.NB_TIMESTEP_LINE_STATUS_REMODIF = 2

        self.helper_action.legal_action = RulesChecker(
            legalActClass=PreventReconnection
        ).legal_action

        # i act a first time on powerline 15
        act = self.helper_action(
            {"set_line_status": arr_line2}, env=self.env, check_legal=True
        )
        _ = self.env.step(act)
        # i compute another time step without doing anything
        _ = self.env.step(self.helper_action({}))

        # i try to react on it, it should throw an IllegalAction exception because we ask the environment to wait
        # at least 2 time steps
        try:
            # i try to react on it, it should throw an IllegalAction exception.
            act = self.helper_action(
                {"set_line_status": arr_line2}, env=self.env, check_legal=True
            )
            raise RuntimeError("This should have thrown an IllegalException")
        except IllegalAction:
            pass

    def test_toporeactionnable_throw(self):
        id_1 = 1
        id_2 = 12
        id_line = 17
        id_line2 = 15

        arr1 = np.array([False, False, False, True, True, True], dtype=dt_bool)
        arr2 = np.array([1, 1, 2, 2], dtype=dt_int)
        arr_line1 = np.full(self.helper_action.n_line, fill_value=False, dtype=dt_bool)
        arr_line1[id_line] = True

        arr_line2 = np.full(self.helper_action.n_line, fill_value=0, dtype=dt_int)
        arr_line2[id_line2] = -1

        self.env._max_timestep_topology_deactivated = 1
        self.helper_action.legal_action = RulesChecker(
            legalActClass=PreventReconnection
        ).legal_action

        # i act a first time on powerline 15
        act = self.helper_action(
            {"set_bus": {"substations_id": [(id_2, arr2)]}},
            env=self.env,
            check_legal=True,
        )
        self.env.step(act)
        try:
            # i try to react on it, it should throw an IllegalAction exception.
            act = self.helper_action(
                {"set_bus": {"substations_id": [(id_2, arr2)]}},
                env=self.env,
                check_legal=True,
            )
            raise RuntimeError("This should have thrown an IllegalException")
        except IllegalAction:
            pass

    def test_toporeactionnable_nothrow(self):
        id_1 = 1
        id_2 = 12
        id_line = 17
        id_line2 = 15

        arr1 = np.array([False, False, False, True, True, True], dtype=dt_bool)
        arr2 = np.array([1, 1, 2, 2], dtype=dt_int)
        arr_line1 = np.full(self.helper_action.n_line, fill_value=False, dtype=dt_bool)
        arr_line1[id_line] = True

        arr_line2 = np.full(self.helper_action.n_line, fill_value=0, dtype=dt_int)
        arr_line2[id_line2] = -1

        self.env._max_timestep_topology_deactivated = 1
        self.helper_action.legal_action = RulesChecker(
            legalActClass=PreventReconnection
        ).legal_action

        # i act a first time on powerline 15
        act = self.helper_action(
            {"set_bus": {"substations_id": [(id_2, arr2)]}},
            env=self.env,
            check_legal=True,
        )
        self.env.step(act)
        # i compute another time step without doing anything
        self.env.step(self.helper_action({}))

        # i try to react on it, it should NOT throw an IllegalAction exception, but
        act = self.helper_action(
            {"set_bus": {"substations_id": [(id_2, arr2)]}},
            env=self.env,
            check_legal=True,
        )

    def test_toporeactionnable_throw_longerperiod(self):
        id_1 = 1
        id_2 = 12
        id_line = 17
        id_line2 = 15

        arr1 = np.array([False, False, False, True, True, True], dtype=dt_bool)
        arr2 = np.array([1, 1, 2, 2], dtype=dt_int)
        arr_line1 = np.full(self.helper_action.n_line, fill_value=False, dtype=dt_bool)
        arr_line1[id_line] = True

        arr_line2 = np.full(self.helper_action.n_line, fill_value=0, dtype=dt_int)
        arr_line2[id_line2] = -1

        self.env._max_timestep_topology_deactivated = 2
        self.helper_action.legal_action = RulesChecker(
            legalActClass=PreventReconnection
        ).legal_action

        # i act a first time on powerline 15
        act = self.helper_action(
            {"set_bus": {"substations_id": [(id_2, arr2)]}},
            env=self.env,
            check_legal=True,
        )
        self.env.step(act)
        # i compute another time step without doing anything
        self.env.step(self.helper_action({}))

        # i try to react on it, it should throw an IllegalAction exception because we ask the environment to wait
        # at least 2 time steps
        try:
            # i try to react on it, it should throw an IllegalAction exception.
            act = self.helper_action(
                {"set_bus": {"substations_id": [(id_2, arr2)]}},
                env=self.env,
                check_legal=True,
            )
            raise RuntimeError("This should have thrown an IllegalException")
        except IllegalAction:
            pass


class TestCooldown(unittest.TestCase):
    def setUp(self):
        params = Parameters()
        params.NB_TIMESTEP_COOLDOWN_LINE = 5
        params.NB_TIMESTEP_COOLDOWN_SUB = 15
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = make(
                "rte_case5_example",
                test=True,
                gamerules_class=DefaultRules,
                param=params,
            )

    def tearDown(self):
        self.env.close()

    def test_cooldown_sub(self):
        sub_id = 2
        act = self.env.action_space(
            {"set_bus": {"substations_id": [(sub_id, [1, 1, 2, 2])]}}
        )
        obs, reward, done, info = self.env.step(act)
        assert not done
        assert obs.time_before_cooldown_sub[sub_id] == 15

        # the next action is illegal because it consist in reconfiguring the same substation
        act = self.env.action_space(
            {"set_bus": {"substations_id": [(sub_id, [1, 1, 1, 1])]}}
        )
        obs, reward, done, info = self.env.step(act)
        assert not done
        assert info["is_illegal"]
        assert obs.time_before_cooldown_sub[sub_id] == 14

    def test_cooldown_line(self):
        line_id = 1
        act = self.env.action_space({"set_line_status": [(line_id, -1)]})
        obs, reward, done, info = self.env.step(act)
        assert not done
        assert obs.time_before_cooldown_line[line_id] == 5
        act = self.env.action_space({"set_line_status": [(line_id, +1)]})
        obs, reward, done, info = self.env.step(act)
        assert not done
        assert info["is_illegal"]
        assert obs.time_before_cooldown_line[line_id] == 4


class TestReconnectionsLegality(unittest.TestCase):
    def test_reconnect_already_connected(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env_case2 = make("rte_case5_example", test=True)
        obs = env_case2.reset()  # reset is good
        obs, reward, done, info = env_case2.step(
            env_case2.action_space()
        )  # do the action, it's valid
        # powerline 5 is connected
        # i fake a reconnection of it
        act_case2 = env_case2.action_space.reconnect_powerline(
            line_id=5, bus_or=2, bus_ex=1
        )
        obs_case2, reward_case2, done_case2, info_case2 = env_case2.step(act_case2)
        # this was illegal before, but test it is still illegal
        assert info_case2["is_illegal"], (
            "action should be illegal as it consists of change both ends of a "
            "powerline, while authorizing only 1 substations change"
        )

    def test_reconnect_disconnected(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            params = Parameters()
            params.MAX_SUB_CHANGED = 0
            params.NO_OVERFLOW_DISCONNECTION = True
            env_case2 = make("rte_case5_example", test=True, param=params)
        obs = env_case2.reset()  # reset is good
        line_id = 5

        # Disconnect the line
        disco_act = env_case2.action_space.disconnect_powerline(line_id=line_id)
        obs, reward, done, info = env_case2.step(disco_act)
        # Line has been disconnected
        assert info["is_illegal"] == False
        assert done == False
        assert np.sum(obs.line_status) == (env_case2.n_line - 1)

        # Reconnect the line
        reco_act = env_case2.action_space.reconnect_powerline(
            line_id=line_id, bus_or=1, bus_ex=2
        )
        obs, reward, done, info = env_case2.step(reco_act)
        # Check reconnecting is legal
        assert info["is_illegal"] == False
        assert done == False
        # Check line has been reconnected
        assert np.sum(obs.line_status) == (env_case2.n_line)

    def test_sub_dont_change(self):
        """test that i cannot reconect a powerline by acting on the substation"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            params = Parameters()
            params.MAX_SUB_CHANGED = 1
            params.MAX_LINE_STATUS_CHANGED = 1
            params.NB_TIMESTEP_COOLDOWN_LINE = 3
            params.NB_TIMESTEP_COOLDOWN_SUB = 3
            params.NO_OVERFLOW_DISCONNECTION = True
            env = make("rte_case5_example", test=True, param=params)
        l_id = 2
        # prepare the actions
        disco_act = env.action_space.disconnect_powerline(line_id=l_id)
        reco_act = env.action_space.reconnect_powerline(
            line_id=l_id, bus_or=1, bus_ex=1
        )
        set_or_1_act = env.action_space({"set_bus": {"lines_or_id": [(l_id, 1)]}})
        set_ex_1_act = env.action_space({"set_bus": {"lines_ex_id": [(l_id, 1)]}})

        obs, reward, done, info = env.step(disco_act)
        assert obs.rho[l_id] == 0.0
        assert obs.time_before_cooldown_line[l_id] == 3
        assert env.backend._grid.line.iloc[l_id]["in_service"] == False

        # i have a cooldown to 2 so i cannot reconnect it
        assert obs.time_before_cooldown_line[l_id] == 3
        obs, reward, done, info = env.step(reco_act)
        assert obs.rho[l_id] == 0.0
        assert info["is_illegal"]
        assert obs.time_before_cooldown_line[l_id] == 2
        assert env.backend._grid.line.iloc[l_id]["in_service"] == False

        # this is not supposed to reconnect it either (cooldown)
        assert obs.time_before_cooldown_line[l_id] == 2
        obs, reward, done, info = env.step(set_or_1_act)
        # pdb.set_trace()
        # assert info["is_illegal"]
        assert env.backend._grid.line.iloc[l_id]["in_service"] == False
        assert obs.rho[l_id] == 0.0
        assert obs.time_before_cooldown_line[l_id] == 1
        assert env.backend._grid.line.iloc[l_id]["in_service"] == False

        # and neither is that (cooldown)
        assert obs.time_before_cooldown_line[l_id] == 1
        obs, reward, done, info = env.step(set_ex_1_act)
        assert obs.rho[l_id] == 0.0
        assert obs.time_before_cooldown_line[l_id] == 0
        assert env.backend._grid.line.iloc[l_id]["in_service"] == False

        # and now i can reconnect
        obs, reward, done, info = env.step(reco_act)
        assert obs.rho[l_id] != 0.0
        assert obs.time_before_cooldown_line[l_id] == 3
        assert env.backend._grid.line.iloc[l_id]["in_service"] == True


class TestSubstationImpactLegality(unittest.TestCase):
    def setUp(self):
        # Create env with custom params
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.params = Parameters()
            self.env = make("rte_case5_example", test=True, param=self.params)

    def tearDown(self):
        self.env.close()

    def test_setbus_line_no_sub_allowed_is_illegal(self):
        # Set 0 allowed substation changes
        self.env._parameters.MAX_SUB_CHANGED = 0
        # Make a setbus
        LINE_ID = 4
        bus_action = self.env.action_space({"set_bus": {"lines_ex_id": [(LINE_ID, 2)]}})
        # Make sure its illegal
        _, _, _, i = self.env.step(bus_action)
        assert i["is_illegal"] == True

    def test_two_setbus_line_one_sub_allowed_is_illegal(self):
        # Set 1 allowed substation changes
        self.env._parameters.MAX_SUB_CHANGED = 1
        # Make a double setbus
        LINE1_ID = 4
        LINE2_ID = 5
        bus_action = self.env.action_space(
            {"set_bus": {"lines_ex_id": [(LINE1_ID, 2), (LINE2_ID, 2)]}}
        )
        # Make sure its illegal
        _, _, _, i = self.env.step(bus_action)
        assert i["is_illegal"] == True

    def test_one_setbus_line_one_sub_allowed_is_legal(self):
        # Set 1 allowed substation changes
        self.env._parameters.MAX_SUB_CHANGED = 1
        # Make a setbus
        LINE1_ID = 4
        bus_action = self.env.action_space(
            {"set_bus": {"lines_ex_id": [(LINE1_ID, 2)]}}
        )
        # Make sure its legal
        _, _, _, i = self.env.step(bus_action)
        assert i["is_illegal"] == False

    def test_two_setbus_line_two_sub_allowed_is_legal(self):
        # Set 2 allowed substation changes
        self.env._parameters.MAX_SUB_CHANGED = 2
        # Make a double setbus
        LINE1_ID = 4
        LINE2_ID = 5
        bus_action = self.env.action_space(
            {"set_bus": {"lines_ex_id": [(LINE1_ID, 2), (LINE2_ID, 2)]}}
        )
        # Make sure its legal
        _, _, _, i = self.env.step(bus_action)
        assert i["is_illegal"] == False

    def test_changebus_line_no_sub_allowed_is_illegal(self):
        # Set 0 allowed substation changes
        self.env._parameters.MAX_SUB_CHANGED = 0
        # Make a changebus
        LINE_ID = 4
        bus_action = self.env.action_space({"change_bus": {"lines_ex_id": [LINE_ID]}})
        # Make sure its illegal
        _, _, _, i = self.env.step(bus_action)
        assert i["is_illegal"] == True

    def test_changebus_line_one_sub_allowed_is_legal(self):
        # Set 1 allowed substation changes
        self.env._parameters.MAX_SUB_CHANGED = 1
        # Make a changebus
        LINE_ID = 4
        bus_action = self.env.action_space({"change_bus": {"lines_ex_id": [LINE_ID]}})
        # Make sure its legal
        _, _, _, i = self.env.step(bus_action)
        assert i["is_illegal"] == False

    def test_changebus_two_line_one_sub_allowed_is_illegal(self):
        # Set 1 allowed substation changes
        self.env._parameters.MAX_SUB_CHANGED = 1
        # Make a changebus
        LINE1_ID = 4
        LINE2_ID = 5
        bus_action = self.env.action_space(
            {"change_bus": {"lines_ex_id": [LINE1_ID, LINE2_ID]}}
        )
        # Make sure its illegal
        _, _, _, i = self.env.step(bus_action)
        assert i["is_illegal"] == True

    def test_changebus_two_line_two_sub_allowed_is_legal(self):
        # Set 2 allowed substation changes
        self.env._parameters.MAX_SUB_CHANGED = 2
        # Make a changebus
        LINE1_ID = 4
        LINE2_ID = 5
        bus_action = self.env.action_space(
            {"change_bus": {"lines_ex_id": [LINE1_ID, LINE2_ID]}}
        )
        # Make sure its legal
        _, _, _, i = self.env.step(bus_action)
        assert i["is_illegal"] == False


class TestLoadingFromInstance(unittest.TestCase):
    def test_correct(self):
        rules = AlwaysLegal()
        rules.TOTO = 1
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                env = make("rte_case5_example", test=True, gamerules_class=rules)
            assert hasattr(env._game_rules.legal_action, "TOTO")
            assert env._game_rules.legal_action.TOTO == 1
        finally:
            env.close()
            

if __name__ == "__main__":
    unittest.main()
