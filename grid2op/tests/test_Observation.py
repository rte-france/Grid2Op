# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import json
import tempfile
import warnings
import copy
import pdb
import unittest

from grid2op.tests.helper_path_test import *

import grid2op
from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Exceptions import *
from grid2op.Observation import ObservationSpace
from grid2op.Reward import (
    L2RPNReward,
    CloseToOverflowReward,
    RedispReward,
    RewardHelper,
)
from grid2op.Action import CompleteAction, PlayableAction

# TODO add unit test for the proper update the backend in the observation [for now there is a "data leakage" as
# the real backend is copied when the observation is built, but i need to make a test to check that's it's properly
# copied]

# temporary deactivation of all the failing test until simulate is fixed
DEACTIVATE_FAILING_TEST = False


class TestBasisObsBehaviour(unittest.TestCase):
    def setUp(self):
        """
        The case file is a representation of the case14 as found in the ieee14 powergrid.
        :return:
        """
        self.tolvect = 1e-2
        self.tol_one = 1e-5

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("rte_case14_test",
                                    test=True,
                                    _add_to_name=type(self).__name__)
            
        self.dict_ = {
            "name_gen": ["gen_1_0", "gen_2_1", "gen_5_2", "gen_7_3", "gen_0_4"],
            "n_busbar_per_sub": "2",
            "name_load": [
                "load_1_0",
                "load_2_1",
                "load_13_2",
                "load_3_3",
                "load_4_4",
                "load_5_5",
                "load_8_6",
                "load_9_7",
                "load_10_8",
                "load_11_9",
                "load_12_10",
            ],
            "name_line": [
                "0_1_0",
                "0_4_1",
                "8_9_2",
                "8_13_3",
                "9_10_4",
                "11_12_5",
                "12_13_6",
                "1_2_7",
                "1_3_8",
                "1_4_9",
                "2_3_10",
                "3_4_11",
                "5_10_12",
                "5_11_13",
                "5_12_14",
                "3_6_15",
                "3_8_16",
                "4_5_17",
                "6_7_18",
                "6_8_19",
            ],
            "name_sub": [
                "sub_0",
                "sub_1",
                "sub_10",
                "sub_11",
                "sub_12",
                "sub_13",
                "sub_2",
                "sub_3",
                "sub_4",
                "sub_5",
                "sub_6",
                "sub_7",
                "sub_8",
                "sub_9",
            ],
            "name_storage": [],
            "glop_version": grid2op.__version__,
            # "env_name": "rte_case14_test",
            "env_name": "rte_case14_testTestBasisObsBehaviour",
            "sub_info": [3, 6, 4, 6, 5, 6, 3, 2, 5, 3, 3, 3, 4, 3],
            "load_to_subid": [1, 2, 13, 3, 4, 5, 8, 9, 10, 11, 12],
            "gen_to_subid": [1, 2, 5, 7, 0],
            "line_or_to_subid": [
                0,
                0,
                8,
                8,
                9,
                11,
                12,
                1,
                1,
                1,
                2,
                3,
                5,
                5,
                5,
                3,
                3,
                4,
                6,
                6,
            ],
            "line_ex_to_subid": [
                1,
                4,
                9,
                13,
                10,
                12,
                13,
                2,
                3,
                4,
                3,
                4,
                10,
                11,
                12,
                6,
                8,
                5,
                7,
                8,
            ],
            "storage_to_subid": [],
            "load_to_sub_pos": [5, 3, 2, 5, 4, 5, 4, 2, 2, 2, 3],
            "gen_to_sub_pos": [4, 2, 4, 1, 2],
            "line_or_to_sub_pos": [
                0,
                1,
                0,
                1,
                1,
                0,
                1,
                1,
                2,
                3,
                1,
                2,
                0,
                1,
                2,
                3,
                4,
                3,
                1,
                2,
            ],
            "line_ex_to_sub_pos": [
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                1,
                2,
                1,
                1,
                2,
                0,
                2,
                3,
                0,
                3,
            ],
            "storage_to_sub_pos": [],
            "load_pos_topo_vect": [8, 12, 55, 18, 23, 29, 39, 42, 45, 48, 52],
            "gen_pos_topo_vect": [7, 11, 28, 34, 2],
            "line_or_pos_topo_vect": [
                0,
                1,
                35,
                36,
                41,
                46,
                50,
                4,
                5,
                6,
                10,
                15,
                24,
                25,
                26,
                16,
                17,
                22,
                31,
                32,
            ],
            "line_ex_pos_topo_vect": [
                3,
                19,
                40,
                53,
                43,
                49,
                54,
                9,
                13,
                20,
                14,
                21,
                44,
                47,
                51,
                30,
                37,
                27,
                33,
                38,
            ],
            "storage_pos_topo_vect": [],
            "gen_type": ["nuclear", "thermal", "solar", "wind", "thermal"],
            "gen_pmin": [0.0, 0.0, 0.0, 0.0, 0.0],
            "gen_pmax": [200.0, 200.0, 40.0, 70.0, 400.0],
            "gen_redispatchable": [True, True, False, False, True],
            "gen_renewable": [False, False, True, True, False],
            "gen_max_ramp_up": [5.0, 10.0, 0.0, 0.0, 10.0],
            "gen_max_ramp_down": [5.0, 10.0, 0.0, 0.0, 10.0],
            "gen_min_uptime": [96, 4, 0, 0, 4],
            "gen_min_downtime": [96, 4, 0, 0, 4],
            "gen_cost_per_MW": [5.0, 10.0, 0.0, 0.0, 10.0],
            "gen_startup_cost": [20.0, 2.0, 0.0, 0.0, 2.0],
            "gen_shutdown_cost": [10.0, 1.0, 0.0, 0.0, 1.0],
            "grid_layout": {
                "sub_0": [-280.0, -81.0],
                "sub_1": [-100.0, -270.0],
                "sub_10": [366.0, -270.0],
                "sub_11": [366.0, -54.0],
                "sub_12": [-64.0, -54.0],
                "sub_13": [-64.0, 54.0],
                "sub_2": [450.0, 0.0],
                "sub_3": [550.0, 0.0],
                "sub_4": [326.0, 54.0],
                "sub_5": [222.0, 108.0],
                "sub_6": [79.0, 162.0],
                "sub_7": [-170.0, 270.0],
                "sub_8": [-64.0, 270.0],
                "sub_9": [222.0, 216.0],
            },
            "name_shunt": ["shunt_8_0"],
            "shunt_to_subid": [8],
            "storage_type": [],
            "storage_Emax": [],
            "storage_Emin": [],
            "storage_max_p_prod": [],
            "storage_max_p_absorb": [],
            "storage_marginal_cost": [],
            "storage_loss": [],
            "storage_charging_efficiency": [],
            "storage_discharging_efficiency": [],
            "_init_subtype": "grid2op.Observation.completeObservation.CompleteObservation",
            "dim_alarms": 0,
            "dim_alerts": 0,
            "alarms_area_names": [],
            "alarms_lines_area": {},
            "alarms_area_lines": [],
            "alertable_line_names": [],
            "alertable_line_ids": [],
            "assistant_warning_type": None,
            "_PATH_GRID_CLASSES": None,
        }

        self.json_ref = {
            "year": [2019],
            "month": [1],
            "day": [6],
            "hour_of_day": [0],
            "minute_of_hour": [0],
            "day_of_week": [6],
            "gen_p": [93.5999984741211, 75.0, 0.0, 7.599999904632568, 77.9990234375],
            "gen_q": [
                65.4969711303711,
                98.51886749267578,
                -12.746061325073242,
                6.789371013641357,
                3.801255941390991,
            ],
            "gen_v": [
                142.10000610351562,
                142.10000610351562,
                0.20000000298023224,
                12.0,
                142.10000610351562,
            ],
            "load_p": [
                21.200000762939453,
                86.9000015258789,
                15.199999809265137,
                45.5,
                7.300000190734863,
                11.699999809265137,
                29.399999618530273,
                8.600000381469727,
                3.5,
                5.599999904632568,
                13.399999618530273,
            ],
            "load_q": [
                14.899999618530273,
                60.099998474121094,
                10.800000190734863,
                31.5,
                5.099999904632568,
                8.300000190734863,
                20.600000381469727,
                6.0,
                2.4000000953674316,
                3.9000000953674316,
                9.399999618530273,
            ],
            "load_v": [
                142.10000610351562,
                142.10000610351562,
                0.19267192482948303,
                133.21652221679688,
                133.5172882080078,
                0.20000000298023224,
                0.20202238857746124,
                0.1999506950378418,
                0.1990993618965149,
                0.19563813507556915,
                0.19479265809059143,
            ],
            "p_or": [
                39.331451416015625,
                38.667572021484375,
                8.023938179016113,
                11.87976360321045,
                -0.621783435344696,
                1.323334813117981,
                3.6911065578460693,
                29.264848709106445,
                44.41638946533203,
                37.75281524658203,
                16.985824584960938,
                -28.051433563232422,
                4.144556999206543,
                7.01623010635376,
                16.03428840637207,
                25.47538185119629,
                16.228321075439453,
                38.895076751708984,
                -7.599999904632568,
                33.075382232666016,
            ],
            "q_or": [
                -15.30455207824707,
                19.10580825805664,
                8.4382963180542,
                10.634726524353027,
                2.3168439865112305,
                0.45094606280326843,
                0.9515853524208069,
                -8.529295921325684,
                23.858327865600586,
                24.905370712280273,
                33.14556884765625,
                3.8572652339935303,
                0.13210760056972504,
                4.54428768157959,
                10.420299530029297,
                10.74376106262207,
                9.365352630615234,
                43.827266693115234,
                -6.606429576873779,
                15.779977798461914,
            ],
            "v_or": [
                142.10000610351562,
                142.10000610351562,
                0.20202238857746124,
                0.20202238857746124,
                0.1999506950378418,
                0.19563813507556915,
                0.19479265809059143,
                142.10000610351562,
                142.10000610351562,
                142.10000610351562,
                142.10000610351562,
                133.21652221679688,
                0.20000000298023224,
                0.20000000298023224,
                0.20000000298023224,
                133.21652221679688,
                133.21652221679688,
                133.5172882080078,
                13.833837509155273,
                13.833837509155273,
            ],
            "a_or": [
                171.47496032714844,
                175.23731994628906,
                33277.5390625,
                45566.9609375,
                6926.5302734375,
                4125.82861328125,
                11297.8642578125,
                123.84978485107422,
                204.8500518798828,
                183.7598419189453,
                151.32354736328125,
                122.71674346923828,
                11970.3818359375,
                24131.24609375,
                55202.73828125,
                119.82522583007812,
                81.20392608642578,
                253.38462829589844,
                420.26788330078125,
                1529.4415283203125,
            ],
            "p_ex": [
                -39.034053802490234,
                -37.70601272583008,
                -7.978217124938965,
                -11.537211418151855,
                0.6268926858901978,
                -1.3184539079666138,
                -3.6627886295318604,
                -28.885826110839844,
                -43.03413772583008,
                -36.650413513183594,
                -16.11812973022461,
                28.161354064941406,
                -4.126892566680908,
                -6.92333459854126,
                -15.772652626037598,
                -25.47538185119629,
                -16.228321075439453,
                -38.895076751708984,
                7.599999904632568,
                -33.075382232666016,
            ],
            "q_ex": [
                10.362568855285645,
                -20.268247604370117,
                -8.31684398651123,
                -9.906070709228516,
                -2.3048837184906006,
                -0.44652995467185974,
                -0.8939293026924133,
                5.273301124572754,
                -23.203140258789062,
                -25.148475646972656,
                -32.26323699951172,
                -3.510542869567871,
                -0.09511636942625046,
                -4.350945949554443,
                -9.905055046081543,
                -9.173547744750977,
                -7.482545852661133,
                -36.142757415771484,
                6.789371013641357,
                -14.266851425170898,
            ],
            "v_ex": [
                142.10000610351562,
                133.5172882080078,
                0.1999506950378418,
                0.19267192482948303,
                0.1990993618965149,
                0.19479265809059143,
                0.19267192482948303,
                142.10000610351562,
                133.21652221679688,
                133.5172882080078,
                133.21652221679688,
                133.5172882080078,
                0.1990993618965149,
                0.19563813507556915,
                0.19479265809059143,
                13.833837509155273,
                0.20202238857746124,
                0.20000000298023224,
                12.0,
                0.20202238857746124,
            ],
            "a_ex": [
                164.0883026123047,
                185.10972595214844,
                33277.5390625,
                45566.9609375,
                6926.5302734375,
                4125.82861328125,
                11297.8642578125,
                119.30233764648438,
                211.88955688476562,
                192.20391845703125,
                156.30455017089844,
                122.71674346923828,
                11970.3818359375,
                24131.24609375,
                55202.73828125,
                1130.0374755859375,
                51070.6328125,
                153273.34375,
                490.3125305175781,
                102943.1796875,
            ],
            "rho": [
                0.4860054850578308,
                0.4966689944267273,
                0.18164825439453125,
                0.2487310916185379,
                0.03780904784798622,
                0.3378177583217621,
                0.061670344322919846,
                0.3510231077671051,
                0.580599308013916,
                0.5208240747451782,
                0.42889103293418884,
                0.347811758518219,
                0.06534133851528168,
                0.13172243535518646,
                0.3013288080692291,
                0.33961644768714905,
                0.23015344142913818,
                0.7181591987609863,
                0.15440839529037476,
                0.5619240403175354,
            ],
            "line_status": [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            "timestep_overflow": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            "topo_vect": [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            "time_before_cooldown_line": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            "time_before_cooldown_sub": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "time_next_maintenance": [
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
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
            ],
            "duration_next_maintenance": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            "target_dispatch": [0.0, 0.0, 0.0, 0.0, 0.0],
            "actual_dispatch": [0.0, 0.0, 0.0, 0.0, 0.0],
            "_shunt_p": [0.0],
            "_shunt_q": [-17.923625946044922],
            "_shunt_v": [0.20202238857746124],
            "_shunt_bus": [1],
            "storage_charge": [],
            "storage_power_target": [],
            "storage_power": [],
            "gen_p_before_curtail": [0.0, 0.0, 0.0, 7.599999904632568, 0.0],
            "curtailment": [0.0, 0.0, 0.0, 0.0, 0.0],
            "curtailment_limit": [1.0, 1.0, 1.0, 1.0, 1.0],
            "curtailment_limit_effective": [1.0, 1.0, 1.0, 1.0, 1.0],
            "theta_ex": [
                -1.3276801109313965,
                -4.100967884063721,
                -10.311812400817871,
                -11.245238304138184,
                -10.119081497192383,
                -10.50421142578125,
                -11.245238304138184,
                -4.473601341247559,
                -4.824699401855469,
                -4.100967884063721,
                -4.824699401855469,
                -4.100967884063721,
                -10.119081497192383,
                -10.39695930480957,
                -10.50421142578125,
                -7.88769006729126,
                -10.060456275939941,
                -9.613715171813965,
                -7.1114115715026855,
                -10.060456275939941,
            ],
            "theta_or": [
                0.0,
                0.0,
                -10.060456275939941,
                -10.060456275939941,
                -10.311812400817871,
                -10.39695930480957,
                -10.50421142578125,
                -1.3276801109313965,
                -1.3276801109313965,
                -1.3276801109313965,
                -4.473601341247559,
                -4.824699401855469,
                -9.613715171813965,
                -9.613715171813965,
                -9.613715171813965,
                -4.824699401855469,
                -4.824699401855469,
                -4.100967884063721,
                -7.88769006729126,
                -7.88769006729126,
            ],
            "gen_theta": [
                -1.3276801109313965,
                -4.473601341247559,
                -9.613715171813965,
                -7.1114115715026855,
                0.0,
            ],
            "load_theta": [
                -1.3276801109313965,
                -4.473601341247559,
                -11.245238304138184,
                -4.824699401855469,
                -4.100967884063721,
                -9.613715171813965,
                -10.060456275939941,
                -10.311812400817871,
                -10.119081497192383,
                -10.39695930480957,
                -10.50421142578125,
            ],
            "storage_theta": [],
            "_thermal_limit": [
                352.8251647949219,
                352.8251647949219,
                183197.6875,
                183197.6875,
                183197.6875,
                12213.1787109375,
                183197.6875,
                352.8251647949219,
                352.8251647949219,
                352.8251647949219,
                352.8251647949219,
                352.8251647949219,
                183197.6875,
                183197.6875,
                183197.6875,
                352.8251647949219,
                352.8251647949219,
                352.8251647949219,
                2721.794189453125,
                2721.794189453125,
            ],
            "support_theta": [True],
            "gen_margin_up": [5.0, 10.0, 0.0, 0.0, 10.0],
            "gen_margin_down": [5.0, 10.0, 0.0, 0.0, 10.0],
            "is_alarm_illegal": [False],
            "time_since_last_alarm": [-1],
            "last_alarm": [],
            "attention_budget": [0.0],
            "was_alarm_used_after_game_over": [False],
            "current_step": [0],
            "max_step": [8064],
            "delta_time": [5.0],
            "time_since_last_alert": [],
            "active_alert": [],
            "alert_duration": [],
            "total_number_of_alert": [],
            "time_since_last_attack": [],
            "was_alert_used_after_attack": [],
            "attack_under_alert": [],
        }
        self.dtypes = np.array(
            [
                dt_int,
                dt_int,
                dt_int,
                dt_int,
                dt_int,
                dt_int,
                dt_float,
                dt_float,
                dt_float,
                dt_float,
                dt_float,
                dt_float,
                dt_float,
                dt_float,
                dt_float,
                dt_float,
                dt_float,
                dt_float,
                dt_float,
                dt_float,
                dt_float,
                dt_bool,
                dt_int,
                dt_int,
                dt_int,
                dt_int,
                dt_int,
                dt_int,
                dt_float,
                dt_float,
                dt_float,
                dt_float,
                dt_float,
                # curtailment
                dt_float,
                dt_float,
                dt_float,
                dt_float,
                # alarm feature
                dt_bool,
                dt_int,
                dt_int,
                dt_float,
                dt_bool,
                # shunts
                dt_float,
                dt_float,
                dt_float,
                dt_int,
                # steps
                dt_int,
                dt_int,
                # delta_time
                dt_float,
                # gen margins
                dt_float,
                dt_float,
                # alert feature
                dt_bool,
                dt_int,
                dt_int,
                dt_int,
                dt_int,
                dt_int,
                dt_int,
            ],
            dtype=object,
        )

        self.dtypes = np.array([np.dtype(el) for el in self.dtypes])

        self.shapes = np.array(
            [
                1,
                1,
                1,
                1,
                1,
                1,
                5,
                5,
                5,
                11,
                11,
                11,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                56,
                20,
                14,
                20,
                20,
                5,
                5,
                0,
                0,
                0,
                # curtailment
                5,
                5,
                5,
                5,
                1,
                1,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                5,
                5,
                # alert
                0,
                0,
                0,
                0,
                0,
                0,
                0
            ]
        )
        self.size_obs = 429 + 4 + 4 + 2 + 1 + 10 + 5 + 0

    def tearDown(self):
        self.env.close()

    def test_sum_shape_equal_size(self):
        obs = self.env.observation_space(self.env)
        assert obs.size() == np.sum(obs.shapes())

    def test_sub_topology(self):
        """test the sub_topology function"""
        obs = self.env.observation_space(self.env)
        # test in normal conditions
        topo = obs.sub_topology(sub_id=1)
        assert np.all(topo == 1)
        # test if i fake a change in the topology
        obs.topo_vect[2] = 2
        topo = obs.sub_topology(sub_id=0)
        assert np.array_equal(topo, [1, 1, 2])
        topo = obs.sub_topology(sub_id=1)
        assert np.all(topo == 1)

    def test_size(self):
        obs = self.env.observation_space(self.env)
        obs.size()

    def test_copy_space(self):
        obs_space2 = self.env.observation_space.copy()
        assert isinstance(obs_space2, ObservationSpace)

    def test_proper_size(self):
        obs = self.env.observation_space(self.env)
        assert obs.size() == self.size_obs, f"{obs.size()} vs {self.size_obs}"

    def test_size_observation_space(self):
        assert self.env.observation_space.size() == self.size_obs, f"{self.env.observation_space.size()} vs {self.size_obs}"

    def aux_test_bus_conn_mat(self, as_csr=False):
        obs = self.env.observation_space(self.env)
        mat1 = obs.bus_connectivity_matrix(as_csr_matrix=as_csr)
        ref_mat = np.array(
            [
                [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            ]
        )
        assert np.all(mat1 == ref_mat)

    def test_bus_conn_mat(self):
        self.aux_test_bus_conn_mat()

    def test_bus_conn_mat_twice(self):
        """test i can call twice the bus_connectivity_matrix"""
        obs = self.env.observation_space(self.env)
        mat1 = obs.bus_connectivity_matrix(as_csr_matrix=False)
        mat2 = obs.bus_connectivity_matrix(as_csr_matrix=True)
        mat3 = obs.bus_connectivity_matrix(as_csr_matrix=False)
        mat4 = obs.bus_connectivity_matrix(as_csr_matrix=True)
        assert np.all(mat1 == mat3)
        assert np.all(mat2.todense() == mat4.todense())
        assert np.all(mat1 == mat2.todense())

    def test_conn_mat_twice(self):
        """test i can call twice the connectivity_matrix"""
        obs = self.env.observation_space(self.env)
        mat1 = obs.connectivity_matrix(as_csr_matrix=False)
        mat2 = obs.connectivity_matrix(as_csr_matrix=True)
        mat3 = obs.connectivity_matrix(as_csr_matrix=False)
        mat4 = obs.connectivity_matrix(as_csr_matrix=True)
        assert np.all(mat1 == mat3)
        assert np.all(mat2.todense() == mat4.todense())
        assert np.all(mat1 == mat2.todense())

    def test_flow_bus_mat_twice(self):
        """test i can call twice the flow_bus_matrix (it crashed before due to a bug of a copy of an array)"""
        obs = self.env.observation_space(self.env)
        mat1, *_ = obs.flow_bus_matrix(as_csr_matrix=False)
        mat2, *_ = obs.flow_bus_matrix(as_csr_matrix=True)
        mat3, *_ = obs.flow_bus_matrix(as_csr_matrix=False)
        mat4, *_ = obs.flow_bus_matrix(as_csr_matrix=True)
        assert np.all(mat1 == mat3)
        assert np.all(mat2.todense() == mat4.todense())
        assert np.all(mat1 == mat2.todense())

    def test_networkx_graph(self):
        obs = self.env.observation_space(self.env)
        graph = obs.get_energy_graph()
        for node_id in graph.nodes:
            # retrieve power (active and reactive) produced at this node
            p_ = graph.nodes[node_id]["p"]
            q_ = graph.nodes[node_id]["q"]

            # get the edges
            edges = graph.edges(node_id)
            p_line = 0
            q_line = 0
            for (k1, k2) in edges:
                if k1 < k2:
                    p_line += graph.edges[(k1, k2)]["p_or"]
                    q_line += graph.edges[(k1, k2)]["q_or"]
                else:
                    p_line += graph.edges[(k1, k2)]["p_ex"]
                    q_line += graph.edges[(k1, k2)]["q_ex"]
            assert abs(p_line - p_) <= 1e-5, "error for kirchoff's law for graph for P"
            assert abs(q_line - q_) <= 1e-5, "error for kirchoff's law for graph for Q"

    def test_bus_conn_mat_csr(self):
        self.aux_test_bus_conn_mat(as_csr=True)

    def test_conn_mat(self):
        obs = self.env.observation_space(self.env)
        mat = obs.connectivity_matrix()
        ref_mat = np.array(
            [
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ]
        )

        assert np.all(mat[:10, :] == ref_mat)
        ind_conn = obs.topo_vect > 0
        assert np.all(mat[ind_conn, ind_conn] == 1)
        mat2 = obs.connectivity_matrix(as_csr_matrix=True)
        assert np.all(mat2[:10, :] == ref_mat)
        assert np.all(mat2[ind_conn, ind_conn] == 1)

        # test disconnected element (1 disconnect)
        disco_powerline = self.env.action_space()
        line_id = 0
        disco_powerline.line_set_status = [(line_id, -1)]
        obs, reward, done, info = self.env.step(disco_powerline)
        assert not done
        mat3 = obs.connectivity_matrix()
        lor_id = self.env.line_or_pos_topo_vect[line_id]
        lex_id = self.env.line_ex_pos_topo_vect[line_id]
        assert np.all(mat3[lor_id, :] == 0)
        assert np.all(mat3[:, lor_id] == 0)
        assert np.all(mat3[lex_id, :] == 0)
        assert np.all(mat3[:, lex_id] == 0)
        ind_conn = obs.topo_vect > 0
        assert np.all(mat3[ind_conn, ind_conn] == 1)

        mat4 = obs.connectivity_matrix(as_csr_matrix=True)
        assert mat4[lor_id, :].nnz == 0
        assert mat4[:, lor_id].nnz == 0
        assert mat4[lex_id, :].nnz == 0
        assert mat4[:, lor_id].nnz == 0

        # test 2 disconnected element (check they are not connected together)
        disco_powerline2 = self.env.action_space()
        line_id2 = 7
        disco_powerline2.line_set_status = [(line_id2, -1)]
        lor_id2 = self.env.line_or_pos_topo_vect[line_id2]
        lex_id2 = self.env.line_ex_pos_topo_vect[line_id2]
        obs, reward, done, info = self.env.step(disco_powerline2)
        assert not done
        assert np.array_equal(obs.line_status[[0, 7]], [False, False])
        mat5 = obs.connectivity_matrix()
        assert np.all(mat5[lor_id, :] == 0)
        assert np.all(mat5[:, lor_id] == 0)
        assert np.all(mat5[lex_id, :] == 0)
        assert np.all(mat5[:, lex_id] == 0)
        assert np.all(mat5[lor_id2, :] == 0)
        assert np.all(mat5[:, lor_id2] == 0)
        assert np.all(mat5[lex_id2, :] == 0)
        assert np.all(mat5[:, lex_id2] == 0)

        mat6 = obs.connectivity_matrix(as_csr_matrix=True)
        assert mat6[lor_id, :].nnz == 0
        assert mat6[:, lor_id].nnz == 0
        assert mat6[lex_id, :].nnz == 0
        assert mat6[:, lor_id].nnz == 0
        assert mat6[lor_id2, :].nnz == 0
        assert mat6[:, lor_id2].nnz == 0
        assert mat6[lex_id2, :].nnz == 0
        assert mat6[:, lor_id2].nnz == 0

    def test_copy_is_done(self):
        """make sure the attribute obs._is_done is properly copied"""
        obs = self.env.observation_space(self.env)
        obs._is_done = True
        obs_cpy = obs.copy()
        assert obs_cpy._is_done == obs._is_done
        
        obs._is_done = False
        obs_cpy = obs.copy()
        assert obs_cpy._is_done == obs._is_done
        
    def aux_test_conn_mat2(self, as_csr=False):
        l_id = 0
        # check line is connected, and matrix is the right size
        ob0 = self.env.get_obs()
        mat0 = ob0.bus_connectivity_matrix(as_csr)
        assert mat0.shape == (14, 14)
        assert mat0[ob0.line_or_to_subid[l_id], ob0.line_ex_to_subid[l_id]] == 1.0
        assert mat0[ob0.line_ex_to_subid[l_id], ob0.line_or_to_subid[l_id]] == 1.0

        # when a powerline is disconnected, check it is disconnected
        obs, reward, done, info = self.env.step(
            self.env.action_space({"set_line_status": [(l_id, -1)]})
        )
        assert not done
        mat = obs.bus_connectivity_matrix(as_csr)
        assert mat.shape == (14, 14)
        assert mat[obs.line_or_to_subid[l_id], obs.line_ex_to_subid[l_id]] == 0.0
        assert mat[obs.line_ex_to_subid[l_id], obs.line_or_to_subid[l_id]] == 0.0

        # when there is a substation counts 2 buses
        obs, reward, done, info = self.env.step(
            self.env.action_space({"set_bus": {"lines_or_id": [(13, 2), (14, 2)]}})
        )
        assert not done, f"failed with error {info['exception']}"
        assert obs.bus_connectivity_matrix(as_csr).shape == (15, 15)
        assert (
            obs.bus_connectivity_matrix(as_csr)[14, 11] == 1.0
        )  # first powerline I modified
        assert (
            obs.bus_connectivity_matrix(as_csr)[14, 12] == 1.0
        )  # second powerline I modified
        assert (
            obs.bus_connectivity_matrix(as_csr)[5, 11] == 0.0
        )  # first powerline modified
        assert (
            obs.bus_connectivity_matrix(as_csr)[5, 12] == 0.0
        )  # second powerline modified

    def test_conn_mat2(self):
        self.aux_test_conn_mat2(as_csr=False)

    def test_conn_mat2_csr(self):
        self.aux_test_conn_mat2(as_csr=True)

    def aux_test_conn_mat3(self, as_csr=False):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("rte_case14_realistic", test=True, _add_to_name=type(self).__name__)
        obs, reward, done, info = env.step(
            env.action_space({"set_bus": {"lines_or_id": [(7, 2), (8, 2)]}})
        )
        mat, (ind_lor, ind_lex) = obs.bus_connectivity_matrix(
            as_csr, return_lines_index=True
        )
        assert mat.shape == (15, 15)
        assert ind_lor[7] == 14
        assert ind_lor[8] == 14
        obs, reward, done, info = env.step(
            env.action_space(
                {"set_bus": {"lines_or_id": [(2, 2)], "lines_ex_id": [(0, 2)]}}
            )
        )
        mat, (ind_lor, ind_lex) = obs.bus_connectivity_matrix(
            as_csr, return_lines_index=True
        )
        assert mat.shape == (16, 16)
        assert ind_lor[7] == 15
        assert ind_lor[8] == 15
        assert ind_lor[2] == 14
        assert ind_lex[0] == 14

    def test_conn_mat3(self):
        self.aux_test_conn_mat3(False)

    def test_conn_mat3_csr(self):
        self.aux_test_conn_mat3(True)

    def test_active_flow_bus_matrix(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.aux_flow_bus_matrix(active_flow=True)

    def test_reactive_flow_bus_matrix(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.aux_flow_bus_matrix(active_flow=False)

    def aux_flow_bus_matrix(self, active_flow):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("rte_case14_realistic", test=True, _add_to_name=type(self).__name__)
        obs, reward, done, info = env.step(
            env.action_space({"set_bus": {"lines_or_id": [(7, 2), (8, 2)]}})
        )
        mat, (load, prod, stor, ind_lor, ind_lex) = obs.flow_bus_matrix(
            active_flow=active_flow, as_csr_matrix=True
        )
        assert mat.shape == (15, 15)
        assert ind_lor[7] == 14
        assert ind_lor[8] == 14
        # check that kirchoff law is met
        if active_flow:
            assert np.max(np.abs(mat.sum(axis=1))) <= self.tol_one
            assert np.abs(mat[0, 0] - obs.prod_p[-1]) <= self.tol_one
            assert np.abs(mat[0, 1] + obs.p_or[0]) <= self.tol_one
            assert np.abs(mat[0, 4] + obs.p_or[1]) <= self.tol_one
        else:
            assert np.max(np.abs(mat.sum(axis=1))) <= self.tol_one
            assert np.abs(mat[0, 0] - obs.prod_q[-1]) <= self.tol_one
            assert np.abs(mat[0, 1] + obs.q_or[0]) <= self.tol_one
            assert np.abs(mat[0, 4] + obs.q_or[1]) <= self.tol_one

        obs, reward, done, info = env.step(
            env.action_space(
                {"set_bus": {"lines_or_id": [(2, 2)], "lines_ex_id": [(0, 2)]}}
            )
        )
        mat, (load, prod, stor, ind_lor, ind_lex) = obs.flow_bus_matrix(
            active_flow=active_flow, as_csr_matrix=True
        )
        assert mat.shape == (16, 16)
        assert ind_lor[7] == 15
        assert ind_lor[8] == 15
        assert ind_lor[2] == 14
        assert ind_lex[0] == 14
        # check that kirchoff law is met
        if active_flow:
            assert np.max(np.abs(mat.sum(axis=1))) <= self.tol_one
            assert np.abs(mat[0, 0] - obs.prod_p[-1]) <= self.tol_one
            assert (
                np.abs(mat[0, 1] - 0) <= self.tol_one
            )  # no powerline connect bus 0 to bus 1 now (because i changed the bus)
            assert (
                np.abs(mat[0, 14] + obs.p_or[0]) <= self.tol_one
            )  # powerline 0 now connects bus 0 and bus 14
            assert (
                np.abs(mat[0, 4] + obs.p_or[1]) <= self.tol_one
            )  # powerline 1 has not moved
        else:
            assert np.max(np.abs(mat.sum(axis=1))) <= self.tol_one
            assert np.abs(mat[0, 0] - obs.prod_q[-1]) <= self.tol_one
            assert (
                np.abs(mat[0, 1] - 0) <= self.tol_one
            )  # no powerline connect bus 0 to bus 1 now (because i changed the bus)
            assert (
                np.abs(mat[0, 14] + obs.q_or[0]) <= self.tol_one
            )  # powerline 0 now connects bus 0 and bus 14
            assert (
                np.abs(mat[0, 4] + obs.q_or[1]) <= self.tol_one
            )  # powerline 1 has not moved
        env.close()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("educ_case14_storage", test=True, action_class=CompleteAction, _add_to_name=type(self).__name__)
        obs = env.reset()
        mat, (load, prod, stor, ind_lor, ind_lex) = obs.flow_bus_matrix(
            active_flow=active_flow, as_csr_matrix=True
        )
        assert mat.shape == (14, 14)
        assert np.max(np.abs(mat.sum(axis=1))) <= self.tol_one

        if active_flow:
            assert np.abs(mat[0, 0] - obs.prod_p[-1]) <= self.tol_one
            assert np.abs(mat[0, 1] + obs.p_or[0]) <= self.tol_one
            assert np.abs(mat[0, 4] + obs.p_or[1]) <= self.tol_one
            assert np.abs(mat[0, 4] + obs.p_or[1]) <= self.tol_one
        else:
            assert np.abs(mat[0, 0] - obs.prod_q[-1]) <= self.tol_one
            assert np.abs(mat[0, 1] + obs.q_or[0]) <= self.tol_one
            assert np.abs(mat[0, 4] + obs.q_or[1]) <= self.tol_one
            assert np.abs(mat[0, 4] + obs.q_or[1]) <= self.tol_one

        array_modif = np.array([1.5, 5.0], dtype=dt_float) * 0.0
        obs, reward, done, info = env.step(
            env.action_space(
                {
                    "set_storage": array_modif,
                    "set_bus": {"lines_or_id": [(7, 2), (8, 2)]},
                }
            )
        )
        assert not info["exception"]
        mat, (load, prod, stor, ind_lor, ind_lex) = obs.flow_bus_matrix(
            active_flow=active_flow, as_csr_matrix=True
        )
        assert mat.shape == (15, 15)
        assert ind_lor[7] == 14
        assert ind_lor[8] == 14
        # check that kirchoff law is met
        if active_flow:
            assert np.max(np.abs(mat.sum(axis=1))) <= self.tol_one
            assert np.abs(mat[0, 0] - obs.prod_p[-1]) <= self.tol_one
            assert np.abs(mat[0, 1] + obs.p_or[0]) <= self.tol_one
            assert np.abs(mat[0, 4] + obs.p_or[1]) <= self.tol_one
            assert np.abs(mat[0, 4] + obs.p_or[1]) <= self.tol_one
        else:
            assert np.max(np.abs(mat.sum(axis=1))) <= self.tol_one
            assert np.abs(mat[0, 0] - obs.prod_q[-1]) <= self.tol_one
            assert np.abs(mat[0, 1] + obs.q_or[0]) <= self.tol_one
            assert np.abs(mat[0, 4] + obs.q_or[1]) <= self.tol_one
            assert np.abs(mat[0, 4] + obs.q_or[1]) <= self.tol_one

        obs, reward, done, info = env.step(
            env.action_space(
                {
                    "set_storage": array_modif,
                    "set_bus": {"lines_or_id": [(2, 2)], "lines_ex_id": [(0, 2)]},
                }
            )
        )
        mat, (load, prod, stor, ind_lor, ind_lex) = obs.flow_bus_matrix(
            active_flow=active_flow, as_csr_matrix=True
        )
        assert mat.shape == (16, 16)
        assert ind_lor[7] == 15
        assert ind_lor[8] == 15
        assert ind_lor[2] == 14
        assert ind_lex[0] == 14
        # check that kirchoff law is met
        assert np.max(np.abs(mat.sum(axis=1))) <= self.tol_one
        if active_flow:
            assert np.abs(mat[0, 0] - obs.prod_p[-1]) <= self.tol_one
            assert (
                np.abs(mat[0, 1] - 0) <= self.tol_one
            )  # no powerline connect bus 0 to bus 1 now (because i changed the bus)
            assert (
                np.abs(mat[0, 14] + obs.p_or[0]) <= self.tol_one
            )  # powerline 0 now connects bus 0 and bus 14
            assert (
                np.abs(mat[0, 4] + obs.p_or[1]) <= self.tol_one
            )  # powerline 1 has not moved
        else:
            assert np.abs(mat[0, 0] - obs.prod_q[-1]) <= self.tol_one
            assert (
                np.abs(mat[0, 1] - 0) <= self.tol_one
            )  # no powerline connect bus 0 to bus 1 now (because i changed the bus)
            assert (
                np.abs(mat[0, 14] + obs.q_or[0]) <= self.tol_one
            )  # powerline 0 now connects bus 0 and bus 14
            assert (
                np.abs(mat[0, 4] + obs.q_or[1]) <= self.tol_one
            )  # powerline 1 has not moved

    def test_observation_space(self):
        obs = self.env.observation_space(self.env)
        assert self.env.observation_space.n == obs.size()

    def test_shape_correct(self):
        obs = self.env.observation_space(self.env)
        assert obs.shapes().shape == obs.dtypes().shape
        assert np.all(obs.dtypes() == self.dtypes)
        assert np.all(obs.shapes() == self.shapes)

    def test_0_load_properly(self):
        # this test aims at checking that everything in setUp is working properly, eg that "ObsEnv" class has enough
        # information for example
        assert type(self.env).shunts_data_available

    def test_1_generating_obs(self):
        # test that helper_obs is abl to generate a valid observation
        assert type(self.env).shunts_data_available
        obs = self.env.observation_space(self.env)
        assert type(self.env).shunts_data_available
        assert type(obs).shunts_data_available

    def test_2_reset(self):
        # test that helper_obs is abl to generate a valid observation
        obs = self.env.observation_space(self.env)
        assert obs.prod_p[0] is not None
        obs.reset()
        assert np.all(np.isnan(obs.prod_p))
        assert np.all(obs.dtypes() == self.dtypes)
        assert np.all(obs.shapes() == self.shapes)

    def test_3_reset(self):
        # test that helper_obs is able to generate a valid observation
        obs = self.env.observation_space(self.env)
        obs2 = obs.copy()
        assert obs == obs2
        obs2.reset()
        assert np.all(np.isnan(obs2.prod_p))
        assert np.all(obs2.dtypes() == self.dtypes)
        assert np.all(obs2.shapes() == self.shapes)
        # assert obs.prod_p is not None

    def test_shapes_types(self):
        obs = self.env.observation_space(self.env)
        dtypes = obs.dtypes()
        assert np.all(dtypes == self.dtypes)
        shapes = obs.shapes()
        assert np.all(shapes == self.shapes)

    def test_4_to_from_vect(self):
        # test that helper_obs is able to generate a valid observation
        obs = self.env.observation_space(self.env)
        obs2 = self.env.observation_space(self.env)
        vect = obs.to_vect()
        assert vect.shape[0] == obs.size()
        obs2.reset()
        obs2.from_vect(vect)
        assert np.all(obs.dtypes() == self.dtypes)
        assert np.all(obs.shapes() == self.shapes)

        # TODO there is not reason that these 2 are equal: reset, will erase everything
        # TODO whereas creating the observation
        # assert obs == obs2
        obs_diff, attr_diff = obs.where_different(obs2)
        for el in attr_diff:
            assert el in obs.attr_list_json, f"{el} should be equal in obs and obs2"
        vect2 = obs2.to_vect()
        assert np.all(vect == vect2)

    def test_5_simulate_proper_timestep(self):
        self.skipTest(
            "This is extensively tested elswhere, and the chronics have been changed."
        )
        obs_orig = self.env.observation_space(self.env)
        action = self.env.action_space({})
        action2 = self.env.action_space({})

        simul_obs, simul_reward, simul_has_error, simul_info = obs_orig.simulate(action)
        real_obs, real_reward, real_has_error, real_info = self.env.step(action2)
        assert not real_has_error, "The powerflow diverged"

        # this is not true for every observation chronics, but we made sure in this files that the forecast were
        # without any noise, maintenance, nor hazards
        assert (
            simul_obs == real_obs
        ), "there is a mismatch in the observation, though they are supposed to be equal"
        assert np.abs(simul_reward - real_reward) <= self.tol_one

    def test_6_simulate_dont_affect_env(self):
        obs_orig = self.env.observation_space(self.env)
        obs_orig = obs_orig.copy()

        for i in range(self.env.backend.n_line):
            # simulate lots of action
            tmp = np.full(self.env.backend.n_line, fill_value=False, dtype=dt_bool)
            tmp[i] = True
            action = self.env.action_space({"change_line_status": tmp})
            simul_obs, simul_reward, simul_has_error, simul_info = obs_orig.simulate(
                action
            )

        obs_after = self.env.observation_space(self.env)
        assert obs_orig == obs_after

    def test_inspect_load(self):
        obs = self.env.observation_space(self.env)
        dict_ = obs.state_of(load_id=0)
        assert "p" in dict_
        assert np.abs(dict_["p"] - 21.2) <= self.tol_one
        assert "q" in dict_
        assert np.abs(dict_["q"] - 14.9) <= self.tol_one
        assert "v" in dict_
        assert np.abs(dict_["v"] - 142.1) <= self.tol_one
        assert "bus" in dict_
        assert dict_["bus"] == 1
        assert "sub_id" in dict_
        assert dict_["sub_id"] == 1

    def test_inspect_gen(self):
        obs = self.env.observation_space(self.env)
        dict_ = obs.state_of(gen_id=0)
        assert "p" in dict_
        assert np.abs(dict_["p"] - 93.6) <= self.tol_one
        assert "q" in dict_
        assert np.abs(dict_["q"] - 65.49697) <= self.tol_one
        assert "v" in dict_
        assert np.abs(dict_["v"] - 142.1) <= self.tol_one
        assert "bus" in dict_
        assert dict_["bus"] == 1
        assert "sub_id" in dict_
        assert dict_["sub_id"] == 1

    def test_inspect_line(self):
        obs = self.env.observation_space(self.env)
        dict_both = obs.state_of(line_id=0)
        assert "origin" in dict_both
        dict_ = dict_both["origin"]
        assert "p" in dict_
        assert np.abs(dict_["p"] - 39.33145) <= self.tol_one
        assert "q" in dict_
        assert np.abs(dict_["q"] - -15.304552) <= self.tol_one
        assert "v" in dict_
        assert np.abs(dict_["v"] - 142.1) <= self.tol_one
        assert "bus" in dict_
        assert dict_["bus"] == 1
        assert "sub_id" in dict_
        assert dict_["sub_id"] == 0

        assert "extremity" in dict_both
        dict_ = dict_both["extremity"]
        assert "p" in dict_
        assert np.abs(dict_["p"] - -39.034054) <= self.tol_one
        assert "q" in dict_
        assert np.abs(dict_["q"] - 10.362568) <= self.tol_one
        assert "v" in dict_
        assert np.abs(dict_["v"] - 142.1) <= self.tol_one
        assert "bus" in dict_
        assert dict_["bus"] == 1
        assert "sub_id" in dict_
        assert dict_["sub_id"] == 1

    def test_inspect_topo(self):
        obs = self.env.observation_space(self.env)
        dict_ = obs.state_of(substation_id=1)
        assert "topo_vect" in dict_
        assert np.all(dict_["topo_vect"] == [1, 1, 1, 1, 1, 1])
        assert "nb_bus" in dict_
        assert dict_["nb_bus"] == 1

    def test_get_obj_connect_to(self):
        dict_ = self.env.observation_space.get_obj_connect_to(substation_id=1)
        assert "loads_id" in dict_
        assert np.all(dict_["loads_id"] == 0)
        assert "generators_id" in dict_
        assert np.all(dict_["generators_id"] == 0)
        assert "lines_or_id" in dict_
        assert np.all(dict_["lines_or_id"] == [7, 8, 9])
        assert "lines_ex_id" in dict_
        assert np.all(dict_["lines_ex_id"] == 0)
        assert "nb_elements" in dict_
        assert dict_["nb_elements"] == 6

    def test_space_to_dict(self):
        dict_ = self.env.observation_space.cls_to_dict()
        for el in dict_:
            assert el in self.dict_, f"missing key {el} in self.dict_"
        for el in self.dict_:
            assert el in dict_, f"missing key {el} in dict_"
            
        for el in self.dict_:
            val = dict_[el]
            val_res = self.dict_[el]
            if val is None and val_res is not None:
                raise AssertionError(f"val is None and val_res is not None: val_res: {val_res}")
            if val is not None and val_res is None:
                raise AssertionError(f"val is not None and val_res is None: val {val}")
            if val is None and val_res is None:
                continue
            
            ok_ = np.array_equal(val, val_res)
            assert ok_, (f"values different for {el}: "
                         f"{dict_[el]} vs "
                         f"{self.dict_[el]}")
            
        # self.maxDiff = None
        # self.assertDictEqual(dict_, self.dict_)

    def test_from_dict(self):
        res = ObservationSpace.from_dict(self.dict_)
        assert res.n_gen == self.env.observation_space.n_gen
        assert res.n_load == self.env.observation_space.n_load
        assert res.n_line == self.env.observation_space.n_line
        assert np.all(res.sub_info == self.env.observation_space.sub_info)
        assert np.all(res.load_to_subid == self.env.observation_space.load_to_subid)
        assert np.all(res.gen_to_subid == self.env.observation_space.gen_to_subid)
        assert np.all(
            res.line_or_to_subid == self.env.observation_space.line_or_to_subid
        )
        assert np.all(
            res.line_ex_to_subid == self.env.observation_space.line_ex_to_subid
        )
        assert np.all(res.load_to_sub_pos == self.env.observation_space.load_to_sub_pos)
        assert np.all(res.gen_to_sub_pos == self.env.observation_space.gen_to_sub_pos)
        assert np.all(
            res.line_or_to_sub_pos == self.env.observation_space.line_or_to_sub_pos
        )
        assert np.all(
            res.line_ex_to_sub_pos == self.env.observation_space.line_ex_to_sub_pos
        )
        assert np.all(
            res.load_pos_topo_vect == self.env.observation_space.load_pos_topo_vect
        )
        assert np.all(
            res.gen_pos_topo_vect == self.env.observation_space.gen_pos_topo_vect
        )
        assert np.all(
            res.line_or_pos_topo_vect
            == self.env.observation_space.line_or_pos_topo_vect
        )
        assert np.all(
            res.line_ex_pos_topo_vect
            == self.env.observation_space.line_ex_pos_topo_vect
        )
        assert issubclass(
            res.observationClass, self.env.observation_space._init_subtype
        )

    def test_json_serializable(self):
        dict_ = self.env.observation_space.cls_to_dict()
        res = json.dumps(obj=dict_, indent=4, sort_keys=True)

    def test_json_loadable(self):
        dict_ = self.env.observation_space.cls_to_dict()
        tmp = json.dumps(obj=dict_, indent=4, sort_keys=True)
        res = ObservationSpace.from_dict(json.loads(tmp))

        assert res.n_gen == self.env.observation_space.n_gen
        assert res.n_load == self.env.observation_space.n_load
        assert res.n_line == self.env.observation_space.n_line
        assert np.all(res.sub_info == self.env.observation_space.sub_info)
        assert np.all(res.load_to_subid == self.env.observation_space.load_to_subid)
        assert np.all(res.gen_to_subid == self.env.observation_space.gen_to_subid)
        assert np.all(
            res.line_or_to_subid == self.env.observation_space.line_or_to_subid
        )
        assert np.all(
            res.line_ex_to_subid == self.env.observation_space.line_ex_to_subid
        )
        assert np.all(res.load_to_sub_pos == self.env.observation_space.load_to_sub_pos)
        assert np.all(res.gen_to_sub_pos == self.env.observation_space.gen_to_sub_pos)
        assert np.all(
            res.line_or_to_sub_pos == self.env.observation_space.line_or_to_sub_pos
        )
        assert np.all(
            res.line_ex_to_sub_pos == self.env.observation_space.line_ex_to_sub_pos
        )
        assert np.all(
            res.load_pos_topo_vect == self.env.observation_space.load_pos_topo_vect
        )
        assert np.all(
            res.gen_pos_topo_vect == self.env.observation_space.gen_pos_topo_vect
        )
        assert np.all(
            res.line_or_pos_topo_vect
            == self.env.observation_space.line_or_pos_topo_vect
        )
        assert np.all(
            res.line_ex_pos_topo_vect
            == self.env.observation_space.line_ex_pos_topo_vect
        )
        assert issubclass(
            res.observationClass, self.env.observation_space._init_subtype
        )

    def test_to_from_json(self):
        """test the to_json, and from_json and make sure these are all  json serializable"""
        obs = self.env.observation_space(self.env)
        obs2 = self.env.observation_space(self.env)
        dict_ = obs.to_json()

        # test that the right dictionary is returned
        for k in dict_:
            assert (
                dict_[k] == self.json_ref[k]
            ), f"error for key {k} (in dict_): {dict_[k]} vs {self.json_ref[k]} "
        for k in self.json_ref:
            assert dict_[k] == self.json_ref[k], f"error for key {k} (in self.json_ref)"
        self.assertDictEqual(dict_, self.json_ref)

        with tempfile.TemporaryDirectory() as tmpdirname:
            # test i can save it (json serializable)
            with open(os.path.join(tmpdirname, "test.json"), "w") as fp:
                json.dump(obj=dict_, fp=fp)
            # test i can properly load it back
            with open(os.path.join(tmpdirname, "test.json"), "r") as fp:
                dict_realoaded = json.load(fp=fp)
        assert dict_realoaded == dict_
        # test i can initialize an observation from it
        obs2.reset()
        obs2.from_json(dict_realoaded)
        assert obs == obs2


class TestUpdateEnvironement(unittest.TestCase):
    def setUp(self):
        # Create env and obs in left hand
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.lenv = grid2op.make("rte_case5_example", test=True, _add_to_name=type(self).__name__)
            self.lobs = self.lenv.reset()

        # Create env and obs in right hand
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.renv = grid2op.make("rte_case5_example", test=True, _add_to_name=type(self).__name__)
            # Step once to make it different
            self.robs, _, _, _ = self.renv.step(self.renv.action_space())

        # Update left obs with right hand side environement
        self.lobs.update(self.renv)

    def tearDown(self):
        self.lenv.close()
        self.renv.close()

    def test_topology_updates(self):
        # Check left observation topology is updated to the right observation topology
        assert np.all(self.lobs.timestep_overflow == self.robs.timestep_overflow)
        assert np.all(self.lobs.line_status == self.robs.line_status)
        assert np.all(self.lobs.topo_vect == self.robs.topo_vect)

    def test_prods_updates(self):
        # Check left loads are updated to the right loads
        assert np.all(self.lobs.prod_p == self.robs.prod_p)
        assert np.all(self.lobs.prod_q == self.robs.prod_q)
        assert np.all(self.lobs.prod_v == self.robs.prod_v)

    def test_loads_updates(self):
        # Check left loads are updated to the right loads
        assert np.all(self.lobs.load_p == self.robs.load_p)
        assert np.all(self.lobs.load_q == self.robs.load_q)
        assert np.all(self.lobs.load_v == self.robs.load_v)

    def test_lines_or_updates(self):
        # Check left loads are updated to the right loads
        assert np.all(self.lobs.p_or == self.robs.p_or)
        assert np.all(self.lobs.q_or == self.robs.q_or)
        assert np.all(self.lobs.v_or == self.robs.v_or)
        assert np.all(self.lobs.a_or == self.robs.a_or)

    def test_lines_ex_updates(self):
        # Check left loads are updated to the rhs loads
        assert np.all(self.lobs.p_ex == self.robs.p_ex)
        assert np.all(self.lobs.q_ex == self.robs.q_ex)
        assert np.all(self.lobs.v_ex == self.robs.v_ex)
        assert np.all(self.lobs.a_ex == self.robs.a_ex)

    def test_forecasts_updates(self):
        # Check left forecasts are updated to the rhs forecasts
        # Check forecasts sizes
        assert len(self.lobs._forecasted_inj) == len(self.robs._forecasted_inj)
        # Check each forecast
        for i in range(len(self.lobs._forecasted_inj)):
            # Check timestamp
            assert self.lobs._forecasted_inj[i][0] == self.robs._forecasted_inj[i][0]
            # Check load_p
            l_load_p = self.lobs._forecasted_inj[i][1]["injection"]["load_p"]
            r_load_p = self.robs._forecasted_inj[i][1]["injection"]["load_p"]
            assert np.all(l_load_p == r_load_p)
            # Check load_q
            l_load_q = self.lobs._forecasted_inj[i][1]["injection"]["load_q"]
            r_load_q = self.robs._forecasted_inj[i][1]["injection"]["load_q"]
            assert np.all(l_load_q == r_load_q)
            # Check prod_p
            l_prod_p = self.lobs._forecasted_inj[i][1]["injection"]["prod_p"]
            r_prod_p = self.robs._forecasted_inj[i][1]["injection"]["prod_p"]
            assert np.all(l_prod_p == r_prod_p)
            # Check prod_v
            l_prod_v = self.lobs._forecasted_inj[i][1]["injection"]["prod_v"]
            r_prod_v = self.robs._forecasted_inj[i][1]["injection"]["prod_v"]
            assert np.all(l_prod_v == r_prod_v)

            # Check maintenance
            # we never forecasted the maintenance anyway
            # l_maintenance = self.lobs._forecasted_inj[i][1]['maintenance']
            # r_maintenance = self.robs._forecasted_inj[i][1]['maintenance']
            # assert np.all(l_maintenance == r_maintenance)

        # Check relative flows
        assert np.all(self.lobs.rho == self.robs.rho)

    def test_cooldowns_updates(self):
        # Check left cooldowns are updated to the rhs CDs
        assert np.all(
            self.lobs.time_before_cooldown_line == self.robs.time_before_cooldown_line
        )
        assert np.all(
            self.lobs.time_before_cooldown_sub == self.robs.time_before_cooldown_sub
        )
        assert np.all(
            self.lobs.time_before_cooldown_line == self.robs.time_before_cooldown_line
        )
        assert np.all(
            self.lobs.time_next_maintenance == self.robs.time_next_maintenance
        )
        assert np.all(
            self.lobs.duration_next_maintenance == self.robs.duration_next_maintenance
        )

    def test_redispatch_updates(self):
        # Check left redispatch are updated to the rhs redispatches
        assert np.all(self.lobs.target_dispatch == self.robs.target_dispatch)
        assert np.all(self.lobs.actual_dispatch == self.robs.actual_dispatch)


class TestSimulateEqualsStep(unittest.TestCase):
    def _make_forecast_perfect(self, env):
        # Set forecasts to actual values so that simulate runs on the same numbers as step
        env.chronics_handler.real_data.data.prod_p_forecast = np.roll(
            self.env.chronics_handler.real_data.data.prod_p, -1, axis=0
        )
        env.chronics_handler.real_data.data.prod_v_forecast = np.roll(
            self.env.chronics_handler.real_data.data.prod_v, -1, axis=0
        )
        env.chronics_handler.real_data.data.load_p_forecast = np.roll(
            self.env.chronics_handler.real_data.data.load_p, -1, axis=0
        )
        env.chronics_handler.real_data.data.load_q_forecast = np.roll(
            self.env.chronics_handler.real_data.data.load_q, -1, axis=0
        )
        obs, _, _, _ = env.step(env.action_space({}))
        return obs

    def setUp(self):
        # Create env
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("rte_case14_realistic", test=True, _add_to_name=type(self).__name__)

        self.obs = self._make_forecast_perfect(self.env)
        self.sim_obs = None
        self.step_obs = None

    def tearDown(self):
        self.env.close()

    def test_do_nothing(self):
        # Create action
        donothing_act = self.env.action_space()
        # Simulate & Step
        self.sim_obs, _, _, _ = self.obs.simulate(donothing_act)
        self.step_obs, _, _, _ = self.env.step(donothing_act)
        # Test observations are the same
        if self.sim_obs != self.step_obs:
            diff_, attr_diff = self.sim_obs.where_different(self.step_obs)
            raise AssertionError(f"Following attributes are different: {attr_diff}")

    def test_change_line_status(self):
        # Get change status vector
        change_status = self.env.action_space.get_change_line_status_vect()
        # Make a change
        change_status[0] = True
        # Create change action
        change_act = self.env.action_space({"change_line_status": change_status})
        # Simulate & Step
        self.sim_obs, reward_sim, done_sim, _ = self.obs.simulate(change_act)
        self.step_obs, reward_real, done_real, _ = self.env.step(change_act)
        assert not done_sim
        assert not done_real
        assert abs(reward_sim - reward_real) <= 1e-7
        # Test observations are the same
        if self.sim_obs != self.step_obs:
            diff_, attr_diff = self.sim_obs.where_different(self.step_obs)
            raise AssertionError(f"Following attributes are different: {attr_diff}")

    def test_set_line_status(self):
        # Get set status vector
        set_status = self.env.action_space.get_set_line_status_vect()
        # Make a change
        set_status[0] = -1 if self.obs.line_status[0] else 1
        # Create set action
        set_act = self.env.action_space({"set_line_status": set_status})
        # Simulate & Step
        self.sim_obs, _, _, _ = self.obs.simulate(set_act)
        self.step_obs, _, _, _ = self.env.step(set_act)
        # Test observations are the same
        if self.sim_obs != self.step_obs:
            diff_, attr_diff = self.sim_obs.where_different(self.step_obs)
            raise AssertionError(f"Following attributes are different: {attr_diff}")

    def test_change_bus(self):
        # Create a change bus action for all types
        change_act = self.env.action_space(
            {
                "change_bus": {
                    "loads_id": [0],
                    "generators_ids": [0],
                    "lines_or_id": [0],
                    "lines_ex_id": [0],
                }
            }
        )
        # Simulate & Step
        self.sim_obs, _, _, _ = self.obs.simulate(change_act)
        self.step_obs, _, _, _ = self.env.step(change_act)
        assert isinstance(
            self.sim_obs, type(self.step_obs)
        ), "sim_obs is not the same type as the step"
        assert isinstance(
            self.step_obs, type(self.sim_obs)
        ), "step is not the same type as the simulation"

        # Test observations are the same
        if self.sim_obs != self.step_obs:
            diff_, attr_diff = self.sim_obs.where_different(self.step_obs)
            raise AssertionError(f"Following attributes are different: {attr_diff}")

    def test_set_bus(self):
        # Increment buses from current topology
        new_load_bus = self.obs.topo_vect[self.obs.load_pos_topo_vect[0]] + 1
        new_gen_bus = self.obs.topo_vect[self.obs.gen_pos_topo_vect[0]] + 1
        new_lor_bus = self.obs.topo_vect[self.obs.line_or_pos_topo_vect[0]] + 1
        new_lex_bus = self.obs.topo_vect[self.obs.line_ex_pos_topo_vect[0]] + 1

        # Create a set bus action for all types
        set_act = self.env.action_space(
            {
                "set_bus": {
                    "loads_id": [(0, new_load_bus)],
                    "generators_ids": [(0, new_gen_bus)],
                    "lines_or_id": [(0, new_lor_bus)],
                    "lines_ex_id": [(0, new_lex_bus)],
                }
            }
        )
        # Simulate & Step
        self.sim_obs, _, _, _ = self.obs.simulate(set_act)
        self.step_obs, _, _, _ = self.env.step(set_act)
        # Test observations are the same
        if self.sim_obs != self.step_obs:
            diff_, attr_diff = self.sim_obs.where_different(self.step_obs)
            raise AssertionError(f"Following attributes are different: {attr_diff}")

    def test_redispatch(self):
        if DEACTIVATE_FAILING_TEST:
            return
        # Find first redispatchable generator
        gen_id = next((i for i, j in enumerate(self.obs.gen_redispatchable) if j), None)
        # Create valid ramp up
        redisp_val = self.obs.gen_max_ramp_up[gen_id] / 2.0
        # Create redispatch action
        redisp_act = self.env.action_space({"redispatch": [(gen_id, redisp_val)]})
        # Simulate & Step
        self.sim_obs, _, _, _ = self.obs.simulate(redisp_act)
        self.step_obs, _, _, _ = self.env.step(redisp_act)
        # Test observations are the same
        if self.sim_obs != self.step_obs:
            diff_, attr_diff = self.sim_obs.where_different(self.step_obs)
            raise AssertionError(f"Following attributes are different: {attr_diff}")

    def test_change_simulate_reward(self):
        """test the env.observation_space.change_other_reward function"""
        # Create env
        other_rewards = {
            "close_overflow": CloseToOverflowReward,
            "l2rpn": L2RPNReward,
            "redisp": RedispReward,
        }
        env = self.env

        # do as if the environment were created with these rewards !
        env.observation_space.change_other_rewards(copy.deepcopy(other_rewards))
        env.other_rewards = {}
        for k, reward in other_rewards.items():
            env.other_rewards[k] = RewardHelper(reward)
            env.other_rewards[k].initialize(env)

        # Set forecasts to actual values so that simulate runs on the same numbers as step
        first_obs = self._make_forecast_perfect(env)

        sim_o, sim_r, sim_d, sim_i = first_obs.simulate(env.action_space())
        for k in other_rewards.keys():
            assert k in sim_i["rewards"]
        obs, reward, done, info = env.step(env.action_space())
        # check rewards are same, this is the case because simulate is in "perfect information"
        assert np.all(sim_o.rho == obs.rho)
        self._aux_comp_reward(info, sim_i)
        assert np.all(sim_o.load_p == obs.load_p)

        env.observation_space.change_other_rewards({})
        sim_o, sim_r, sim_d, sim_i = obs.simulate(env.action_space())
        # check the rewards have disappeared
        for k in other_rewards.keys():
            assert k not in sim_i["rewards"]

        # check they are still present on real environment
        obs, reward, done, info = env.step(env.action_space())
        for k in other_rewards.keys():
            assert k in info["rewards"]

        env.observation_space.change_other_rewards(other_rewards)
        sim_o, sim_r, sim_d, sim_i = obs.simulate(env.action_space())
        for k in other_rewards.keys():
            assert k in sim_i["rewards"]
        obs, reward, done, info = env.step(env.action_space())
        
        # check rewards are same, this is the case because simulate is in "perfect information"
        assert np.all(sim_o.rho == obs.rho)
        self._aux_comp_reward(info, sim_i)
        assert np.all(sim_o.load_p == obs.load_p)

    def _aux_comp_reward(self, info, sim_info):
        for el in info["rewards"]:
            tmp_info = info["rewards"][el]
            tmp_sinfo = sim_info["rewards"][el]
            assert np.allclose(tmp_info, tmp_sinfo), f"error for {el}: in info: {tmp_info}, in simulated info: {tmp_sinfo}"
        
    def _multi_actions_sample(self):
        actions = []
        ## do_nothing action
        donothing_act = self.env.action_space()
        actions.append(donothing_act)

        ## change_status action
        # Get change status vector
        change_status = self.env.action_space.get_change_line_status_vect()
        # Make a change
        change_status[0] = True
        # Register change action
        change_act = self.env.action_space({"change_line_status": change_status})
        actions.append(change_act)

        ## set_status action
        # Get set status vector
        set_status = self.env.action_space.get_set_line_status_vect()
        # Make a change
        set_status[0] = -1 if self.obs.line_status[0] else 1
        # Register set action
        set_act = self.env.action_space({"set_line_status": set_status})
        actions.append(set_act)

        ## change_bus action
        # Register a change bus action for all types
        change_bus_act = self.env.action_space(
            {
                "change_bus": {
                    "loads_id": [0],
                    "generators_ids": [0],
                    "lines_or_id": [0],
                    "lines_ex_id": [0],
                }
            }
        )
        actions.append(change_bus_act)

        ## set_bus_action
        # Increment buses from current topology
        new_load_bus = self.obs.topo_vect[self.obs.load_pos_topo_vect[0]] + 1
        new_gen_bus = self.obs.topo_vect[self.obs.gen_pos_topo_vect[0]] + 1
        new_lor_bus = self.obs.topo_vect[self.obs.line_or_pos_topo_vect[0]] + 1
        new_lex_bus = self.obs.topo_vect[self.obs.line_ex_pos_topo_vect[0]] + 1
        # Create a set bus action for all types
        set_bus_act = self.env.action_space(
            {
                "set_bus": {
                    "loads_id": [(0, new_load_bus)],
                    "generators_ids": [(0, new_gen_bus)],
                    "lines_or_id": [(0, new_lor_bus)],
                    "lines_ex_id": [(0, new_lex_bus)],
                }
            }
        )
        actions.append(set_bus_act)

        ## redispatch action
        # Find first redispatchable generator
        gen_id = next((i for i, j in enumerate(self.obs.gen_redispatchable) if j), None)
        # Create valid ramp up
        redisp_val = self.obs.gen_max_ramp_up[gen_id] / 2.0
        # Create redispatch action
        redisp_act = self.env.action_space({"redispatch": [(gen_id, redisp_val)]})
        actions.append(redisp_act)

        return actions

    def test_multi_simulate_last_do_nothing(self):
        if DEACTIVATE_FAILING_TEST:
            return
        actions = self._multi_actions_sample()

        # Add do_nothing last
        actions.append(self.env.action_space())

        # Simulate all actions
        for act in actions:
            self.sim_obs, _, _, _ = self.obs.simulate(act)
        # Step with last action
        self.step_obs, _, _, _ = self.env.step(actions[-1])
        # Test observations are the same
        if self.sim_obs != self.step_obs:
            diff_, attr_diff = self.sim_obs.where_different(self.step_obs)
            raise AssertionError(f"Following attributes are different: {attr_diff}")

    def test_multi_simulate_last_change_line_status(self):
        if DEACTIVATE_FAILING_TEST:
            return
        actions = self._multi_actions_sample()

        ## Add change_line_status last
        # Get change status vector
        change_status = self.env.action_space.get_change_line_status_vect()
        # Make a change
        change_status[1] = True
        # Register change action
        change_act = self.env.action_space({"change_line_status": change_status})
        actions.append(change_act)

        # Simulate all actions
        for act in actions:
            self.sim_obs, _, _, _ = self.obs.simulate(act)
        # Step with last action
        self.step_obs, _, _, _ = self.env.step(actions[-1])
        # Test observations are the same
        if self.sim_obs != self.step_obs:
            diff_, attr_diff = self.sim_obs.where_different(self.step_obs)
            raise AssertionError(f"Following attributes are different: {attr_diff}")

    def test_multi_simulate_last_set_line_status(self):
        if DEACTIVATE_FAILING_TEST:
            return
        actions = self._multi_actions_sample()
        ## Add set_status action last
        # Get set status vector
        set_status = self.env.action_space.get_set_line_status_vect()
        # Make a change
        set_status[1] = -1 if self.obs.line_status[1] else 1
        # Register set action
        set_act = self.env.action_space({"set_line_status": set_status})
        actions.append(set_act)

        # Simulate all actions
        for act in actions:
            self.sim_obs, _, _, _ = self.obs.simulate(act)
        # Step with last action
        self.step_obs, _, _, _ = self.env.step(actions[-1])
        # Test observations are the same
        if self.sim_obs != self.step_obs:
            diff_, attr_diff = self.sim_obs.where_different(self.step_obs)
            raise AssertionError(f"Following attributes are different: {attr_diff}")

    def test_multi_simulate_last_change_bus(self):
        if DEACTIVATE_FAILING_TEST:
            return
        actions = self._multi_actions_sample()

        ## Add change_bus action last
        # Register a change bus action for all types
        change_bus_act = self.env.action_space(
            {
                "change_bus": {
                    "loads_id": [1],
                    "generators_ids": [1],
                    "lines_or_id": [1],
                    "lines_ex_id": [1],
                }
            }
        )
        actions.append(change_bus_act)

        # Simulate all actions
        for act in actions:
            self.sim_obs, _, _, _ = self.obs.simulate(act)
        # Step with last action
        self.step_obs, _, _, _ = self.env.step(actions[-1])
        # Test observations are the same
        if self.sim_obs != self.step_obs:
            diff_, attr_diff = self.sim_obs.where_different(self.step_obs)
            raise AssertionError(f"Following attributes are different: {attr_diff}")

    def test_multi_simulate_last_set_bus(self):
        if DEACTIVATE_FAILING_TEST:
            return
        actions = self._multi_actions_sample()
        ## Add set_bus_action last
        # Increment buses from current topology
        new_load_bus = self.obs.topo_vect[self.obs.load_pos_topo_vect[1]] + 1
        new_gen_bus = self.obs.topo_vect[self.obs.gen_pos_topo_vect[1]] + 1
        new_lor_bus = self.obs.topo_vect[self.obs.line_or_pos_topo_vect[1]] + 1
        new_lex_bus = self.obs.topo_vect[self.obs.line_ex_pos_topo_vect[1]] + 1
        # Create a set bus action for all types
        set_bus_act = self.env.action_space(
            {
                "set_bus": {
                    "loads_id": [(1, new_load_bus)],
                    "generators_ids": [(1, new_gen_bus)],
                    "lines_or_id": [(1, new_lor_bus)],
                    "lines_ex_id": [(1, new_lex_bus)],
                }
            }
        )
        actions.append(set_bus_act)

        # Simulate all actions
        for act in actions:
            self.sim_obs, _, _, _ = self.obs.simulate(act)
        # Step with last action
        self.step_obs, _, _, _ = self.env.step(actions[-1])
        # Test observations are the same
        if self.sim_obs != self.step_obs:
            diff_, attr_diff = self.sim_obs.where_different(self.step_obs)
            raise AssertionError(f"Following attributes are different: {attr_diff}")

    def test_multi_simulate_last_redispatch(self):
        if DEACTIVATE_FAILING_TEST:
            return
        actions = self._multi_actions_sample()

        ## Add redispatch action last
        # Find second redispatchable generator
        matches = 0
        gen_id = -1
        for i, j in enumerate(self.obs.gen_redispatchable):
            if j:
                matches += 1
                gen_id = i
            if matches == 2:
                break
        # Make sure we have a generator
        assert gen_id != -1
        # Create valid ramp up
        redisp_val = self.obs.gen_max_ramp_up[gen_id] / 2.0
        # Create redispatch action
        redisp_act = self.env.action_space({"redispatch": [(gen_id, redisp_val)]})
        actions.append(redisp_act)

        # Simulate all actions
        # for act in actions:
        #     self.sim_obs, _, _, _ = self.obs.simulate(act)
        self.sim_obs, _, _, _ = self.obs.simulate(actions[-1])   
         
        # Step with last action
        self.step_obs, _, _, _ = self.env.step(actions[-1])
        
        # Test observations are the same
        if self.sim_obs != self.step_obs:
            diff_, attr_diff = self.sim_obs.where_different(self.step_obs)
            raise AssertionError(f"Following attributes are different: {attr_diff}")

    def test_forecasted_inj(self):
        sim_obs, _, _, _ = self.obs.simulate(self.env.action_space())
        prod_p_f, prod_v_f, load_p_f, load_q_f = self.obs.get_forecasted_inj()
        assert np.sum(np.abs(prod_v_f - sim_obs.prod_v)) < 1e-5
        assert np.sum(np.abs(load_p_f - sim_obs.load_p)) < 1e-5
        assert np.sum(np.abs(load_q_f - sim_obs.load_q)) < 1e-5
        # test all prod p are equal, of course we remove the slack bus...
        assert np.sum(np.abs(prod_p_f[:-1] - sim_obs.prod_p[:-1])) < 1e-5

    def _check_equal(self, obs1, obs2):
        tol = 1e-8
        assert np.all(np.abs(obs1.prod_p - obs2.prod_p) <= tol), "issue with prod_p"
        assert np.all(np.abs(obs1.prod_v - obs2.prod_v) <= tol), "issue with prod_v"
        assert np.all(np.abs(obs1.prod_q - obs2.prod_q) <= tol), "issue with prod_q"
        assert np.all(np.abs(obs1.load_p - obs2.load_p) <= tol), "issue with load_p"
        assert np.all(np.abs(obs1.load_q - obs2.load_q) <= tol), "issue with load_q"
        assert np.all(np.abs(obs1.load_v - obs2.load_v) <= tol), "issue with load_v"
        assert np.all(np.abs(obs1.rho - obs2.rho) <= tol), "issue with rho"
        assert np.all(np.abs(obs1.p_or - obs2.p_or) <= tol), "issue with p_or"
        assert np.all(np.abs(obs1.q_or - obs2.q_or) <= tol), "issue with q_or"
        assert np.all(np.abs(obs1.v_or - obs2.v_or) <= tol), "issue with v_or"
        assert np.all(np.abs(obs1.a_or - obs2.a_or) <= tol), "issue with a_or"
        assert np.all(np.abs(obs1.p_ex - obs2.p_ex) <= tol), "issue with p_ex"
        assert np.all(np.abs(obs1.q_ex - obs2.q_ex) <= tol), "issue with q_ex"
        assert np.all(np.abs(obs1.v_ex - obs2.v_ex) <= tol), "issue with v_ex"
        assert np.all(np.abs(obs1.a_ex - obs2.a_ex) <= tol), "issue with a_ex"
        assert np.all(
            np.abs(obs1.storage_power - obs2.storage_power) <= tol
        ), "issue with storage_power"

    def test_simulate_current_ts(self):
        sim_obs, _, _, _ = self.obs.simulate(self.env.action_space(), time_step=0)
        # check that the observations are equal
        self._check_equal(sim_obs, self.obs)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            obs = self.env.reset()
        sim_obs1, rew1, done1, _ = obs.simulate(
            self.env.action_space.disconnect_powerline(line_id=2)
        )
        sim_obs2, rew2, done2, _ = obs.simulate(self.env.action_space(), time_step=0)
        sim_obs3, rew3, done3, _ = obs.simulate(
            self.env.action_space.disconnect_powerline(line_id=2)
        )
        assert not done1
        assert not done2
        assert not done3
        self._check_equal(sim_obs2, obs)
        assert abs(rew1 - rew3) <= 1e-8, "issue with reward"
        self._check_equal(sim_obs1, sim_obs3)

    def test_nb_step(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            obs = self.env.reset()
        sim_obs1, rew1, done1, _ = obs.simulate(
            self.env.action_space.disconnect_powerline(line_id=2)
        )
        sim_obs2, rew2, done2, _ = obs.simulate(self.env.action_space(), time_step=0)
        sim_obs3, rew3, done3, _ = obs.simulate(
            self.env.action_space.disconnect_powerline(line_id=2)
        )
        assert obs.current_step == 0
        assert sim_obs1.current_step == 1
        assert sim_obs2.current_step == 1
        assert sim_obs3.current_step == 1

        obs, *_ = self.env.step(self.env.action_space())
        sim_obs1, rew1, done1, _ = obs.simulate(
            self.env.action_space.disconnect_powerline(line_id=2)
        )
        assert obs.current_step == 1
        assert sim_obs1.current_step == 2

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            obs = self.env.reset()
        sim_obs1, rew1, done1, _ = obs.simulate(
            self.env.action_space.disconnect_powerline(line_id=2)
        )
        sim_obs2, rew2, done2, _ = obs.simulate(self.env.action_space(), time_step=0)
        sim_obs3, rew3, done3, _ = obs.simulate(
            self.env.action_space.disconnect_powerline(line_id=2)
        )
        assert obs.current_step == 0
        assert sim_obs1.current_step == 1
        assert sim_obs2.current_step == 1
        assert sim_obs3.current_step == 1


class TestSimulateEqualsStepStorageCurtail(TestSimulateEqualsStep):
    def setUp(self):
        # Create env
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                "educ_case14_storage", test=True, action_class=PlayableAction,
                _add_to_name=type(self).__name__
            )
        self.obs = self._make_forecast_perfect(self.env)
        self.sim_obs = None
        self.step_obs = None
        
    def tearDown(self):
        self.env.close()

    def test_storage_act(self):
        """test i can do storage actions in simulate"""
        act = self.env.action_space()
        act.storage_power = [(0, 3)]
        obs = self.env.get_obs()
        sim_obs1, rew1, done1, _ = obs.simulate(act)
        assert not done1
        sim_obs2, rew2, done2, _ = obs.simulate(self.env.action_space(), time_step=0)
        assert not done2
        sim_obs3, rew3, done3, _ = obs.simulate(act)
        assert not done3
        real_obs, real_rew, real_done, _ = self.env.step(act)
        assert not real_done

        self._check_equal(sim_obs1, sim_obs3)
        self._check_equal(sim_obs2, obs)
        assert abs(rew1 - rew3) <= 1e-8, "issue with reward"
        self._check_equal(sim_obs1, sim_obs3)
        assert abs(rew1 - real_rew) <= 1e-8, "issue with reward"
        if real_obs != sim_obs3:
            diff_, attr_diff = real_obs.where_different(sim_obs3)
            raise AssertionError(f"Following attributes are different: {attr_diff}")

    def test_curtail_act(self):
        """test i can do a curtailment actions in simulate"""
        act = self.env.action_space()
        act.curtail = [(2, 0.1)]
        obs = self.env.get_obs()
        sim_obs1, rew1, done1, _ = obs.simulate(act)
        assert not done1
        sim_obs2, rew2, done2, _ = obs.simulate(self.env.action_space(), time_step=0)
        assert not done2
        sim_obs3, rew3, done3, _ = obs.simulate(act)
        assert not done3
        real_obs, real_rew, real_done, _ = self.env.step(act)
        assert not real_done

        self._check_equal(sim_obs1, sim_obs3)
        self._check_equal(sim_obs2, obs)
        assert (
            abs(rew1 - rew3) <= 1e-8
        ), "issue with reward between the two simulated actions"
        self._check_equal(sim_obs1, sim_obs3)
        assert abs(rew1 - real_rew) <= 1e-8, (
            "issue with reward between simulate (first curtail) "
            "and step (with curtail)"
        )

        if real_obs != sim_obs3:
            diff_, attr_diff = real_obs.where_different(sim_obs3)
            raise AssertionError(f"Following attributes are different: {attr_diff}")


if __name__ == "__main__":
    unittest.main()
