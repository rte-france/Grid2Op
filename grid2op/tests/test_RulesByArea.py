# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from itertools import chain
from grid2op.tests.helper_path_test import *
from grid2op.Exceptions import *
from grid2op.Parameters import Parameters
from grid2op.Rules.rulesByArea import *
from grid2op.MakeEnv import make

import warnings

class TestDefaultRulesByArea(unittest.TestCase):
    def setUp(self):
        n_sub = 14
        self.rules_1area = RulesByArea([[int(k) for k in range(n_sub)]])
        self.rules_2areas = RulesByArea([[k for k in np.arange(n_sub,dtype=int)[:8]],[k for k in np.arange(n_sub,dtype=int)[8:]]])
        self.rules_3areas = RulesByArea([[k for k in np.arange(n_sub,dtype=int)[:4]],[k for k in np.arange(n_sub,dtype=int)[4:9]],[k for k in np.arange(n_sub,dtype=int)[9:]]])
        
    def test_legal_when_islegal(self):
        params = Parameters()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            #noaction
            self.env = make(
                    "l2rpn_case14_sandbox",
                    test=True,
                    param=params,
                    gamerules_class = self.rules_1area
                )
            self.helper_action = self.env._helper_action_env
            self.env._parameters.MAX_SUB_CHANGED = 1
            self.env._parameters.MAX_LINE_STATUS_CHANGED = 1
            act = {}
            _ = self.helper_action(
                    act,
                    env=self.env,
                    check_legal=True,
            )
            
            #test allowance max action in all areas over the grid
            for rules in [self.rules_2areas, self.rules_3areas]:
                self.env = make(
                        "l2rpn_case14_sandbox",
                        test=True,
                        param=params,
                        gamerules_class = rules
                    )
                self.helper_action = self.env._helper_action_env
                lines_by_area = self.env._game_rules.legal_action.lines_id_by_area
                line_select = [[int(k) for k in np.random.choice(list_ids, size=3, replace=False)] for list_ids in lines_by_area.values()]
                
                #one line one sub with one action per area per item per area
                self.env._parameters.MAX_SUB_CHANGED = 1
                self.env._parameters.MAX_LINE_STATUS_CHANGED = 1
                act = {
                        "set_line_status": [(LINE_ID, -1) for LINE_ID in list(chain(*[list_ids[:1] for list_ids in line_select]))],
                        "change_bus" : {"lines_or_id":[LINE_ID for LINE_ID in list(chain(*[list_ids[1:2] for list_ids in line_select]))]}
                }
                _ = self.helper_action(
                        act,
                        env=self.env,
                        check_legal=True,
                )

                #two lines one sub with two actions per line one per sub per area
                self.env._parameters.MAX_SUB_CHANGED = 1
                self.env._parameters.MAX_LINE_STATUS_CHANGED = 2
                act = {
                        "set_line_status": [(LINE_ID, -1) for LINE_ID in list(chain(*[list_ids[:2] for list_ids in line_select]))],
                        "change_bus" : {"lines_or_id":[LINE_ID for LINE_ID in list(chain(*[list_ids[2:] for list_ids in line_select]))]}
                }
                _ = self.helper_action(
                        act,
                        env=self.env,
                        check_legal=True,
                )

                #one line two sub with one action per line two per sub per area
                self.env._parameters.MAX_SUB_CHANGED = 2
                self.env._parameters.MAX_LINE_STATUS_CHANGED = 1
                act = {
                        "set_line_status": [(LINE_ID, -1) for LINE_ID in list(chain(*[list_ids[:1] for list_ids in line_select]))],
                        "change_bus" : {"lines_or_id":[LINE_ID for LINE_ID in list(chain(*[list_ids[1:] for list_ids in line_select]))]}
                }
                _ = self.helper_action(
                        act,
                        env=self.env,
                        check_legal=True,
                )
                self.env.close()
                
    def test_illegal_when_illegal(self):
        params = Parameters()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = make(
                    "l2rpn_case14_sandbox",
                    test=True,
                    param=params,
                    gamerules_class = self.rules_3areas
                    )
            self.env._parameters.MAX_SUB_CHANGED = 1
            self.env._parameters.MAX_LINE_STATUS_CHANGED = 1
            self.helper_action = self.env._helper_action_env
            lines_by_area = [list_ids for list_ids in self.env._game_rules.legal_action.lines_id_by_area.values()]
            
            #illegal action in one area due to lines
            with self.assertRaises(IllegalAction):
                act= {
                    "set_line_status": [(LINE_ID, -1) for LINE_ID in lines_by_area[0][:2]] + \
                     [(LINE_ID, -1) for LINE_ID in [lines_by_area[1][2], lines_by_area[2][2]]],
                }
                i = self.helper_action(
                        act,
                        env=self.env,
                        check_legal=True,
                )
            
            #illegal action in one area due to substations
            with self.assertRaises(IllegalAction):
                
                area0_sorted_ids = np.argsort(self.env.line_or_to_subid[lines_by_area[0]])
                aff_2subs_lines_ids = np.array(lines_by_area[0])[area0_sorted_ids][[0,-1]]
                act= {
                    "change_bus" : {"lines_or_id":[LINE_ID for LINE_ID in aff_2subs_lines_ids] + \
                        [LINE_ID for LINE_ID in [lines_by_area[1][2], lines_by_area[2][2]]]},
                }
                i = self.helper_action(
                        act,
                        env=self.env,
                        check_legal=True,
                )
            
            #illegal action in one area but still do action another area
            
            self.env.close()


if __name__ == "__main__":
    unittest.main()