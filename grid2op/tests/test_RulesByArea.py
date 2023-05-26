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

    def test_rules_areas(self):
        params = Parameters()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            for rules in [self.rules_1area, self.rules_2areas, self.rules_3areas]:
                self.env = make(
                        "l2rpn_case14_sandbox",
                        test=True,
                        param=params,
                        gamerules_class = rules
                    )
                self.helper_action = self.env._helper_action_env
                lines_by_area = self.env._game_rules.legal_action.lines_id_by_area
                line_select = [[int(k) for k in np.random.choice(list_ids, size=3, replace=False)] for list_ids in lines_by_area.values()]
                #two lines one sub by area with 1 action in one area per item
                try:
                    self.env._parameters.MAX_SUB_CHANGED = 1
                    self.env._parameters.MAX_LINE_STATUS_CHANGED = 1
                    for line_select_byarea in line_select:
                        act = {
                            "set_line_status": [(LINE_ID, -1) for LINE_ID in line_select_byarea[:2]],
                            "change_bus" : {"lines_or_id":[LINE_ID for LINE_ID in line_select_byarea[2:]]}
                        }
                        _ = self.helper_action(
                            act,
                            env=self.env,
                            check_legal=True,
                        )
                    raise RuntimeError("This should have thrown an IllegalException")
                except IllegalAction:
                    pass

                #one line two sub by area with 1 action in one area per item
                try:
                    self.env._parameters.MAX_SUB_CHANGED = 1
                    self.env._parameters.MAX_LINE_STATUS_CHANGED = 1
                    for line_select_byarea in line_select:
                        act = {
                            "set_line_status": [(LINE_ID, -1) for LINE_ID in line_select_byarea[:1]],
                            "change_bus" : {"lines_or_id":[LINE_ID for LINE_ID in line_select_byarea[1:]]}
                        }
                        _ = self.helper_action(
                            act,
                            env=self.env,
                            check_legal=True,
                        )
                    raise RuntimeError("This should have thrown an IllegalException")
                except IllegalAction:
                    pass

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


if __name__ == "__main__":
    unittest.main()