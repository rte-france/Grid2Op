
# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
from grid2op.Parameters import Parameters
import warnings
import unittest

class Issue511Tester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
            "l2rpn_idf_2023",
            test=True,
            _add_to_name=type(self).__name__
        )
        return super().setUp()
    
    def tearDown(self):
        self.env.close()

    def test_issue_set_bus(self):
        act = {
            "set_bus": {
                "lines_or_id": [(0, 2)],
                "loads_id": [(0, 2)],
            },
        }

        topo_action = self.env.action_space(act)
        as_dict =  topo_action.as_dict()
        assert len(as_dict['set_bus_vect']['0']) == 2  # two objects modified

    def test_issue_change_bus(self):
        act = {
            "change_bus": {
                "lines_or_id": [0],
                "loads_id": [0],
            },
        }

        topo_action = self.env.action_space(act)
        as_dict =  topo_action.as_dict()
        assert len(as_dict['change_bus_vect']['0']) == 2  # two objects modified
        

if __name__ == '__main__':
    unittest.main()
