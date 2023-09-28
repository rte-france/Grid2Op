
# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
import warnings
import unittest
import pdb


class Issue533Tester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make('l2rpn_neurips_2020_track1',
                                          test=True,
                                          )
        self.env.seed(0)
        return super().setUp()
    
    def tearDown(self):
        self.env.close()
        
    def test_issue_as_serializable_dict(self):
        actions = self.env.action_space.get_all_unitary_topologies_set(self.env.action_space, sub_id=1)   
        actions[1].as_serializable_dict()


if __name__ == '__main__':
    unittest.main()
