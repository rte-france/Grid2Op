# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
import grid2op
import unittest
import warnings
import pdb


class Issue494Tester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_idf_2023", test=True, _add_to_name=type(self).__name__)
        self.env.seed(0)
        self.env.set_id(0)
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_act_legal(self):
        obs = self.env.reset()
        for sub_id in [24, 98, 100]:
            obs, reward, done, info = self.env.step(self.env.action_space({"set_bus": {"substations_id": [(sub_id, np.ones(type(obs).sub_info[sub_id], dtype=int))]}}))
            assert not info["exception"], f'for {sub_id=} {info["exception"]} vs []'


if __name__ == '__main__':
    unittest.main()
