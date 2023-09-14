# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings

import grid2op
from grid2op.Backend import PandaPowerBackend
from grid2op.Space import DetailedTopoDescription


class _PPBkForTestDetTopo(PandaPowerBackend):
    def load_grid(self, path=None, filename=None):
        super().load_grid(path, filename)
        self.detailed_topo_desc = DetailedTopoDescription.from_init_grid(self)


class DetailedTopoTester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
            "l2rpn_case14_sandbox",
            test=True,
            backend=_PPBkForTestDetTopo(),
            _add_to_name="_BaseTestNames",
        )
        return super().setUp()
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_init_ok(self):
        obs = self.env.reset()
        assert type(obs).detailed_topo_desc is not None
        assert  type(obs).detailed_topo_desc.load_to_busbar_id == [
            (1, 15), (2, 16), (3, 17), (4, 18), (5, 19), (8, 22), (9, 23), (10, 24), (11, 25), (12, 26), (13, 27)
        ]
        assert type(obs).detailed_topo_desc.gen_to_busbar_id == [(1, 15), (2, 16), (5, 19), (5, 19), (7, 21), (0, 14)]

    # TODO detailed topo
        
 
if __name__ == "__main__":
    unittest.main()
   