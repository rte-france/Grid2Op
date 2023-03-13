# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
from grid2op.gym_compat import GymEnv
import unittest
import warnings

class Issue418Tester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # this needs to be tested with pandapower backend
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True)
        self.env.seed(0)
        self.env.set_id(0)

    def test_can_make(self):
        obs = self.env.reset()
        complete_graph = obs.get_elements_graph()
        cls = type(obs)
        assert len(complete_graph.nodes) == cls.n_sub + 2*cls.n_sub + cls.n_load + cls.n_gen + cls.n_line + cls.n_storage + cls.n_shunt

# TODO disconnected lines
# TODO change topo
# TODO with shunts
# TODO with storage units
# TODO basic prop: complete_graph.edges[54, obs.gen_to_subid[complete_graph.nodes[54]["id"]] + obs.n_sub]
# TODO kirchhoff
# TODO doc in the code with an example

if __name__ == "__main__":
    unittest.main()
