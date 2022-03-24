# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings
import numpy as np
import networkx

import grid2op

import pdb


class TestNetworkXGraph(unittest.TestCase):
    """this class test the networkx representation of an observation."""

    def setUp(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_neurips_2020_track1", test=True)
        self.tol = 1e-5

    def test_kirchhoff(self):
        """
        test kirchhoff law

        in case of parallel lines
        """
        obs = self.env.reset()
        graph = obs.as_networkx()
        assert isinstance(graph, networkx.Graph), "graph should be a networkx object"
        ps = np.array([graph.nodes[el]["p"] for el in graph.nodes])
        qs = np.array([graph.nodes[el]["q"] for el in graph.nodes])

        p_out = np.zeros(ps.shape[0])
        q_out = np.zeros(ps.shape[0])
        for or_, ex_ in graph.edges:
            me = graph.edges[(or_, ex_)]
            p_out[or_] += me["p_or"]
            q_out[or_] += me["q_or"]
            p_out[ex_] += me["p_ex"]
            q_out[ex_] += me["q_ex"]

        assert np.max(np.abs(ps - p_out)) <= self.tol, "error for active flow"
        assert np.max(np.abs(qs - q_out)) <= self.tol, "error for reactive flow"


if __name__ == "__main__":
    unittest.main()
