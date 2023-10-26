# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt and https://github.com/rte-france/Grid2Op/pull/319
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
import unittest
import warnings
import numpy as np
import pdb

class Issue380Tester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__)
        self.env.seed(0)
        self.env.set_id(0)
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_limit_cs_margin(self):
        obs = self.env.reset()
        act = self.env.action_space({"curtail": [(2, 0.), (3, 0.), (4, 0.)]})
        # this action "cut" 36.4 MW, I have a margin of 30
        
        # margin is enough, i just cut the 6.4
        res, res_add_curtailed, res_add_storage = act.limit_curtail_storage(obs, 0, do_copy=True)
        assert np.sum(np.abs(res_add_storage)) == 0.
        assert abs(np.sum(res_add_curtailed * obs.gen_pmax) - 6.4) <= 1e-4
        # margin is relatively low, i just got 5. MW, so i "cut" 31.4
        res, res_add_curtailed, res_add_storage = act.limit_curtail_storage(obs, 25, do_copy=True)
        assert np.sum(np.abs(res_add_storage)) == 0.
        assert abs(np.sum(res_add_curtailed * obs.gen_pmax) - 31.4) <= 1e-4
        # margin is extremely low, i cut almost everything
        res, res_add_curtailed, res_add_storage = act.limit_curtail_storage(obs, 29.8, do_copy=True)
        assert np.sum(np.abs(res_add_storage)) == 0.
        assert abs(np.sum(res_add_curtailed * obs.gen_pmax) - 36.2) <= 1e-4
        # margin is extremely low, i cut almost everything
        res, res_add_curtailed, res_add_storage = act.limit_curtail_storage(obs, 29.89, do_copy=True, _tol_equal=0.1)
        assert np.sum(np.abs(res_add_storage)) == 0.
        assert abs(np.sum(res_add_curtailed * obs.gen_pmax) - 36.29) <= 1e-4
        # margin is extremely low, i cut everything (due to precision _tol_equal)
        res, res_add_curtailed, res_add_storage = act.limit_curtail_storage(obs, 29.91, do_copy=True, _tol_equal=0.1)
        assert np.sum(np.abs(res_add_storage)) == 0.
        assert abs(np.sum(res_add_curtailed * obs.gen_pmax) - 36.4) <= 1e-4
        # margin is null, i cut everything
        res, res_add_curtailed, res_add_storage = act.limit_curtail_storage(obs, 30., do_copy=True, _tol_equal=0.1)
        assert np.sum(np.abs(res_add_storage)) == 0.
        assert abs(np.sum(res_add_curtailed * obs.gen_pmax) - 36.4) <= 1e-4
    

        # Change of the _tol_equal
        # margin is extremely low, i cut almost everything
        res, res_add_curtailed, res_add_storage = act.limit_curtail_storage(obs, 29.989, do_copy=True, _tol_equal=0.01)
        assert np.sum(np.abs(res_add_storage)) == 0.
        assert abs(np.sum(res_add_curtailed * obs.gen_pmax) - 36.389) <= 1e-4
        # margin is extremely low, i cut everything (due to precision _tol_equal)
        res, res_add_curtailed, res_add_storage = act.limit_curtail_storage(obs, 29.991, do_copy=True, _tol_equal=0.01)
        assert np.sum(np.abs(res_add_storage)) == 0.
        assert abs(np.sum(res_add_curtailed * obs.gen_pmax) - 36.4) <= 1e-4
        
        
if __name__ == "__main__":
    unittest.main()
