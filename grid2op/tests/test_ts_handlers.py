# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import pdb
import time
import warnings

from grid2op.tests.helper_path_test import *

import grid2op
from grid2op.Chronics.time_series_from_handlers import CSVHandler, FromHandlers

import warnings

# TODO check when there is also redispatching


class TestCSVHandlerEnv(HelperTests):
    """test the env part of the storage functionality"""
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env1 = grid2op.make("l2rpn_case14_sandbox")  # regular env
            self.env2 = grid2op.make("l2rpn_case14_sandbox",
                                     data_feeding_kwargs={"gridvalueClass": FromHandlers,
                                                          "gen_p_handler": CSVHandler("prod_p"),
                                                          "load_p_handler": CSVHandler("load_p"),
                                                          "gen_v_handler": CSVHandler("prod_v"),
                                                          "load_q_handler": CSVHandler("load_q"),
                                                          },
                                     _add_to_name="test")  # regular env
            
        for env in [self.env1, self.env2]:
            env.set_id(0)
            env.seed(0)
            env.reset()
            
        return super().setUp()

    def _run_then_compare(self):
        for k in range(10):
            obs1, reward1, done1, info1 = self.env1.step(self.env1.action_space())
            obs2, reward2, done2, info2 = self.env2.step(self.env2.action_space())
            assert done2 == done1, f"error at iteration {k}"
            assert reward1 == reward2, f"error at iteration {k}"
            for attr_nm in ["load_p", "load_q", "gen_p", "gen_v", "rho"]:
                assert np.all(getattr(obs1, attr_nm) == getattr(obs2, attr_nm)), f"error for {attr_nm} at iteration {k}"
                
    def test_step_equal(self):
        self._run_then_compare()
        self.env1.reset()
        self.env2.reset()
        self._run_then_compare()
        self.env1.reset()
        self.env2.reset()
        self._run_then_compare()
    
    
# TODO:
# test when "names_chronics_to_backend"
# test runner
# test env copy
# test when max_iter `env.set_max_iter`
# test when "set_chunk"
# test with forecasts
# test with maintenance

if __name__ == "__main__":
    unittest.main()

    