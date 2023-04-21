# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
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


class TestForecastFromArrays(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox")
        return super().setUp()
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()

    def test_basic_behaviour(self):
        obs = self.env.reset()
        
        nb_ts = 15
        load_p_forecasted = np.tile(obs.load_p, nb_ts).reshape(nb_ts, -1)
        load_q_forecasted = np.tile(obs.load_p, nb_ts).reshape(nb_ts, -1)
        gen_p_forecasted = np.tile(obs.gen_p, nb_ts).reshape(nb_ts, -1)
        gen_v_forecasted = np.tile(obs.gen_v, nb_ts).reshape(nb_ts, -1)
            
        forcast_env = obs.get_env_from_external_forecasts(load_p_forecasted,
                                                          load_q_forecasted,
                                                          gen_p_forecasted,
                                                          gen_v_forecasted)
        sim_obs = forcast_env.reset()
        assert sim_obs.max_iter == nb_ts + 1
        
if __name__ == "__main__":
    unittest.main()
