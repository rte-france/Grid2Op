# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings

import grid2op
from grid2op.Reward import L2RPNWCCI2022ScoreFun

import pdb


class WCCI2022Tester(unittest.TestCase):
    def setUp(self) -> None:
        self.seed = 0
        self.scen_id = 0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("educ_case14_storage", test=True, reward_class=L2RPNWCCI2022ScoreFun)
    
    def _aux_reset_env(self):    
        self.env.seed(self.seed)
        self.env.set_id(self.scen_id)
        obs = self.env.reset()
        return obs
    
    def test_storage_cost(self):
        score_fun = L2RPNWCCI2022ScoreFun()
        score_fun.initialize(self.env)
        th_val = 10. * 10. / 12.
        
        obs = self._aux_reset_env()
        act = self.env.action_space({"set_storage": [(0, -5.), (1, 5.)]})
        obs, reward, done, info = self.env.step(act)
        rew = score_fun(act, self.env, False, False, False, False)
        margin_cost =  score_fun._get_marginal_cost(self.env)
        assert margin_cost == 70.
        storage_cost = score_fun._get_storage_cost(self.env, margin_cost)
        assert abs(storage_cost - th_val) <= 1e-5  # (10 MWh )* (10 € / MW )* (1/12. step / h)
        gen_p = 1.0 * obs.gen_p
        
        _ = self._aux_reset_env()
        obs, reward_dn, done, info = self.env.step(self.env.action_space())
        gen_p_dn = 1.0 * obs.gen_p
        
        assert reward >= reward_dn
        assert abs(reward - (reward_dn + storage_cost + (gen_p.sum() - gen_p_dn.sum()) * margin_cost / 12. )) <= 1e-6
        

if __name__ == "__main__":
    unittest.main()        
