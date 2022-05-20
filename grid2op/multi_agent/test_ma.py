# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings
from grid2op import make
from grid2op.multi_agent.multiAgentEnv import MultiAgentEnv
import pdb
import numpy as np

class MATester(unittest.TestCase):
    def setUp(self) -> None:
        
        self.env = make("l2rpn_case14_sandbox", test = True)

        self.action_domains = {
            'agent_0' : [0,1,2,3, 4],
            'agent_1' : [5,6,7,8,9,10,11,12,13]
        }
        self.observation_domains = {
            'agent_0' : self.action_domains['agent_1'],
            'agent_1' : self.action_domains['agent_0']
        }
        # run redispatch agent on one scenario for 100 timesteps
        self.ma_env = MultiAgentEnv(self.env, self.observation_domains, self.action_domains)
        return super().setUp()
    
    def test_build_subgrids_action_domains(self):
        """Tests that the action_domains are correctly defined 
            in MultiAgentEnv._build_subgrids method
        """
        
        
        assert self.ma_env._action_domains['agent_0']['sub_id'] == self.action_domains['agent_0']
        assert self.ma_env._action_domains['agent_1']['sub_id'] == self.action_domains['agent_1']
        
        assert self.ma_env._action_domains['agent_0']['mask_load'] == [True,  True,  True,  True, False, False, False, False, False, False, False]
        assert self.ma_env._action_domains['agent_1']['mask_load'] == np.invert([True,  True,  True,  True, False, False, False, False, False, False, False])
        
        assert self.ma_env._action_domains['agent_0']['mask_gen'] == [ True,  True, False, False, False,  True]
        assert self.ma_env._action_domains['agent_1']['mask_gen'] == np.invert([ True,  True, False, False, False,  True])
        
        assert self.ma_env._action_domains['agent_0']['mask_storage'] == []
        assert self.ma_env._action_domains['agent_1']['mask_storage'] == []
        
        assert self.ma_env._action_domains['agent_0']['mask_line_ex'] == [ True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False,False, False]
        assert self.ma_env._action_domains['agent_1']['mask_line_ex'] == np.invert([ True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False,False, False])
        
        assert self.ma_env._action_domains['agent_0']['mask_line_or'] == [ True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False,  True,  True,  True, False, False]
        assert self.ma_env._action_domains['agent_1']['mask_line_or'] == np.invert([ True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False,  True,  True,  True, False, False])
        
        assert self.ma_env._action_domains['agent_0']['mask_shunt'] == [False]
        assert self.ma_env._action_domains['agent_1']['mask_shunt'] == [True]
        
    def test_build_subgrids_observation_domains(self):
        """Tests that the observation_domains are correctly defined 
            in MultiAgentEnv._build_subgrids method
        """
        
        
        assert self.ma_env._action_domains['agent_1']['sub_id'] == self.action_domains['agent_0']
        assert self.ma_env._action_domains['agent_0']['sub_id'] == self.action_domains['agent_1']
        
        assert self.ma_env._action_domains['agent_1']['mask_load'] == [True,  True,  True,  True, False, False, False, False, False, False, False]
        assert self.ma_env._action_domains['agent_0']['mask_load'] == np.invert([True,  True,  True,  True, False, False, False, False, False, False, False])
        
        assert self.ma_env._action_domains['agent_1']['mask_gen'] == [ True,  True, False, False, False,  True]
        assert self.ma_env._action_domains['agent_0']['mask_gen'] == np.invert([ True,  True, False, False, False,  True])
        
        assert self.ma_env._action_domains['agent_1']['mask_storage'] == []
        assert self.ma_env._action_domains['agent_0']['mask_storage'] == []
        
        assert self.ma_env._action_domains['agent_1']['mask_line_ex'] == [ True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False,False, False]
        assert self.ma_env._action_domains['agent_0']['mask_line_ex'] == np.invert([ True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False,False, False])
        
        assert self.ma_env._action_domains['agent_1']['mask_line_or'] == [True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False,  True,  True,  True, False, False]
        assert self.ma_env._action_domains['agent_0']['mask_line_or'] == np.invert([ True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False,  True,  True,  True, False, False])
        
        assert self.ma_env._action_domains['agent_1']['mask_shunt'] == [False]
        assert self.ma_env._action_domains['agent_0']['mask_shunt'] == [True]
    


if __name__ == "__main__":
    unittest.main()
