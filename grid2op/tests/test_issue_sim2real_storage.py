# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import unittest

import grid2op
from grid2op.tests.helper_path_test import *
from grid2op.Backend import PandaPowerBackend
from lightsim2grid import LightSimBackend


class _AuxTestSim2realStorage:  
    def get_name(self):
        return "educ_case14_storage_diffgrid"
    
    def setUp(self) -> None:
        # print(f"\n\n\nfor {type(self.get_backend())}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # self.env = grid2op.make(os.path.join(PATH_DATA_TEST, "educ_case14_storage_diffgrid"),
            self.env = grid2op.make(os.path.join(PATH_DATA_TEST, self.get_name()),
                                    test=True,
                                    backend=self.get_backend())
        self.env.seed(0)
        self.env.set_id(0)
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_reset(self):
        obs = self.env.reset()
        assert obs.storage_Emax is not None
        assert obs.storage_max_p_prod is not None
        assert obs.n_storage == 2
    
    def test_simulate(self):
        obs = self.env.reset()
        sim_obs, *_ = obs.simulate(self.env.action_space())
        assert sim_obs.storage_Emax is not None
        assert sim_obs.storage_max_p_prod is not None
        assert obs.n_storage == 2


class TestSim2realStorageLS(_AuxTestSim2realStorage, unittest.TestCase):  
    def get_backend(self):
        return LightSimBackend()
    
    
class TestSim2realStoragePP(_AuxTestSim2realStorage, unittest.TestCase):  
    def get_backend(self):
        return PandaPowerBackend()


class TestSim2realStorageLSDiffObs(_AuxTestSim2realStorage, unittest.TestCase):  
    """add this test for https://github.com/rte-france/Grid2Op/issues/518"""
    def get_backend(self):
        return LightSimBackend()
    
    def get_name(self):
        return "educ_case14_storage_diffgrid_diff_obs"
    
    
class TestSim2realStoragePPDiffObs(_AuxTestSim2realStorage, unittest.TestCase):  
    """add this test for https://github.com/rte-france/Grid2Op/issues/518"""
    def get_backend(self):
        return PandaPowerBackend()
    
    def get_name(self):
        return "educ_case14_storage_diffgrid_diff_obs"
    
    
if __name__ == '__main__':
    unittest.main()
