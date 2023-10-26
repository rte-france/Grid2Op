# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import pdb
import warnings
import os
import grid2op
import numpy as np
from grid2op.Chronics import FromChronix2grid
import unittest
import pkg_resources
from lightsim2grid import LightSimBackend

DEV_DATA_FOLDER = pkg_resources.resource_filename("grid2op", "data")

class TestFromChronix2Grid(unittest.TestCase):
    def _aux_reset_env(self):
        self.env.seed(self.seed_)
        self.env.set_id(self.scen_nm)
        return self.env.reset()
    
    def setUp(self) -> None:
        self.seed_ = 0
        self.env_nm = "l2rpn_wcci_2022_dev"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(self.env_nm,
                                    test=True,
                                    backend=LightSimBackend(),
                                    chronics_class=FromChronix2grid,
                                    data_feeding_kwargs={"env_path": os.path.join(DEV_DATA_FOLDER, self.env_nm),  # os.path.join(grid2op.get_current_local_dir(), self.env_nm),
                                                         "with_maintenance": True,
                                                         "max_iter": 10,
                                                         "with_loss": False
                                                        },
                                    _add_to_name=type(self).__name__
                                    )
    
    
    def test_ok(self):
        """test it can be created"""
        assert self.env.chronics_handler.real_data._gen_p.shape == (12, 62)
        assert np.all(np.isfinite(self.env.chronics_handler.real_data._gen_p))
        
    def test_seed_setid(self):
        """test env.seed(...) and env.set_id(...)"""
        id_ref = '2525122259@2050-02-28'
        id_ref = '377638611@2050-02-28'
        # test tell_id
        sum_prod_ref = 42340.949878
        sum_prod_ref = 41784.477161
        self.env.seed(self.seed_)
        self.env.reset()
        id_ = self.env.chronics_handler.get_id()
        assert id_ == id_ref, f"wrong id {id_} instead of {id_ref}"
        assert abs(self.env.chronics_handler.real_data._gen_p.sum() - sum_prod_ref) <= 1e-4, f"{self.env.chronics_handler.real_data._gen_p.sum():.2f}"
        self.env.reset()
        # assert abs(self.env.chronics_handler.real_data._gen_p.sum() - 38160.833356999996) <= 1e-4
        assert abs(self.env.chronics_handler.real_data._gen_p.sum() - 37662.206248999995) <= 1e-4, f"{self.env.chronics_handler.real_data._gen_p.sum():.2f}"
        self.env.set_id(id_ref)
        self.env.reset()
        assert abs(self.env.chronics_handler.real_data._gen_p.sum() - sum_prod_ref) <= 1e-4, f"{self.env.chronics_handler.real_data._gen_p.sum():.2f}"
        
        # test seed
        self.env.seed(self.seed_)
        self.env.reset()
        assert abs(self.env.chronics_handler.real_data._gen_p.sum() - sum_prod_ref) <= 1e-4, f"{self.env.chronics_handler.real_data._gen_p.sum():.2f}"
        
    def test_episode(self):
        """test that the episode can go until the end"""
        self.env.seed(1)
        obs = self.env.reset()
        assert obs.current_step == 0
        assert obs.max_step == 10
        for i in range(obs.max_step - 1):
            obs, reward, done, info = self.env.step(self.env.action_space())
            assert not done, f"episode should not be over at iteration {i}"
        obs, reward, done, info = self.env.step(self.env.action_space())
        assert obs.current_step == 10
        assert obs.max_step == 10
        assert done, "episode should have terminated"
        obs = self.env.reset()
        assert obs.max_step == 10
        assert obs.current_step == 0
    
    def test_maintenance(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(self.env_nm,
                                    test=True,
                                    _add_to_name=type(self).__name__,
                                    chronics_class=FromChronix2grid,
                                    data_feeding_kwargs={"env_path": os.path.join(DEV_DATA_FOLDER, self.env_nm),
                                                         "with_maintenance": True,
                                                         "max_iter": 2 * 288,
                                                         "with_loss": False
                                                        }
                                    )
        self.env.seed(0)
        id_ref = '0@2050-08-08'
        self.env.set_id(id_ref)
        obs = self.env.reset()
        assert np.all(obs.time_next_maintenance[[43, 126]] == [107, 395])
        assert np.all(obs.time_next_maintenance[:43] == -1)
        assert np.all(obs.time_next_maintenance[127:] == -1)
        assert np.all(obs.time_next_maintenance[44:126] == -1)
        assert self.env.chronics_handler.real_data.maintenance is not None
        assert self.env.chronics_handler.real_data.maintenance.sum() == 192
        assert self.env.chronics_handler.real_data.maintenance_time is not None
        assert self.env.chronics_handler.real_data.maintenance_duration is not None
        
        
if __name__ == "__main__":
    unittest.main()