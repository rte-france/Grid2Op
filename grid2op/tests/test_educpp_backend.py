# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import warnings
import unittest

import grid2op
from grid2op.Backend.educPandaPowerBackend import EducPandaPowerBackend
import pdb

class EducPPTester(unittest.TestCase):
    """mainly to test that a backend that do not support shunts_data can
    be loaded properly.
    """
    def test_make(self):
        for env_name in grid2op.list_available_test_env():
            if (env_name == "l2rpn_icaps_2021" or 
                env_name == "l2rpn_neurips_2020_track1" or
                env_name == "l2rpn_wcci_2020" or
                env_name == "l2rpn_wcci_2022_dev" or
                env_name == "l2rpn_wcci_2022"
            ):
                # does not work because of generators name
                # in the redispatching data
                # or name in the powerlines (when maintenance)
                continue
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                env = grid2op.make(env_name,
                                   test=True,
                                   backend=EducPandaPowerBackend(),
                                   _add_to_name=type(self).__name__+"educppbk")
                assert type(env).n_shunt is None, f"error for {env_name}"
                assert not type(env).shunts_data_available, f"error for {env_name}"
            env.close()
            
            
if __name__ == "__main__":
    unittest.main()
