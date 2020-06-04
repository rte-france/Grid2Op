# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import tempfile

from grid2op.tests.helper_path_test import *
from grid2op.Environment import MultiMixEnvironment
from grid2op.Observation import CompleteObservation
from grid2op.Exceptions import EnvError

class TestMultiMixEnvironment(unittest.TestCase):
    def test_creation(self):        
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX)
        assert mme.current_obs is not None
        assert mme.current_env is not None

    def test_create_fail(self):
        with self.assertRaises(EnvError):
            mme = MultiMixEnvironment("/tmp/error")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(EnvError):
                mme = MultiMixEnvironment(tmpdir)

    def test_reset(self):
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX)
        mme.reset()
        assert mme.current_obs is not None
        assert mme.current_env is not None
        
    def test_seeding(self):
        mme1 = MultiMixEnvironment(PATH_DATA_MULTIMIX)
        seeds_1 = mme1.seed(2)
        mme1.close()
        
        mme2 = MultiMixEnvironment(PATH_DATA_MULTIMIX)
        seeds_2 = mme2.seed(42)
        mme2.close()
        
        mme3 = MultiMixEnvironment(PATH_DATA_MULTIMIX)
        seeds_3 = mme3.seed(2)
        mme3.close()

        assert np.all(seeds_1 == seeds_3)
        assert np.any(seeds_1 != seeds_2)

    def test_step(self):
        mme = MultiMixEnvironment(PATH_DATA_MULTIMIX)
        dn = mme.current_env.action_space()

        obs, r, done, info  = mme.step(dn)
        assert obs is not None
        assert r is not None
        assert isinstance(info, dict)
        assert done is not True
        
if __name__ == "__main__":
    unittest.main()
