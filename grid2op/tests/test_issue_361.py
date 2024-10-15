# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt and https://github.com/Grid2Op/grid2op/pull/319
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
import unittest
import warnings

from grid2op.Runner import Runner


class Issue361Tester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__)
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_copy(self):
        
        env_cpy = self.env.copy()
        assert hasattr(self.env.backend, "_my_kwargs")
        assert hasattr(env_cpy.backend, "_my_kwargs")
        env_cpy.close()
    
    def test_runner(self):
        env_cpy = self.env.copy()
        runner = Runner(**env_cpy.get_params_for_runner())
        res = runner.run(nb_episode=1, max_iter=10)
        # this crashed above
        env_cpy.close()

if __name__ == "__main__":
    unittest.main()
