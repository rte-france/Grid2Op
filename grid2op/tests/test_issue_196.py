# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings
import numpy as np

import grid2op

from grid2op.gym_compat import GymEnv
from grid2op.gym_compat import ScalerAttrConverter


class Issue196Tester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_neurips_2020_track1", test=True, _add_to_name=type(self).__name__)
            env_gym = GymEnv(env)
            ob_space = env_gym.observation_space
            ob_space = ob_space.keep_only_attr(
                ["rho", "gen_p", "load_p", "topo_vect", "actual_dispatch"]
            )
            ob_space = ob_space.reencode_space(
                "actual_dispatch",
                ScalerAttrConverter(substract=0.0, divide=env.gen_pmax),
            )
            ob_space = ob_space.reencode_space(
                "gen_p", ScalerAttrConverter(substract=0.0, divide=env.gen_pmax)
            )
            ob_space = ob_space.reencode_space(
                "load_p", ScalerAttrConverter(substract=0.0, divide=-1.0)
            )
            env_gym.observation_space = ob_space
            self.env_gym = env_gym

    def test_issue_196_loadp(self):
        assert np.all(
            self.env_gym.observation_space["load_p"].low
            <= self.env_gym.observation_space["load_p"].high
        )

    def test_issue_196_genp(self):
        # not great test as it passes with the bug... but just in the case... cannot hurt
        obs, *_ = self.env_gym.reset()
        assert obs in self.env_gym.observation_space
        
    
if __name__ == "__main__":
    unittest.main()
