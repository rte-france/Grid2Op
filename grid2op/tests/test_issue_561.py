# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
import grid2op
from grid2op.Backend import PandaPowerBackend
import warnings
import unittest


class PandaPowerNoShunt_Test(PandaPowerBackend):
    shunts_data_available = False
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shunts_data_available = False
        
    def _init_private_attrs(self) -> None:
        super()._init_private_attrs()
        self.shunts_data_available = False


class Issue561Tester(unittest.TestCase):
    def test_update_from_obs(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox",
                               test=True,
                               backend=PandaPowerNoShunt_Test(),
                               _add_to_name=type(self).__name__)
        obs_init = env.reset()
        assert not type(obs_init).shunts_data_available
        assert not type(env.backend).shunts_data_available
        backend = env.backend.copy()
        backend1 = env.backend.copy()
        obs, *_ = env.step(env.action_space())
        obs.load_p[:] += 1.  # to make sure everything changes
        backend.update_from_obs(obs)
        assert np.all(backend._grid.load["p_mw"] != backend1._grid.load["p_mw"])


if __name__ == "__main__":
    unittest.main()
