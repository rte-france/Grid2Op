# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
from grid2op.Backend import PandaPowerBackend
import warnings
import unittest


class PandaPowerNoShunt_Test550(PandaPowerBackend):
    shunts_data_available = False  # class attribute (only one used)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _init_private_attrs(self) -> None:
        super()._init_private_attrs()


class Issue550Tester(unittest.TestCase):
    def test_no_shunt(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox",
                               test=True,
                               backend=PandaPowerNoShunt_Test550(),
                               _add_to_name=type(self).__name__)
        obs_init = env.reset()
        assert not type(obs_init).shunts_data_available
        assert not type(env.backend).shunts_data_available
        
    def test_with_shunt(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox",
                               test=True,
                               backend=PandaPowerBackend(),
                               _add_to_name=type(self).__name__)
        obs_init = env.reset()
        assert type(obs_init).shunts_data_available
        assert type(env.backend).shunts_data_available


if __name__ == "__main__":
    unittest.main()
