# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings
import grid2op

import pdb


class BasicTestActSpace(unittest.TestCase):
    def test_is_legal_None(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__)

            is_legal, reason = env.action_space._is_legal(None, None)
            assert reason is None
            assert is_legal


if __name__ == "__main__":
    unittest.main()
