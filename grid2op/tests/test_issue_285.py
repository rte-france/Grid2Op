# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings
import re

import grid2op
from grid2op.Chronics import MultifolderWithCache
from grid2op.Runner import Runner


class Issue285Tester(unittest.TestCase):
    def setUp(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make(
                "rte_case5_example", test=True, chronics_class=MultifolderWithCache,
                _add_to_name=type(self).__name__
            )
        self.env.chronics_handler.real_data.set_filter(
            lambda x: re.match(".*0$", x) is not None
        )
        self.env.chronics_handler.real_data.reset()

    def tearDown(self):
        self.env.close()

    def test_runner_ok(self):
        """test that the runner works ok"""
        runner = Runner(**self.env.get_params_for_runner())
        res = runner.run(nb_episode=2, max_iter=5)
        assert res[0][1] != res[1][1]
        assert res[0][1] == "00"
        assert res[1][1] == "10"


if __name__ == "__main__":
    unittest.main()
