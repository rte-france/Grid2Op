# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
import warnings
import unittest
from grid2op.Chronics import GridStateFromFile

class Issue593Tester(unittest.TestCase):
    def test_issue_593(self):
        # parameters is read from the config file, 
        # it should be removed "automatically"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            env = grid2op.make("l2rpn_idf_2023",
                               test=True,
                               data_feeding_kwargs={"gridvalueClass": GridStateFromFile,
                                                    })
            
    def test_issue_593_should_break(self):
        # user did something wrong
        # there should be an error 
        with self.assertRaises(TypeError):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                env = grid2op.make("l2rpn_idf_2023",
                                   test=True,
                                   data_feeding_kwargs={"gridvalueClass": GridStateFromFile,
                                                     "h_forecast": [5]
                                                     })

if __name__ == "__main__":
    unittest.main()