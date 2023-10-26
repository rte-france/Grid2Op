# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt and https://github.com/rte-france/Grid2Op/pull/319
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
import unittest
import warnings


class Issue364Tester(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_wcci_2022", test=True, _add_to_name=type(self).__name__)
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
    
    def test_copy(self):
        act = self.env.action_space({"curtail": [(1, 0.)]})
        dict_ = act.impact_on_objects()
        assert act._modif_curtailment
        assert dict_["curtailment"]["changed"]
        assert not dict_["storage"]["changed"]

if __name__ == "__main__":
    unittest.main()
