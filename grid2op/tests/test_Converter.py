# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import time
import warnings
import pandapower as pp

from grid2op.tests.helper_path_test import *

import grid2op
from grid2op.MakeEnv import make
from grid2op.Parameters import Parameters
from grid2op.Converter import ConnectivityConverter
import pdb


class TestConverter(HelperTests):
    def setUp(self):
        """
        The case file is a representation of the case14 as found in the ieee14 powergrid.
        :return:
        """
        param = Parameters()
        param.init_from_dict({"NO_OVERFLOW_DISCONNECTION": True})
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = make("rte_case14_redisp", test=True, param=param)

    def tearDown(self):
        self.env.close()

    def test_ConnectivityConverter(self):
        converter = ConnectivityConverter(self.env.action_space)
        converter.init_converter()
        assert np.all(converter.subs_ids == np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3,
                                                      3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5,
                                                      5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]))
        assert len(converter.obj_type) == converter.n

        pdb.set_trace()
        assert len(set(converter.obj_type)) == converter.n