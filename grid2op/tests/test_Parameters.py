# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import unittest
import tempfile
import warnings
import json

from grid2op.Parameters import Parameters

import warnings
warnings.simplefilter("error")


class TestParameters(unittest.TestCase):
    def test_default_builds(self):
        p = Parameters()

    def test_to_dict(self):
        p = Parameters()
        p_dict = p.to_dict()
        assert isinstance(p_dict, dict)

    def test_init_from_dict(self):
        p = Parameters()
        p_dict = p.to_dict()

        p_dict["NB_TIMESTEP_OVERFLOW_ALLOWED"] = 42

        p.init_from_dict(p_dict)
        assert p.NB_TIMESTEP_OVERFLOW_ALLOWED == 42

    def test_from_json(self):
        p = Parameters()
        p_dict = p.to_dict()
        p_dict["NB_TIMESTEP_OVERFLOW_ALLOWED"] = 42
        p_json = json.dumps(p_dict, indent=2)
        tf = tempfile.NamedTemporaryFile(delete=False)
        tf.write(bytes(p_json, "utf-8"))
        tf.close()

        p2 = Parameters.from_json(tf.name)
        assert p2.NB_TIMESTEP_OVERFLOW_ALLOWED == 42

        os.remove(tf.name)

    def test_init_from_json(self):
        p = Parameters()
        p_dict = p.to_dict()
        p_dict["NB_TIMESTEP_OVERFLOW_ALLOWED"] = 42
        p_json = json.dumps(p_dict, indent=2)
        tf = tempfile.NamedTemporaryFile(delete=False)
        tf.write(bytes(p_json, "utf-8"))
        tf.close()

        p.init_from_json(tf.name)
        assert p.NB_TIMESTEP_OVERFLOW_ALLOWED == 42

        os.remove(tf.name)
        
if __name__ == "__main__":
    unittest.main()
