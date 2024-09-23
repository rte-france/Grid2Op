# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest

from grid2op.Backend import PandaPowerBackend

class PandaPowerNoShunt(PandaPowerBackend):
    shunts_data_available = False
    
from grid2op._create_test_suite import create_test_suite
from grid2op.tests.aaa_test_backend_interface import AAATestBackendAPI
class TestBackendAPI_PPNoShuntTester(AAATestBackendAPI, unittest.TestCase):        
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return  PandaPowerNoShunt(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)
    
    
# and run it with `python -m unittest gridcal_backend_tests.py`
if __name__ == "__main__":
    unittest.main()