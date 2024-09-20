# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from aaa_test_backend_interface import AAATestBackendAPI
from grid2op.Backend import PandaPowerBackend
from grid2op.Converter import BackendConverter


BKclass1 = PandaPowerBackend
BKclass2 = PandaPowerBackend

class TestPandapowerBkInterface(AAATestBackendAPI, unittest.TestCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        return PandaPowerBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)


class TestPandapowerCpyBkInterface(AAATestBackendAPI, unittest.TestCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        tmp = PandaPowerBackend(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)
        return tmp.copy()
    
    
class TestConverterBkInterface(AAATestBackendAPI, unittest.TestCase):
    def make_backend(self, detailed_infos_for_cascading_failures=False):
        backend = BackendConverter(
            source_backend_class=BKclass1,
            target_backend_class=BKclass2,
            target_backend_grid_path=None,
            detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures,
        )
        return backend


if __name__ == "__main__":
    unittest.main()
