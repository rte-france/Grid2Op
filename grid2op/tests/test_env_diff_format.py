# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
from typing import Optional, Union
import unittest

from grid2op.tests.helper_path_test import PATH_DATA_TEST
from grid2op.tests.aaa_test_backend_interface import AAATestBackendAPI
from grid2op.Backend import PandaPowerBackend
FILE_FORMAT = "json_custom"


class BackendDiffFormatTester(PandaPowerBackend):
    def __init__(self,
                 detailed_infos_for_cascading_failures: bool = False,
                 lightsim2grid: bool = False,
                 dist_slack: bool = False,
                 max_iter: int = 10,
                 can_be_copied: bool = True,
                 with_numba: bool = False):
        super().__init__(detailed_infos_for_cascading_failures, lightsim2grid, dist_slack, max_iter, can_be_copied, with_numba)
        self.supported_grid_format = ("json_custom", )
    
    # def load_grid(self, path: Union[os.PathLike, str], filename: Union[os.PathLike, str, None] = None) -> None:
    #     full_path = self.make_complete_path(path, filename)
    #     full_path = full_path.rstrip("_custom")
    #     return super().load_grid(full_path)


class TestBackendAPI_BackendDiffFormatTester(AAATestBackendAPI, unittest.TestCase):   
    def get_path(self):
        return os.path.join(PATH_DATA_TEST, "5bus_fake_grid_format")
    
    def get_casefile(self):
        return "grid.json_custom"   # or `grid.xml` or any other format

    def make_backend(self, detailed_infos_for_cascading_failures=False):
        # the function that will create your backend
        # do not put "PandaPowerBackend" of course, but the class you coded as a backend !
        backend = BackendDiffFormatTester(detailed_infos_for_cascading_failures=detailed_infos_for_cascading_failures)
        assert FILE_FORMAT in backend.supported_grid_format, f"your backend does not recognize the '{FILE_FORMAT}' extension, grid2op will not work"
        return backend

    def setUp(self):
        self.tests_skipped = ("test_01load_grid", "test_22_islanded_grid_make_divergence")
        return super().setUp()
if __name__ == "__main__":
    unittest.main()
