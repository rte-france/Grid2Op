# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import os
import cProfile
import unittest
import numpy as np
from grid2op.tests.helper_path_test import *

from grid2op.tests.test_compute_switch_pos import AuxTestComputeSwitchPos


CPROF = False


dir_out  = "test_topo_ampl"
dir_out  = "test_topo_ampl2"


class TestComputeSwitchPosExt(unittest.TestCase):
    def debug_test_one_case(self):
        # el = 25
        # el = 35
        # el = 22
        # el = 37  # really long, does not work
        # el = 42
        # el = 65
        # el = 57
        # el = 148
        el = 201 # bug in the data
        el = 109
        el = 110
        el = 38
        el = 37
        tmp = AuxTestComputeSwitchPos._aux_read_case(f"{el}", dir=".", nm_dir=dir_out)
        if tmp is None:
            print(f"error for {el}: some elements (or busbar) not controlled by any switch")
            return
        dtd, target, result = tmp
        print(target)
        AuxTestComputeSwitchPos._aux_test_switch_topo(dtd, target, result)
        print("test done")
        dtd._aux_compute_busbars_sections()
        if CPROF:
            cp = cProfile.Profile()
            cp.enable()
        switches = dtd.compute_switches_position(target)
        if CPROF:
            cp.disable()
            nm_f, ext = os.path.splitext(__file__)
            nm_out = f"{nm_f}_{el}.prof"
            cp.dump_stats(nm_out)
            print("You can view profiling results with:\n\tsnakeviz {}".format(nm_out))
        AuxTestComputeSwitchPos._aux_test_switch_topo(dtd, target, switches)
        
    def test_cases(self):
        # for el in range(2, 22):
        for el in range(1, 222):
            print(f"test {el}")
            if el == 37:
                # too long (all night)
                continue
            elif el == 201:
                # error when reading the case
                continue
            # tmp = self._aux_read_case(f"{el}", dir=".", nm_dir=dir_out)
            tmp = AuxTestComputeSwitchPos._aux_read_case(f"{el}")
            if tmp is None:
                raise RuntimeError(f"Impossible to read case {el}")
            dtd, target, result = tmp
            AuxTestComputeSwitchPos._aux_test_switch_topo(dtd, target, result)
            dtd._aux_compute_busbars_sections()
            switches = dtd.compute_switches_position(target)
            AuxTestComputeSwitchPos._aux_test_switch_topo(dtd, target, switches)
    
    
if __name__ == "__main__":
    unittest.main()
    