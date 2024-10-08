# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from _aux_test_gym_compat import (AuxilliaryForTest,
                                  _AuxTestGymCompatModule,
                                  _AuxTestBoxGymObsSpace,
                                  _AuxTestBoxGymActSpace,
                                  _AuxTestMultiDiscreteGymActSpace,
                                  _AuxTestDiscreteGymActSpace,
                                  _AuxTestAllGymActSpaceWithAlarm,
                                  _AuxTestGOObsInRange)
import unittest


class TestGymCompatModule(_AuxTestGymCompatModule, AuxilliaryForTest, unittest.TestCase):
    pass

class TestBoxGymObsSpace(_AuxTestBoxGymObsSpace, AuxilliaryForTest, unittest.TestCase):
    pass

class TestBoxGymActSpace(_AuxTestBoxGymActSpace, AuxilliaryForTest, unittest.TestCase):
    pass

class TestMultiDiscreteGymActSpace(_AuxTestMultiDiscreteGymActSpace, AuxilliaryForTest, unittest.TestCase):
    pass

class TestDiscreteGymActSpace(_AuxTestDiscreteGymActSpace, AuxilliaryForTest, unittest.TestCase):
    pass

class TestAllGymActSpaceWithAlarm(_AuxTestAllGymActSpaceWithAlarm, AuxilliaryForTest, unittest.TestCase):
    pass

class TestGOObsInRange(_AuxTestGOObsInRange, AuxilliaryForTest, unittest.TestCase):
    pass

if __name__ == "__main__":
    unittest.main()
