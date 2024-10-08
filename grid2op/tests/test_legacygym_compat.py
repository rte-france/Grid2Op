# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from _aux_test_gym_compat import (GYM_AVAILABLE,
                                  _AuxTestGymCompatModule,
                                  _AuxTestBoxGymObsSpace,
                                  _AuxTestBoxGymActSpace,
                                  _AuxTestMultiDiscreteGymActSpace,
                                  _AuxTestDiscreteGymActSpace,
                                  _AuxTestAllGymActSpaceWithAlarm,
                                  _AuxTestGOObsInRange)
import unittest

class AuxilliaryForTestLegacyGym:
    def _aux_GymEnv_cls(self):
        from grid2op.gym_compat import GymEnv_Modern
        return GymEnv_Modern
    
    def _aux_ContinuousToDiscreteConverter_cls(self):
        from grid2op.gym_compat import ContinuousToDiscreteConverterLegacyGym
        return ContinuousToDiscreteConverterLegacyGym
    
    def _aux_ScalerAttrConverter_cls(self):
        from grid2op.gym_compat import ScalerAttrConverterLegacyGym
        return ScalerAttrConverterLegacyGym
    
    def _aux_MultiToTupleConverter_cls(self):
        from grid2op.gym_compat import MultiToTupleConverterLegacyGym
        return MultiToTupleConverterLegacyGym
    
    def _aux_BoxGymObsSpace_cls(self):
        from grid2op.gym_compat import BoxLegacyGymObsSpace
        return BoxLegacyGymObsSpace
    
    def _aux_BoxGymActSpace_cls(self):
        from grid2op.gym_compat import BoxLegacyGymActSpace
        return BoxLegacyGymActSpace
    
    def _aux_MultiDiscreteActSpace_cls(self):
        from grid2op.gym_compat import MultiDiscreteActSpaceLegacyGym
        return MultiDiscreteActSpaceLegacyGym
    
    def _aux_DiscreteActSpace_cls(self):
        from grid2op.gym_compat import DiscreteActSpaceLegacyGym
        return DiscreteActSpaceLegacyGym
    
    def _aux_Box_cls(self):
        if GYM_AVAILABLE:
            from gym.spaces import Box
            return Box
    
    def _aux_MultiDiscrete_cls(self):
        if GYM_AVAILABLE:
            from gym.spaces import MultiDiscrete
            return MultiDiscrete
    
    def _aux_Discrete_cls(self):
        if GYM_AVAILABLE:
            from gym.spaces import Discrete
            return Discrete
        
    def _aux_Tuple_cls(self):
        if GYM_AVAILABLE:
            from gym.spaces import Tuple
            return Tuple
        
    def _aux_Dict_cls(self):
        if GYM_AVAILABLE:
            from gym.spaces import Dict
            return Dict
            
    def _skip_if_no_gym(self):
        if not GYM_AVAILABLE:
            self.skipTest("Gym is not available")
            

class TestLegacyGymCompatModule(_AuxTestGymCompatModule, AuxilliaryForTestLegacyGym, unittest.TestCase):
    pass

class TestBoxLegacyGymObsSpace(_AuxTestBoxGymObsSpace, AuxilliaryForTestLegacyGym, unittest.TestCase):
    pass

class TestBoxLegacyGymActSpace(_AuxTestBoxGymActSpace, AuxilliaryForTestLegacyGym, unittest.TestCase):
    pass

class TestMultiDiscreteLegacyGymActSpace(_AuxTestMultiDiscreteGymActSpace, AuxilliaryForTestLegacyGym, unittest.TestCase):
    pass

class TestDiscreteLegacyGymActSpace(_AuxTestDiscreteGymActSpace, AuxilliaryForTestLegacyGym, unittest.TestCase):
    pass

class TestAllLegacyGymActSpaceWithAlarm(_AuxTestAllGymActSpaceWithAlarm, AuxilliaryForTestLegacyGym, unittest.TestCase):
    pass

class TestGOObsInRangeLegacyGym(_AuxTestGOObsInRange, AuxilliaryForTestLegacyGym, unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
