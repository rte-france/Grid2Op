# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.


from _aux_test_gym_compat import (GYMNASIUM_AVAILABLE,
                                  _AuxTestGymCompatModule,
                                  _AuxTestBoxGymObsSpace,
                                  _AuxTestBoxGymActSpace,
                                  _AuxTestMultiDiscreteGymActSpace,
                                  _AuxTestDiscreteGymActSpace,
                                  _AuxTestAllGymActSpaceWithAlarm,
                                  _AuxTestGOObsInRange)
import unittest

class AuxilliaryForTestGymnasium:
    def _aux_GymEnv_cls(self):
        from grid2op.gym_compat import GymnasiumEnv
        return GymnasiumEnv
    
    def _aux_ContinuousToDiscreteConverter_cls(self):
        from grid2op.gym_compat import ContinuousToDiscreteConverterGymnasium
        return ContinuousToDiscreteConverterGymnasium
    
    def _aux_ScalerAttrConverter_cls(self):
        from grid2op.gym_compat import ScalerAttrConverterGymnasium
        return ScalerAttrConverterGymnasium
    
    def _aux_MultiToTupleConverter_cls(self):
        from grid2op.gym_compat import MultiToTupleConverterGymnasium
        return MultiToTupleConverterGymnasium
    
    def _aux_BoxGymObsSpace_cls(self):
        from grid2op.gym_compat import BoxGymnasiumObsSpace
        return BoxGymnasiumObsSpace
    
    def _aux_BoxGymActSpace_cls(self):
        from grid2op.gym_compat import BoxGymnasiumActSpace
        return BoxGymnasiumActSpace
    
    def _aux_MultiDiscreteActSpace_cls(self):
        from grid2op.gym_compat import MultiDiscreteActSpaceGymnasium
        return MultiDiscreteActSpaceGymnasium
    
    def _aux_DiscreteActSpace_cls(self):
        from grid2op.gym_compat import DiscreteActSpaceGymnasium
        return DiscreteActSpaceGymnasium
    
    def _aux_Box_cls(self):
        if GYMNASIUM_AVAILABLE:
            from gymnasium.spaces import Box
            return Box
    
    def _aux_MultiDiscrete_cls(self):
        if GYMNASIUM_AVAILABLE:
            from gymnasium.spaces import MultiDiscrete
            return MultiDiscrete
    
    def _aux_Discrete_cls(self):
        if GYMNASIUM_AVAILABLE:
            from gymnasium.spaces import Discrete
            return Discrete
        
    def _aux_Tuple_cls(self):
        if GYMNASIUM_AVAILABLE:
            from gymnasium.spaces import Tuple
            return Tuple
        
    def _aux_Dict_cls(self):
        if GYMNASIUM_AVAILABLE:
            from gymnasium.spaces import Dict
            return Dict
            
    def _skip_if_no_gym(self):
        if not GYMNASIUM_AVAILABLE:
            self.skipTest("Gym is not available")
            

class TestGymnasiumCompatModule(_AuxTestGymCompatModule, AuxilliaryForTestGymnasium, unittest.TestCase):
    pass

class TestBoxGymnasiumObsSpace(_AuxTestBoxGymObsSpace, AuxilliaryForTestGymnasium, unittest.TestCase):
    pass

class TestBoxGymnasiumActSpace(_AuxTestBoxGymActSpace, AuxilliaryForTestGymnasium, unittest.TestCase):
    pass

class TestMultiDiscreteGymnasiumActSpace(_AuxTestMultiDiscreteGymActSpace, AuxilliaryForTestGymnasium, unittest.TestCase):
    pass

class TestDiscreteGymnasiumActSpace(_AuxTestDiscreteGymActSpace, AuxilliaryForTestGymnasium, unittest.TestCase):
    def test_class_different_from_multi_discrete(self):
        from grid2op.gym_compat import (DiscreteActSpaceGymnasium,
                                        MultiDiscreteActSpaceGymnasium)
        assert DiscreteActSpaceGymnasium is not MultiDiscreteActSpaceGymnasium
        assert DiscreteActSpaceGymnasium.__doc__ != MultiDiscreteActSpaceGymnasium.__doc__
        assert DiscreteActSpaceGymnasium.__name__ != MultiDiscreteActSpaceGymnasium.__name__

class TestAllGymnasiumActSpaceWithAlarm(_AuxTestAllGymActSpaceWithAlarm, AuxilliaryForTestGymnasium, unittest.TestCase):
    pass

class TestGOObsInRangeGymnasium(_AuxTestGOObsInRange, AuxilliaryForTestGymnasium, unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
