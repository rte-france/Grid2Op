# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

__all__ = [
    "BaseGymAttrConverter",
    "GymEnv",
    "GymObservationSpace",
    "GymActionSpace",
    "ScalerAttrConverter",
    "MultiToTupleConverter",
    "ContinuousToDiscreteConverter",
    "BoxGymObsSpace",
    "BoxGymActSpace",
    "MultiDiscreteActSpace",
    "DiscreteActSpace",
]

from grid2op.gym_compat.utils import _MAX_GYM_VERSION_RANDINT, GYM_VERSION, GYMNASIUM_AVAILABLE, GYM_AVAILABLE

# base for all gym converter
from grid2op.gym_compat.base_gym_attr_converter import BaseGymAttrConverter
if GYMNASIUM_AVAILABLE:
    from grid2op.gym_compat.base_gym_attr_converter import BaseGymnasiumAttrConverter
    __all__.append("BaseGymnasiumAttrConverter")
if GYM_AVAILABLE:
    from grid2op.gym_compat.base_gym_attr_converter import BaseGymLegacyAttrConverter
    __all__.append("BaseGymLegacyAttrConverter")

# the environment (by default with dict encoding)
if GYMNASIUM_AVAILABLE:
    from grid2op.gym_compat.gymenv import GymnasiumEnv
    __all__.append("GymnasiumEnv")
if GYM_AVAILABLE:
    from grid2op.gym_compat.gymenv import GymEnv_Legacy, GymEnv_Modern
    __all__.append("GymEnv_Legacy")
    __all__.append("GymEnv_Modern")

# define the default env to use
if GYMNASIUM_AVAILABLE:
    GymEnv = GymnasiumEnv
else:
    if GYM_VERSION <= _MAX_GYM_VERSION_RANDINT:
        GymEnv = GymEnv_Legacy
    else:
        GymEnv = GymEnv_Modern
    
# action space (as Dict)
from grid2op.gym_compat.gym_act_space import GymActionSpace
if GYMNASIUM_AVAILABLE:
    from grid2op.gym_compat.gym_act_space import GymnasiumActionSpace
    __all__.append("GymnasiumActionSpace")
if GYM_AVAILABLE:
    from grid2op.gym_compat.gym_act_space import GymLegacyActionSpace
    __all__.append("GymLegacyActionSpace")
    
# observation space (as Dict)
from grid2op.gym_compat.gym_obs_space import GymObservationSpace
if GYMNASIUM_AVAILABLE:
    from grid2op.gym_compat.gym_obs_space import GymnasiumObservationSpace
    __all__.append("GymnasiumObservationSpace")
if GYM_AVAILABLE:
    from grid2op.gym_compat.gym_obs_space import GymLegacyObservationSpace
    __all__.append("GymLegacyObservationSpace")
    
from grid2op.gym_compat.scaler_attr_converter import ScalerAttrConverter
if GYMNASIUM_AVAILABLE:
    from grid2op.gym_compat.scaler_attr_converter import ScalerAttrConverterGymnasium
    __all__.append("ScalerAttrConverterGymnasium")
if GYM_AVAILABLE:
    from grid2op.gym_compat.scaler_attr_converter import ScalerAttrConverterGymLegacy
    __all__.append("ScalerAttrConverterGymLegacy")


from grid2op.gym_compat.multi_to_tuple_converter import MultiToTupleConverter
if GYMNASIUM_AVAILABLE:
    from grid2op.gym_compat.multi_to_tuple_converter import MultiToTupleConverterGymnasium
    __all__.append("MultiToTupleConverterGymnasium")
if GYM_AVAILABLE:
    from grid2op.gym_compat.multi_to_tuple_converter import MultiToTupleConverterGymLegacy
    __all__.append("MultiToTupleConverterGymLegacy")
    

from grid2op.gym_compat.continuous_to_discrete import ContinuousToDiscreteConverter
if GYMNASIUM_AVAILABLE:
    from grid2op.gym_compat.continuous_to_discrete import ContinuousToDiscreteConverterGymnasium
    __all__.append("ContinuousToDiscreteConverterGymnasium")
if GYM_AVAILABLE:
    from grid2op.gym_compat.continuous_to_discrete import ContinuousToDiscreteConverterGymLegacy
    __all__.append("ContinuousToDiscreteConverterGymLegacy")
   
# observation space (as Box)
from grid2op.gym_compat.box_gym_obsspace import BoxGymObsSpace
if GYMNASIUM_AVAILABLE:
    from grid2op.gym_compat.box_gym_obsspace import BoxGymnasiumObsSpace
    __all__.append("BoxGymnasiumObsSpace")
if GYM_AVAILABLE:
    from grid2op.gym_compat.box_gym_obsspace import BoxGymLegacyObsSpace
    __all__.append("BoxGymLegacyObsSpace")


from grid2op.gym_compat.box_gym_actspace import BoxGymActSpace
if GYMNASIUM_AVAILABLE:
    from grid2op.gym_compat.box_gym_actspace import BoxGymnasiumActSpace
    __all__.append("BoxGymnasiumActSpace")
if GYM_AVAILABLE:
    from grid2op.gym_compat.box_gym_actspace import BoxGymLegacyActSpace
    __all__.append("BoxGymLegacyActSpace")


from grid2op.gym_compat.multidiscrete_gym_actspace import MultiDiscreteActSpace
if GYMNASIUM_AVAILABLE:
    from grid2op.gym_compat.multidiscrete_gym_actspace import MultiDiscreteActSpaceGymnasium
    __all__.append("MultiDiscreteActSpaceGymnasium")
if GYM_AVAILABLE:
    from grid2op.gym_compat.multidiscrete_gym_actspace import MultiDiscreteActSpaceGymLegacy
    __all__.append("MultiDiscreteActSpaceGymLegacy")
    

from grid2op.gym_compat.discrete_gym_actspace import DiscreteActSpace
if GYMNASIUM_AVAILABLE:
    from grid2op.gym_compat.discrete_gym_actspace import DiscreteActSpaceGymnasium
    __all__.append("DiscreteActSpaceGymnasium")
if GYM_AVAILABLE:
    from grid2op.gym_compat.discrete_gym_actspace import DiscreteActSpaceGymLegacy
    __all__.append("DiscreteActSpaceGymLegacy")


# TODO doc and test

