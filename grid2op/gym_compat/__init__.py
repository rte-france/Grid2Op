# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

__all__ = [
    "BaseGymAttrConverter",
    "GymEnv_Legacy",
    "GymEnv_Modern",
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

from grid2op.gym_compat.base_gym_attr_converter import BaseGymAttrConverter
from grid2op.gym_compat.gymenv import GymEnv_Legacy, GymEnv_Modern

from grid2op.gym_compat.utils import _MAX_GYM_VERSION_RANDINT, GYM_VERSION
if GYM_VERSION <= _MAX_GYM_VERSION_RANDINT:
    GymEnv = GymEnv_Legacy
else:
    GymEnv = GymEnv_Modern
    
    
from grid2op.gym_compat.gym_act_space import GymActionSpace
from grid2op.gym_compat.gym_obs_space import GymObservationSpace
from grid2op.gym_compat.scaler_attr_converter import ScalerAttrConverter
from grid2op.gym_compat.multi_to_tuple_converter import MultiToTupleConverter
from grid2op.gym_compat.continuous_to_discrete import ContinuousToDiscreteConverter
from grid2op.gym_compat.box_gym_obsspace import BoxGymObsSpace
from grid2op.gym_compat.box_gym_actspace import BoxGymActSpace
from grid2op.gym_compat.multidiscrete_gym_actspace import MultiDiscreteActSpace
from grid2op.gym_compat.discrete_gym_actspace import DiscreteActSpace
