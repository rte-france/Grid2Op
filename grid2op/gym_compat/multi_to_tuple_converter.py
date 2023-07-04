# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
# from gym.spaces import Tuple, MultiBinary, MultiDiscrete, Discrete

from grid2op.dtypes import dt_int
from grid2op.gym_compat.utils import sample_seed
from grid2op.gym_compat.utils import GYM_AVAILABLE, GYMNASIUM_AVAILABLE


class __AuxFixedTuple:
    """I simply overload the "seed" function because the default one behaves
    really really poorly
    see issue https://github.com/openai/gym/issues/2166
    """

    def seed(self, seed=None):
        """Seed the PRNG of this space.
        see issue https://github.com/openai/gym/issues/2166
        of openAI gym
        """
        seeds = super(type(self)._TupleType, self).seed(seed)
        sub_seeds = seeds
        max_ = np.iinfo(dt_int).max
        for i, space in enumerate(self.spaces):
            sub_seed = sample_seed(max_, self.np_random)
            sub_seeds.append(space.seed(sub_seed))
        return sub_seeds


if GYM_AVAILABLE:
    from gym.spaces import Tuple
    FixedTupleLegacyGym = type("FixedTupleLegacyGym",
                               (__AuxFixedTuple, Tuple, ),
                               {"_gymnasium": False,
                                "_TupleType": Tuple})
    FixedTuple = FixedTupleLegacyGym
        

if GYMNASIUM_AVAILABLE:
    from gymnasium.spaces import Tuple
    # I has been fixed in gymnasium so I reuse it
    # FixedTupleGymnasium = type("FixedTupleGymnasium",
    #                            (__AuxFixedTuple, Tuple, ),
    #                            {"_gymnasium": True,
    #                             "_TupleType": Tuple})
    FixedTupleGymnasium = Tuple 
    FixedTuple = Tuple


class __AuxMultiToTupleConverter:
    """
    Some framework, for example ray[rllib] do not support MultiBinary nor MultiDiscrete gym
    action space. Apparently this is not going to change in a near
    future (see https://github.com/ray-project/ray/issues/1519).

    We choose to encode some variable using `MultiBinary` variable in grid2op. This allows for easy
    manipulation of them if using these frameworks.

    MultiBinary are encoded with gym Tuple of gym Discrete variables.

    .. warning::
        Depending on the presence absence of gymnasium and gym packages this class might behave differently.
        
        In grid2op we tried to maintain compatibility both with gymnasium (newest) and gym (legacy, 
        no more maintained) RL packages. The behaviour is the following:
        
        - :class:`MultiToTupleConverter` will inherit from gymnasium if it's installed 
          (in this case it will be :class:`MultiToTupleConverterGymnasium`), otherwise it will
          inherit from gym (and will be exactly :class:`MultiToTupleConverterLegacyGym`)
        - :class:`MultiToTupleConverterGymnasium` will inherit from gymnasium if it's available and never from
          from gym
        - :class:`MultiToTupleConverterLegacyGym` will inherit from gym if it's available and never from
          from gymnasium
        
        See :ref:`gymnasium_gym` for more information
        
    TODO add code example
    """

    def __init__(self, init_space=None):
        self.size = None
        type(self)._BaseGymAttrConverterType.__init__(self, space=None)
        if init_space is not None:
            self.initialize_space(init_space)

        self.previous_fun = self._previous_fun
        self.after_fun = self._after_fun

    def _previous_fun(self, x):
        return x

    def _after_fun(self, x):
        return x

    def initialize_space(self, init_space):
        if isinstance(init_space, (type(self)._MultiBinaryType, type(self)._MultiDiscreteType)):
            pass
        elif isinstance(init_space, type(self)._BaseGymAttrConverterType):
            self.previous_fun = init_space.g2op_to_gym
            self.after_fun = init_space.gym_to_g2op
            if isinstance(init_space.my_space, (type(self)._MultiBinaryType, type(self)._MultiDiscreteType)):
                init_space = init_space.my_space
            else:
                raise RuntimeError(
                    "Bad converter used. It should be of type MultiBinary or MultiDiscrete"
                )
        else:
            raise RuntimeError(
                "Impossible to convert a gym space of type {} to a Tuple (it should be of "
                "type space.MultiBinary or space.MultiDiscrete)"
                "".format(type(init_space))
            )
        if isinstance(init_space, type(self)._MultiBinaryType):
            self.size = init_space.n
        else:
            # then it's a MultiDiscrete
            self.size = init_space.nvec.shape[0]
        li = []
        for i in range(self.size):
            tmp_sz = 2
            if isinstance(init_space, type(self)._MultiDiscreteType):
                tmp_sz = init_space.nvec[i]
            li.append(type(self)._DiscreteType(tmp_sz))
        self.base_initialize(space=type(self)._FixedTupleType(li), g2op_to_gym=None, gym_to_g2op=None)

    def gym_to_g2op(self, gym_object):
        tmp = np.array(gym_object).astype(dt_int)
        return self.after_fun(tmp)

    def g2op_to_gym(self, g2op_object):
        tmp = self.previous_fun(g2op_object)  # TODO
        return tuple(tmp.astype(dt_int))

    def close(self):
        pass


if GYM_AVAILABLE:
    from gym.spaces import (MultiBinary as LegacyGymMultiBinary,
                            MultiDiscrete as LegacyGymMultiDiscrete, 
                            Discrete as LegacyGymDiscrete)
    from grid2op.gym_compat.base_gym_attr_converter import BaseLegacyGymAttrConverter
    MultiToTupleConverterLegacyGym = type("MultiToTupleConverterLegacyGym",
                                          (__AuxMultiToTupleConverter, BaseLegacyGymAttrConverter, ),
                                          {"_gymnasium": False,
                                           "_FixedTupleType": FixedTupleLegacyGym,
                                           "_BaseGymAttrConverterType": BaseLegacyGymAttrConverter,
                                           "_MultiDiscreteType": LegacyGymMultiDiscrete,
                                           "_MultiBinaryType": LegacyGymMultiBinary,
                                           "_DiscreteType": LegacyGymDiscrete,
                                           "__module__": __name__
                                           })
    MultiToTupleConverterLegacyGym.__doc__ = __AuxMultiToTupleConverter.__doc__
    MultiToTupleConverter = MultiToTupleConverterLegacyGym
    MultiToTupleConverter.__doc__ = __AuxMultiToTupleConverter.__doc__
        

if GYMNASIUM_AVAILABLE:
    from gymnasium.spaces import MultiBinary, MultiDiscrete, Discrete, Tuple
    from grid2op.gym_compat.base_gym_attr_converter import BaseGymnasiumAttrConverter
    MultiToTupleConverterGymnasium = type("MultiToTupleConverterGymnasium",
                                          (__AuxMultiToTupleConverter, BaseGymnasiumAttrConverter, ),
                                          {"_gymnasium": True,
                                           "_FixedTupleType": Tuple,
                                           "_BaseGymAttrConverterType": BaseGymnasiumAttrConverter,
                                           "_MultiDiscreteType": MultiDiscrete,
                                           "_MultiBinaryType": MultiBinary,
                                           "_DiscreteType": Discrete,
                                           "__module__": __name__
                                           })
    MultiToTupleConverterGymnasium.__doc__ = __AuxMultiToTupleConverter.__doc__
    MultiToTupleConverter = MultiToTupleConverterGymnasium
    MultiToTupleConverter.__doc__ = __AuxMultiToTupleConverter.__doc__
