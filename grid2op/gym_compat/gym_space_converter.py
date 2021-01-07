# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from gym import spaces
import numpy as np
import copy

from grid2op.dtypes import dt_int, dt_bool, dt_float


class _BaseGymSpaceConverter(spaces.Dict):
    """
    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
        Used as a base class to convert grid2op state to gym state (wrapper for some useful function
        for both the action space and the observation space).

    """
    def __init__(self, dict_gym_space, dict_variables=None):
        spaces.Dict.__init__(self, dict_gym_space)
        self._keys_encoding = {}
        if dict_variables is not None:
            for k, v in dict_variables.items():
                self._keys_encoding[k] = v

    @staticmethod
    def _generic_gym_space(dt, sh, low=None, high=None):
        if low is None:
            low = np.iinfo(dt).min
        if high is None:
            high = np.iinfo(dt).max
        shape = (sh,)
        my_type = spaces.Box(low=dt.type(low), high=dt.type(high), shape=shape, dtype=dt)
        return my_type

    @staticmethod
    def _boolean_type(sh):
        return spaces.MultiBinary(n=sh)

    @staticmethod
    def _extract_obj_grid2op(vect, dtype):
        if len(vect) == 1:
            res = vect[0]
            # convert the types for json serializable
            # this is not automatically done by gym...
            if dtype == dt_int or dtype == np.int64 or dtype == np.int32:
                res = int(res)
            elif dtype == dt_float or dtype == np.float64 or dtype == np.float32:
                res = float(res)
            elif dtype == dt_bool:
                res = bool(res)
        else:
            res = vect
        return res

    def _base_to_gym(self, keys, obj, dtypes, converter=None):
        res = spaces.dict.OrderedDict()
        for k in keys:
            conv_k = k
            if converter is not None:
                # for converting the names between internal names and "human readable names"
                conv_k = converter[k]

            obj_raw = obj._get_array_from_attr_name(conv_k)
            if k in self._keys_encoding:
                if self._keys_encoding[k] is None:
                    # keys is deactivated
                    continue
                else:
                    # i need to process the "function" part in the keys
                    obj_json_cleaned = self._keys_encoding[k].g2op_to_gym(obj_raw)
            else:
                obj_json_cleaned = self._extract_obj_grid2op(obj_raw, dtypes[k])
            res[k] = obj_json_cleaned
        return res

    def get_dict_encoding(self):
        return copy.deepcopy(self._keys_encoding)

    def reencode_space(self, key, func):
        raise NotImplementedError("This should be implemented in the GymActionSpace and GymObservationSpace")

    def reenc(self, key, fun):
        """
        shorthand for :func:`GymObservationSpace.reencode_space` or
        :func:`GymActionSpace.reencode_space`
        """
        return self.reencode_space(key, fun)

    def keep_only_attr(self, attr_names):
        """
        keep only a certain part of the observation
        """
        if isinstance(attr_names, str):
            attr_names = [attr_names]

        dict_ = self.spaces.keys()
        res = self
        for k in dict_:
            if k not in attr_names:
                res = res.reencode_space(k, None)
        return res

    def ignore_attr(self, attr_names):
        """
        ignore some attribute names from the space
        """
        if isinstance(attr_names, str):
            attr_names = [attr_names]
        res = self
        for k in self.spaces.keys():
            if k in attr_names:
                res = res.reencode_space(k, None)
        return res
