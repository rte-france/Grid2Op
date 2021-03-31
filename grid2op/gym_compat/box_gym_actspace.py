# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
import warnings
import numpy as np
from gym.spaces import Box

from grid2op.Action import BaseAction, ActionSpace
from grid2op.dtypes import dt_int, dt_bool, dt_float

# TODO test that it works normally
# TODO test the casting in dt_int or dt_float depending on the data
# TODO test the scaling
# TODO doc
# TODO test the function part

from grid2op.gym_compat.utils import ALL_ATTR, ATTR_DISCRETE


class BoxGymActSpace(Box):
    """
    TODO

    """
    def __init__(self,
                 grid2op_action_space,
                 attr_to_keep=ALL_ATTR,
                 add={},
                 multiply={},
                 functs={}):
        if not isinstance(grid2op_action_space, ActionSpace):
            raise RuntimeError(f"Impossible to create a BoxGymActSpace without providing a "
                               f"grid2op action_space. You provided {type(grid2op_action_space)}"
                               f"as the \"grid2op_action_space\" attribute.")

        if attr_to_keep == ALL_ATTR:
            # by default, i remove all the attributes that are not supported by the action type
            # i do not do that if the user specified specific attributes to keep. This is his responsibility in
            # in this case
            attr_to_keep = {el for el in attr_to_keep if grid2op_action_space.supports_type(el)}

        for el in ATTR_DISCRETE:
            if el in attr_to_keep:
                warnings.warn(f"The class \"BoxGymActSpace\" should mainly be used to consider only continuous "
                              f"actions (eg. redispatch, set_storage or curtail). Though it is possible to use "
                              f"\"{el}\" when building it, be aware that this discrete action will be treated "
                              f"as continuous. Consider using the \"MultiDiscreteActSpace\" for these attributes."
                              )

        self._attr_to_keep = attr_to_keep

        act_sp = grid2op_action_space
        self._act_space = copy.deepcopy(grid2op_action_space)

        low_gen = -1.0 * act_sp.gen_max_ramp_down
        high_gen = 1.0 * act_sp.gen_max_ramp_up
        low_gen[~act_sp.gen_redispatchable] = 0.
        high_gen[~act_sp.gen_redispatchable] = 0.
        curtail = np.full(shape=(act_sp.n_gen,), fill_value=0., dtype=dt_float)
        curtail[~act_sp.gen_renewable] = 1.0
        curtail_mw = np.full(shape=(act_sp.n_gen,), fill_value=0., dtype=dt_float)
        curtail_mw[~act_sp.gen_renewable] = act_sp.gen_pmax[~act_sp.gen_renewable]
        self.dict_properties = {
            "set_line_status": (np.full(shape=(act_sp.n_line,), fill_value=-1, dtype=dt_int),
                                np.full(shape=(act_sp.n_line,), fill_value=1, dtype=dt_int),
                                (act_sp.n_line,),
                                dt_int),
            "change_line_status": (np.full(shape=(act_sp.n_line,), fill_value=0, dtype=dt_int),
                                   np.full(shape=(act_sp.n_line,), fill_value=1, dtype=dt_int),
                                   (act_sp.n_line,),
                                   dt_int),
            "set_bus": (np.full(shape=(act_sp.dim_topo,), fill_value=-1, dtype=dt_int),
                        np.full(shape=(act_sp.dim_topo,), fill_value=1, dtype=dt_int),
                        (act_sp.dim_topo,),
                        dt_int),
            "change_bus": (np.full(shape=(act_sp.dim_topo,), fill_value=0, dtype=dt_int),
                           np.full(shape=(act_sp.dim_topo,), fill_value=1, dtype=dt_int),
                           (act_sp.dim_topo,),
                           dt_int),
            "redispatch": (low_gen,
                           high_gen,
                           (act_sp.n_gen,),
                           dt_float),
            "set_storage": (-1.0 * act_sp.storage_max_p_prod,
                            1.0 * act_sp.storage_max_p_absorb,
                            (act_sp.n_storage,),
                            dt_float),
            "curtail": (curtail,
                        np.full(shape=(act_sp.n_gen,), fill_value=1., dtype=dt_float),
                        (act_sp.n_gen,),
                        dt_float),
            "curtail_mw": (curtail_mw,
                           1.0 * act_sp.gen_pmax,
                           (act_sp.n_gen,),
                           dt_float),
        }
        self._key_dict_to_proptype = {"set_line_status": dt_int,
                                      "change_line_status": dt_bool,
                                      "set_bus": dt_int,
                                      "change_bus": dt_bool,
                                      "redispatch": dt_float,
                                      "set_storage": dt_float,
                                      "curtail": dt_float,
                                      "curtail_mw": dt_float}
        self._add = add
        self._multiply = multiply

        # handle the "functional" part
        self.__func = {}

        self._dims = None
        self._dtypes = None
        low, high, shape, dtype = self._get_info(functs)

        # initialize the base container
        Box.__init__(self, low=low, high=high, shape=shape, dtype=dtype)

    def _get_info(self, functs):
        low = None
        high = None
        shape = None
        dtype = None
        self._dims = []
        self._dtypes = []
        for el in self._attr_to_keep:
            if el in functs:
                # the attribute name "el" has been put in the functs
                callable_, low_, high_, shape_, dtype_ = functs[el]

                if dtype_ is None:
                    dtype_ = dt_float
                if shape_ is None:
                    raise RuntimeError(f"Error: if you use the \"functs\" keyword for the action space, "
                                       f"you need to provide the shape of the vector you expect. See some "
                                       f"examples in the official documentation.")
                if low_ is None:
                    low_ = np.full(shape_, fill_value=-np.inf, dtype=dtype_)
                elif isinstance(low_, float):
                    low_ = np.full(shape_, fill_value=low_, dtype=dtype_)

                if high_ is None:
                    high_ = np.full(shape_, fill_value=np.inf, dtype=dtype_)
                elif isinstance(high_, float):
                    high_ = np.full(shape_, fill_value=high_, dtype=dtype_)

                # simulate a vector in the range low_, high_ and the right shape to test the function given
                # by the user
                vect_right_properties = np.random.uniform(size=shape_)
                finite_both = np.isfinite(low_) & np.isfinite(high_)
                fintte_low = np.isfinite(low_) & ~np.isfinite(high_)
                fintte_high = ~np.isfinite(low_) & np.isfinite(high_)
                vect_right_properties[finite_both] = vect_right_properties[finite_both] * \
                                                     (high_[finite_both] - low_[finite_both]) +\
                                                     low_[finite_both]
                vect_right_properties[fintte_high] += low_[fintte_high]

                try:
                    tmp = callable_(vect_right_properties)
                except Exception as exc_:
                    raise RuntimeError(f"Error for the function your provided with key \"{el}\". "
                                       f"The error was :\n {exc_}")
                if not isinstance(tmp, BaseAction):
                    raise RuntimeError(f"The function you provided in the \"functs\" argument for key \"{el}\" "
                                       f"should take a")

                self.__func[el] = callable_
                    
            elif el in self.dict_properties:
                # el is an attribute of an observation, for example "load_q" or "topo_vect"
                low_, high_, shape_, dtype_ = self.dict_properties[el]
            else:
                li_keys = '\n\t- '.join(sorted(list(self.dict_properties.keys()) +
                                              list(self.__func.keys()))
                                       )
                raise RuntimeError(f"Unknown action attributes \"{el}\". Supported attributes are: "
                                   f"\n\t- {li_keys}")

            # handle the data type
            if dtype is None:
                dtype = dtype_
            else:
                if dtype_ == dt_float:
                    dtype = dt_float

            # handle the shape
            if shape is None:
                shape = shape_
            else:
                shape = (shape[0] + shape_[0],)

            # handle low / high
            # NB: the formula is: glop = gym * multiply + add
            if el in self._add:
                low_ -= self._add[el]
                high_ -= self._add[el]
            if el in self._multiply:
                # special case if a 0 were entered
                arr_ = 1.0 * self._multiply[el]
                is_nzero = arr_ != 0.
                low_[is_nzero] /= arr_[is_nzero]
                high_[is_nzero] /= arr_[is_nzero]
            if low is None:
                low = low_
                high = high_
            else:
                low = np.concatenate((low.astype(dtype), low_.astype(dtype))).astype(dtype)
                high = np.concatenate((high.astype(dtype), high_.astype(dtype))).astype(dtype)

            # remember where this need to be stored
            self._dims.append(shape[0])
            self._dtypes.append(dtype_)

        return low, high, shape, dtype

    def _handle_attribute(self, res, gym_act_this, attr_nm):
        """
        TODO

        Parameters
        ----------
        res
        gym_act_this
        attr_nm

        Returns
        -------

        """
        if attr_nm in self._multiply:
            gym_act_this *= self._multiply[attr_nm]
        if attr_nm in self._add:
            gym_act_this += self._add[attr_nm]
        setattr(res, attr_nm, gym_act_this)
        return res

    def from_gym(self, gym_act):
        """
        TODO

        Parameters
        ----------
        gym_act

        Returns
        -------

        """
        res = self._act_space()
        prev = 0
        for attr_nm, where_to_put, dtype in zip(self._attr_to_keep, self._dims, self._dtypes):
            this_part = 1 * gym_act[prev:where_to_put]
            if attr_nm in self.__func:
                glop_act_tmp = self.__func[attr_nm](this_part)
                res += glop_act_tmp
            elif hasattr(res, attr_nm):
                glop_dtype = self._key_dict_to_proptype[attr_nm]
                if glop_dtype == dt_int:
                    # convert floating point actions to integer.
                    # NB: i round first otherwise it is cut.
                    this_part = np.round(this_part, 0).astype(dtype)
                elif glop_dtype == dt_bool:
                    # convert floating point actions to bool.
                    # NB: it's important here the numbers are between 0 and 1
                    this_part = (this_part >= 0.5).astype(dt_bool)

                self._handle_attribute(res, this_part, attr_nm)
            else:
                raise RuntimeError(f"Unknown attribute \"{attr_nm}\".")
            prev = where_to_put
        return res
