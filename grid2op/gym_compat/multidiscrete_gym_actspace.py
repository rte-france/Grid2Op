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
from gym.spaces import MultiDiscrete

from grid2op.Action import BaseAction
from grid2op.dtypes import dt_int, dt_bool, dt_float

from grid2op.gym_compat.utils import ALL_ATTR, ATTR_DISCRETE
# TODO test that it works normally
# TODO test the casting in dt_int or dt_float depending on the data
# TODO test the scaling
# TODO doc
# TODO test the function part


class MultiDiscreteActSpace(MultiDiscrete):
    """
    TODO

    """
    def __init__(self,
                 grid2op_action_space,
                 attr_to_keep=ALL_ATTR,
                 nb_bins={"redispatch": 7, "set_storage": 7, "curtail": 7, "curtail_mw": 7}
                 ):

        if attr_to_keep == ALL_ATTR:
            # by default, i remove all the attributes that are not supported by the action type
            # i do not do that if the user specified specific attributes to keep. This is his responsibility in
            # in this case
            attr_to_keep = {el for el in attr_to_keep if grid2op_action_space.supports_type(el)}

        for el in attr_to_keep:
            if el not in ATTR_DISCRETE:
                warnings.warn(f"The class \"MultiDiscreteActSpace\" should mainly be used to consider only discrete "
                              f"actions (eg. set_line_status, set_bus or change_bus). Though it is possible to use "
                              f"\"{el}\" when building it, be aware that this continuous action will be treated "
                              f"as discrete by splitting it into bins. "
                              f"Consider using the \"BoxGymActSpace\" for these attributes."
                              )

        self._attr_to_keep = attr_to_keep

        act_sp = grid2op_action_space
        self._act_space = copy.deepcopy(grid2op_action_space)

        low_gen = -1.0 * act_sp.gen_max_ramp_down
        high_gen = 1.0 * act_sp.gen_max_ramp_up
        low_gen[~act_sp.gen_redispatchable] = 0.
        high_gen[~act_sp.gen_redispatchable] = 0.

        # nb, dim
        self.dict_properties = {
            "set_line_status": ([3 for _ in range(act_sp.n_line)],
                                act_sp.n_line),
            "change_line_status": ([2 for _ in range(act_sp.n_line)],
                                   act_sp.n_line),
            "set_bus": ([3 for _ in range(act_sp.n_line)],
                        act_sp.dim_topo),
            "change_bus": (2,
                           act_sp.dim_topo),
            "sub_set_bus": (None, act_sp.n_sub),  # dimension will be computed on the fly, if the stuff is used
            "sub_change_bus": (None, act_sp.n_sub),  # dimension will be computed on the fly, if the stuff is used

        }

        for el in ["redispatch", "set_storage", "curtail", "curtail_mw"]:
            if el in attr_to_keep:
                if el not in nb_bins:
                    raise RuntimeError(f"The attribute you want to keep \"{el}\" is not present in the "
                                       f"\"nb_bins\". This attribute is continuous, you have to specify in how "
                                       f"how to convert it to a discrete space. See the documentation "
                                       f"for more information.")
                if el == "redispatch":
                    self.dict_properties[el] = ([nb_bins[el] if redisp else 1 for redisp in act_sp.gen_redispatchable],
                                                act_sp.n_gen)
                elif el == "curtail" or el == "curtail_mw":
                    self.dict_properties[el] = ([nb_bins[el] if renew else 1 for renew in act_sp.gen_renewable],
                                                act_sp.n_gen)
                elif el == "set_storage":
                    self.dict_properties[el] = ([nb_bins[el] for _ in range(act_sp.n_storage)],
                                                act_sp.n_gen)
                else:
                    raise RuntimeError(f"Unknown attribute \"{el}\"")

        self._dims = None
        nvec = self._get_info()

        # initialize the base container
        MultiDiscrete.__init__(self, nvec=nvec)

    def _get_info(self):
        nvec = None
        self._dims = []
        for el in self._attr_to_keep:
            if el in self.dict_properties:
                # el is an attribute of an observation, for example "load_q" or "topo_vect"
                nvec_ = self.dict_properties[el]
            else:
                li_keys = '\n\t-'.join(sorted(list(self.dict_properties.keys())))
                raise RuntimeError(f"Unknown action attributes \"{el}\". Supported attributes are: "
                                   f"\n{li_keys}")

            # TODO
            if nvec is not None:
                nvec += nvec_
            else:
                nvec = nvec_
            self._dims.append(nvec_)
        return nvec

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
                    # NB: i suppose here the numbers are between 0 and 1
                    this_part = (this_part >= 0.5).astype(dt_bool)

                self._handle_attribute(res, this_part, attr_nm)
            else:
                raise RuntimeError(f"Unknown attribute \"{attr_nm}\".")
            prev = where_to_put
        return res
