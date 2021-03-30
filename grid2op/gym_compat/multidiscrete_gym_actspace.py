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
from gym.spaces import MultiDiscrete, Box

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
    ATTR_CHANGE = 0
    ATTR_SET = 1
    ATTR_NEEDBUILD = 2
    ATTR_NEEDBINARIZED = 3

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

        # nb, dim, []
        self.dict_properties = {
            "set_line_status": ([3 for _ in range(act_sp.n_line)],
                                act_sp.n_line, self.ATTR_SET),
            "change_line_status": ([2 for _ in range(act_sp.n_line)],
                                   act_sp.n_line, self.ATTR_CHANGE),
            "set_bus": ([4 for _ in range(act_sp.dim_topo)],
                        act_sp.dim_topo, self.ATTR_SET),
            "change_bus": ([2 for _ in range(act_sp.dim_topo)],
                           act_sp.dim_topo, self.ATTR_CHANGE),
            "sub_set_bus": (None, act_sp.n_sub, self.ATTR_NEEDBUILD),  # dimension will be computed on the fly, if the stuff is used
            "sub_change_bus": (None, act_sp.n_sub, self.ATTR_NEEDBUILD),  # dimension will be computed on the fly, if the stuff is used
            "one_sub_set": (None, 1, self.ATTR_NEEDBUILD),  # dimension will be computed on the fly, if the stuff is used
            "one_sub_change": (None, 1, self.ATTR_NEEDBUILD),  # dimension will be computed on the fly, if the stuff is used
        }
        self._nb_bins = nb_bins
        for el in ["redispatch", "set_storage", "curtail", "curtail_mw"]:
            if el in attr_to_keep:
                if el not in nb_bins:
                    raise RuntimeError(f"The attribute you want to keep \"{el}\" is not present in the "
                                       f"\"nb_bins\". This attribute is continuous, you have to specify in how "
                                       f"how to convert it to a discrete space. See the documentation "
                                       f"for more information.")
                if el == "redispatch":
                    self.dict_properties[el] = ([nb_bins[el] if redisp else 1 for redisp in act_sp.gen_redispatchable],
                                                act_sp.n_gen, self.ATTR_NEEDBINARIZED)
                elif el == "curtail" or el == "curtail_mw":
                    self.dict_properties[el] = ([nb_bins[el] if renew else 1 for renew in act_sp.gen_renewable],
                                                act_sp.n_gen, self.ATTR_NEEDBINARIZED)
                elif el == "set_storage":
                    self.dict_properties[el] = ([nb_bins[el] for _ in range(act_sp.n_storage)],
                                                act_sp.n_gen, self.ATTR_NEEDBINARIZED)
                else:
                    raise RuntimeError(f"Unknown attribute \"{el}\"")

        self._dims = None
        self._functs = None  # final functions that is applied to the gym action to map it to a grid2Op action
        self._binarizers = None  # contains all the stuff to binarize the data
        self._types = None
        nvec = self._get_info()

        # initialize the base container
        MultiDiscrete.__init__(self, nvec=nvec)

    @staticmethod
    def _funct_set(vect):
        # gym encodes: 
        # for set_bus: 0 -> -1, 1-> 0 (don't change)), 2-> 1, 3 -> 2
        # for set_status: 0 -> -1, 1-> 0 (don't change)), 2-> 1 [3 do not exist for set_line_status !]
        vect -= 1
        return vect
    
    @staticmethod
    def _funct_change(vect):
        # gym encodes 0 -> False, 1 -> True
        vect = vect.astype(dt_bool)
        return vect

    def _funct_substations(self, orig_act, attr_nm, vect):
        """
        Used for "sub_set_bus" and "sub_change_bus"
        """
        vect_act = self._sub_modifiers[attr_nm]
        for sub_id, act_id in enumerate(vect):
            orig_act += vect_act[sub_id][act_id]

    def _funct_one_substation(self, orig_act, attr_nm, vect):
        """
        Used for "one_sub_set" and "one_sub_change"
        """
        orig_act += self._sub_modifiers[attr_nm][int(vect)]

    def _get_info(self):
        nvec = None
        self._dims = []
        self._functs = []
        self._binarizers = {}
        self._sub_modifiers = {}
        self._types = []
        box_space = None
        dim = 0
        for el in self._attr_to_keep:
            if el in self.dict_properties:
                nvec_, dim_, type_ = self.dict_properties[el]
                if type_ == self.ATTR_CHANGE:
                    # I can convert them directly into discrete attributes because it's a 
                    # recognize "change" attribute
                    funct = self._funct_change
                elif type_ == self.ATTR_SET:
                    # I can convert them directly into discrete attributes because it's a 
                    # recognize "set" attribute
                    funct = self._funct_set
                elif type_ == self.ATTR_NEEDBINARIZED:
                    # base action was continuous, i need to convert it to discrete action thanks
                    # to "binarization", that is done automatically here
                    from grid2op.gym_compat.box_gym_actspace import BoxGymActSpace
                    from grid2op.gym_compat.continuous_to_discrete import ContinuousToDiscreteConverter
                    if box_space is None:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore")
                            box_space = BoxGymActSpace(self._act_space,
                                                       attr_to_keep=["redispatch", "set_storage", "curtail", "curtail_mw"]
                                                      )

                    if el not in box_space.dict_properties:
                        raise RuntimeError(f"Impossible to dertmine lowest and maximum value for "
                                           f"key \"{el}\".")
                    low_, high_, shape_, dtype_ = box_space.dict_properties[el]
                    tmp_box = Box(low=low_, high=high_, dtype=dtype_)
                    tmp_binarizer = ContinuousToDiscreteConverter(init_space=tmp_box,
                                                                  nb_bins=self._nb_bins[el])
                    self._binarizers[el] = tmp_binarizer
                    funct = tmp_binarizer.gym_to_g2op
                elif type_ == self.ATTR_NEEDBUILD:
                    # attributes comes from substation manipulation, i need to build the entire space
                    nvec_ = []
                    self._sub_modifiers[el] = []
                    if el == "sub_set_bus": 
                        # one action per substations, using "set"
                        for sub_id in range(self._act_space.n_sub):
                            act_this_sub = self._act_space.get_all_unitary_topologies_set(self._act_space,
                                                                                          sub_id=sub_id)
                            if len(act_this_sub) == 0:
                                # no action can be done at this substation
                                act_this_sub.append(self._act_space())
                            nvec_.append(len(act_this_sub))
                            self._sub_modifiers[el].append(act_this_sub)
                        funct = self._funct_substations
                    elif el == "sub_change_bus":
                        # one action per substation, using "change"
                        for sub_id in range(self._act_space.n_sub):
                            acts_this_sub = self._act_space.get_all_unitary_topologies_change(self._act_space,
                                                                                              sub_id=sub_id)
                            if len(act_this_sub) == 0:
                                # no action can be done at this substation
                                act_this_sub.append(self._act_space())
                            nvec_.append(len(act_this_sub))
                            self._sub_modifiers[el].append(acts_this_sub)
                        funct = self._funct_substations
                    elif el == "one_sub_set":
                        # an action change only one substation, using "set"
                        self._sub_modifiers[el] = self._act_space.get_all_unitary_topologies_set(self._act_space)
                        funct = self._funct_one_substation
                        nvec_ = [len(self._sub_modifiers[el])]
                    elif el == "one_sub_change":
                        # an action change only one substation, using "change"
                        self._sub_modifiers[el] = self._act_space.get_all_unitary_topologies_change(self._act_space)
                        funct = self._funct_one_substation
                        nvec_ = [len(self._sub_modifiers[el])]
                    else:
                        raise RuntimeError(f"Unsupported attribute \"{el}\" when dealing with "
                                           f"action on substation")
                    
                else:
                    raise RuntimeError(f"Unknown way to build the action.")
            else:
                li_keys = '\n\t- '.join(sorted(list(self.dict_properties.keys())))
                raise RuntimeError(f"Unknown action attributes \"{el}\". Supported attributes are: "
                                   f"\n\t- {li_keys}")
            dim += dim_
            if nvec is not None:
                nvec += nvec_
            else:
                nvec = nvec_
            self._dims.append(dim)
            self._functs.append(funct)
            self._types.append(type_)
        return nvec

    def _handle_attribute(self, res, gym_act_this, attr_nm, funct, type_):
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
        # TODO code that !
        vect = 1 * gym_act_this
        if type_ == self.ATTR_NEEDBUILD:
            funct(res, attr_nm, vect)
        else:
            setattr(res, attr_nm, funct(vect))
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
        for attr_nm, where_to_put, funct, type_ in \
            zip(self._attr_to_keep, self._dims, self._functs, self._types):
            this_part = 1 * gym_act[prev:where_to_put]
            if attr_nm in self.dict_properties:
                self._handle_attribute(res, this_part, attr_nm, funct, type_)
            else:
                raise RuntimeError(f"Unknown attribute \"{attr_nm}\".")
            prev = where_to_put
        return res
