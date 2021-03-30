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
from gym.spaces import Discrete, Box


from grid2op.Action import BaseAction
from grid2op.dtypes import dt_int, dt_bool, dt_float
from grid2op.Converter import IdToAct

from grid2op.gym_compat.utils import ALL_ATTR, ATTR_DISCRETE
# TODO test that it works normally
# TODO test the casting in dt_int or dt_float depending on the data
# TODO test the scaling
# TODO doc
# TODO test the function part


class DiscreteActSpace(Discrete):
    """
    TODO

    """
    def __init__(self,
                 grid2op_action_space,
                 attr_to_keep=ALL_ATTR,
                 nb_bins={"redispatch": 7, "set_storage": 7, "curtail": 7}
                 ):
        act_sp = grid2op_action_space
        self.action_space = copy.deepcopy(act_sp)

        if attr_to_keep == ALL_ATTR:
            # by default, i remove all the attributes that are not supported by the action type
            # i do not do that if the user specified specific attributes to keep. This is his responsibility in
            # in this case
            attr_to_keep = {el for el in attr_to_keep if grid2op_action_space.supports_type(el)}

        for el in attr_to_keep:
            if el not in ATTR_DISCRETE:
                warnings.warn(f"The class \"DiscreteActSpace\" should mainly be used to consider only discrete "
                              f"actions (eg. set_line_status, set_bus or change_bus). Though it is possible to use "
                              f"\"{el}\" when building it, be aware that this continuous action will be treated "
                              f"as discrete by splitting it into bins. "
                              f"Consider using the \"BoxGymActSpace\" for these attributes."
                              )

        self._attr_to_keep = attr_to_keep
        self._nb_bins = nb_bins

        self.dict_properties = {
            "set_line_status": act_sp.get_all_unitary_line_set,
            "change_line_status": act_sp.get_all_unitary_line_change,
            "set_bus": act_sp.get_all_unitary_topologies_set,
            "change_bus": act_sp.get_all_unitary_topologies_change,
            "redispatch": act_sp.get_all_unitary_redispatch,
            "set_storage": act_sp.get_all_unitary_storage,
            "curtail": act_sp.get_all_unitary_curtail,
            "curtail_mw": act_sp.get_all_unitary_curtail
        }

        self.converter = None
        n_act = self._get_info()
        
        # initialize the base container
        Discrete.__init__(self, n=n_act)

    def _get_info(self):
        converter = IdToAct(self.action_space)
        li_act = [self.action_space()]
        for attr_nm in self._attr_to_keep:
            if attr_nm in self.dict_properties:
                if attr_nm not in self._nb_bins:
                    li_act += self.dict_properties[attr_nm](self.action_space)
                else:
                    if attr_nm == "curtail" or attr_nm == "curtail_mw":
                        li_act += self.dict_properties[attr_nm](self.action_space, num_bin=self._nb_bins[attr_nm])
                    else:
                        li_act += self.dict_properties[attr_nm](self.action_space,
                                                                num_down=self._nb_bins[attr_nm],
                                                                num_up=self._nb_bins[attr_nm])
            else:
                li_keys = '\n\t- '.join(sorted(list(self.dict_properties.keys())))
                raise RuntimeError(f"Unknown action attributes \"{attr_nm}\". Supported attributes are: "
                                   f"\n\t- {li_keys}")

        converter.init_converter(li_act)
        self.converter = converter
        return self.converter.n

    def from_gym(self, gym_act):
        """
        TODO

        Parameters
        ----------
        gym_act

        Returns
        -------

        """
        res = self.converter.all_actions[int(gym_act)]
        return res
