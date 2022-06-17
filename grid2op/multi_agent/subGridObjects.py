# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Space.GridObjects import GridObjects
import numpy as np


class SubGridObjects(GridObjects) :

    sub_orig_ids = None
    local2global = dict()
    mask_sub = None
    mask_load = None
    mask_gen = None
    mask_storage = None
    mask_line_or = None
    mask_line_ex = None
    mask_shunt = None
    mask_interco = None
    agent_name : str = None
    
    interco_to_subid = None
    interco_to_lineid = None
    interco_to_sub_pos = None
    interco_is_origin = None
    interco_pos_topo_vect = None
    name_interco = None
    n_interco = -1
    
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def _make_cls_dict_extended(cls, res, as_list=True, copy_=True):
        super()._make_cls_dict_extended(cls, res, as_list=as_list, copy_=copy_)
        res["sub_orig_ids"] = cls.sub_orig_ids
        res["local2global"] = cls.local2global
        res["mask_sub"] = cls.mask_sub
        res["mask_load"] = cls.mask_load
        res["mask_gen"] = cls.mask_gen
        res["mask_storage"] = cls.mask_storage
        res["mask_line_or"] = cls.mask_line_or
        res["mask_line_ex"] = cls.mask_line_ex
        res["mask_shunt"] = cls.mask_shunt
        res["agent_name"] = cls.agent_name
