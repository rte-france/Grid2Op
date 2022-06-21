# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Space.GridObjects import GridObjects
from grid2op.Space.space_utils import extract_from_dict, save_to_dict


class SubGridObjects(GridObjects) :

    sub_orig_ids = None
    load_orig_ids = None
    gen_orig_ids = None
    storage_orig_ids = None
    shunt_orig_ids = None
    line_orig_ids = None
    
    #local2global = dict()
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
        
        res["n_interco"] = cls.n_interco
        res["agent_name"] = str(cls.agent_name)
        
        save_to_dict(
            res,
            cls,
            "name_interco",
            (lambda arr: [str(el) for el in arr]) if as_list else None,
            copy_,
        )
        
        save_to_dict(
            res,
            cls,
            "interco_pos_topo_vect",
            (lambda arr: [int(el) for el in arr]) if as_list else None,
            copy_,
        )
        
        save_to_dict(
            res,
            cls,
            "interco_is_origin",
            (lambda arr: [bool(el) for el in arr]) if as_list else None,
            copy_,
        )
        
        save_to_dict(
            res,
            cls,
            "interco_to_sub_pos",
            (lambda arr: [int(el) for el in arr]) if as_list else None,
            copy_,
        )
        
        save_to_dict(
            res,
            cls,
            "interco_to_lineid",
            (lambda arr: [int(el) for el in arr]) if as_list else None,
            copy_,
        )
        
        save_to_dict(
            res,
            cls,
            "interco_to_subid",
            (lambda arr: [int(el) for el in arr]) if as_list else None,
            copy_,
        )
        
        save_to_dict(
            res,
            cls,
            "sub_orig_ids",
            (lambda arr: [int(el) for el in arr]) if as_list else None,
            copy_,
        )
        
        save_to_dict(
            res,
            cls,
            "load_orig_ids",
            (lambda arr: [int(el) for el in arr]) if as_list else None,
            copy_,
        )
        
        save_to_dict(
            res,
            cls,
            "gen_orig_ids",
            (lambda arr: [int(el) for el in arr]) if as_list else None,
            copy_,
        )
        
        save_to_dict(
            res,
            cls,
            "line_orig_ids",
            (lambda arr: [int(el) for el in arr]) if as_list else None,
            copy_,
        )
        
        save_to_dict(
            res,
            cls,
            "storage_orig_ids",
            (lambda arr: [int(el) for el in arr]) if as_list else None,
            copy_,
        )
        
        save_to_dict(
            res,
            cls,
            "shunt_orig_ids",
            (lambda arr: [int(el) for el in arr]) if as_list else None,
            copy_,
        )
        
        save_to_dict(
            res,
            cls,
            "mask_sub",
            (lambda arr: [bool(el) for el in arr]) if as_list else None,
            copy_,
        )
        
        save_to_dict(
            res,
            cls,
            "mask_load",
            (lambda arr: [bool(el) for el in arr]) if as_list else None,
            copy_,
        )
        
        save_to_dict(
            res,
            cls,
            "mask_gen",
            (lambda arr: [bool(el) for el in arr]) if as_list else None,
            copy_,
        )
        
        save_to_dict(
            res,
            cls,
            "mask_storage",
            (lambda arr: [bool(el) for el in arr]) if as_list else None,
            copy_,
        )
        
        save_to_dict(
            res,
            cls,
            "mask_line_or",
            (lambda arr: [bool(el) for el in arr]) if as_list else None,
            copy_,
        )
        
        save_to_dict(
            res,
            cls,
            "mask_line_ex",
            (lambda arr: [bool(el) for el in arr]) if as_list else None,
            copy_,
        )
        
        save_to_dict(
            res,
            cls,
            "mask_shunt",
            (lambda arr: [bool(el) for el in arr]) if as_list else None,
            copy_,
        )
        
    # TODO BEN (not later)
    @staticmethod
    def from_dict(dict_):
        GridObjects.from_dict(dict_)
        
        class res(SubGridObjects):
            pass

        cls = res
        #cls.sub_info = extract_from_dict(
        #    dict_, "sub_info", lambda x: np.array(x).astype(dt_int)
        #)
        
    # TODO BEN (later)
    @staticmethod
    def init_grid_from_dict_for_pickle(name_res, orig_cls, cls_attr):
        GridObjects.init_grid_from_dict_for_pickle(name_res, orig_cls, cls_attr)
    
    # TODO BEN (later)
    @classmethod
    def _get_full_cls_str(cls):
        GridObjects._get_full_cls_str(cls)
