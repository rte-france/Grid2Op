# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np

from grid2op.dtypes import dt_int
from grid2op.Exceptions import Grid2OpException
from grid2op.Space.GridObjects import GridObjects
from grid2op.Space.space_utils import extract_from_dict, save_to_dict


class SubGridObjects(GridObjects):
    INTERCO_COL = GridObjects.DIM_OBJ_SUB
    DIM_OBJ_SUB = INTERCO_COL + 1

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
            (lambda li: [bool(el) for el in li]) if as_list else None,
            copy_,
        )
        res["agent_name"] = cls.agent_name
    
    @classmethod
    def get_obj_substations(cls, _sentinel=None, substation_id=None):
        """
        Same as for the base class but adds a columns:
        Returns
        -------
        res: ``numpy.ndarray``
            A matrix with as many rows as the number of element of the substation and 6 columns:

              1. column 0: the id of the substation
              2. column 1: -1 if this object is not a load, or `LOAD_ID` if this object is a load (see example)
              3. column 2: -1 if this object is not a generator, or `GEN_ID` if this object is a generator (see example)
              4. column 3: -1 if this object is not the origin end of a line, or `LOR_ID` if this object is the
                 origin end of a powerline(see example)
              5. column 4: -1 if this object is not a extremity end, or `LEX_ID` if this object is the extremity
                 end of a powerline
              6. column 5: -1 if this object is not a storage unit, or `STO_ID` if this object is one
              7. column 6: -1 if this object is not an "interco" , or `INTERCO_ID` if this object is one [ADDED]

        """
        tmp_ = cls._inheritance_get_obj_substations(_sentinel, substation_id)
        dict_ = cls.get_obj_connect_to(_sentinel, substation_id)
        res = np.full((dict_["nb_elements"], 7), fill_value=-1, dtype=dt_int)
        res[:, :GridObjects.DIM_OBJ_SUB] = tmp_
        res[cls.interco_to_sub_pos[dict_["intercos_id"]], cls.INTERCO_COL] = dict_[
            "intercos_id"
        ]
        return res
    
    @classmethod
    def _compute_nb_element(cls) -> int:
        return cls.n_load + cls.n_gen + 2 * cls.n_line + cls.n_storage + cls.n_interco
    
    @classmethod
    def get_obj_connect_to(cls, _sentinel=None, substation_id=None):
        res = cls._inheritance_get_obj_connect_to(_sentinel, substation_id)
        res["intercos_id"] = np.where(cls.interco_to_subid == substation_id)[0]
        return res
    
    @classmethod
    def _nb_obj_per_sub(cls) -> int:
        obj_per_sub = cls._inheritance_nb_obj_per_sub() 
        for sub_id in cls.interco_to_subid:
            obj_per_sub[sub_id] += 1
        return obj_per_sub
    
    @classmethod
    def _concat_topo_vect(cls):
        # used in assert_grid_correct_cls to check that the topo_vect
        # is consistent
        res = cls._inheritance_concat_topo_vect()
        res = np.concatenate((res, cls.interco_pos_topo_vect.flatten()))
        return res
    
    @classmethod
    def _compute_pos_big_topo_cls(cls):
        cls._inheritance_compute_pos_big_topo()
        cls.interco_pos_topo_vect = cls._aux_pos_big_topo(
            cls.interco_to_subid, cls.interco_to_sub_pos
        ).astype(dt_int)
        
    @classmethod
    def _check_for_gen_loads(cls):
        # real complete powergrid should have generators and loads
        # but that's not the case for "subgrid" (see SubGridObjects) => they can "NOT HAVE" 
        # generators and that's fine
        pass
    
    @classmethod
    def _check_load_size(cls):
        # in case of subgrid, there can be "no load"
        # in the subgrid, and following checks fails 
        # usually they are performed in _check_sub_id
        pass
        
    @classmethod
    def _check_gen_size(cls):
        # in case of subgrid, there can be "no gen"
        # in the subgrid, and following checks fails 
        # usually they are performed in _check_sub_id
        pass
    
    # TODO BEN (not later)
    @staticmethod
    def from_dict(dict_):
        GridObjects.from_dict(dict_)
        #
        #class res(SubGridObjects):
        #    pass

        #cls = res
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
    
    # TODO BEN (later)
    @classmethod
    def _clear_class_attribute(cls):
        GridObjects._clear_class_attribute(cls)
        