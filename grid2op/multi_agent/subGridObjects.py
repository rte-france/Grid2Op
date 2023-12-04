# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import pdb
import numpy as np
from grid2op.Exceptions import (EnvError, 
                                IncorrectNumberOfElements, 
                                IncorrectNumberOfGenerators, 
                                IncorrectNumberOfLines, 
                                IncorrectNumberOfLoads, 
                                IncorrectNumberOfStorages)

from grid2op.dtypes import dt_int
from grid2op.Space.GridObjects import GridObjects
from grid2op.Space.space_utils import extract_from_dict, save_to_dict
import re


class SubGridObjects(GridObjects):
    INTERCO_COL = GridObjects.DIM_OBJ_SUB
    DIM_OBJ_SUB = INTERCO_COL + 1

    sub_orig_ids = None
    load_orig_ids = None
    gen_orig_ids = None
    storage_orig_ids = None
    shunt_orig_ids = None
    line_orig_ids = None
    
    mask_sub = None
    mask_load = None
    mask_gen = None
    mask_storage = None
    mask_line = None
    mask_shunt = None
    mask_interco = None
    agent_name : str = None
    mask_orig_pos_topo_vect = None
    
    interco_to_subid = None
    interco_to_lineid = None
    interco_to_sub_pos = None
    interco_is_origin = None
    interco_pos_topo_vect = None
    name_interco = None
    n_interco = -1
    
    def __init__(self):
        GridObjects.__init__(self)
    
    @staticmethod
    def _make_cls_dict_extended(cls, res, as_list=True, copy_=True):
        GridObjects._make_cls_dict_extended(cls, res, as_list=as_list, copy_=copy_)
        res["agent_name"] = str(cls.agent_name)
        
        # interco
        res["n_interco"] = int(cls.n_interco)
        save_to_dict(
            res,
            cls,
            "mask_interco",
            (lambda li: [bool(el) for el in li]) if as_list else None,
            copy_,
        )
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
        
        # original id (from main grid)
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
        
        # masks (extraction of the data from original grid)
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
            "mask_line",
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
        save_to_dict(
            res,
            cls,
            "mask_orig_pos_topo_vect",
            (lambda li: [bool(el) for el in li]) if as_list else None,
            copy_,
        )
    
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
        tmp_ = cls._get_obj_substations_gridobjects(_sentinel, substation_id)
        dict_ = cls.get_obj_connect_to(_sentinel, substation_id)
        res = np.full((dict_["nb_elements"], SubGridObjects.DIM_OBJ_SUB), fill_value=-1, dtype=dt_int)
        res[:, :GridObjects.DIM_OBJ_SUB] = tmp_
        res[cls.interco_to_sub_pos[dict_["intercos_id"]], cls.INTERCO_COL] = dict_[
            "intercos_id"
        ]
        return res
    
    @classmethod
    def init_grid(cls, gridobj, force=False, extra_name=None, force_module=None):
        extr_nms = []
        if gridobj.agent_name is not None:
            extr_nms .append(f"{gridobj.agent_name}")
        if extra_name is not None:
            extr_nms.append(f"{extra_name}")
            
        if extr_nms:
            extr_nms = "_".join(extr_nms)
        else:
            extr_nms = None
        return cls.init_grid_gridobject(gridobj, force, extr_nms, force_module)
    
    @classmethod
    def _compute_nb_element(cls) -> int:
        return cls.n_load + cls.n_gen + 2 * cls.n_line + cls.n_storage + cls.n_interco
    
    @classmethod
    def get_obj_connect_to(cls, _sentinel=None, substation_id=None):
        res = cls._get_obj_connect_to_gridobjects(_sentinel, substation_id)
        res["intercos_id"] = np.where(cls.interco_to_subid == substation_id)[0]
        return res
    
    @classmethod
    def _nb_obj_per_sub(cls) -> int:
        obj_per_sub = cls._nb_obj_per_sub_gridobjects() 
        for sub_id in cls.interco_to_subid:
            obj_per_sub[sub_id] += 1
        return obj_per_sub
    
    @classmethod
    def _concat_topo_vect(cls):
        # used in assert_grid_correct_cls to check that the topo_vect
        # is consistent
        res = cls._concat_topo_vect_gridobjects()
        res = np.concatenate((res, cls.interco_pos_topo_vect.flatten()))
        return res
    
    @classmethod
    def _compute_pos_big_topo_cls(cls):
        cls._compute_pos_big_topo_gridobjects()
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
        if cls.n_load > 0:
            cls._check_load_size_gridobjects()
        
    @classmethod
    def _check_gen_size(cls):
        # in case of subgrid, there can be "no gen"
        # in the subgrid, and following checks fails 
        # usually they are performed in _check_sub_id
        if cls.n_gen > 0:
            cls._check_gen_size_gridobjects()
    
    @classmethod
    def _check_for_lines(cls):
        # in case of subgrid, there can be "no line"
        # in the subgrid, and following checks fails 
        # usually they are performed in assert_grid_correct_cls
        if cls.n_line + cls.n_interco <= 0:
            raise EnvError(
                "There should be at least one line or one interco for each "
                "zone of your grid."
            )
    
    @classmethod
    def _check_powerline_size(cls):
        # in case of subgrid, there can be "no line"
        # in the subgrid, and following checks fails 
        # usually they are performed in _check_sub_id
        # there is no line if there are only interconnections !
        if cls.n_line > 0:
            cls._check_powerline_size_gridobjects()
        
    @classmethod
    def _check_sub_id_other_elements(cls):
        # if you added some elements in a subgrid for example
        # this is the place to perform the appropriate checkings
        # for this class, we added the interco, so we check here that everything works
        if len(cls.interco_to_subid) != cls.n_interco:
            raise IncorrectNumberOfElements("Incorrect number of interconnections")

        if cls.n_interco > 0:
            if np.min(cls.interco_to_subid) < 0:
                raise EnvError("Some interco is connected to a negative substation id.")
            if np.max(cls.interco_to_subid) > cls.n_sub:
                raise EnvError(
                    "Some interco is supposed to be connected to substations with id {} which"
                    "is greater than the number of substations of the grid, which is {}."
                    "".format(np.max(cls.interco_to_subid), cls.n_sub)
                )
    
    @classmethod
    def _assert_grid_correct_other_elements(cls):
        # some checks for the interconnections
        if len(cls.name_interco) != cls.n_interco:
            raise EnvError("len(self.name_sub) != self.n_sub")
        
        if len(cls.interco_to_sub_pos) != cls.n_interco:
            raise EnvError("len(self.interco_to_sub_pos) != self.n_load")
        
        if len(cls.interco_pos_topo_vect) != cls.n_interco:
            raise EnvError(
                "len(self.interco_pos_topo_vect) != self.n_interco"
            )
            
        for i, (sub_id, sub_pos) in enumerate(
            zip(cls.interco_to_subid, cls.interco_to_sub_pos)
        ):
            if sub_pos >= cls.sub_info[sub_id]:
                raise EnvError("for interco {}".format(i))
            
        interco_pos_big_topo = cls._aux_pos_big_topo(
            cls.interco_to_subid, cls.interco_to_sub_pos
        )
        if not np.all(interco_pos_big_topo == cls.interco_pos_topo_vect):
            raise EnvError(
                "Mismatch between interco_to_subid, "
                "interco_to_sub_pos and interco_pos_topo_vect"
            )     
            
        # pos topo vect
        if cls.mask_orig_pos_topo_vect is None:
            raise EnvError("mask_orig_pos_topo_vect should not be None")
        if cls.mask_orig_pos_topo_vect.sum() != cls.dim_topo:
            raise EnvError("mask_orig_pos_topo_vect counts more active component than the "
                           "number of elements on the subgrid")
    
    # TODO BEN: use the "type(name_res, (cls,), cls_attr_as_dict)" 
    # syntax in the function below
    @classmethod
    def make_local(cls, BaseClass):
        class NewClass(SubGridObjects, BaseClass):
            def __init__(self, *args ,**kwargs):
                SubGridObjects.__init__(self)
                BaseClass.__init__(self, *args, **kwargs)
                
        tmp_cls = NewClass.init_grid(cls)
        tmp_cls.__name__ = re.sub("NewClass", BaseClass.__name__, tmp_cls.__name__)
        return tmp_cls

    # TODO BEN: later             
    @staticmethod
    def from_dict(dict_):
        return GridObjects.from_dict(dict_)
        #
        #class res(SubGridObjects):
        #    pass

        #cls = res
        #cls.sub_info = extract_from_dict(
        #    dict_, "sub_info", lambda x: np.array(x).astype(dt_int)
        #)
    
    # TODO BEN: later   
    @staticmethod
    def init_grid_from_dict_for_pickle(name_res, orig_cls, cls_attr):
        return GridObjects.init_grid_from_dict_for_pickle(name_res, orig_cls, cls_attr)
    
    # TODO BEN: later   
    @classmethod
    def _get_full_cls_str(cls):
        return GridObjects._get_full_cls_str(cls)
    
    # TODO BEN: later   
    @classmethod
    def _clear_class_attribute(cls):
        cls._clear_class_attribute_gridobjects()
        
        cls.INTERCO_COL = GridObjects.DIM_OBJ_SUB
        cls.DIM_OBJ_SUB = cls.INTERCO_COL + 1

        cls.sub_orig_ids = None
        cls.load_orig_ids = None
        cls.gen_orig_ids = None
        cls.storage_orig_ids = None
        cls.shunt_orig_ids = None
        cls.line_orig_ids = None

        cls.mask_sub = None
        cls.mask_load = None
        cls.mask_gen = None
        cls.mask_storage = None
        cls.mask_line = None
        cls.mask_shunt = None
        cls.mask_interco = None
        cls.mask_orig_pos_topo_vect = None
        cls.agent_name : str = None

        cls.interco_to_subid = None
        cls.interco_to_lineid = None
        cls.interco_to_sub_pos = None
        cls.interco_is_origin = None
        cls.interco_pos_topo_vect = None
        cls.name_interco = None
        cls.n_interco = -1
