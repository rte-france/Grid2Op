# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.


import numpy as np

from grid2op.dtypes import dt_int

from grid2op.Space.space_utils import extract_from_dict, save_to_dict


class DetailedTopoDescription(object):
    """This class represent the detail description of the 
    switches in the grid.
    
    It does not say whether switches / breakers / etc. are opened or closed
    just that it exists a switch between this and this
    
    It is const, should be initialized by the backend and never modified afterwards.
    
    It is a const member of the class (not the object, the class !)
    """
    SUB_COL = 0
    OBJ_TYPE_COL = 1
    OBJ_ID_COL = 2
    BUSBAR_ID_COL = 3
    
    LOAD_ID = 0
    GEN_ID = 1
    STORAGE_ID = 2
    LINE_OR_ID = 3
    LINE_EX_ID = 4
    SHUNT_ID = 5
    
    def __init__(self):        
        self.busbar_name = None  # id / name / whatever for each busbars
        self.busbar_to_subid = None  # which busbar belongs to which substation
        
        self.busbar_connectors = None  # for each element that connects busbars, tells which busbars its connect (by id)
    
        self.switches = None  # a matrix of 'n_switches' rows and 4 columns
        # col 0 give the substation id
        # col 1 give the object type it connects (0 = LOAD, etc.)
    
        self.load_to_busbar_id = None  # for each loads, you have a tuple of busbars to which it can be connected
        self.gen_to_busbar_id = None
        self.line_or_to_busbar_id = None
        self.line_ex_to_busbar_id = None
        self.storage_to_busbar_id = None
        self.shunt_to_busbar_id = None
    
    @classmethod
    def from_init_grid(cls, init_grid):
        """For now, suppose that the grid comes from ieee"""
        n_sub = init_grid.n_sub
        
        res = cls()
        res.busbar_name = np.array([f"busbar_{i}" for i in range(2 * init_grid.n_sub)])
        res.busbar_to_subid = np.arange(n_sub) % init_grid.n_sub  
        
        # in current environment, there are 2 busbars per substations, 
        # and 1 connector allows to connect both of them
        nb_connector = n_sub
        res.busbar_connectors = np.zeros((nb_connector, 2), dtype=dt_int)
        res.busbar_connectors[:,0] = np.arange(n_sub)
        res.busbar_connectors[:,1] = np.arange(n_sub) + n_sub
        
        # for each element (load, gen, etc.)
        # gives the id of the busbar to which this element can be connected thanks to a
        # switches
        # in current grid2op environment, there are 2 switches for each element
        # one that connects it to busbar 1
        # another one that connects it to busbar 2
        n_shunt = init_grid.n_shunt if init_grid.shunts_data_available else 0
        res.switches = np.zeros((2*(init_grid.dim_topo + n_shunt), 4), dtype=dt_int)
        # add the shunts (considered as element here !)
        sub_info = 1 * init_grid.sub_info
        if init_grid.shunts_data_available:
            for sub_id in init_grid.shunt_to_subid:
                sub_info[sub_id] += 1
        # now fill the switches: 2 switches per element, everything stored in the res.switches matrix
        res.switches[:, cls.SUB_COL] = np.repeat(np.arange(n_sub), 2 * sub_info)
        ars = [init_grid.load_to_subid,
               init_grid.gen_to_subid,
               init_grid.line_or_to_subid,
               init_grid.line_ex_to_subid,
               init_grid.storage_to_subid,
               ]
        ids = [cls.LOAD_ID, cls.GEN_ID, cls.LINE_OR_ID, cls.LINE_EX_ID, cls.STORAGE_ID]
        if init_grid.shunts_data_available:
            ars.append(init_grid.shunt_to_subid)
            ids.append(cls.SHUNT_ID)
        prev_el = 0
        for sub_id in range(n_sub):    
            for arr, obj_col in zip(ars, ids):
                nb_el = (arr == sub_id).sum()
                res.switches[prev_el : (prev_el + 2 * nb_el), cls.OBJ_TYPE_COL] = obj_col
                res.switches[prev_el : (prev_el + 2 * nb_el), cls.OBJ_ID_COL] = np.repeat(np.where(arr == sub_id)[0], 2)
                res.switches[prev_el : (prev_el + 2 * nb_el), cls.BUSBAR_ID_COL] = np.tile(np.array([1, 2]), nb_el)
                prev_el += 2 * nb_el
        
        # and also fill some extra information
        res.load_to_busbar_id = [(load_sub, load_sub + n_sub) for load_id, load_sub in enumerate(init_grid.load_to_subid)]
        res.gen_to_busbar_id = [(gen_sub, gen_sub + n_sub) for gen_id, gen_sub in enumerate(init_grid.gen_to_subid)]
        res.line_or_to_busbar_id = [(line_or_sub, line_or_sub + n_sub) for line_or_id, line_or_sub in enumerate(init_grid.line_or_to_subid)]
        res.line_ex_to_busbar_id = [(line_ex_sub, line_ex_sub + n_sub) for line_ex_id, line_ex_sub in enumerate(init_grid.line_ex_to_subid)]
        res.storage_to_busbar_id = [(storage_sub, storage_sub + n_sub) for storage_id, storage_sub in enumerate(init_grid.storage_to_subid)]
        if init_grid.shunts_data_available:
            res.shunt_to_busbar_id = [(shunt_sub, shunt_sub + n_sub) for shunt_id, shunt_sub in enumerate(init_grid.shunt_to_subid)]
        return res
    
    def compute_switches_position(self, topo_vect, shunt_bus):
        # TODO detailed topo
        # TODO in reality, for more complex environment, this requires a routine to compute it
        # but for now in grid2op as only ficitive grid are modeled then 
        # this is not a problem
        switches_state = np.zeros(self.switches.shape[0], dtype=bool)
        busbar_connectors_state = np.zeros(self.busbar_connectors.shape[0], dtype=bool)
        
        return switches_state
        
    def check_validity(self):
        # TODO detailed topo
        pass
    
    def save_to_dict(self, res, as_list=True, copy_=True):
        # TODO detailed topo
        save_to_dict(
            res,
            self,
            "busbar_name",
            (lambda arr: [str(el) for el in arr]) if as_list else None,
            copy_,
        )
        save_to_dict(
            res,
            self,
            "busbar_to_subid",
            (lambda arr: [int(el) for el in arr]) if as_list else None,
            copy_,
        )
        save_to_dict(
            res,
            self,
            "busbar_connectors",
            (lambda arr: [int(el) for el in arr]) if as_list else lambda arr: arr.flatten(),
            copy_,
        )
        save_to_dict(
            res,
            self,
            "switches",
            (lambda arr: [int(el) for el in arr]) if as_list else lambda arr: arr.flatten(),
            copy_,
        )
        
        # for the switches per element
        save_to_dict(
            res,
            self,
            "load_to_busbar_id",
            lambda arr: [(int(el1), int(el2)) for el1, el2 in arr],
            copy_,
        )
        save_to_dict(
            res,
            self,
            "gen_to_busbar_id",
            lambda arr: [(int(el1), int(el2)) for el1, el2 in arr],
            copy_,
        )
        save_to_dict(
            res,
            self,
            "line_or_to_busbar_id",
            lambda arr: [(int(el1), int(el2)) for el1, el2 in arr],
            copy_,
        )
        save_to_dict(
            res,
            self,
            "line_ex_to_busbar_id",
            lambda arr: [(int(el1), int(el2)) for el1, el2 in arr],
            copy_,
        )
        save_to_dict(
            res,
            self,
            "storage_to_busbar_id",
            lambda arr: [(int(el1), int(el2)) for el1, el2 in arr],
            copy_,
        )
        if self.shunt_to_busbar_id is not None:
            save_to_dict(
                res,
                self,
                "shunt_to_busbar_id",
                lambda arr: [(int(el1), int(el2)) for el1, el2 in arr],
                copy_,
            )
        # TODO detailed topo
        
    @classmethod
    def from_dict(cls, dict_):
        """
        INTERNAL

        .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
            This is used internally only to build the "DetailedTopoDescription"
            when the classes are serialized. You should not modify this under any circumstances.
            
        """ 
        res = cls()
        
        res.busbar_name = extract_from_dict(
            dict_, "busbar_name", lambda x: np.array(x).astype(str)
        )
        res.busbar_to_subid = extract_from_dict(
            dict_, "busbar_to_subid", lambda x: np.array(x).astype(dt_int)
        )
        res.busbar_connectors = extract_from_dict(
            dict_, "busbar_connectors", lambda x: np.array(x).astype(dt_int)
        )
        res.busbar_connectors = res.busbar_connectors.reshape((-1, 2))
        
        res.switches = extract_from_dict(
            dict_, "switches", lambda x: np.array(x).astype(dt_int)
        )
        res.switches = res.switches.reshape((-1, 4))
        
        
        res.load_to_busbar_id = extract_from_dict(
            dict_, "load_to_busbar_id", lambda x: x
        )
        res.gen_to_busbar_id = extract_from_dict(
            dict_, "gen_to_busbar_id", lambda x: x
        )
        res.line_or_to_busbar_id = extract_from_dict(
            dict_, "line_or_to_busbar_id", lambda x: x
        )
        res.line_ex_to_busbar_id = extract_from_dict(
            dict_, "line_ex_to_busbar_id", lambda x: x
        )
        res.storage_to_busbar_id = extract_from_dict(
            dict_, "storage_to_busbar_id", lambda x: x
        )
        if "shunt_to_busbar_id" in dict_:
            res.shunt_to_busbar_id = extract_from_dict(
                dict_, "shunt_to_busbar_id", lambda x: x
            )
        
        # TODO detailed topo
        return res