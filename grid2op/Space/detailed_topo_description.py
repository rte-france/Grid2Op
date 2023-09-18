# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.


import numpy as np

from grid2op.dtypes import dt_int, dt_bool

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
        # col 0 gives the substation id
        # col 1 gives the object type it connects (0 = LOAD, etc.)
        # col 2 gives the ID of the object it connects (number between 0 and n_load-1 if previous column is 0 for example)
        # col 3 gives the busbar id that this switch connects its element to
        
        # for each switches says which element in the "topo_vect" it concerns [-1 for shunt]
        self.switches_to_topovect_id = None
        self.switches_to_shunt_id = None
        
        # whether the switches connects an element represented in the topo_vect vector  (unused atm)
        self.in_topo_vect = None
        
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
        res.switches_to_topovect_id = np.zeros(np.sum(sub_info) * 2, dtype=dt_int) - 1
        if init_grid.shunts_data_available:
            res.switches_to_shunt_id = np.zeros(np.sum(sub_info) * 2, dtype=dt_int) - 1
        # res.in_topo_vect = np.zeros(np.sum(sub_info), dtype=dt_int)
        
        arrs_subid = [init_grid.load_to_subid,
                      init_grid.gen_to_subid,
                      init_grid.line_or_to_subid,
                      init_grid.line_ex_to_subid,
                      init_grid.storage_to_subid,
                      ]
        ars2 = [init_grid.load_pos_topo_vect,
               init_grid.gen_pos_topo_vect,
               init_grid.line_or_pos_topo_vect,
               init_grid.line_ex_pos_topo_vect,
               init_grid.storage_pos_topo_vect,
               ]
        ids = [cls.LOAD_ID, cls.GEN_ID, cls.LINE_OR_ID, cls.LINE_EX_ID, cls.STORAGE_ID]
        if init_grid.shunts_data_available:
            arrs_subid.append(init_grid.shunt_to_subid)
            ars2.append(np.array([-1] * init_grid.n_shunt))
            ids.append(cls.SHUNT_ID)
        prev_el = 0
        # prev_el1 = 0
        for sub_id in range(n_sub):    
            for arr_subid, pos_topo_vect, obj_col in zip(arrs_subid, ars2, ids):
                nb_el = (arr_subid == sub_id).sum()
                where_el = np.where(arr_subid == sub_id)[0]
                res.switches[prev_el : (prev_el + 2 * nb_el), cls.OBJ_TYPE_COL] = obj_col
                res.switches[prev_el : (prev_el + 2 * nb_el), cls.OBJ_ID_COL] = np.repeat(where_el, 2)
                res.switches[prev_el : (prev_el + 2 * nb_el), cls.BUSBAR_ID_COL] = np.tile(np.array([1, 2]), nb_el)
                res.switches_to_topovect_id[prev_el : (prev_el + 2 * nb_el)] = np.repeat(pos_topo_vect[arr_subid == sub_id], 2)
                if init_grid.shunts_data_available and obj_col == cls.SHUNT_ID:
                    res.switches_to_shunt_id[prev_el : (prev_el + 2 * nb_el)] = np.repeat(where_el, 2)
                    
                # if obj_col != cls.SHUNT_ID:
                #     # object is modeled in topo_vect
                #     res.in_topo_vect[prev_el1 : (prev_el1 + nb_el)] = 1
                prev_el += 2 * nb_el
                # prev_el1 += nb_el
        
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
        switches_state = np.zeros(self.switches.shape[0], dtype=dt_bool)
        busbar_connectors_state = np.zeros(self.busbar_connectors.shape[0], dtype=dt_bool)  # we can always say they are opened 
        
        # compute the position for the switches of the "topo_vect" elements 
        # only work for current grid2op modelling !
        
        # TODO detailed topo vectorize this ! (or cython maybe ?)
        for el_id, bus_id in enumerate(topo_vect):
            mask_el = self.switches_to_topovect_id == el_id
            if mask_el.any():
                # it's a regular element
                if bus_id == 1:
                    mask_el[np.where(mask_el)[0][1]] = False  # I open the switch to busbar 2 in this case
                    switches_state[mask_el] = True
                elif bus_id == 2:
                    mask_el[np.where(mask_el)[0][0]] = False  # I open the switch to busbar 1 in this case
                    switches_state[mask_el] = True
                    
        if self.switches_to_shunt_id is not None:
            # read the switches associated with the shunts
            for el_id, bus_id in enumerate(shunt_bus):
                # it's an element not in the topo_vect (for now only switches)
                mask_el = self.switches_to_shunt_id == el_id
                if mask_el.any():
                    # it's a shunt
                    if bus_id == 1:
                        mask_el[np.where(mask_el)[0][1]] = False  # I open the switch to busbar 2 in this case
                        switches_state[mask_el] = True
                    elif bus_id == 2:
                        mask_el[np.where(mask_el)[0][0]] = False  # I open the switch to busbar 1 in this case
                        switches_state[mask_el] = True
        return busbar_connectors_state, switches_state
    
    def from_switches_position(self):
        # TODO detailed topo
        # opposite of `compute_switches_position`
        topo_vect = None
        shunt_bus = None
        return topo_vect, shunt_bus
        
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
        save_to_dict(
            res,
            self,
            "switches_to_topovect_id",
            (lambda arr: [int(el) for el in arr]) if as_list else lambda arr: arr.flatten(),
            copy_,
        )
        if self.switches_to_topovect_id is not None:
            save_to_dict(
                res,
                self,
                "switches_to_shunt_id",
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
        
        res.switches_to_topovect_id = extract_from_dict(
            dict_, "switches_to_topovect_id", lambda x: np.array(x).astype(dt_int)
        )
        
        if "switches_to_shunt_id" in dict_:
            res.switches_to_shunt_id = extract_from_dict(
                dict_, "switches_to_shunt_id", lambda x: np.array(x).astype(dt_int)
            )
        else:
            # shunts are not supported
            res.switches_to_shunt_id = None
        
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