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
    def __init__(self):        
        self.busbar_name = None  # id / name / whatever for each busbars
        self.busbar_to_subid = None  # which busbar belongs to which substation
        
        self.busbar_connectors = None  # for each element that connects busbars, tells which busbars its connect (by id)
    
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
        
        # for each connector, give which busbars it can connect together
        nb_connector = n_sub
        res.busbar_connectors = np.zeros((nb_connector, 2), dtype=dt_int)
        res.busbar_connectors[:,0] = np.arange(n_sub)
        res.busbar_connectors[:,1] = np.arange(n_sub) + n_sub
        
        # for each element (load, gen, etc.)
        # gives the id of the busbar to which this element can be connected thanks to a
        # switches
        res.load_to_busbar_id = [(load_sub, load_sub + n_sub) for load_id, load_sub in enumerate(init_grid.load_to_subid)]
        res.gen_to_busbar_id = [(gen_sub, gen_sub + n_sub) for gen_id, gen_sub in enumerate(init_grid.gen_to_subid)]
        res.line_or_to_busbar_id = [(line_or_sub, line_or_sub + n_sub) for line_or_id, line_or_sub in enumerate(init_grid.line_or_to_subid)]
        res.line_ex_to_busbar_id = [(line_ex_sub, line_ex_sub + n_sub) for line_ex_id, line_ex_sub in enumerate(init_grid.line_ex_to_subid)]
        res.storage_to_busbar_id = [(storage_sub, storage_sub + n_sub) for storage_id, storage_sub in enumerate(init_grid.storage_to_subid)]
        if init_grid.shunts_data_available:
            res.shunt_to_busbar_id = [(shunt_sub, shunt_sub + n_sub) for shunt_id, shunt_sub in enumerate(init_grid.shunt_to_subid)]
        return res
    
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