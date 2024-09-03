# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from typing import Optional
import numpy as np

from grid2op.dtypes import dt_int, dt_bool

from grid2op.Space.space_utils import extract_from_dict, save_to_dict


class DetailedTopoDescription(object):
    """This class represent the detail description of the 
    switches in the grid. It allows to use new types of actions (`act.set_switches = ..` # TODO detailed topo)
    and to get some extra information in the observation (`obs.switches_state` # TODO detailed topo).
    
    This class only stores the existence of switches. It just informs
    the user that "just that it exists a switch between this and this". It does 
    not say whether switches / breakers / etc. are opened or closed (for that you need to have 
    a look at the observation) and it does not allow to modify the switches state (for that you
    need to use the action).
    
    If set, it is "const" / "read only" / immutable.
    It should be initialized by the backend and never modified afterwards.
    
    It is a const member of the main grid2op classes (not the object, the class !), just like the `n_sub` or 
    `lines_or_pos_topo_vect` property for example.
    
    In order to fill a :class:`DetailedTopoDescription` you need to fill the 
    following attribute:
    
    - :attr:`DetailedTopoDescription.conn_node_name`: 
    - :attr:`DetailedTopoDescription.conn_node_to_subid`
    - (deprecated) :attr:`DetailedTopoDescription.conn_node_connectors`
    - :attr:`DetailedTopoDescription.switches`
    - :attr:`DetailedTopoDescription.switches_to_topovect_id`
    - :attr:`DetailedTopoDescription.switches_to_shunt_id` 
    - :attr:`DetailedTopoDescription.load_to_conn_node_id` 
    - :attr:`DetailedTopoDescription.gen_to_conn_node_id`
    - :attr:`DetailedTopoDescription.line_or_to_conn_node_id`
    - :attr:`DetailedTopoDescription.line_ex_to_conn_node_id`
    - :attr:`DetailedTopoDescription.storage_to_conn_node_id`
    - :attr:`DetailedTopoDescription.shunt_to_conn_node_id`

    To create a "detailed description of the swtiches", somewhere in the implementation of your
    backend you have a piece of code looking like:
    
    .. code-block:: python
    
        import os
        from grid2op.Backend import Backend
        from typing import Optional, Union, Tuple
        
        class MyBackend(Backend):
            # some implementation of other methods...
            
            def load_grid(self,
                          path : Union[os.PathLike, str],
                          filename : Optional[Union[os.PathLike, str]]=None) -> None:
                # do the regular implementation of the load_grid function
                ...
                ...
                
                # once done, then you can create a detailed topology
                detailed_topo_desc = DetailedTopoDescription()
                
                # you fill it with the data in the grid you read
                # (at this stage you tell grid2op what the grid is made of)
                detailed_topo_desc.conn_node_name = ...
                detailed_topo_desc.conn_node_to_subid = ...
                # (deprecated) detailed_topo_desc.conn_node_connectors = ...
                detailed_topo_desc.switches = ...
                detailed_topo_desc.switches_to_topovect_id = ...
                detailed_topo_desc.switches_to_shunt_id = ...
                detailed_topo_desc.load_to_conn_node_id = ...
                detailed_topo_desc.gen_to_conn_node_id = ...
                detailed_topo_desc.line_or_to_conn_node_id = ...
                detailed_topo_desc.line_ex_to_conn_node_id = ...
                detailed_topo_desc.storage_to_conn_node_id = ...
                detailed_topo_desc.shunt_to_conn_node_id = ...
                
                # and then you assign it as a member of this class
                self.detailed_topo_desc =  detailed_topo_desc
        
            # some other implementation of other methods

    Examples
    --------
    
    Unfortunately, most of the ieee grid (used in released grid2op environments) does not
    come with a detailed description of the topology. They only describe the "nodal" topology (buses)
    and not how things are wired together with switches.
    
    If you want to use this feature with released grid2op environment, 
    you can create a new backend class, and use it to create a new environment like this:
    
    .. code-block:: python
    
        import grid2op
        from grid2op.Space import AddDetailedTopoIEEE
        from grid2op.Backend import PandaPowerBackend  # or any other backend (*eg* lightsim2grid)
        
        class PandaPowerBackendWithDetailedTopo(AddDetailedTopoIEEE, PandaPowerBackend):
            pass
        
        env_name = "l2rpn_case14_sandbox"
        env = grid2op.make(env_name, backend=PandaPowerBackendWithDetailedTopo())
        # do wathever you want

        
    """
    #: In the :attr:`DetailedTopoDescription.switches` table, tells that column 0
    #: concerns the substation
    SUB_COL = 0 
    
    #: In the :attr:`DetailedTopoDescription.switches` table, tells that column 1
    #: concerns the type of object (0 for Load, see the `xxx_ID` (*eg* :attr:`DetailedTopoDescription.LOAD_ID`))
    OBJ_TYPE_COL = 1
    
    #: In the :attr:`DetailedTopoDescription.switches` table, tells that column 2
    #: concerns the id of object that this switches connects / disconnects
    OBJ_ID_COL = 2
    
    #: In the :attr:`DetailedTopoDescription.switches` table, tells that column 2
    #: concerns the id of the connection node that this switches connects / disconnects
    CONN_NODE_ID_COL = 3
    
    #: In the :attr:`DetailedTopoDescription.switches` table, column 2
    #: if a 0 is present, then this switch will connect a load to a connection node
    LOAD_ID = 0
    
    #: In the :attr:`DetailedTopoDescription.switches` table, column 2
    #: if a 1 is present, then this switch will connect a generator to a connection node
    GEN_ID = 1
    
    #: In the :attr:`DetailedTopoDescription.switches` table, column 2
    #: if a 2 is present, then this switch will connect a storage unit to a connection node
    STORAGE_ID = 2
    
    #: In the :attr:`DetailedTopoDescription.switches` table, column 2
    #: if a 3 is present, then this switch will connect a line (origin side) to a connection node
    LINE_OR_ID = 3
    
    #: In the :attr:`DetailedTopoDescription.switches` table, column 2
    #: if a 4 is present, then this switch will connect a line (extremity side) to a connection node
    LINE_EX_ID = 4
    
    #: In the :attr:`DetailedTopoDescription.switches` table, column 2
    #: if a 5 is present, then this switch will connect a shunt to a connection node
    SHUNT_ID = 5
    
    #: In the :attr:`DetailedTopoDescription.switches` table, column 2
    #: if a 5 is present, then this switch will connect a standard "connection node"
    #: to another connection node. There isn't anything special about any
    #: of the "connection node".
    OTHER = 6
    
    def __init__(self):        
        #: vector of string that has the size of the number of connection nodes on your grid
        #: and for each connection node it gives... its name
        self.conn_node_name = None
        
        #: vector of int that has the size of the number of connection nodes on
        #: your grid and for each connection node it gives the substation id [0...n_sub] to which
        #: the connection node belongs to.
        self.conn_node_to_subid = None
        
        # #: A matrix representing the "switches" between the connection nodes.
        # #: It counts 2 columns and as many rows as the number of "switches" between
        # #: the connection nodes. And for each "connection node switches" it gives the id of the
        # #: connection nodes it can connect / disconnect.
        # self.conn_node_connectors = None 
    
        #: It is a matrix describing each switches. This matrix has 'n_switches' rows and 4 columns. 
        #: Each column provides an information about the switch:
        #:     
        #:     - col 0 gives the substation id
        #:     - col 1 gives the object type it connects (0 = LOAD, etc.) see :attr:`DetailedTopoDescription.LOAD_ID`, 
        #:       :attr:`DetailedTopoDescription.GEN_ID`, :attr:`DetailedTopoDescription.STORAGE_ID`, 
        #:       :attr:`DetailedTopoDescription.LINE_OR_ID`, :attr:`DetailedTopoDescription.LINE_EX_ID` 
        #:       or :attr:`DetailedTopoDescription.SHUNT_ID` or :attr:`DetailedTopoDescription.OTHER`
        #:     - col 2 gives the ID of the connection node it connects (number between 0 and n_conn_node-1)
        #:     - col 3 gives the other ID of the connection node it connects
        self.switches = None
        
        #: This is a vector of integer having the same size as the number of switches in your grid.
        #: For each switches it gives the ID of the element this switch controls in the `topo_vect` vector
        #: When `-1` it means the element is not reprensented in the `topo_vect` (for example it's a shunt
        #: or a standard "connection node")
        self.switches_to_topovect_id = None
        
        #: This is a vector of integer having the same size as the number of switches in your grid.
        #: For each switches it says "-1" if the switch does not control a shunt or the shunt id (=>0)
        #: if the switch does control a shunt.
        self.switches_to_shunt_id = None
        
        #: A list of tuple that has the same size as the number of loads on the grid.
        #: For each loads, it gives the connection node ids to which (thanks to a switch) a load can be
        #: connected. For example if `type(env)..detailed_topo_desc.load_to_conn_node_id[0]` is the tuple `(1, 15)` this means that load
        #: id 0 can be connected to either connection node id 1 or connection node id 15.
        #: This information is redundant with the one provided in :attr:`DetailedTopoDescription.switches`
        self.load_to_conn_node_id = None
        
        #: Same as :attr:`DetailedTopoDescription.load_to_conn_node_id` but for generators
        self.gen_to_conn_node_id = None
        
        #: Same as :attr:`DetailedTopoDescription.load_to_conn_node_id` but for lines (or side)
        self.line_or_to_conn_node_id = None
        
        #: Same as :attr:`DetailedTopoDescription.load_to_conn_node_id` but for lines (ex side)
        self.line_ex_to_conn_node_id = None
        
        #: Same as :attr:`DetailedTopoDescription.load_to_conn_node_id` but for storage unit
        self.storage_to_conn_node_id = None
        
        #: Same as :attr:`DetailedTopoDescription.load_to_conn_node_id` but for shunt
        self.shunt_to_conn_node_id = None
    
    @classmethod
    def from_ieee_grid(cls, init_grid):
        """For now, suppose that the grid comes from ieee"""
        n_sub = init_grid.n_sub
        
        res = cls()
        res.conn_node_name = np.array([f"conn_node_{i}" for i in range(2 * init_grid.n_sub)])
        res.conn_node_to_subid = np.arange(n_sub) % init_grid.n_sub  
        
        # in current environment, there are 2 conn_nodes per substations, 
        # and 1 connector allows to connect both of them
        nb_connector = n_sub
        res.conn_node_connectors = np.zeros((nb_connector, 2), dtype=dt_int)
        res.conn_node_connectors[:, 0] = np.arange(n_sub)
        res.conn_node_connectors[:, 1] = np.arange(n_sub) + n_sub
        
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
        for sub_id in range(n_sub):    
            for arr_subid, pos_topo_vect, obj_col in zip(arrs_subid, ars2, ids):
                nb_el = (arr_subid == sub_id).sum()
                where_el = np.where(arr_subid == sub_id)[0]
                res.switches[prev_el : (prev_el + 2 * nb_el), cls.OBJ_TYPE_COL] = obj_col
                res.switches[prev_el : (prev_el + 2 * nb_el), cls.OBJ_ID_COL] = np.repeat(where_el, 2)
                res.switches[prev_el : (prev_el + 2 * nb_el), cls.CONN_NODE_ID_COL] = np.tile(np.array([1, 2]), nb_el)
                res.switches_to_topovect_id[prev_el : (prev_el + 2 * nb_el)] = np.repeat(pos_topo_vect[arr_subid == sub_id], 2)
                if init_grid.shunts_data_available and obj_col == cls.SHUNT_ID:
                    res.switches_to_shunt_id[prev_el : (prev_el + 2 * nb_el)] = np.repeat(where_el, 2)
                prev_el += 2 * nb_el
        
        # and also fill some extra information
        res.load_to_conn_node_id = [(load_sub, load_sub + n_sub) for load_id, load_sub in enumerate(init_grid.load_to_subid)]
        res.gen_to_conn_node_id = [(gen_sub, gen_sub + n_sub) for gen_id, gen_sub in enumerate(init_grid.gen_to_subid)]
        res.line_or_to_conn_node_id = [(line_or_sub, line_or_sub + n_sub) for line_or_id, line_or_sub in enumerate(init_grid.line_or_to_subid)]
        res.line_ex_to_conn_node_id = [(line_ex_sub, line_ex_sub + n_sub) for line_ex_id, line_ex_sub in enumerate(init_grid.line_ex_to_subid)]
        res.storage_to_conn_node_id = [(storage_sub, storage_sub + n_sub) for storage_id, storage_sub in enumerate(init_grid.storage_to_subid)]
        if init_grid.shunts_data_available:
            res.shunt_to_conn_node_id = [(shunt_sub, shunt_sub + n_sub) for shunt_id, shunt_sub in enumerate(init_grid.shunt_to_subid)]
        return res
    
    def compute_switches_position(self, topo_vect: np.ndarray, shunt_bus: Optional[np.ndarray]=None):
        """This function compute a plausible switches configuration
        from a given `topo_vect` representation.
        
        .. danger::
            At time of writing, it only works if the detailed topology has been generated
            from :func:`DetailedTopoDescription.from_ieee_grid`

        Parameters
        ----------
        topo_vect : `np.ndarray`
            The `topo_vect` detailing on which bus each element of the grid is connected
        shunt_bus : `np.ndarray`
            The busbar on which each shunt is connected.

        Returns
        -------
        Tuple of 2 elements:
        
        - `busbar_connectors_state` state of each busbar_connector
        - `switches_state` state of each switches
        
        """
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