# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from typing import Optional
import numpy as np

import grid2op
from grid2op.dtypes import dt_int, dt_bool
from grid2op.Exceptions import Grid2OpException
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
    CONN_NODE_1_ID_COL = 2
    
    #: In the :attr:`DetailedTopoDescription.switches` table, tells that column 2
    #: concerns the id of the connection node that this switches connects / disconnects
    CONN_NODE_2_ID_COL = 3
    
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
    def from_ieee_grid(cls, init_grid : "grid2op.Space.GridObjects.GridObjects"):
        """For now, suppose that the grid comes from ieee"""
        init_grid_cls = type(init_grid)
        
        n_sub = init_grid_cls.n_sub
        n_bb_per_sub = init_grid_cls.n_busbar_per_sub
        if n_bb_per_sub < 2:
            raise NotImplementedError("This function has not been implemented for less "
                                      "than 2 busbars per subs at the moment.")
        res = cls()
        
        # define the "connection nodes"
        # for ieee grid we model:
        # one connection node per busbar (per sub)
        # for each element (side of powerline, load, generator, storage, shunt etc.) 2 connection nodes
        # (status of the element) 
        # conn node for each busbar
        bb_conn_node = sum([[f"conn_node_sub_{subid}_busbar_{bb_i}" for bb_i in range(n_bb_per_sub)] for subid in range(n_sub)],
                             start=[])
        el_conn_node = ([f"conn_node_load_{i}" for i in range(init_grid_cls.n_load)] + 
                        [f"conn_node_gen_{i}" for i in range(init_grid_cls.n_gen)] +
                        [f"conn_node_line_or_{i}" for i in range(init_grid_cls.n_line)] +
                        [f"conn_node_line_ex_{i}" for i in range(init_grid_cls.n_line)] +
                        [f"conn_node_storage_{i}" for i in range(init_grid_cls.n_storage)] +
                        [f"conn_node_shunt_{i}" for i in range(init_grid_cls.n_shunt)] if init_grid_cls.shunts_data_available else []
                        )
        el_breaker_conn_node = ([f"conn_node_breaker_load_{i}" for i in range(init_grid_cls.n_load)] + 
                                [f"conn_node_breaker_gen_{i}" for i in range(init_grid_cls.n_gen)] +
                                [f"conn_node_breaker_line_or_{i}" for i in range(init_grid_cls.n_line)] +
                                [f"conn_node_breaker_line_ex_{i}" for i in range(init_grid_cls.n_line)] +
                                [f"conn_node_breaker_storage_{i}" for i in range(init_grid_cls.n_storage)] +
                                [f"conn_node_breaker_shunt_{i}" for i in range(init_grid_cls.n_shunt)] if init_grid_cls.shunts_data_available else []
                                )
        res.conn_node_name = np.array(bb_conn_node + 
                                      el_conn_node +
                                      el_breaker_conn_node)
        res.conn_node_to_subid = np.array(sum([[subid for bb_i in range(n_bb_per_sub)] for subid in range(n_sub)], start=[]) +
                                          2* (init_grid_cls.load_to_subid.tolist() +
                                              init_grid_cls.gen_to_subid.tolist() +
                                              init_grid_cls.line_or_to_subid.tolist() +
                                              init_grid_cls.line_ex_to_subid.tolist() +
                                              init_grid_cls.storage_to_subid.tolist() +
                                              init_grid_cls.shunt_to_subid.tolist() if init_grid_cls.shunts_data_available else []
                                              )
                                          )
        
        # add the switches : there are 1 switches that connects all pairs
        # of busbars in the substation, plus for each element:
        # - 1 switch for the status of the element ("conn_node_breaker_xxx_i")
        # - 1 breaker connecting the element to each busbar
        n_shunt = init_grid_cls.n_shunt if init_grid_cls.shunts_data_available else 0
        nb_switch_bb_per_sub = (n_bb_per_sub * (n_bb_per_sub - 1)) // 2  #  switches between busbars
        nb_switch_busbars = n_sub * nb_switch_bb_per_sub # switches between busbars at each substation
        nb_switch_total = nb_switch_busbars + (init_grid_cls.dim_topo + n_shunt) * (1 + n_bb_per_sub)
        res.switches = np.zeros((nb_switch_total, 4), dtype=dt_int)
        
        # add the shunts in the "sub_info" (considered as element here !)
        sub_info = 1 * init_grid_cls.sub_info
        if init_grid_cls.shunts_data_available:
            for sub_id in init_grid_cls.shunt_to_subid:
                sub_info[sub_id] += 1
        # now fill the switches matrix
        # fill with the switches between busbars
        res.switches[:nb_switch_busbars, cls.SUB_COL] = np.repeat(np.arange(n_sub), nb_switch_bb_per_sub)
        res.switches[:nb_switch_busbars, cls.OBJ_TYPE_COL] = cls.OTHER
        li_or_bb_switch = sum([[j for i in range(j+1, n_bb_per_sub)] for j in range(n_bb_per_sub - 1)], start=[])  # order relative to the substation
        li_ex_bb_switch = sum([[i for i in range(j+1, n_bb_per_sub)] for j in range(n_bb_per_sub - 1)], start=[])  # order relative to the substation
        add_sub_id_unique_id = np.repeat(np.arange(n_sub), nb_switch_bb_per_sub) * n_bb_per_sub  # make it a unique substation labelling
        res.switches[:nb_switch_busbars, cls.CONN_NODE_1_ID_COL] = np.array(n_sub * li_or_bb_switch) + add_sub_id_unique_id
        res.switches[:nb_switch_busbars, cls.CONN_NODE_2_ID_COL] = np.array(n_sub * li_ex_bb_switch) + add_sub_id_unique_id
        
        # and now fill the switches for all elements
        res.switches_to_topovect_id = np.zeros(nb_switch_total, dtype=dt_int) - 1
        if init_grid_cls.shunts_data_available:
            res.switches_to_shunt_id = np.zeros(nb_switch_total, dtype=dt_int) - 1
        
        arrs_subid = [init_grid_cls.load_to_subid,
                      init_grid_cls.gen_to_subid,
                      init_grid_cls.line_or_to_subid,
                      init_grid_cls.line_ex_to_subid,
                      init_grid_cls.storage_to_subid,
                      ]
        ars2 = [init_grid_cls.load_pos_topo_vect,
                init_grid_cls.gen_pos_topo_vect,
                init_grid_cls.line_or_pos_topo_vect,
                init_grid_cls.line_ex_pos_topo_vect,
                init_grid_cls.storage_pos_topo_vect,
               ]
        ids = [cls.LOAD_ID, cls.GEN_ID, cls.LINE_OR_ID, cls.LINE_EX_ID, cls.STORAGE_ID]
        if init_grid_cls.shunts_data_available:
            arrs_subid.append(init_grid_cls.shunt_to_subid)
            ars2.append(np.array([-1] * init_grid_cls.n_shunt))
            ids.append(cls.SHUNT_ID)
            
        prev_el = nb_switch_busbars   
        handled = 0
        for arr_subid, pos_topo_vect, obj_col in zip(arrs_subid, ars2, ids):
            nb_el = arr_subid.shape[0]
            next_el = prev_el + (1 + n_bb_per_sub) * nb_el
            
            # fill the object type
            res.switches[prev_el : next_el, cls.OBJ_TYPE_COL] = cls.OTHER
            res.switches[prev_el : next_el : (1 + n_bb_per_sub), cls.OBJ_TYPE_COL] = obj_col
            
            # fill the substation id
            res.switches[prev_el : next_el, cls.SUB_COL] = np.repeat(arr_subid, (1 + n_bb_per_sub))
            
            conn_node_breaker_ids = (len(bb_conn_node) + len(el_conn_node) + handled + np.arange(nb_el))
            # fill the switches that connect the element to each busbars (eg)
            # `conn_node_breaker_load_{i}` to `conn_node_sub_{subid}_busbar_{bb_i}`
            # nb some values here are erased by the following statement (but I did not want to make a for loop in python)
            res.switches[prev_el : next_el, cls.CONN_NODE_1_ID_COL] = np.repeat(conn_node_breaker_ids, 1 + n_bb_per_sub)
            res.switches[prev_el : next_el, cls.CONN_NODE_2_ID_COL] = (np.tile(np.arange(-1, n_bb_per_sub), nb_el) +
                                                                       np.repeat(arr_subid * n_bb_per_sub, n_bb_per_sub+1))
            
            # fill the breaker that connect (eg):
            # `conn_node_load_{i}` to `conn_node_breaker_load_{i}`
            res.switches[prev_el : next_el : (1 + n_bb_per_sub), cls.CONN_NODE_1_ID_COL] = len(bb_conn_node) + handled + np.arange(nb_el)
            res.switches[prev_el : next_el : (1 + n_bb_per_sub), cls.CONN_NODE_2_ID_COL] = conn_node_breaker_ids
            
            # TODO detailed topo : fill switches_to_topovect_id and switches_to_shunt_id
            # res.switches_to_topovect_id[prev_el : (prev_el + 2 * nb_el)] = np.repeat(pos_topo_vect[arr_subid == sub_id], 2)
            # if init_grid_cls.shunts_data_available and obj_col == cls.SHUNT_ID:
            #     res.switches_to_shunt_id[prev_el : (prev_el + 2 * nb_el)] = np.repeat(where_el, 2)
            prev_el = next_el
            handled += nb_el
        
        # and also fill some extra information
        res.load_to_conn_node_id = 1 * res.switches[res.switches[:,cls.OBJ_TYPE_COL] == cls.LOAD_ID, cls.CONN_NODE_1_ID_COL]
        res.gen_to_conn_node_id = 1 * res.switches[res.switches[:,cls.OBJ_TYPE_COL] == cls.GEN_ID, cls.CONN_NODE_1_ID_COL]
        res.line_or_to_conn_node_id = 1 * res.switches[res.switches[:,cls.OBJ_TYPE_COL] == cls.LINE_OR_ID, cls.CONN_NODE_1_ID_COL]
        res.line_ex_to_conn_node_id = 1 * res.switches[res.switches[:,cls.OBJ_TYPE_COL] == cls.LINE_EX_ID, cls.CONN_NODE_1_ID_COL]
        res.storage_to_conn_node_id = 1 * res.switches[res.switches[:,cls.OBJ_TYPE_COL] == cls.STORAGE_ID, cls.CONN_NODE_1_ID_COL]
        if init_grid_cls.shunts_data_available:
            res.shunt_to_conn_node_id = 1 * res.switches[res.switches[:,cls.OBJ_TYPE_COL] == cls.SHUNT_ID, cls.CONN_NODE_1_ID_COL]
        # TODO detailed topo: have a function to compute the above things
        # TODO detailed topo: have a function to compute the switches `sub_id` columns from the `conn_node_to_subid`
        return res
    
    def compute_switches_position(self,
                                  topo_vect: np.ndarray,
                                  shunt_bus: Optional[np.ndarray]=None):
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
        # busbar_connectors_state = np.zeros(self.busbar_connectors.shape[0], dtype=dt_bool)  # we can always say they are opened 
        
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
        return switches_state
    
    def from_switches_position(self):
        # TODO detailed topo
        # opposite of `compute_switches_position`
        topo_vect = None
        shunt_bus = None
        return topo_vect, shunt_bus
        
    def check_validity(self):
        if self.conn_node_to_subid.shape != self.conn_node_name.shape:
            raise Grid2OpException(f"Inconsistencies found on the connectivity nodes: "
                                   f"you declared {len(self.conn_node_to_subid)} connectivity nodes "
                                   f"in `self.conn_node_to_subid` but "
                                   f"{len( self.conn_node_name)} connectivity nodes in "
                                   "`self.conn_node_name`")
        if self.switches[:,type(self).CONN_NODE_1_ID_COL].max() >= len(self.conn_node_to_subid):
            raise Grid2OpException("Inconsistencies found in the switches: some switches are "
                                   "mapping unknown connectivity nodes for 'CONN_NODE_1_ID_COL' (too high)")
        if self.switches[:,type(self).CONN_NODE_2_ID_COL].max() >= len(self.conn_node_to_subid):
            raise Grid2OpException("Inconsistencies found in the switches: some switches are "
                                   "mapping unknown connectivity nodes for 'CONN_NODE_2_ID_COL' (too high)")
        if self.switches[:,type(self).CONN_NODE_1_ID_COL].min() < 0:
            raise Grid2OpException("Inconsistencies found in the switches: some switches are "
                                   "mapping unknown connectivity nodes for 'CONN_NODE_1_ID_COL' (too low)")
        if self.switches[:,type(self).CONN_NODE_2_ID_COL].max() >= len(self.conn_node_to_subid):
            raise Grid2OpException("Inconsistencies found in the switches: some switches are "
                                   "mapping unknown connectivity nodes for 'CONN_NODE_2_ID_COL' (too low)")
            
        if (self.conn_node_to_subid[self.switches[:,type(self).CONN_NODE_1_ID_COL]] != 
            self.conn_node_to_subid[self.switches[:,type(self).CONN_NODE_2_ID_COL]]).any():
            raise Grid2OpException("Inconsistencies found in the switches mapping. Some switches are "
                                   "mapping connectivity nodes that belong to different substation id.") 
        # TODO detailed topo other tests
        # TODO detailed topo proper exception class and not Grid2OpException
    
    def save_to_dict(self, res, as_list=True, copy_=True):
        # TODO detailed topo
        save_to_dict(
            res,
            self,
            "conn_node_name",
            (lambda arr: [str(el) for el in arr]) if as_list else None,
            copy_,
        )
        save_to_dict(
            res,
            self,
            "conn_node_to_subid",
            (lambda arr: [int(el) for el in arr]) if as_list else None,
            copy_,
        )
        # save_to_dict(
        #     res,
        #     self,
        #     "conn_node_connectors",
        #     (lambda arr: [int(el) for el in arr]) if as_list else lambda arr: arr.flatten(),
        #     copy_,
        # )
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
            "load_to_conn_node_id",
            (lambda arr: [int(el) for el in arr]) if as_list else None,
            copy_,
        )
        save_to_dict(
            res,
            self,
            "gen_to_conn_node_id",
            (lambda arr: [int(el) for el in arr]) if as_list else None,
            copy_,
        )
        save_to_dict(
            res,
            self,
            "line_or_to_conn_node_id",
            (lambda arr: [int(el) for el in arr]) if as_list else None,
            copy_,
        )
        save_to_dict(
            res,
            self,
            "line_ex_to_conn_node_id",
            (lambda arr: [int(el) for el in arr]) if as_list else None,
            copy_,
        )
        save_to_dict(
            res,
            self,
            "storage_to_conn_node_id",
            (lambda arr: [int(el) for el in arr]) if as_list else None,
            copy_,
        )
        if self.shunt_to_conn_node_id is not None:
            save_to_dict(
                res,
                self,
                "shunt_to_conn_node_id",
                (lambda arr: [int(el) for el in arr]) if as_list else None,
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
        res.conn_node_name = extract_from_dict(
            dict_, "conn_node_name", lambda x: np.array(x).astype(str)
        )
        res.conn_node_to_subid = extract_from_dict(
            dict_, "conn_node_to_subid", lambda x: np.array(x).astype(dt_int)
        )
        # res.busbar_connectors = extract_from_dict(
        #     dict_, "busbar_connectors", lambda x: np.array(x).astype(dt_int)
        # )
        # res.busbar_connectors = res.busbar_connectors.reshape((-1, 2))
        
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
        
        res.load_to_conn_node_id = extract_from_dict(
            dict_, "load_to_conn_node_id", lambda x: x
        )
        res.gen_to_conn_node_id = extract_from_dict(
            dict_, "gen_to_conn_node_id", lambda x: x
        )
        res.line_or_to_conn_node_id = extract_from_dict(
            dict_, "line_or_to_conn_node_id", lambda x: x
        )
        res.line_ex_to_conn_node_id = extract_from_dict(
            dict_, "line_ex_to_conn_node_id", lambda x: x
        )
        res.storage_to_conn_node_id = extract_from_dict(
            dict_, "storage_to_conn_node_id", lambda x: x
        )
        if "shunt_to_conn_node_id" in dict_:
            res.shunt_to_conn_node_id = extract_from_dict(
                dict_, "shunt_to_conn_node_id", lambda x: x
            )
        
        # TODO detailed topo
        return res