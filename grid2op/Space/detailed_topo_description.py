# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from typing import List, Optional
import numpy as np
import networkx as nx
import copy
from functools import lru_cache

import grid2op
from grid2op.dtypes import dt_int, dt_bool
from grid2op.Exceptions import Grid2OpException, ImpossibleTopology
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
    
    # TODO detailed topo: remove the switches to topo_vect id and make a "conn node id to topo_vect id"
    # TODO detailed topo: remove in the doc the requirement of conn_node_1 and conn_node_2
    # TODO detailed topo: remove the element types from the switch matrix
    
    In order to fill a :class:`DetailedTopoDescription` you need to fill the 
    following attributes :
    
    - :attr:`DetailedTopoDescription.conn_node_name` : for each connectivity node, you provide a name. For now we 
      recommend using it (at least for debug purpose) but later this vector might contain None for internal connectivity 
      node. 
    - :attr:`DetailedTopoDescription.conn_node_to_subid` : for each connectiviy node, you provide the substation to 
      which it is connected. The substation should exist in the grid. All substation should have a least one connectivity
      node at the moment.
    - :attr:`DetailedTopoDescription.switches` : this is the "main" information about detailed topology. It provide the 
      information about each switches on your grid. It is a matrix with 4 columns:
      
        - the first is the substation id to which this switches belong. As of now you have to fill it manually 
          and this information should match the one given by the connectivity node this switch
          represent. TODO detailed topo: have a routine to add it automatically afterwards
        - the second one is an information about the element - *eg* load or generator or side of powerline- it concerns (if any)
        - the third one is the ID of one of the connectivity node this switch is attached to
        - the fourth one is the ID of the other connectivity node this switch is attached to
        
    - :attr:`DetailedTopoDescription.conn_node_to_topovect_id` : for each connectivity node, it gives the index in the 
      topo_vect vector to which this connectivity node is connected. Put -1 for conn node not represented in 
      the "topo_vect" vector
      otherwise the id of the topo_vect converned by this switch (should be -1 for everything except for
      switch whose conn_node_id_1 represents element modeled in the topo_vect eg load, generator or side of powerline)
      (TODO detailed topo: something again that for now you should manually process but that will
      be automatically processed by grid2op in the near future).
    - :attr:`DetailedTopoDescription.conn_node_to_shunt_id` : for each connectivity node, it gives the index of the shunt it
      concerns (should be -1 except for connectivity node that concerns shunts)
      (TODO detailed topo: something again that for now you should manually process but that will
      be automatically processed by grid2op in the near future).
    - :attr:`DetailedTopoDescription.load_to_conn_node_id` : for each load, it gives by which connectivity
      node it is represented. It should match the info in the colum 2 (third column) of the switches matrix.
      (TODO detailed topo: something again that for now you should manually process but that will
      be automatically processed by grid2op in the near future).
    - :attr:`DetailedTopoDescription.gen_to_conn_node_id` : for each generator, it gives by which connectivity
      node it is represented. It should match the info in the colum 2 (third column) of the switches matrix.
      (TODO detailed topo: something again that for now you should manually process but that will
      be automatically processed by grid2op in the near future).
    - :attr:`DetailedTopoDescription.line_or_to_conn_node_id` : for each "origin" side of powerline, 
      it gives by which connectivity
      node it is represented. It should match the info in the colum 2 (third column) of the switches matrix.
      (TODO detailed topo: something again that for now you should manually process but that will
      be automatically processed by grid2op in the near future).
    - :attr:`DetailedTopoDescription.line_ex_to_conn_node_id` : for each "extremity" side of powerline, 
      it gives by which connectivity
      node it is represented. It should match the info in the colum 2 (third column) of the switches matrix.
      (TODO detailed topo: something again that for now you should manually process but that will
      be automatically processed by grid2op in the near future).
    - :attr:`DetailedTopoDescription.storage_to_conn_node_id` : for each storage unit, 
      it gives by which connectivity
      node it is represented. It should match the info in the colum 2 (third column) of the switches matrix.
      (TODO detailed topo: something again that for now you should manually process but that will
      be automatically processed by grid2op in the near future).
    - :attr:`DetailedTopoDescription.shunt_to_conn_node_id` : for each shunt, 
      it gives by which connectivity
      node it is represented. It should match the info in the colum 2 (third column) of the switches matrix.
      (TODO detailed topo: something again that for now you should manually process but that will
      be automatically processed by grid2op in the near future).
    - :attr:`DetailedTopoDescription.busbar_section_to_conn_node_id` : this vector has the size of the number 
      of "busbar sections" in the grid. And for each busbar section, it gives the information for which
      connectivity node it is represented.
    - :attr:`DetailedTopoDescription.busbar_section_to_subid` : this vector has the same size as the
      :attr:`DetailedTopoDescription.busbar_section_to_conn_node_id` and give the information of 
      the substation id each busbar section is part of. It should match the 
      information in `self.switches` too
      (TODO detailed topo: something again that for now you should manually process but that will
      be automatically processed by grid2op in the near future).
    
    .. warning::
        If a switch connects an element - *eg* load or generator or side of powerline- on one of it side, the 
        connectivity node of this element should be on the 3rd column (index 2 in python) in the switches
        matrix and not on the 4th column (index 4 in python)
    
    .. danger::
        As opposed to some other elements of grid2op, by default, connectivity nodes should be labeled
        in a "global" way. This means that there is exactly one connectivity node labeled `1` 
        for the whole grid (as opposed to 1 per substation !). 
        
        They are labelled the same way as *eg* `load` (there is a unique `load 1`) and not like `busbar in the
        substation` where thare are "as many busbar 1 as there are substation".
        
        TODO detailed topo: this is `True` for now but there would be nothing (except some added tests 
        and maybe a bit of code) to allow the "substation local" labelling.
    
    .. danger::
        As of writing, we suppose that there exist a breaker controlling 
        the state of each element. This breaker should be unique.
        
        This means that for every connectiviy node representing an element
        of the grid (*eg* the side of a powerline or a generator etc.) there
        exist a switch connecting this element to the rest of the graph. And that,
        if this switch is opened, the element is disconnected.
        
        This switch does not control any other element.
        (you can write a feature request if this is a problem for you, we did 
        this hypothesis out of simplicity for disconnected element, but the routine
        to compute the switch state can be adapted without it).
        
    To create a "detailed description of the swtiches", somewhere in the implementation of your
    backend you have a piece of code looking like:
    
    .. code-block:: python
    
        import os
        from grid2op.Backend import Backend
        from typing import Optional, Union, Tuple
        from grid2op.Space import DetailedTopoDescription
        
        class MyBackendType(Backend):
            # some implementation of other methods...
            
            def load_grid(self,
                          path : Union[os.PathLike, str],
                          filename : Optional[Union[os.PathLike, str]]=None) -> None:
                # do the regular implementation of the load_grid function
                ...
                ...
                
                # once done, then you can create a detailed topology
                dtd = DetailedTopoDescription()
                
                # you fill it with the data in the grid you read
                # (at this stage you tell grid2op what the grid is made of)
                dtd.conn_node_name = ...
                dtd.conn_node_to_subid = ...
                dtd.switches = ...
                dtd.conn_node_to_topovect_id = ...
                dtd.conn_node_to_shunt_id = ...
                dtd.load_to_conn_node_id = ...
                dtd.gen_to_conn_node_id = ...
                dtd.line_or_to_conn_node_id = ...
                dtd.line_ex_to_conn_node_id = ...
                dtd.storage_to_conn_node_id = ...
                dtd.shunt_to_conn_node_id = ...
                dtd.busbar_section_to_conn_node_id = ...
                dtd.busbar_section_to_subid = ...
                
                # and then you assign it as a member of this class
                self.detailed_topo_desc =  dtd
        
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
    
    #: In the :attr:`DetailedTopoDescription.switches` table, tells that column 2
    #: concerns the id of object that this switches connects / disconnects
    CONN_NODE_1_ID_COL = 1
    
    #: In the :attr:`DetailedTopoDescription.switches` table, tells that column 2
    #: concerns the id of the connection node that this switches connects / disconnects
    CONN_NODE_2_ID_COL = 2
    
    #: TODO detailed topo doc
    LOAD_ID = 0
    
    #: TODO detailed topo doc
    GEN_ID = 1
    
    #: TODO detailed topo doc
    STORAGE_ID = 2
    
    #: TODO detailed topo doc
    LINE_OR_ID = 3
    
    #: TODO detailed topo doc
    LINE_EX_ID = 4
    
    #: TODO detailed topo doc
    SHUNT_ID = 5
    
    #: TODO detailed topo doc
    OTHER = 6
    
    def __init__(self):        
        #: vector of string that has the size of the number of connection nodes on your grid
        #: and for each connection node it gives... its name
        self.conn_node_name = None
        
        #: vector of int that has the size of the number of connection nodes on
        #: your grid and for each connection node it gives the substation id [0...n_sub] to which
        #: the connection node belongs to.
        self.conn_node_to_subid = None
    
        #: It is a matrix describing each switches. This matrix has 'n_switches' rows and 4 columns. 
        #: Each column provides an information about the switch:
        #:     
        #:     - col 0 gives the substation id
        #:     - col 1 TODO detailed topo doc
        #:     - col 2 TODO detailed topo doc
        self.switches = None
        
        #: TODO
        self.conn_node_to_topovect_id = None
        
        #: TODO
        self.conn_node_to_shunt_id = None
        
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
        
        #: For each busbar section, it gives the connection node id
        #: that represent this busbar section
        self.busbar_section_to_conn_node_id = None
        
        #: For each busbar section, it gives the substation id to which it
        #: is connected
        self.busbar_section_to_subid = None
        
        #: flag to detect that the detailed topo have been built with the 
        #: :func:`.DetailedTopoDescriptionfrom_ieee_grid`
        #: which enables some feature that will be more generic in the future.
        self._from_ieee_grid = False
        
        #: number of substation on the grid
        #: this is automatically set when the detailed topo description
        #: is processed
        self._n_sub : int = -1
        
        #: dimension of the "topology vector" (in grid2op)
        #: this is automatically set when the detailed topo description
        #: is processed
        self._dim_topo : int = -1
        
        #: number of shunt in the grid
        #: this is automatically set when the detailed topo description
        #: is processed
        self._n_shunt : int = -1
        
        #: INTERNAL
        self._conn_node_to_bbs_conn_node_id = None
        
        #: INTERNAL
        self._connectivity_graph : List[nx.Graph] = None
        # TODO detailed topo: list per substation !

        #: INTERNAL
        self._cn_pos_in_sub : np.ndarray = None
        
        #: INTERNAL
        self._sw_pos_in_sub : np.ndarray = None
        
    @classmethod
    def from_ieee_grid(cls, init_grid : "grid2op.Space.GridObjects.GridObjects"):
        """For now, suppose that the grid comes from ieee grids.
        
        See doc of :class:`AddDetailedTopoIEEE` for more information.
        
        """
        init_grid_cls = type(init_grid)
        
        n_sub = init_grid_cls.n_sub
        n_bb_per_sub = init_grid_cls.n_busbar_per_sub
        if n_bb_per_sub < 2:
            raise NotImplementedError("This function has not been implemented for less "
                                      "than 2 busbars per subs at the moment.")
        res = cls()
        res._from_ieee_grid = True
        res._n_sub = n_sub
        res._dim_topo = init_grid_cls.dim_topo
        res._n_shunt = init_grid_cls.n_shunt
        
        # define the "connection nodes"
        # for ieee grid we model:
        # one connection node per busbar (per sub)
        # for each element (side of powerline, load, generator, storage, shunt etc.) 2 connection nodes
        # (status of the element) 
        # conn node for each busbar
        bb_conn_node = sum([[f"conn_node_sub_{subid}_busbar_{bb_i}" for bb_i in range(n_bb_per_sub)] for subid in range(n_sub)],
                             start=[])
        res.busbar_section_to_subid = np.repeat(np.arange(n_sub),n_bb_per_sub)
        res.busbar_section_to_conn_node_id = np.arange(len(bb_conn_node))
        
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
        n_conn_nodes = res.conn_node_name.shape[0]
        
        # add the switches : there are 1 switches that connects all pairs
        # of busbars in the substation, plus for each element:
        # - 1 switch for the status of the element ("conn_node_breaker_xxx_i")
        # - 1 breaker connecting the element to each busbar
        n_shunt = init_grid_cls.n_shunt if init_grid_cls.shunts_data_available else 0
        nb_switch_bb_per_sub = (n_bb_per_sub * (n_bb_per_sub - 1)) // 2  #  switches between busbars
        nb_switch_busbars = n_sub * nb_switch_bb_per_sub # switches between busbars at each substation
        nb_switch_total = nb_switch_busbars + (init_grid_cls.dim_topo + n_shunt) * (1 + n_bb_per_sub)
        res.switches = np.zeros((nb_switch_total, 3), dtype=dt_int)
        
        # add the shunts in the "sub_info" (considered as element here !)
        sub_info = 1 * init_grid_cls.sub_info
        if init_grid_cls.shunts_data_available:
            for sub_id in init_grid_cls.shunt_to_subid:
                sub_info[sub_id] += 1
        # now fill the switches matrix
        # fill with the switches between busbars
        res.switches[:nb_switch_busbars, cls.SUB_COL] = np.repeat(np.arange(n_sub), nb_switch_bb_per_sub)
        
        li_or_bb_switch = sum([[j for i in range(j+1, n_bb_per_sub)] for j in range(n_bb_per_sub - 1)], start=[])  # order relative to the substation
        li_ex_bb_switch = sum([[i for i in range(j+1, n_bb_per_sub)] for j in range(n_bb_per_sub - 1)], start=[])  # order relative to the substation
        add_sub_id_unique_id = np.repeat(np.arange(n_sub), nb_switch_bb_per_sub) * n_bb_per_sub  # make it a unique substation labelling
        res.switches[:nb_switch_busbars, cls.CONN_NODE_1_ID_COL] = np.array(n_sub * li_or_bb_switch) + add_sub_id_unique_id
        res.switches[:nb_switch_busbars, cls.CONN_NODE_2_ID_COL] = np.array(n_sub * li_ex_bb_switch) + add_sub_id_unique_id
        
        # and now fill the switches for all elements
        res.conn_node_to_topovect_id = np.zeros(n_conn_nodes, dtype=dt_int) - 1
        if init_grid_cls.shunts_data_available:
            res.conn_node_to_shunt_id = np.zeros(n_conn_nodes, dtype=dt_int) - 1
        
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
            this_conn_nodes = np.arange(len(bb_conn_node) + handled,
                                        len(bb_conn_node) + handled + nb_el)
            if obj_col == cls.LOAD_ID:
                res.load_to_conn_node_id = this_conn_nodes
            elif obj_col == cls.GEN_ID:
                res.gen_to_conn_node_id = this_conn_nodes
            elif obj_col == cls.LINE_OR_ID:
                res.line_or_to_conn_node_id = this_conn_nodes
            elif obj_col == cls.LINE_EX_ID:
                res.line_ex_to_conn_node_id = this_conn_nodes
            elif obj_col == cls.STORAGE_ID:
                res.storage_to_conn_node_id = this_conn_nodes
            elif obj_col == cls.SHUNT_ID and init_grid_cls.shunts_data_available:
                res.shunt_to_conn_node_id = this_conn_nodes
    
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
            
            if obj_col != cls.SHUNT_ID:
                res.conn_node_to_topovect_id[this_conn_nodes] = pos_topo_vect
            if init_grid_cls.shunts_data_available and obj_col == cls.SHUNT_ID:
                res.conn_node_to_shunt_id[this_conn_nodes] = np.arange(nb_el)
            prev_el = next_el
            handled += nb_el
        
        # TODO detailed topo: have a function to compute the switches `sub_id` columns from the `conn_node_to_subid`
        # TODO detailed topo: have a function for the "conn_node_to_topovect_id" and  "switches_to_shunt_id"
        return res
    
    def _aux_compute_busbars_sections(self):
        # TODO detailed topo: speed optimization: install graph-tool (but not available with pip...)
        cls = type(self)
        # import time
        # beg_ = time.perf_counter()
        self._connectivity_graph = []
        self._conn_node_to_bbs_conn_node_id = [set() for _ in range(self.conn_node_name.shape[0])]
        for sub_id in range(self._n_sub):
            g_this_sub = nx.Graph()
            g_this_sub.add_edges_from([(el[1], el[2], {"id": switch_id}) for switch_id, el in enumerate(self.switches) if el[cls.SUB_COL] == sub_id])
            bbs_this_sub = self.busbar_section_to_conn_node_id[self.busbar_section_to_subid == sub_id]
            # je veux isoler les elements qui, si on enleve les busbar, peuvent atteindre les autres busbars
            for busbar_id in bbs_this_sub:
                tmp_g = copy.deepcopy(g_this_sub)
                tmp_g.remove_nodes_from([el for el in bbs_this_sub if el != busbar_id])
                conn_nodes = nx.node_connected_component(tmp_g, busbar_id)
                for el in conn_nodes:
                    self._conn_node_to_bbs_conn_node_id[el].add(busbar_id)
            self._connectivity_graph.append(g_this_sub)
        # print(time.perf_counter() - beg_)  # 2ms for 1 sub
        
        # compute the position of each connectivity node in the substation
        self._cn_pos_in_sub = np.zeros(self.conn_node_to_subid.shape[0], dtype=dt_int) - 1
        for subid in range(self._n_sub):
            cn_this_sub = (self.conn_node_to_subid == subid).nonzero()[0]
            if len(cn_this_sub) == 0:
                raise Grid2OpException(f"There are no connectivity node at substation {subid}")
            self._cn_pos_in_sub[cn_this_sub] = cn_this_sub.argsort()
        if self._cn_pos_in_sub.min() < 0:
            raise Grid2OpException("Impossible to compute the position of some "
                                   "connectivity nodes in their substation.")
            
        # compute the position of each switch in its substation
        self._sw_pos_in_sub = np.zeros(self.switches.shape[0], dtype=dt_int) - 1
        for subid in range(self._n_sub):
            sw_this_sub = (self.switches[:, cls.SUB_COL] == subid).nonzero()[0]
            if len(sw_this_sub) == 0:
                raise Grid2OpException(f"There are no switch at substation {subid}")
            self._sw_pos_in_sub[sw_this_sub] = sw_this_sub.argsort()
        if self._sw_pos_in_sub.min() < 0:
            raise Grid2OpException("Impossible to compute the position of some "
                                   "switches in their substation.")

    def get_switch_id_ieee(self, conn_node_id: int):
        """TODO detailed topo

        Parameters
        ----------
        conn_node_id : int
            _description_
        """
        switch_id = (self.switches[:, type(self).CONN_NODE_1_ID_COL] == conn_node_id).nonzero()[0]
        if switch_id.shape[0] == 0:
            raise Grid2OpException(f"Cannot find a switch for connectivity node {conn_node_id}")
        if switch_id.shape[0] > 1:
            raise Grid2OpException(f"Found multiple switch for connectivity node {conn_node_id}")
        switch_id = switch_id[0]
        return switch_id
    
    def _aux_compute_switches_pos_ieee(self, 
                                       bus_vect,  # topo_vect
                                       el_to_conn_node_id,   # load_to_conn_node_id
                                       conn_node_to_bus_id,  # conn_node_to_topo_vect_id
                                       switches_state,  # result
                                       ):
        if not self._from_ieee_grid:
            raise NotImplementedError("This function is only implemented for detailed topology "
                                      "generated from ieee grids. You can use `compute_switches_position` "
                                      "for a more generic function")
            
        # compute the position for the switches of the "topo_vect" elements 
        # only work for current grid2op modelling !
        
        # TODO detailed topo vectorize this ! (or cython maybe ?)
        for conn_node in el_to_conn_node_id:
            switch_id = self.get_switch_id_ieee(conn_node)
            my_bus = bus_vect[conn_node_to_bus_id[conn_node]]
            if my_bus == -1:
                # I init the swith at False, so nothing to do in this case
                continue
            switches_state[switch_id] = True  # connector is connected
            switches_state[switch_id + my_bus] = True  # connector to busbar is connected
    
    def compute_switches_position_ieee(self, topo_vect, shunt_bus):
        if not self._from_ieee_grid:
            raise NotImplementedError("This function is only implemented for detailed topology "
                                      "generated from ieee grids. You can use `compute_switches_position` "
                                      "for a more generic function.")
        switches_state = np.zeros(self.switches.shape[0], dtype=dt_bool)
        
        # compute the position for the switches of the "topo_vect" elements
        self._aux_compute_switches_pos_ieee(topo_vect, self.load_to_conn_node_id, self.conn_node_to_topovect_id, switches_state)
        self._aux_compute_switches_pos_ieee(topo_vect, self.gen_to_conn_node_id, self.conn_node_to_topovect_id, switches_state)
        self._aux_compute_switches_pos_ieee(topo_vect, self.line_or_to_conn_node_id, self.conn_node_to_topovect_id, switches_state)
        self._aux_compute_switches_pos_ieee(topo_vect, self.line_ex_to_conn_node_id, self.conn_node_to_topovect_id, switches_state)
        self._aux_compute_switches_pos_ieee(topo_vect, self.storage_to_conn_node_id, self.conn_node_to_topovect_id, switches_state)
            
        if self.conn_node_to_shunt_id is None or shunt_bus is None or self._n_shunt == 0:
            # no need to process the shunts in these cases
            return switches_state
        
        # compute the position for the switches of the "shunts" elements
        self._aux_compute_switches_pos_ieee(shunt_bus, self.shunt_to_conn_node_id, self.conn_node_to_shunt_id, switches_state)
        return switches_state
            
    def compute_switches_position(self,
                                  topo_vect: np.ndarray,
                                  shunt_bus: Optional[np.ndarray]=None,
                                  subs_changed : Optional[np.ndarray]=None):
        """This function compute a plausible switches configuration
        from a given `topo_vect` representation.
        
        TODO detailed topo: documentation

        Parameters
        ----------
        topo_vect : `np.ndarray`
            The `topo_vect` detailing on which bus each element of the grid is connected
        shunt_bus : `np.ndarray`
            The busbar on which each shunt is connected.

        Returns
        -------
        `switches_state` state (connected = ``True`` or disconnected = ``False``) of each switches as 
        a numpy boolean array.
        
        """
        # TODO detailed topo: input the previous switch state
        
        if topo_vect.shape[0] != self._dim_topo:
            raise Grid2OpException("Incorrect input size for the topology vector.")
        if shunt_bus is not None and shunt_bus.shape[0] != self._n_shunt:
            raise Grid2OpException("Incorrect size for the shunt bus vector.")
        conn_topo_vect = topo_vect[topo_vect != -1]
        if len(conn_topo_vect):
            if conn_topo_vect.min() < 1:
                raise Grid2OpException("In grid2op buses are labelled starting from 1 and not 0 "
                                    "(check your `topo_vect` input)")
        if self._n_shunt > 0 and shunt_bus is not None:
            conn_shunt = shunt_bus[shunt_bus != -1]
            if conn_shunt.shape[0]:
                if conn_shunt.min() < 1:
                    raise Grid2OpException("In grid2op buses are labelled starting from 1 and not 0 "
                                           "(check your `shunt_bus` input)")
        if np.unique(conn_topo_vect).shape[0] > self.busbar_section_to_subid.shape[0]:
            raise ImpossibleTopology("You ask for more independant buses than there are "
                                     "busbar section on this substation")
        if self._from_ieee_grid:
            # specific case for IEEE grid, consistent with the AddDetailedTopoIEEE
            # class
            return self.compute_switches_position_ieee(topo_vect, shunt_bus)
        
        if subs_changed is None:
            subs_changed = np.ones(self._n_sub, dtype=dt_bool)
        
        if subs_changed.shape[0] != self._n_sub:
            raise Grid2OpException("Incorrect size for the substation mask")
        
        if self._conn_node_to_bbs_conn_node_id is None:
            self._aux_compute_busbars_sections()
            
        full_res = np.zeros(self.switches.shape[0], dtype=dt_bool)
        for sub_id, is_sub_modif in enumerate(subs_changed):
            if not is_sub_modif:
                continue
            mask_this_sub = self.switches[:, type(self).SUB_COL] == sub_id
            res_this_sub = self._aux_compute_switches_position_one_sub(sub_id, topo_vect, shunt_bus)
            full_res[mask_this_sub] = res_this_sub
        return full_res
    
    def _aux_compute_switches_position_one_sub(self,
                                               sub_id,
                                               topo_vect: np.ndarray,
                                               shunt_bus: Optional[np.ndarray]=None):
        
        # by default they are False
        nb_switch = self.switches[self.switches[:, type(self).SUB_COL] == sub_id].shape[0]
        nb_conn_node = self.conn_node_name[self.conn_node_to_subid == sub_id].shape[0]
        
        # results
        switches_state = np.zeros(nb_switch, dtype=dt_bool)  
        # whether the switch is already assigned to a bus
        switch_visited = np.zeros(nb_switch, dtype=dt_bool)
                
        # whether the connectivity node is assigned to a bus
        conn_node_visited = np.zeros(nb_conn_node, dtype=dt_bool)
        conn_node_to_bus_id = np.zeros(nb_conn_node, dtype=dt_int)
        all_pos = ((self.conn_node_to_topovect_id != -1) & (self.conn_node_to_subid == sub_id)).nonzero()[0]
        
        if self._n_shunt > 0 and shunt_bus is not None:
            # add the shunts
            all_pos = np.concatenate((all_pos, 
                                      ((self.conn_node_to_shunt_id != -1) & (self.conn_node_to_subid == sub_id)).nonzero()[0]))
            topo_vect = np.concatenate((topo_vect, shunt_bus))
            
        # traverse all objects
        main_obj_id = 0
        li_bbs = [self._conn_node_to_bbs_conn_node_id[el] for el in all_pos]
        # perf optim
        # loop through elements in a given order:
        # first start by element that are the most constrained
        order_pos = np.argsort(li_bbs)
        # order_pos = np.arange(len(li_bbs))  # debug
        # end perf optim
        # TODO detailed topo: be even smarter by looking at object bus after bus
        # and not in a "random" order
        
        cn_can_be_connected = np.ones((nb_conn_node, self.busbar_section_to_subid.shape[0]))
        try:
            # TODO detailed topo: in the df_compute_switches, make clearer what is const and what is not
            res = self._dfs_compute_switches_position(topo_vect, 
                                                      self._connectivity_graph[sub_id],
                                                      main_obj_id, 
                                                      all_pos, 
                                                      switch_visited, 
                                                      switches_state, 
                                                      conn_node_visited,
                                                      conn_node_to_bus_id,
                                                      order_pos,
                                                    #   cn_can_be_connected
                                                      )
        except RecursionError as exc_:
            raise ImpossibleTopology(f"For substation {sub_id}: "
                                     "No topology found, maybe the substation is "
                                     "too large or there is a bug in the implementation. "
                                     "It is most likely due to the fact that does not exist "
                                     "a valid switch state for the input topology, but we "
                                     "exclude a bug or a substation too large.") from exc_
        except Exception as exc_:
            raise Grid2OpException(f"Error in the `compute_switches_position` "
                                   f"for sub {sub_id}") from exc_
        if res is None:
            raise ImpossibleTopology(f"For substation {sub_id}")
        return res
    
    def _order_bbs(self, cn_bbs_possible, conn_node_to_bus_id, my_bus):
        # order to favor bbs with same bus, then 
        # bbs without anything
        # then the other bbs (with possibly other element to other bus)
        cn_bbs_possible = list(cn_bbs_possible)
        def mysort(el):
            tmp = conn_node_to_bus_id[el]
            if tmp == my_bus:
                return 1
            elif tmp == 0:
                return 2
            return 3
        res = sorted(cn_bbs_possible, key=mysort)
        return res
    
    def _order_switches(self, switches, cns_, bbs_cn_this_sub):
        # when visiting switches configuration
        # try to visit configuration with as little busbar coupler as possible
        # and (2nd criteria) with as little switch as possible
        both_ = [(sw, cn) for sw, cn in zip(switches, cns_)]
        both_ = sorted(both_, key=lambda el: ((np.isin(el[1], bbs_cn_this_sub)).sum(), len(el[0])))
        return both_
    
    def _aux_dfs_compute_switches_position_disco(self,
                                                 conn_node_visited,
                                                 conn_node_to_bus_id,
                                                 el_cn_id_is,
                                                 all_pos,
                                                 topo_vect,
                                                 conn_graph_this_sub,
                                                 switch_visited,
                                                 switches_state,
                                                 main_obj_id,
                                                 order_pos):
        # the object is disconnected, I suppose here that there exist
        # a switch that directly control this element.
        # With this hyp. this switch will never be changed
        # so there is nothing to do.
        
        # TODO detailed topo: speed optim: this is probably copied too many times
        # switch_visited = copy.deepcopy(switch_visited)
        # switches_state = copy.deepcopy(switches_state)
        conn_node_to_bus_id = copy.deepcopy(conn_node_to_bus_id)
        conn_node_visited = copy.deepcopy(conn_node_visited)

        conn_node_visited[el_cn_id_is] = True
        conn_node_to_bus_id[el_cn_id_is] = -1
        main_obj_id_new = self._aux_find_next_el_id(main_obj_id, all_pos, conn_node_visited, order_pos)
        assert main_obj_id_new != main_obj_id   # TODO detailed topo debug
        if main_obj_id_new is not None:
            # I still need to visit some other elements
            this_res = self._dfs_compute_switches_position(topo_vect,
                                                           conn_graph_this_sub,
                                                           main_obj_id_new,
                                                           all_pos,
                                                           switch_visited,
                                                           switches_state,
                                                           conn_node_visited,
                                                           conn_node_to_bus_id,
                                                           order_pos)
            return this_res
        # all elements have been visited
        return switches_state
    
    def _aux_dfs_compute_switches_position_connect_bbs(self,
                                                       conn_node_to_bus_id,
                                                       my_bus,
                                                       conn_graph_this_sub,
                                                       cn_bbs,
                                                       switch_visited,
                                                       switches_state,
                                                       bbs_cn_this_sub,
                                                       conn_node_visited,
                                                       topo_vect,
                                                       main_obj_id,
                                                       all_pos,
                                                       order_pos
                                                       ):               
        # there is already an element connected to "my" bus, so I need to connect 
        # cn_bbs to some busbar sections, which are of the right color
        which_other_bbs = (conn_node_to_bus_id[self.busbar_section_to_conn_node_id] == my_bus).nonzero()[0]
        other_bbs_cn_ids = self.busbar_section_to_conn_node_id[which_other_bbs]
        
        # remove the nodes that would connect elements to other buses
        # this_tmp_g = conn_graph_this_sub
        this_tmp_g : nx.Graph = copy.deepcopy(conn_graph_this_sub)
        this_tmp_g.remove_nodes_from([el for el in this_tmp_g.nodes if conn_node_to_bus_id[self._cn_pos_in_sub[el]] != 0 and conn_node_to_bus_id[self._cn_pos_in_sub[el]] != my_bus])
        # li_nodes_ok = [el for el in conn_graph_this_sub.nodes 
        #                if (conn_node_to_bus_id[self._cn_pos_in_sub[el]] == 0 
        #                    or 
        #                    conn_node_to_bus_id[self._cn_pos_in_sub[el]] == my_bus)
        #                ]
        # this_tmp_g = conn_graph_this_sub.subgraph(li_nodes_ok)
        # if cn_bbs not in li_nodes_ok:
        #     # no way to connect both busbar in this case
        #     continue                 
        
        for debug_id, other_bbs_cn in enumerate(other_bbs_cn_ids):
            # I try to conenct cn_bbs to other_bbs_cn by avoiding all
            # connectiviy nodes connected to other buses
            other_bbs_cn_is = self._cn_pos_in_sub[other_bbs_cn]
            bid_other = conn_node_to_bus_id[other_bbs_cn_is]
            if bid_other != 0 and bid_other != my_bus:
                # this busbar section is already of the wrong color
                continue
            
            # TODO detailed topo: speed optim, maybe unnecessary to copy here
            n_switch_visited = copy.deepcopy(switch_visited)
            n_switches_state = copy.deepcopy(switches_state)
            n_conn_node_to_bus_id = copy.deepcopy(conn_node_to_bus_id)
            n_conn_node_visited = copy.deepcopy(conn_node_visited)
            
            n_conn_node_to_bus_id[other_bbs_cn_is] = my_bus
            n_conn_node_visited[other_bbs_cn_is] = True
            # so I need to check if a path between cn_bbs and other_bbs_cn
            # of the color `my_bus` exists
            bbs_switch, bbs_cn = self._aux_connect_el_to_switch(conn_graph_this_sub,
                                                                other_bbs_cn,
                                                                cn_bbs,
                                                                n_switch_visited,
                                                                n_switches_state,
                                                                this_tmp_g)
            # speed optim:
            # try first to look for a good results
            # that involves only a minimum amount of busbar coupler
            both_ = self._order_switches(bbs_switch, bbs_cn, bbs_cn_this_sub)
            # end speed optim
            for debug_id2, (bbs_sw, bbs_cn_) in enumerate(both_):
                # there is a way to connect both busbar sections
                # we see if it works out until the end     
                
                # TODO detailed topo: speed optim, maybe unnecessary to copy here
                nn_switch_visited = copy.deepcopy(n_switch_visited)
                nn_switches_state = copy.deepcopy(n_switches_state)
                nn_conn_node_to_bus_id = copy.deepcopy(n_conn_node_to_bus_id)
                nn_conn_node_visited = copy.deepcopy(n_conn_node_visited)
                       
                nn_switch_visited[bbs_sw] = True
                nn_switches_state[bbs_sw] = True
                nn_conn_node_visited[bbs_cn_] = True
                nn_conn_node_to_bus_id[bbs_cn_] = my_bus
                this_res = self._dfs_compute_switches_position(topo_vect,
                                                               conn_graph_this_sub,
                                                               main_obj_id,
                                                               all_pos,
                                                               nn_switch_visited,
                                                               nn_switches_state,
                                                               nn_conn_node_visited,
                                                               nn_conn_node_to_bus_id,
                                                               order_pos)
                if this_res is not None:
                    # I found a solution by connecting the 
                    # busbar cn_bbs to other_bbs_cn
                    return this_res
                # print(f"\t\t for bbs {debug_id}: {debug_id2} fail to connect bbs {cn_bbs} and {other_bbs_cn}")
        # I cannot connect cn_bbs
        # to a busbar connected to bus `my_bus`
        # for all other `other_bbs_cn` so 
        # I need to backtrack before `cn_bbs` is chosen
        return None
        
    def _dfs_compute_switches_position(self,
                                       topo_vect,  # full topo vect
                                       conn_graph_this_sub,
                                       main_obj_id,  # obj id in the substation
                                       all_pos,  # all position to handle in the sub
                                       switch_visited,  # in the sub
                                       switches_state,  # in the sub
                                       conn_node_visited,  # in the sub
                                       conn_node_to_bus_id,  # in the sub
                                       order_pos
                                       ):
        """should be use for one substation only, otherwise it will not work !"""
        # print(f"_dfs_compute_switches_position: {main_obj_id} / {len(all_pos)}")
        if main_obj_id >= len(all_pos):
            # I affected all objects: a solution has been found !
            return switch_visited
        
        if switch_visited.all():
            # TODO detailed topo do I have to check if result topo is correct ?
            return None
        
        # TODO detailed topo: compute this once and for all
        bbs_cn_this_sub = [el 
                           for el in self.busbar_section_to_conn_node_id 
                           if el in conn_graph_this_sub.nodes]   
        bbs_cn_this_sub = self._cn_pos_in_sub[bbs_cn_this_sub]
        # end todo
        
        el_cn_id = all_pos[order_pos[main_obj_id]]
        el_cn_id_is = self._cn_pos_in_sub[el_cn_id]  # element connectivity node id, in the substation
        my_bus = topo_vect[self.conn_node_to_topovect_id[el_cn_id]]
        cn_bbs_possible = self._conn_node_to_bbs_conn_node_id[el_cn_id]

        # TODO detailed topo: speed optim: this is probably copied too many times
        # DEBUG: make sure input data are not modified
        # switch_visited = copy.deepcopy(switch_visited)
        # switches_state = copy.deepcopy(switches_state)
        # conn_node_to_bus_id = copy.deepcopy(conn_node_to_bus_id)
        # conn_node_visited = copy.deepcopy(conn_node_visited)
        
        str_debug = main_obj_id * "  "
        if conn_node_visited[el_cn_id_is]:
            # object has already been visited, and if so
            # without any issue. I can go to the next one
            if conn_node_to_bus_id[el_cn_id_is] != my_bus:
                # This is not a solution
                return None
            main_obj_id_new = self._aux_find_next_el_id(main_obj_id,
                                                        all_pos,
                                                        conn_node_visited,
                                                        order_pos)
            assert main_obj_id_new != main_obj_id  # TODO detailed topo debug
            # assert main_obj_id > main_obj_prev
            if main_obj_id_new is not None:
                # still some work to do
                return self._dfs_compute_switches_position(topo_vect,
                                                           conn_graph_this_sub,
                                                           main_obj_id_new,
                                                           all_pos,
                                                           switch_visited,
                                                           switches_state,
                                                           conn_node_visited,
                                                           conn_node_to_bus_id,
                                                           order_pos)     
            else:
                # a solution has been found
                return switches_state    
            
        if my_bus == -1:
            # special case if the element is disconnected
            # remember I specified in the requirement that an element
            # should be controled by a unique switch. 
            # otherwise I would have to make a loop here too !
            return self._aux_dfs_compute_switches_position_disco(
                                                 conn_node_visited,
                                                 conn_node_to_bus_id,
                                                 el_cn_id_is,
                                                 all_pos,
                                                 topo_vect,
                                                 conn_graph_this_sub,
                                                 switch_visited,
                                                 switches_state,
                                                 main_obj_id,
                                                 order_pos)
            
        # speed optim: reorder the exploration of the busbar section
        better_order = self._order_bbs(cn_bbs_possible, conn_node_to_bus_id, my_bus)
        # end speed optim
        
        for cn_bbs in better_order:  # chose a busbar section                
            # TODO detailed topo: speed optim: this is probably copied too many times
            if main_obj_id <= 5:
                print(f"obj {main_obj_id}, bbs {cn_bbs}, bus : {my_bus}, {conn_node_to_bus_id}")
            n_switch_visited = copy.deepcopy(switch_visited)
            n_switches_state = copy.deepcopy(switches_state)
            n_conn_node_to_bus_id = copy.deepcopy(conn_node_to_bus_id)
            n_conn_node_visited = copy.deepcopy(conn_node_visited)
            
            cn_bbs_is = self._cn_pos_in_sub[cn_bbs]  # position in the substation
            if n_conn_node_visited[cn_bbs_is]:
                if my_bus != n_conn_node_to_bus_id[cn_bbs_is]:
                    # cannot assign on the same busbar section two objects not on the same bus
                    # so I need to "backtrack"
                    continue
                
                # cn_bbs has already the right color, I need to find a direct 
                # path between cn_bbs and the current element
                # and return it 
                
            elif (n_conn_node_to_bus_id == my_bus).any():
                # n_conn_node_visited[cn_bbs_is] = True
                # n_conn_node_to_bus_id[cn_bbs_is] = my_bus
                # # me
                # n_conn_node_visited[el_cn_id_is] = True
                # n_conn_node_to_bus_id[el_cn_id_is] = my_bus
                tmp = self._aux_dfs_compute_switches_position_connect_bbs(
                                                       n_conn_node_to_bus_id,
                                                       my_bus,
                                                       conn_graph_this_sub,
                                                       cn_bbs,
                                                       n_switch_visited,
                                                       n_switches_state,
                                                       bbs_cn_this_sub,
                                                       n_conn_node_visited,
                                                       topo_vect,
                                                       main_obj_id,
                                                       all_pos,
                                                       order_pos)
                if tmp is not None:
                    # a solution has been found that connect 
                    # cn_bbs to another busbar
                    return tmp
                else:
                    # I cannot connect cn_bbs
                    # to a busbar connected to bus `my_bus`
                    # for all other `other_bbs_cn` so 
                    # I need to backtrack before `cn_bbs` is chosen
                    continue
            
            # this is a new bus, I try to connect it to some busbar and 
            # see if it leads to some infeasibility
            n_conn_node_visited[el_cn_id_is] = True
            n_conn_node_to_bus_id[el_cn_id_is] = my_bus
            # graph with all busbars remove except the "correct" one
            tmp_g : nx.Graph = copy.deepcopy(conn_graph_this_sub)
            tmp_g.remove_nodes_from([el for el in self.busbar_section_to_conn_node_id if el != cn_bbs])
            
            # check if "main" element can be connected to this busbar
            possible_switches_tmp, cn_visited_tmp = self._aux_connect_el_to_switch(conn_graph_this_sub,
                                                                                   el_cn_id,
                                                                                   cn_bbs,
                                                                                   n_switch_visited,
                                                                                   n_switches_state,
                                                                                   tmp_g)
            if len(possible_switches_tmp) == 0:
                # this is not possible, I should move to other choice
                # cn_bbs is not correct
                continue
            
            # speed optim: run through the combination in a "smarter"
            # order
            new_order = self._order_switches(possible_switches_tmp, cn_visited_tmp, bbs_cn_this_sub)
            # end speed optim
            
            something_works = False
            this_res = None
            this_tmp_g = conn_graph_this_sub
            
            if main_obj_id <= 2:
                print(f"{str_debug} obj {main_obj_id}, bbs {cn_bbs}, bus : {my_bus}, {conn_node_to_bus_id}: connect element {len(new_order)} different trials")
            # this_tmp_g : nx.Graph = copy.deepcopy(conn_graph_this_sub)
            # this_tmp_g.remove_nodes_from([el for el in this_tmp_g.nodes if n_conn_node_to_bus_id[self._cn_pos_in_sub[el]] != 0 and n_conn_node_to_bus_id[self._cn_pos_in_sub[el]] != my_bus])
            for path, cn_path in new_order:      
                nn_switch_visited = copy.deepcopy(n_switch_visited)
                nn_switches_state = copy.deepcopy(n_switches_state)
                nn_conn_node_to_bus_id = copy.deepcopy(n_conn_node_to_bus_id)
                nn_conn_node_visited = copy.deepcopy(n_conn_node_visited)
                
                nn_switch_visited[path] = True
                nn_switches_state[path] = True
                nn_conn_node_to_bus_id[cn_path] = my_bus
                nn_conn_node_visited[cn_path] = True
                is_working = True
                
                if False:
                    for other_cn_id in all_pos:
                        oth_cn_id_is = self._cn_pos_in_sub[other_cn_id]
                        if nn_conn_node_visited[oth_cn_id_is]:
                            # node already visited, it is assumed to be correct
                            continue
                        
                        # find if all other elements can be assigned to this path (just an assessment for now)
                        bus_other = topo_vect[self.conn_node_to_topovect_id[other_cn_id]]
                        if bus_other != my_bus:
                            # nothing to do if the object is not on the same bus
                            # TODO detailed topo: actually we can do something if an element
                            # in this case is forced to be connected on my_bus it is not possible
                            continue
                                
                        ps_tmp, cns_tmp = self._aux_connect_el_to_switch(conn_graph_this_sub,
                                                                         other_cn_id,
                                                                         cn_bbs,
                                                                         nn_switch_visited,
                                                                         nn_switches_state,
                                                                         this_tmp_g, 
                                                                         assessment=True)
                        if len(ps_tmp) == 0:
                            is_working = False
                            break
                        
                        if len(ps_tmp) == 1:
                            # both objects are on the same bus and there is only one path 
                            # to connect this object to the main object, so I necessarily
                            # toggle all switches on this path and continue
                            tmp_path = ps_tmp[0]
                            nn_switch_visited[tmp_path] = True
                            nn_switches_state[tmp_path] = True
                            nn_conn_node_visited[cns_tmp[0]] = True
                            nn_conn_node_to_bus_id[cns_tmp[0]] = my_bus
                            
                if not is_working:
                    # this path is not working, I don't use it
                    continue
                else:
                    # this seems to work, I try to see if I can 
                    # handle all the remaining elements
                    main_obj_id_new = self._aux_find_next_el_id(main_obj_id, all_pos, nn_conn_node_visited, order_pos)
                    assert main_obj_id_new != main_obj_id  # TODO detailed topo debug
                    if main_obj_id_new is not None:
                        # I still need to visit some other elements
                        this_res = self._dfs_compute_switches_position(topo_vect,
                                                                       conn_graph_this_sub,
                                                                       main_obj_id_new,
                                                                       all_pos,
                                                                       nn_switch_visited,
                                                                       nn_switches_state,
                                                                       nn_conn_node_visited,
                                                                       nn_conn_node_to_bus_id,
                                                                       order_pos)
                    else:
                        # I found a correct path
                        return nn_switches_state
                    if this_res is not None:
                        something_works = True
                        return this_res
                    else:
                        # I need to back track
                        something_works = False
                
            if something_works:
                # I found a solution valid for everything
                return this_res
            else:
                # no solution found, this bus is not possible
                continue
        # If I end up here it's because 
        # none of the `cn_bbs in cn_bbs_possible` are working
        # so there is not solution
        return None
    
    def _aux_find_next_el_id(self, main_obj_id, all_pos, n_conn_node_visited, order_pos):
        still_more_els = True
        while n_conn_node_visited[self._cn_pos_in_sub[all_pos[order_pos[main_obj_id]]]]:
            main_obj_id += 1
            if main_obj_id >= len(all_pos):
                still_more_els = False
                break
        if still_more_els:
            return main_obj_id
        return None           
        
    def _aux_connect_el_to_switch(self,
                                  conn_graph_this_sub,
                                  el_cn_id,
                                  cn_bbs, 
                                  switch_visited,
                                  switches_state,
                                  tmp_g,
                                  assessment=False):
        """connect the connectivity node `el_cn_id` (representing an element) to 
        the connectivity node representing a busbar `cn_bbs` and should return all possible ways
        to connect it without having to traverse another busbar
        """     
        # TODO detailed topo: time optim: in tmp_g,
        # you can remove the edges that does not satisfy 
        # `switches_state[sws_is] | ~switch_visited[sws_is]`
        # directly in the graph, so that I don't need to check 
        # that in post processing
        
        # TODO detailed topo: label the "tmp_g" directly with the position
        # in the substation and not the "global" position
        
        # TODO detailed topo: add the information about the current bus to target
        # and the already assigned buses
        # so that this function can also check that no other "buses"
        # are in the path
        
        res_switch = []
        res_cn = []
        for cn_path in nx.all_simple_paths(tmp_g, el_cn_id, cn_bbs):
            # retrieve the switch id
            sws_id = np.array([conn_graph_this_sub.edges[cn_path[i], cn_path[i+1]]["id"] for i in range(len(cn_path)-1)])
            
            sws_is = self._sw_pos_in_sub[sws_id]
            if not (switches_state[sws_is] | ~switch_visited[sws_is]).all():
                continue
            
            res_switch.append(sws_is)
            res_cn.append(self._cn_pos_in_sub[np.array(cn_path)])
            if assessment and len(res_switch) >= 2:
                # at least two paths that will be visited later
                # when the element id will be the "main" object
                # (in this case, only one is enough; i put two so that
                # the rest of the code works - if only one then the dfs routine
                # does some stuff)
                break
        return res_switch, res_cn        
        
    def from_switches_position(self,
                               switches_state : np.ndarray,
                               subs_changed : Optional[np.ndarray]=None):
        if switches_state.shape[0] != self.switches.shape[0]:
            raise Grid2OpException("Impossible to compute the nodal topology from "
                                   "the switches as you did not provide the state "
                                   "of the correct number of switches: "
                                   f"expected {self.switches.shape[0]} "
                                   f"found {switches_state.shape[0]}")
        if subs_changed is None:
            subs_changed = np.ones(self._n_sub, dtype=dt_bool)
            
        if subs_changed.shape[0] != self._n_sub:
            raise Grid2OpException("Incorrect number of substation provided in the "
                                   "subs_changed argument (it should be a mask "
                                   "indicating for each one whether this substation "
                                   "has been modified or not)")
        
        # TODO detailed topo
        # opposite of `compute_switches_position`
        topo_vect = np.zeros(self._dim_topo, dtype=dt_int)
        if self.conn_node_to_shunt_id is not None:
            shunt_bus = np.zeros(self._n_shunt, dtype=dt_int)
        else:
            shunt_bus = None
            
        # TODO detailed topo: find a way to accelarate it
        for sub_id in range(self._n_sub):
            if not subs_changed[sub_id]:
                continue
            
            bbs_this_sub = self.busbar_section_to_subid == sub_id  # bbs = busbar section
            bbs_id = bbs_this_sub.nonzero()[0]
            bbs_id_inv = np.zeros(bbs_id.max() + 1, dtype=dt_int) - 1
            bbs_id_inv[bbs_id] = np.arange(bbs_id.shape[0])
            bbs_handled = np.zeros(bbs_id.shape[0], dtype=dt_bool)
            mask_s_this_sub = self.switches[:, type(self).SUB_COL] == sub_id
            switches_this_sub = self.switches[mask_s_this_sub,:]
            switches_state_this_sub = switches_state[mask_s_this_sub]
            mask_cn_this_sub = self.conn_node_to_subid == sub_id
            cn_to_tv_id = self.conn_node_to_topovect_id[mask_cn_this_sub]
            # by default elements of this subs are disconnected
            topo_vect[cn_to_tv_id[cn_to_tv_id != -1]] = -1
            
            if self.conn_node_to_shunt_id is not None:
                cn_to_sh_id = self.conn_node_to_shunt_id[mask_cn_this_sub]
                # by default all shunts are connected
                shunt_bus[cn_to_sh_id[cn_to_sh_id != -1]] = -1
            bbs_id_this_sub = 0
            bbs_node_id = 1
            while True:
                if bbs_handled[bbs_id_this_sub]:
                    # this busbar section has already been process
                    bbs_id_this_sub += 1
                    continue
                
                connected_conn_node = np.array([bbs_id[bbs_id_this_sub]])
                # now find all "connection node" connected to this busbar section
                while True:
                    add_conn_2 = np.isin(switches_this_sub[:, type(self).CONN_NODE_1_ID_COL], connected_conn_node) & switches_state_this_sub
                    add_conn_1 = np.isin(switches_this_sub[:, type(self).CONN_NODE_2_ID_COL], connected_conn_node) & switches_state_this_sub
                    if add_conn_1.any() or add_conn_2.any():
                        size_bef = connected_conn_node.shape[0] 
                        connected_conn_node = np.concatenate((connected_conn_node,
                                                              switches_this_sub[add_conn_2, type(self).CONN_NODE_2_ID_COL]))
                        connected_conn_node = np.concatenate((connected_conn_node,
                                                              switches_this_sub[add_conn_1, type(self).CONN_NODE_1_ID_COL]))
                        connected_conn_node = np.unique(connected_conn_node)
                        if connected_conn_node.shape[0] == size_bef:
                            # nothing added
                            break
                    else:
                        break
                    
                # now connect all real element link to the connection node to the right bus id
                topo_vect_id = self.conn_node_to_topovect_id[connected_conn_node]  # keep only connected "connection node" that are connected to an element
                topo_vect_id = topo_vect_id[topo_vect_id != -1]  
                topo_vect_id = topo_vect_id[topo_vect[topo_vect_id] == -1]  # remove element already assigned on a bus
                topo_vect[topo_vect_id] = bbs_node_id  # assign the current bus bar section id
                # now handle the shunts
                if self.conn_node_to_shunt_id is not None:
                    shunt_id = self.conn_node_to_shunt_id[connected_conn_node]  # keep only connected "connection node" that are connected to an element
                    shunt_id = shunt_id[shunt_id != -1]  
                    shunt_id = shunt_id[shunt_bus[shunt_id] == -1]  # remove element already assigned on a bus
                    shunt_bus[shunt_id] = bbs_node_id  # assign the current bus bar section id
                
                # say we go to the next bus id
                bbs_node_id += 1
                
                # now find the next busbar section at this substation not handled
                bbs_conn_this = connected_conn_node[np.isin(connected_conn_node, bbs_id)]
                bbs_handled[bbs_id_inv[bbs_conn_this]] = True
                stop = False
                while True:
                    bbs_id_this_sub += 1
                    if bbs_id_this_sub >= bbs_handled.shape[0]:
                        stop = True
                        break
                    if not bbs_handled[bbs_id_this_sub]:
                        stop = False
                        break
                if stop:
                    # go to next substation as all the busbar sections to 
                    # this substation have been processed
                    break
        return topo_vect, shunt_bus
    
    def _aux_check_pos_topo_vect(self,
                                 el_ids,  # eg load_to_conn_node_id
                                 vect_pos_tv, # eg gridobj_cls.load_pos_topo_vect
                                 el_nm, # eg "load"
                                 ):
        el_tv_id = self.conn_node_to_topovect_id[el_ids]
        if (vect_pos_tv != el_tv_id).any():
            raise Grid2OpException(f"Inconsistency in `conn_node_to_topovect_id` and `switch` for {el_nm}: "
                                   f"Some switch representing {el_nm} do not have the same "
                                   f"`conn_node_to_topovect_id` and `gridobj_cls.{el_nm}_pos_topo_vect`")
        
    def check_validity(self, gridobj_cls: "grid2op.Space.GridObjects.GridObjects"):
        cls = type(self)
        if self._n_sub is None or self._n_sub == -1:
            self._n_sub = gridobj_cls.n_sub
        if self._n_sub != gridobj_cls.n_sub:
            raise Grid2OpException("Incorrect number of susbtation registered "
                                   "in the detailed topology description")
        if self._dim_topo is None or self._dim_topo == -1:
            self._dim_topo = gridobj_cls.dim_topo
        if self._dim_topo != gridobj_cls.dim_topo:
            raise Grid2OpException("Incorrect size for the topology vector registered "
                                   "in the detailed topology description")
        if self._n_shunt is None or self._n_shunt == -1:
            self._n_shunt = gridobj_cls.n_shunt
        if self._n_shunt != gridobj_cls.n_shunt:
            raise Grid2OpException("Incorrect number of shunts registered "
                                   "in the detailed topology description")
            
        if self.conn_node_to_subid.max() != gridobj_cls.n_sub - 1:
            raise Grid2OpException("There are some 'connectivity node' connected to unknown substation, check conn_node_to_subid")
        if self.conn_node_name.shape[0] != self.conn_node_to_subid.shape[0]:
            raise Grid2OpException(f"There are {self.conn_node_name.shape[0]} according to `conn_node_name` "
                                   f"but {self.conn_node_to_subid.shape[0]} according to `conn_node_to_subid`.")
        arr = self.conn_node_to_subid
        arr = arr[arr != -1]
        arr.sort()
        if (np.unique(arr) != np.arange(gridobj_cls.n_sub)).any():
            raise Grid2OpException("There are no 'connectivity node' on some substation, check conn_node_to_subid")
            
        if self.conn_node_to_subid.shape != self.conn_node_name.shape:
            raise Grid2OpException(f"Inconsistencies found on the connectivity nodes: "
                                   f"you declared {len(self.conn_node_to_subid)} connectivity nodes "
                                   f"in `self.conn_node_to_subid` but "
                                   f"{len( self.conn_node_name)} connectivity nodes in "
                                   "`self.conn_node_name`")
            
        nb_conn_node = self.conn_node_name.shape[0]
        all_conn_nodes = np.arange(nb_conn_node)
        if not (np.isin(self.busbar_section_to_conn_node_id, all_conn_nodes)).all():
            raise Grid2OpException("Some busbar are connected to unknown connectivity nodes. Check `busbar_section_to_conn_node_id`")
        if not (np.isin(self.switches[:,cls.CONN_NODE_1_ID_COL], all_conn_nodes)).all():
            raise Grid2OpException(f"Some busbar are connected to unknown connectivity nodes. Check `switches` "
                                   f"(column {cls.CONN_NODE_1_ID_COL})")
        if not (np.isin(self.switches[:,cls.CONN_NODE_2_ID_COL], all_conn_nodes)).all():
            raise Grid2OpException(f"Some busbar are connected to unknown connectivity nodes. Check `switches` "
                                   f"(column {cls.CONN_NODE_2_ID_COL})")
            
        if self.switches[:,cls.CONN_NODE_1_ID_COL].max() >= len(self.conn_node_to_subid):
            raise Grid2OpException("Inconsistencies found in the switches: some switches are "
                                   "mapping unknown connectivity nodes for 'CONN_NODE_1_ID_COL' (too high)")
        if self.switches[:,cls.CONN_NODE_2_ID_COL].max() >= len(self.conn_node_to_subid):
            raise Grid2OpException("Inconsistencies found in the switches: some switches are "
                                   "mapping unknown connectivity nodes for 'CONN_NODE_2_ID_COL' (too high)")
        if self.switches[:,cls.CONN_NODE_1_ID_COL].min() < 0:
            raise Grid2OpException("Inconsistencies found in the switches: some switches are "
                                   "mapping unknown connectivity nodes for 'CONN_NODE_1_ID_COL' (too low)")
        if self.switches[:,cls.CONN_NODE_2_ID_COL].max() >= len(self.conn_node_to_subid):
            raise Grid2OpException("Inconsistencies found in the switches: some switches are "
                                   "mapping unknown connectivity nodes for 'CONN_NODE_2_ID_COL' (too low)")
        
        # check connectivity node info is consistent
        if (self.conn_node_to_subid[self.switches[:,cls.CONN_NODE_1_ID_COL]] != 
            self.conn_node_to_subid[self.switches[:,cls.CONN_NODE_2_ID_COL]]).any():
            raise Grid2OpException("Inconsistencies found in the switches mapping. Some switches are "
                                   "mapping connectivity nodes that belong to different substation.") 
        if (self.conn_node_to_subid[self.switches[:,cls.CONN_NODE_1_ID_COL]] != 
            self.switches[:,cls.SUB_COL]
            ).any():
            raise Grid2OpException(f"Inconsistencies detected between `conn_node_to_subid` and `switches`. "
                                   f"There are some switches declared to belong to some substation (col {cls.SUB_COL}) "
                                   f"or `switches` that connects connectivity node not belonging to this substation "
                                   f"`conn_node_to_subid[switches[:,{cls.CONN_NODE_1_ID_COL}]]`")
        if (self.conn_node_to_subid[self.switches[:,cls.CONN_NODE_2_ID_COL]] != 
            self.switches[:,cls.SUB_COL]
            ).any():
            raise Grid2OpException(f"Inconsistencies detected between `conn_node_to_subid` and `switches`. "
                                   f"There are some switches declared to belong to some substation (col {cls.SUB_COL}) "
                                   f"or `switches` that connects connectivity node not belonging to this substation "
                                   f"`conn_node_to_subid[switches[:,{cls.CONN_NODE_2_ID_COL}]]`")
        
        # check topo vect is consistent
        arr = self.conn_node_to_topovect_id[self.conn_node_to_topovect_id != -1]
        dim_topo = gridobj_cls.dim_topo
        if arr.max() != dim_topo - 1:
            raise Grid2OpException("Inconsistency in `self.conn_node_to_topovect_id`: some objects in the "
                                   "topo_vect are not connected to any switch")
        if arr.shape[0] != dim_topo:
            raise Grid2OpException("Inconsistencies in `self.conn_node_to_topovect_id`: some elements of "
                                   "topo vect are not controlled by any switches.")
        arr.sort() 
        if (arr != np.arange(dim_topo)).any():
            raise Grid2OpException("Inconsistencies in `self.conn_node_to_topovect_id`: two or more swtiches "
                                   "are pointing to the same element")
        self._aux_check_pos_topo_vect(self.load_to_conn_node_id, gridobj_cls.load_pos_topo_vect, "load")
        self._aux_check_pos_topo_vect(self.gen_to_conn_node_id, gridobj_cls.gen_pos_topo_vect, "gen")
        self._aux_check_pos_topo_vect(self.line_or_to_conn_node_id, gridobj_cls.line_or_pos_topo_vect, "line_or")
        self._aux_check_pos_topo_vect(self.line_ex_to_conn_node_id, gridobj_cls.line_ex_pos_topo_vect, "line_ex")
        self._aux_check_pos_topo_vect(self.storage_to_conn_node_id, gridobj_cls.storage_pos_topo_vect, "storage")
            
        # check "el to connectivity nodes" are consistent
        if self.load_to_conn_node_id.shape[0] != gridobj_cls.n_load:
            raise Grid2OpException("load_to_conn_node_id is not with a size of n_load")
        if self.gen_to_conn_node_id.shape[0] != gridobj_cls.n_gen:
            raise Grid2OpException("gen_to_conn_node_id is not with a size of n_gen")
        if self.line_or_to_conn_node_id.shape[0] != gridobj_cls.n_line:
            raise Grid2OpException("line_or_to_conn_node_id is not with a size of n_line")
        if self.line_ex_to_conn_node_id.shape[0] != gridobj_cls.n_line:
            raise Grid2OpException("line_ex_to_conn_node_id is not with a size of n_line")
        if self.storage_to_conn_node_id.shape[0] != gridobj_cls.n_storage:
            raise Grid2OpException("storage_to_conn_node_id is not with a size of n_storage")
        if self.shunt_to_conn_node_id is not None:
            if self.shunt_to_conn_node_id.shape[0] != gridobj_cls.n_shunt:
                raise Grid2OpException("storage_to_conn_node_id is not with a size of n_shunt")
            
        # check some info about the busbars
        if self.busbar_section_to_subid.max() != gridobj_cls.n_sub - 1:
            raise Grid2OpException("There are some 'busbar section' connected to unknown substation, check busbar_section_to_subid")
        arr = self.busbar_section_to_subid
        arr = arr[arr != -1]
        arr.sort()
        if (np.unique(arr) != np.arange(gridobj_cls.n_sub)).any():
            raise Grid2OpException("There are no 'busbar section' on some substation, check busbar_section_to_subid")
        if self.busbar_section_to_subid.shape[0] != self.busbar_section_to_conn_node_id.shape[0]:
            raise Grid2OpException("Wrong size detected for busbar_section_to_subid or busbar_section_to_conn_node_id")
        
        # test "unicity" of connectivity node
        # eg. 1 connectivity nodes cannot represent 2 different objects
        tup =(self.load_to_conn_node_id,
              self.gen_to_conn_node_id,
              self.line_or_to_conn_node_id,
              self.line_ex_to_conn_node_id,
              self.storage_to_conn_node_id,
              self.busbar_section_to_conn_node_id)
        shape_th = (gridobj_cls.n_load + 
                    gridobj_cls.n_gen +
                    2 * gridobj_cls.n_line +
                    gridobj_cls.n_storage +
                    self.busbar_section_to_conn_node_id.shape[0])
        if self.shunt_to_conn_node_id is not None:
            tup = tup + (self.shunt_to_conn_node_id,)
            shape_th += self._n_shunt
        conn_nodes = np.concatenate(tup)
        if np.unique(conn_nodes).shape[0] != shape_th:
            raise Grid2OpException("It appears the same connectivity node represent "
                                   "different element (for example it could represent "
                                   "at the same time a load and a busbar section or "
                                   "a generator and the origin side of a powerline)")
            
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
        res["_from_ieee_grid"] = self._from_ieee_grid
        res["_n_sub"] = int(self._n_sub)
        res["_dim_topo"] = int(self._dim_topo)
        res["_n_shunt"] = int(self._n_shunt)
        
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
            "conn_node_to_topovect_id",
            (lambda arr: [int(el) for el in arr]) if as_list else lambda arr: arr.flatten(),
            copy_,
        )
        if self.conn_node_to_shunt_id is not None:
            save_to_dict(
                res,
                self,
                "conn_node_to_shunt_id",
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
        save_to_dict(
            res,
            self,
            "busbar_section_to_conn_node_id",
            (lambda arr: [int(el) for el in arr]) if as_list else None,
            copy_,
        )
        save_to_dict(
            res,
            self,
            "busbar_section_to_subid",
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
        
        res.switches = extract_from_dict(
            dict_, "switches", lambda x: np.array(x).astype(dt_int)
        )
        res.switches = res.switches.reshape((-1, 3))
        
        res.conn_node_to_topovect_id = extract_from_dict(
            dict_, "conn_node_to_topovect_id", lambda x: np.array(x).astype(dt_int)
        )
        res._from_ieee_grid = bool(dict_["_from_ieee_grid"])
        res._n_sub = int(dict_["_n_sub"])
        res._dim_topo = int(dict_["_dim_topo"])
        res._n_shunt = int(dict_["_n_shunt"])
        
        if "conn_node_to_shunt_id" in dict_:
            res.conn_node_to_shunt_id = extract_from_dict(
                dict_, "conn_node_to_shunt_id", lambda x: np.array(x).astype(dt_int)
            )
        else:
            # shunts are not supported
            res.conn_node_to_shunt_id = None
        
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
        res.busbar_section_to_conn_node_id = extract_from_dict(
            dict_, "busbar_section_to_conn_node_id", lambda x: x
        )
        res.busbar_section_to_subid = extract_from_dict(
            dict_, "busbar_section_to_subid", lambda x: x
        )
        if "shunt_to_conn_node_id" in dict_:
            res.shunt_to_conn_node_id = extract_from_dict(
                dict_, "shunt_to_conn_node_id", lambda x: x
            )
        
        # TODO detailed topo
        return res