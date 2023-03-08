# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

"""
This script provides a possible implementation, based on pandapower of 
the "change topology part of the "grid2op backend loop".

It get back the loading function from Step3, implements 
the "apply_action" relevant for the "change topology" (and copy paste
from Step 4 the "change status").

This script also implement the final "getter" the "get_topo_vect".

NB: the "runpf" is taken from CustomBackend_Step2

"""
import copy
import pandas as pd
import numpy as np

from Step4_modify_line_status import CustomBackend_Step4


class CustomBackend_Minimal(CustomBackend_Step4):
    # We need to perform a "trick" for this functionality to work properly. This is because
    # in the "pandapower representation" there is no explicit substation. You only have "bus".
    # But in grid2op, we assume that it is possible to have two buses per substations.
    # Implementing this using pandapower is rather easy: we will double the number of buses
    # available and say that the first half will be the "bus 1" (at their respective substation)
    # and the second half will be the "bus 2" (at their respective substation)
    # This is what we do in the "load_grid" function below.
    def load_grid(self, path, filename=None):
        # loads and generators are modified in the previous script
        super().load_grid(path, filename)
        
        # please read the note above, this part is specific to pandapower !
        add_topo = copy.deepcopy(self._grid.bus)
        add_topo.index += add_topo.shape[0]
        add_topo["in_service"] = False
        self._grid.bus = pd.concat((self._grid.bus, add_topo))
        
    def _aux_change_bus_or_disconnect(self, new_bus, dt, key, el_id):
        if new_bus == -1:
            dt["in_service"].iloc[el_id] = False
        else:
            dt["in_service"].iloc[el_id] = True
            dt[key].iloc[el_id] = new_bus
            
    # As a "bonus" (see the comments above the "load_grid" function), we can also use the
    # grid2op built-in "***_global()" functions that allows to retrieve the global id
    # (from 0 to n_total_bus-1) directly (instead of manipulating local bus id that
    # are either 1 or 2)
    def apply_action(self, action):
        # the following few lines are highly recommended
        if action is None:
            return
        
        # loads and generators are modified in the previous script
        super().apply_action(action)
                
        # handle the load (see the comment above the definition of this
        # function as to why it's possible to use the get_loads_bus_global)
        loads_bus = action.get_loads_bus_global()
        for load_id, new_bus in loads_bus:
            self._aux_change_bus_or_disconnect(new_bus,
                                               self._grid.load,
                                               "bus",
                                               load_id)
        
        # handle the generators (see the comment above the definition of this
        # function as to why it's possible to use the get_loads_bus_global)
        gens_bus = action.get_gens_bus_global()
        for gen_id, new_bus in gens_bus:
            self._aux_change_bus_or_disconnect(new_bus,
                                               self._grid.gen,
                                               "bus",
                                               gen_id)
        
        # handle the powerlines (largely inspired from the Step4...) (see the comment above the definition of this
        # function as to why it's possible to use the get_lines_or_bus_global)
        n_line_pp = self._grid.line.shape[0]
        
        lines_or_bus = action.get_lines_or_bus_global()
        for line_id, new_bus in lines_or_bus:
            if line_id < n_line_pp:
                dt = self._grid.line
                key = "from_bus"
                line_id_pp = line_id
            else:
                dt = self._grid.trafo
                key = "hv_bus"
                line_id_pp = line_id - n_line_pp
                
            self._aux_change_bus_or_disconnect(new_bus,
                                               dt,
                                               key,
                                               line_id_pp)
        
        lines_ex_bus = action.get_lines_ex_bus_global()
        for line_id, new_bus in lines_ex_bus:
            if line_id < n_line_pp:
                dt = self._grid.line
                key = "to_bus"
                line_id_pp = line_id
            else:
                dt = self._grid.trafo
                key = "lv_bus"
                line_id_pp = line_id - n_line_pp
                
            self._aux_change_bus_or_disconnect(new_bus,
                                               dt,
                                               key,
                                               line_id_pp)        
            
        # and now handle the bus data frame status (specific to pandapower)    
        # we reuse the fact that there is n_sub substation on the grid, 
        # the bus (pandapower) for substation i will be bus i and bus i + n_sub
        # as we explained.
        (
            active_bus,
            (prod_p, prod_v, load_p, load_q, storage),
            _,
            shunts__,
        ) = action()
        bus_is = self._grid.bus["in_service"]
        for i, (bus1_status, bus2_status) in enumerate(active_bus):
            bus_is[i] = bus1_status
            bus_is[i + type(self).n_sub] = bus2_status
    
    def _aux_get_topo_vect(self, res, dt, key, pos_topo_vect, add_id=0):
        # we loop through each element of the table
        # (each table representing either the loads, or the generators or the powerlines or the trafos)
        # then we assign the right bus (local - eg 1 or 2) to the right
        # component of the vector "res"  (the component is given by the "pos_topo_vect" - eg self.load_pos_topo_vect
        # when we look at the loads)
        el_id = 0
        for (status, bus_id) in dt[["in_service", key]].values:
            my_pos_topo_vect = pos_topo_vect[el_id + add_id]
            if status:
                local_bus = self.global_bus_to_local_int(bus_id, my_pos_topo_vect)
            else:
                local_bus = -1
            res[my_pos_topo_vect] = local_bus
            el_id += 1
        
    # it should return, in the correct order, on which bus each element is connected        
    def get_topo_vect(self):
        res = np.full(self.dim_topo, fill_value=-2, dtype=int)
        # read results for load
        self._aux_get_topo_vect(res, self._grid.load, "bus", self.load_pos_topo_vect)
        # then for generators
        self._aux_get_topo_vect(res, self._grid.gen, "bus", self.gen_pos_topo_vect)
        # then each side of powerlines
        self._aux_get_topo_vect(res, self._grid.line, "from_bus", self.line_or_pos_topo_vect)
        self._aux_get_topo_vect(res, self._grid.line, "to_bus", self.line_ex_pos_topo_vect)
        
        # then for the trafos, but remember pandapower trafos are powerlines in grid2Op....
        # so we need to trick it a bit
        # (we can do this trick because we put the trafo "at the end" of the powerline in grid2op
        # in the Step1_loading.py)
        n_line_pp = self._grid.line.shape[0]
        self._aux_get_topo_vect(res, self._grid.trafo, "hv_bus", self.line_or_pos_topo_vect, add_id=n_line_pp)
        self._aux_get_topo_vect(res, self._grid.trafo, "lv_bus", self.line_ex_pos_topo_vect, add_id=n_line_pp)            
        return res
                

if __name__ == "__main__":
    import grid2op
    import os
    from Step0_make_env import make_env_for_backend
    
    path_grid2op = grid2op.__file__
    path_data_test = os.path.join(os.path.split(path_grid2op)[0], "data")
    
    env_name = "l2rpn_wcci_2022_dev"
    # one of:
    # - rte_case5_example: the grid in the documentation (completely fake grid)
    # - l2rpn_case14_sandbox: inspired from IEEE 14
    # - l2rpn_neurips_2020_track1: inspired from IEEE 118 (only a third of it)
    # - l2rpn_wcci_2022_dev: inspired from IEEE 118 (entire grid)
    env, obs = make_env_for_backend(env_name, CustomBackend_Minimal)
    
    a_grid = os.path.join(path_data_test, env_name, "grid.json")
    
    # we highly recommend to do these 3 steps (this is done automatically by grid2op... of course. See an example of the "complete" 
    # backend)
    backend = CustomBackend_Minimal()
    backend.load_grid(a_grid)
    backend.assert_grid_correct()  
    #########
    
    # this is how "user" manipute the grid
    if env_name == "rte_case5_example":
        sub_id = 0
        local_topo = (1, 2, 1, 2, 1, 2)
    elif env_name == "l2rpn_case14_sandbox":
        sub_id = 2
        local_topo = (1, 2, 1, 2)
    elif env_name == "l2rpn_neurips_2020_track1":
        sub_id = 1
        local_topo = (1, 2, 1, 2, 1, 2)
    elif env_name == "l2rpn_wcci_2022_dev":
        sub_id = 3
        local_topo = (1, 2, 1, 2, 1)
    else:
        raise RuntimeError(f"Unknown grid2op environment name {env_name}")
    action = env.action_space({"set_bus": {"substations_id": [(sub_id, local_topo)]}})
    #############################    
    
    # this is technical to grid2op
    bk_act = env._backend_action_class()
    bk_act += action
    ####################################
    
    # this is what the backend receive:
    backend.apply_action(bk_act)
    
    # now run a powerflow
    conv, exc_ = backend.runpf()
    assert conv, f"Power flow diverged with error:\n\t{exc_}"
    
    # and retrieve the results
    p_or, q_or, v_or, a_or = backend.lines_or_info()
    
    print(f"{p_or = }")
    print(f"{q_or = }")
    print(f"{v_or = }")
    print(f"{a_or = }")
    
    topo_vect = backend.get_topo_vect()
    beg_ = np.sum(env.sub_info[:sub_id])
    end_ = beg_ + env.sub_info[sub_id]
    assert np.all(topo_vect[beg_:end_] == local_topo)
    
    # and you can also make a "more powerful" test
    # that test if, from grid2op point of view, the KCL are met or not
    p_subs, q_subs, p_bus, q_bus, diff_v_bus = backend.check_kirchoff()
    # p_subs: active power mismatch at the substation level [shape: nb_substation ]
    # p_subs: reactive power mismatch at the substation level [shape: nb_substation ]
    # p_bus: active power mismatch at the bus level [shape: (nb substation, 2)]
    # p_bus: reactive power mismatch at the bus level [shape: (nb substation, 2)]
    # diff_v_bus: difference between the highest voltage level and the lowest voltage level for 
    #             among all elements connected to the same bus
    # if your "solver" meets the KCL then it should all be 0. (*ie* less than a small tolerance)
    tol = 1e-4
    assert np.all(p_subs <= tol)
    # assert np.all(q_subs <= tol)  # does not work if there are shunts on the grid (not yet coded in the backend)
    assert np.all(p_bus <= tol) 
    # assert np.all(q_bus <= tol)  # does not work if there are shunts on the grid (not yet coded in the backend)
    assert np.all(diff_v_bus <= tol)
