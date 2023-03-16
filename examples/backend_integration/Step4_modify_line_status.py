# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

"""
This script provides a possible implementation, based on pandapower of 
the "change the line status" part of the "grid2op backend loop".

It get back the loading function from Step3, implements 
the "apply_action" relevant for the "change line status" 
and the "lines_or_info" and "lines_ex_info".

NB: the "runpf" is taken from CustomBackend_Step2

"""
import numpy as np

from Step3_modify_gen import CustomBackend_Step3


class CustomBackend_Step4(CustomBackend_Step3):
    def apply_action(self, action):
        # the following few lines are highly recommended
        if action is None:
            return
        
        # loads and generators are modified in the previous script
        super().apply_action(action)
        
        # disconnected powerlines are indicated because they are
        # connected to bus "-1" in the `get_lines_or_bus()` and
        # `get_lines_ex_bus()`
        # NB : at time of writing, grid2op side a powerline disconnected
        # on a side (*eg* "or" side or "ext" side) is
        # disconnected on both.
        
        # the only difficulty here is that grid2op considers that
        # trafo are also powerline.
        # We already "solved" that by saying that the "k" last "lines"
        # from grid2op point of view will indeed be trafos.
        
        n_line_pp = self._grid.line.shape[0]
        
        # handle the disconnection on "or" side
        lines_or_bus = action.get_lines_or_bus()
        for line_id, new_bus in lines_or_bus:
            if line_id < n_line_pp:
                # a pandapower powerline has bee disconnected in grid2op
                dt = self._grid.line
                line_id_db = line_id
            else:
                # a pandapower trafo has bee disconnected in grid2op
                dt = self._grid.trafo
                line_id_db = line_id - n_line_pp

            if new_bus == -1:
                # element was disconnected
                dt["in_service"].iloc[line_id_db] = False
            else:
                # element was connected
                dt["in_service"].iloc[line_id_db] = True

        lines_ex_bus = action.get_lines_ex_bus()
        for line_id, new_bus in lines_ex_bus:
            if line_id < n_line_pp:
                # a pandapower powerline has bee disconnected in grid2op
                dt = self._grid.line
                line_id_db = line_id
            else:
                # a pandapower trafo has bee disconnected in grid2op
                dt = self._grid.trafo
                line_id_db = line_id - n_line_pp

            if new_bus == -1:
                # element was disconnected
                dt["in_service"].iloc[line_id_db] = False
            else:
                # element was connected
                dt["in_service"].iloc[line_id_db] = True
                
    def _aux_get_line_info(self, colname_powerline, colname_trafo):
        """
        concatenate the information of powerlines and trafo using 
        the convention that "powerlines go first" then trafo
        """
        res = np.concatenate(
            (
                self._grid.res_line[colname_powerline].values,
                self._grid.res_trafo[colname_trafo].values,
            )
        )
        return res

    def lines_or_info(self):
        """
        Main method to retrieve the information at the "origin" side of the powerlines and transformers.

        We simply need to follow the convention we adopted:

        - origin side (grid2op) will be "from" side for pandapower powerline
        - origin side (grid2op) will be "hv" side for pandapower trafo
        - we chose to first have powerlines, then transformers

        (convention chosen in :func:`EducPandaPowerBackend.load_grid`)

        """
        p_or = self._aux_get_line_info("p_from_mw", "p_hv_mw")
        q_or = self._aux_get_line_info("q_from_mvar", "q_hv_mvar")
        v_or = self._aux_get_line_info("vm_from_pu", "vm_hv_pu")  # in pu
        a_or = self._aux_get_line_info("i_from_ka", "i_hv_ka") * 1000  # grid2op expects amps (A) pandapower returns kilo-amps (kA)
        
        # get the voltage in kV (and not in pu)
        bus_id = np.concatenate(
            (
                self._grid.line["from_bus"].values,
                self._grid.trafo["hv_bus"].values,
            )
        )
        v_or *= self._grid.bus.iloc[bus_id]["vn_kv"].values
        
        # there would be a bug in v_or because of the way pandapower
        # internally looks at the extremity of powerlines / trafos.
        # we fix it here:
        status = np.concatenate(
            (
                self._grid.line["in_service"].values,
                self._grid.trafo["in_service"].values,
            )
        )
        v_or[~status] = 0.
        return p_or, q_or, v_or, a_or

    def lines_ex_info(self):
        """
        Main method to retrieve the information at the "extremity" side of the powerlines and transformers.

        We simply need to follow the convention we adopted:

        - extremity side (grid2op) will be "to" side for pandapower powerline
        - extremity side (grid2op) will be "lv" side for pandapower trafo
        - we chose to first have powerlines, then transformers

        (convention chosen in function :func:`EducPandaPowerBackend.load_grid`)

        """
        p_ex = self._aux_get_line_info("p_to_mw", "p_lv_mw")
        q_ex = self._aux_get_line_info("q_to_mvar", "q_lv_mvar")
        v_ex = self._aux_get_line_info("vm_to_pu", "vm_lv_pu")  # in pu
        a_ex = self._aux_get_line_info("i_to_ka", "i_lv_ka") * 1000  # grid2op expects amps (A) pandapower returns kilo-amps (kA)
        
        # get the voltage in kV (and not in pu)
        bus_id = np.concatenate(
            (
                self._grid.line["to_bus"].values,
                self._grid.trafo["lv_bus"].values,
            )
        )
        v_ex *= self._grid.bus.iloc[bus_id]["vn_kv"].values
        
        # there would be a bug in v_ex because of the way pandapower
        # internally looks at the extremity of powerlines / trafos.
        # we fix it here:
        status = np.concatenate(
            (
                self._grid.line["in_service"].values,
                self._grid.trafo["in_service"].values,
            )
        )
        v_ex[~status] = 0.
        return p_ex, q_ex, v_ex, a_ex


if __name__ == "__main__":
    import grid2op
    import os
    from Step0_make_env import make_env_for_backend
    
    path_grid2op = grid2op.__file__
    path_data_test = os.path.join(os.path.split(path_grid2op)[0], "data")
    
    env_name = "rte_case5_example"
    # one of:
    # - rte_case5_example: the grid in the documentation (completely fake grid)
    # - l2rpn_case14_sandbox: inspired from IEEE 14
    # - l2rpn_neurips_2020_track1: inspired from IEEE 118 (only a third of it)
    # - l2rpn_wcci_2022_dev: inspired from IEEE 118 (entire grid)
    env, obs = make_env_for_backend(env_name, CustomBackend_Step4)
    
    a_grid = os.path.join(path_data_test, env_name, "grid.json")
    
    # we highly recommend to do these 3 steps (this is done automatically by grid2op... of course. See an example of the "complete" 
    # backend)
    backend = CustomBackend_Step4()
    backend.load_grid(a_grid)
    backend.assert_grid_correct()  
    #########
    
    # this is how "user" manipute the grid
    # in this I disconnect powerline 0
    action = env.action_space({"set_line_status": [(0, -1)]})
    
    # this is technical to grid2op
    bk_act = env._backend_action_class()
    bk_act += action
    #############
    
    # this is what the backend receive:
    backend.apply_action(bk_act)
    
    # now run a powerflow
    conv, exc_ = backend.runpf()
    assert conv, "powerflow has diverged"
    
    # and retrieve the results
    p_or, q_or, v_or, a_or = backend.lines_or_info()
    
    print("After disconnecting powerline 0: ")
    print(f"{p_or = }")
    print(f"{q_or = }")
    print(f"{v_or = }")
    print(f"{a_or = }")
    assert p_or[0] == 0.
    assert q_or[0] == 0.
    assert v_or[0] == 0.
    assert a_or[0] == 0.
    
    # this is how "user" manipute the grid
    # in this I reconnect powerline 0
    action = env.action_space({"set_line_status": [(0, 1)]})
    
    # this is technical to grid2op (done internally)
    bk_act = env._backend_action_class()
    bk_act += action
    #############
    
    # this is what the backend receive:
    backend.apply_action(bk_act)
    assert conv, "powerflow has diverged"
    
    # now run a powerflow
    conv, exc_ = backend.runpf()
    
    # and retrieve the results
    p_or, q_or, v_or, a_or = backend.lines_or_info()
    
    print("\nAfter reconnecting powerline 0: ")
    print(f"{p_or = }")
    print(f"{q_or = }")
    print(f"{v_or = }")
    print(f"{a_or = }")


    # this is how "user" manipute the grid
    # in this I disconnect the last powerline
    line_id = env.n_line - 1
    action = env.action_space({"set_line_status": [(line_id, -1)]})
    
    # this is technical to grid2op (done internally)
    bk_act = env._backend_action_class()
    bk_act += action
    #############
    
    # this is what the backend receive:
    backend.apply_action(bk_act)
    
    # now run a powerflow
    conv, exc_ = backend.runpf()
    assert conv, "powerflow has diverged"
    
    # and retrieve the results
    p_or, q_or, v_or, a_or = backend.lines_or_info()
    
    print(f"\nAfter disconnecting powerline id {line_id}")
    print(f"{p_or = }")
    print(f"{q_or = }")
    print(f"{v_or = }")
    print(f"{a_or = }")
    assert p_or[line_id] == 0.
    assert q_or[line_id] == 0.
    assert v_or[line_id] == 0.
    assert a_or[line_id] == 0.
