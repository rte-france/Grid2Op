# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

"""
This script provides a possible implementation, based on pandapower of the "change the load" part of the "grid2op backend loop".

It get back the loading function from Step1, implements the "apply_action" relevant for the "change_load", the "runpf" method
(to compute the powerflow) and then the "loads_info"

"""
import numpy as np
import pandapower as pp
from typing import Optional, Tuple, Union

from Step1_loading import CustomBackend_Step1


class CustomBackend_Step2(CustomBackend_Step1):
    def apply_action(self, backendAction: Union["grid2op.Action._backendAction._BackendAction", None]) -> None:
        # the following few lines are highly recommended
        if backendAction is None:
            return
        
        (
            active_bus,
            (prod_p, prod_v, load_p, load_q, storage),
            _,
            shunts__,
        ) = backendAction()
        
        # change the active values of the loads
        for load_id, new_p in load_p:
            self._grid.load["p_mw"].iloc[load_id] = new_p
        # change the reactive values of the loads
        for load_id, new_q in load_q:
            self._grid.load["q_mvar"].iloc[load_id] = new_q
            
    def runpf(self, is_dc : bool=False) -> Tuple[bool, Union[Exception, None]]:
        # possible implementation of the runpf function
        try:
            if is_dc:
                pp.rundcpp(self._grid, check_connectivity=False)
            else:
                pp.runpp(self._grid, check_connectivity=False)
            return self._grid.converged, None
        except pp.powerflow.LoadflowNotConverged as exc_:
            # of the powerflow has not converged, results are Nan
            return False, exc_
    
    def loads_info(self)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # carefull with copy / deep copy
        load_p = self._grid.res_load["p_mw"].values  # in MW
        load_q = self._grid.res_load["q_mvar"].values  # in MVAr
        
        # load_v is the voltage magnitude at the bus at which the load is connected.
        # in pandapower this is not straightforward. We first need to retrieve the
        # voltage in per unit of the bus to which each load is connected.
        # And then we convert the pu to kV. This is what is done below.
        load_v = self._grid.res_bus.iloc[self._grid.load["bus"].values]["vm_pu"].values  # in pu
        load_v *= self._grid.bus.iloc[self._grid.load["bus"].values]["vn_kv"].values # in kv
        return load_p, load_q, load_v


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
    env, obs = make_env_for_backend(env_name, CustomBackend_Step2)
    
    a_grid = os.path.join(path_data_test, env_name, "grid.json")
    
    # we highly recommend to do these 3 steps (this is done automatically by grid2op... of course. See an example of the "complete" 
    # backend)
    backend = CustomBackend_Step2()
    backend.load_grid(a_grid)
    backend.assert_grid_correct()  
    #########
       
    # this is how "user" manipute the grid
    new_load_p = obs.load_p * 1.1
    new_load_q = obs.load_q * 0.9
    action = env.action_space({"injection": {"load_p": new_load_p,
                                             "load_q": new_load_q}})
    # we could have written 
    # > action = env.action_space({"injection": {"load_p": [9. , 7.9, 7.7],
    # >                                          "load_q": [6.3, 5.5, 5.4]}})
    # for the environment "rte_case5_example" but we want this script to be usable
    # with the other environments that have different number of loads (so you need different
    # vector of different size... this is why we use the obs.load_p and obs.load_q that already
    # have the proper size)
    
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
    load_p, load_q, load_v = backend.loads_info()
    
    print(f"{load_p = }")
    print(f"{load_q = }")
    print(f"{load_v = }")
    assert np.isclose(np.sum(load_p), np.sum(new_load_p))
    assert np.isclose(np.sum(load_q), np.sum(new_load_q))
    