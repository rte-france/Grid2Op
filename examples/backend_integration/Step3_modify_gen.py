# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

"""
This script provides a possible implementation, based on pandapower of 
the "change the generators" part of the "grid2op backend loop".

It get back the loading function from Step2, implements 
the "apply_action" relevant for the "change generators" 
and the "generators_info".

NB: the "runpf" is taken from CustomBackend_Step2 

"""
import numpy as np
from typing import Optional, Tuple, Union

from Step2_modify_load import CustomBackend_Step2


class CustomBackend_Step3(CustomBackend_Step2):
    def apply_action(self, backendAction: Union["grid2op.Action._backendAction._BackendAction", None]) -> None:
        # the following few lines are highly recommended
        if backendAction is None:
            return
        
        # loads are modified in the previous script
        super().apply_action(backendAction)
        
        (
            active_bus,
            (prod_p, prod_v, load_p, load_q, storage),
            _,
            shunts__,
        ) = backendAction()
        
        # change the active value of generators
        for gen_id, new_p in prod_p:
            self._grid.gen["p_mw"].iloc[gen_id] = new_p
            
        # for the voltage magnitude, pandapower expects pu but grid2op provides kV,
        # so we need a bit of change
        for gen_id, new_v in prod_v:
            self._grid.gen["vm_pu"].iloc[gen_id] = new_v  # but new_v is not pu !
            self._grid.gen["vm_pu"].iloc[gen_id] /= self._grid.bus["vn_kv"][
                self.gen_to_subid[gen_id]
            ]  # now it is :-)
    
    def generators_info(self)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        prod_p = self._grid.res_gen["p_mw"].values  # in MW
        prod_q = self._grid.res_gen["q_mvar"].values  # in MVAr
        
        # same as for load, gen_v is not directly accessible in pandapower
        # we first retrieve the per unit voltage, then convert it to kV
        prod_v = self._grid.res_gen["vm_pu"].values
        prod_v *= (
            self._grid.bus["vn_kv"].iloc[self.gen_to_subid].values
        )  # in kV
        return prod_p, prod_q, prod_v


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
    env, obs = make_env_for_backend(env_name, CustomBackend_Step3)
    
    a_grid = os.path.join(path_data_test, env_name, "grid.json")
    
    # we highly recommend to do these 3 steps (this is done automatically by grid2op... of course. See an example of the "complete" 
    # backend)
    backend = CustomBackend_Step3()
    backend.load_grid(a_grid)
    backend.assert_grid_correct()  
    #########
    
    new_gen_p = obs.gen_p * 1.1
    new_gen_v = obs.gen_v * 1.05
    # this is how "user" manipute the grid
    action = env.action_space({"injection": {"prod_p": new_gen_p,
                                             "prod_v": new_gen_v}})
    # we could have written 
    # > action = env.action_space({"injection": {"prod_p": [ 0.99    , 29.688126],
    # >                                          "prod_v": [107.1, 107.1]}})
    # for the environment "rte_case5_example" but we want this script to be usable
    # with the other environments that have different number of generators (so you need different
    # vector of different size... this is why we use the obs.gen_p and obs.gen_v that already
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
    gen_p, gen_q, gen_v = backend.generators_info()
    
    print(f"{gen_p = }")
    print(f"{gen_q = }")
    print(f"{gen_v = }")
    # some gen_p might be slightly different than the setpoint 
    # due to the slack ! (this is why we cannot assert things based on gen_p...)
    assert np.allclose(gen_v, new_gen_v) 
    # the assertion above works because there is no limit on reactive power absorbed / produced by generators in pandapower.
