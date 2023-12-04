# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

"""
This script :
1) provide a possible implementation of the "load_grid" function based on pandapower, see
   the implementation of `def load_grid(self, path, filename=None):`
2) emulate the behaviour of grid2op when this funciton is called, see what happens after the
   `if __name__ == "__main__":`

I can also be used as "test" to make sure your backend can be loaded properly for example.

Of course, this script is highly inefficient and can be optimized in multiple ways. Its only
purpose is to be as clear and minimal as possible.

"""

import os
import numpy as np
import grid2op
from typing import Optional, Tuple, Union

from grid2op.Backend import Backend   # required

# to serve as an example
import pandapower as pp


class CustomBackend_Step1(Backend):
    def load_grid(self,
                  path : Union[os.PathLike, str],
                  filename : Optional[Union[os.PathLike, str]]=None) -> None:
        # first load the grid from the file
        full_path = path
        if filename is not None:
            full_path = os.path.join(full_path, filename)
        self._grid = pp.from_json(full_path)
        
        # then fill the "n_sub" and "sub_info"
        self.n_sub = self._grid.bus.shape[0]
        
        # then fill the number and location of loads
        self.n_load = self._grid.load.shape[0]
        self.load_to_subid = np.zeros(self.n_load, dtype=int)
        for load_id in range(self.n_load):
            self.load_to_subid[load_id] = self._grid.load.iloc[load_id]["bus"]
            
        # then fill the number and location of generators
        self.n_gen = self._grid.gen.shape[0]
        self.gen_to_subid = np.zeros(self.n_gen, dtype=int)
        for gen_id in range(self.n_gen):
            self.gen_to_subid[gen_id] = self._grid.gen.iloc[gen_id]["bus"]
            
        # then fill the number and location of storage units
        # self.n_storage = self._grid.storage.shape[0]
        # self.storage_to_subid = np.zeros(self.n_storage, dtype=int)
        # for storage_id in range(self.n_storage):
        #     self.storage_to_subid[storage_id] = self._grid.storage.iloc[storage_id]["bus"]
        
        # WARNING
        # for storage, their description is loaded in a different file (see 
        # the doc of Backend.load_storage_data)
        # to start we recommend you to ignore the storage unit of your grid with:
        self.set_no_storage()
        
        # finally handle powerlines
        # NB: grid2op considers that trafos are powerlines.
        # so we decide here to say: first n "powerlines" of grid2Op
        # will be pandapower powerlines and
        # last k "powerlines" of grid2op will be the trafos of pandapower.
        self.n_line = self._grid.line.shape[0] + self._grid.trafo.shape[0]
        self.line_or_to_subid = np.zeros(self.n_line, dtype=int)
        self.line_ex_to_subid = np.zeros(self.n_line, dtype=int)
        for line_id in range(self._grid.line.shape[0]):
            self.line_or_to_subid[line_id] = self._grid.line.iloc[line_id]["from_bus"]
            self.line_ex_to_subid[line_id] = self._grid.line.iloc[line_id]["to_bus"]
        
        nb_powerline = self._grid.line.shape[0]
        for trafo_id in range(self._grid.trafo.shape[0]):
            self.line_or_to_subid[trafo_id + nb_powerline] = self._grid.trafo.iloc[trafo_id]["hv_bus"]
            self.line_ex_to_subid[trafo_id + nb_powerline] = self._grid.trafo.iloc[trafo_id]["lv_bus"]
            
        # and now the thermal limit
        self.thermal_limit_a = 1000. * np.concatenate(
            (
                self._grid.line["max_i_ka"].values,
                self._grid.trafo["sn_mva"].values
                / (np.sqrt(3) * self._grid.trafo["vn_hv_kv"].values),
            )
        )
            
        self._compute_pos_big_topo()

    def apply_action(self, backendAction: Union["grid2op.Action._backendAction._BackendAction", None]) -> None:
        raise NotImplementedError("Will be detailed in another example script")
    
    def runpf(self, is_dc : bool=False) -> Tuple[bool, Union[Exception, None]]:
        raise NotImplementedError("Will be detailed in another example script")
    
    def get_topo_vect(self) -> np.ndarray:
        raise NotImplementedError("Will be detailed in another example script")
    
    def generators_info(self)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError("Will be detailed in another example script")
    
    def loads_info(self)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError("Will be detailed in another example script")
    
    def lines_or_info(self)-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError("Will be detailed in another example script")
    
    def lines_ex_info(self)-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError("Will be detailed in another example script")


if __name__ == "__main__":
    path_grid2op = grid2op.__file__
    path_data_test = os.path.join(os.path.split(path_grid2op)[0], "data")
    
    env_name = "rte_case5_example"
    # one of:
    # - rte_case5_example: the grid in the documentation (completely fake grid)
    # - l2rpn_case14_sandbox: inspired from IEEE 14
    # - l2rpn_neurips_2020_track1: inspired from IEEE 118 (only a third of it)
    # - l2rpn_wcci_2022_dev: inspired from IEEE 118 (entire grid)
    
    a_grid = os.path.join(path_data_test, env_name, "grid.json")
    
    backend = CustomBackend_Step1()
    backend.load_grid(a_grid)
    
    # grid2op then performs basic check to make sure that the grid is "consistent"
    backend.assert_grid_correct()
    
    # and you can check all the attribute that are required by grid2op (exhaustive list in the
    # GridObjects class)
    
    # name_load
    # name_gen
    # name_line
    # name_sub
    # name_storage
    
    # to which substation is connected each element
    # load_to_subid
    # gen_to_subid
    # line_or_to_subid
    # line_ex_to_subid
    # storage_to_subid
    
    # # which index has this element in the substation vector
    # load_to_sub_pos
    # gen_to_sub_pos
    # line_or_to_sub_pos
    # line_ex_to_sub_pos
    # storage_to_sub_pos

    # # which index has this element in the topology vector
    # load_pos_topo_vect
    # gen_pos_topo_vect
    # line_or_pos_topo_vect
    # line_ex_pos_topo_vect
    # storage_pos_topo_vect
    
    # for example
    print(type(backend).name_load)
    print(type(backend).load_to_subid)
    print(type(backend).load_to_sub_pos)
    print(type(backend).load_pos_topo_vect)
    