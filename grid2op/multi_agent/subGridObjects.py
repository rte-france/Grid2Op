# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Space.GridObjects import GridObjects
import numpy as np


class SubGridObjects(GridObjects) :

    sub_orig_ids = None
    local2global = dict()
    mask_load = None
    mask_gen = None
    mask_storage = None
    mask_line_or = None
    mask_line_ex = None
    mask_shunt = None
    agent_name : str = None
    n_line_or = -1
    n_line_ex = -1
    
    def __init__(self):
        super().__init__()
        
        
        #self.mask_load = np.isin(self.grid.load_to_subid, self.sub_ids)
        #self.mask_gen = np.isin(self.grid.gen_to_subid, self.sub_ids)
        #self.mask_storage = np.isin(self.grid.storage_to_subid, self.sub_ids)
        #
        #self.name_load = self.grid.name_load[self.mask_load]
        #self.n_load = len(self.name_load)
        #self.name_gen = self.grid.name_gen[self.mask_gen]
        #self.n_gen = len(self.name_gen)
        #self.name_storage = self.grid.name_storage[self.mask_storage]
        #self.n_storage = len(self.name_storage)
        #
        #self.local2global['sub'] = dict(
        #    zip(
        #        list(range(len(self.name_sub))), self.sub_ids
        #    )
        #)
        #self.local2global['gen'] = dict(
        #    zip(
        #        list(range(len(self.name_gen))), list(np.where(np.array(self.mask_gen) == True))
        #    )
        #)
        #self.local2global['storage'] = dict(
        #    zip(
        #        list(range(len(self.name_storage))), list(np.where(np.array(self.mask_storage) == True))
        #    )
        #)
    
    