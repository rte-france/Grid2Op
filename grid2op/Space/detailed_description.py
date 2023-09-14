# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

class DetailedDescription(object):
    """This class represent the detail description of the 
    switches in the grid.
    
    It does not say whether switches / breakers / etc. are opened or closed
    just that it exists a switch between this and this
    
    It is const, should be initialized by the backend and never modified afterwards.
    
    It is a const member of the class (not the object, the class !)
    """
    def __init__(self, n_sub: int):
        self.substation = np.arange(n_sub)
        self.busbars = None
        self.busbar_to_subid = None
    
    @classmethod
    def from_init_grid(cls, init_grid):
        res = cls(init_grid.n_sub)
        res.busbars = np.arange(2 * init_grid.n_sub)
        res.busbar_to_subid = res.busbars % init_grid.n_sub
        
        res.load_to_busbar_id = ...
        res.gen_to_busbar_id = ...
        res.line_or_to_busbar_id = ...
        res.line_ex_to_busbar_id = ...
        res.storage_to_busbar_id = ...
        res.shunt_to_busbar_id = ...
        return res