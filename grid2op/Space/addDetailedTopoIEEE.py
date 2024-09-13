# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Space.detailed_topo_description import DetailedTopoDescription


class AddDetailedTopoIEEE:    
    """This class allows to add some detailed topology for the ieee networks, because 
    most of the time this information is not present in the released grid (only
    buses information is present in the description of the IEEE grid used for grid2op
    environment as of writing).
    
    If you want to use it, you can by doing the following (or something similar)
    
    .. code-block:: python
    
        import grid2op
        from grid2op.Space import AddDetailedTopoIEEE
        from grid2op.Backend import PandaPowerBackend  # or any other backend (*eg* lightsim2grid)
        
        class PandaPowerBackendWithDetailedTopo(AddDetailedTopoIEEE, PandaPowerBackend):
            pass
        
        env_name = "l2rpn_case14_sandbox"
        env = grid2op.make(env_name, backend=PandaPowerBackendWithDetailedTopo())
        # do wathever you want, with the possibility to operate switches.
        
    More specifically, this class will build each substation in the following way, 
    with each substation :
    
    - counting as many busbars as there are of `n_busbar_per_substation` on the grid
      (2 by default, but can be changed with `env = grid2op.make(..., n_busbar=XXX)`
    - having the possibility to connect each pairs of busbar together with an 
      appropriate switch (so you will have, **per substation** exactly 
      `n_busbar * (n_busbar - 1) // 2` switches allowing to connect them)
    - having the possibility to disconnect each element of the grid independantly of
      anything else. This means there is `n_load + n_gen + n_storage + 2 * n_line + n_shunt`
      such switch like this in total
    - having the possibility to connect each element to each busbar. This means 
      there is `n_busbar * (n_load + n_gen + n_storage + 2 * n_line + n_shunt)` such
      switches on the grid.
        
    Here is the number of switches for some released grid2op environment (with 2 busbars - the default- per substation ):
    
    - `l2rpn_case14_sandbox`: 188
    - `l2rpn_neurips_2020_track1`: 585
    - `l2rpn_neurips_2020_track2`: 1759
    - `l2rpn_wcci_2022`: 1756
    - `l2rpn_idf_2023`: 1780
    
    .. warning::
        As you can see, by using directly the switches to control the grid, the action space blows up. In this case you can 
        achieve exactly the same as the "classic" grid2op representation, but instead of having 
        an action space with a size of `n_load + n_gen + n_storage + 2 * n_line + n_shunt` (for chosing on which busbar you
        want to connect the element) and again `n_load + n_gen + n_storage + 2 * n_line + n_shunt` (for chosing if you 
        want to connect / disconnect each element) you end up with an action space of
        `(n_busbar + 1) * (n_load + n_gen + n_storage + 2 * n_line + n_shunt) + n_sub * (n_busbar * (n_busbar - 1) // 2)`
        
        This is of course to represent **exactly** the same actions: there are no more (and no less) action you can
        do with the switches that you cannot do in the "original" grid2op representation.
        
        This gives, for some grid2op environments:
        
        ==========================  =======================  =============
        env name                    original representation  with switches
        ==========================  =======================  =============
        l2rpn_case14_sandbox        116                      188
        l2rpn_neurips_2020_track1   366                      585
        l2rpn_neurips_2020_track2   1094                     1759
        l2rpn_wcci_2022             1092                     1756
        l2rpn_idf_2023              1108                     1780
        ==========================  =======================  =============
        
        
    """
    def load_grid(self, path=None, filename=None):
        super().load_grid(path, filename)
        self.detailed_topo_desc = DetailedTopoDescription.from_ieee_grid(self)
