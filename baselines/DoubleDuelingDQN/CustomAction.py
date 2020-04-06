# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Action import BaseAction

class CustomAction(BaseAction):
    def __init__(self, gridobj,
                 setSubset=True,
                 changeSubset=True,
                 redispatchSubset=True):
        super().__init__(gridobj)

        self.attr_list_vect = []
        if setSubset:
            self.attr_list_vect.append("_set_line_status")
            self.attr_list_vect.append("_set_topo_vect")
        if changeSubset:
            self.attr_list_vect.append("_change_bus_vect")
            self.attr_list_vect.append("_switch_line_status")
        if redispatchSubset:
            self.attr_list_vect.append("_redispatch")

    def __call__(self):
        return super().__call__()
