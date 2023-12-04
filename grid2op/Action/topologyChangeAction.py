# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Action.playableAction import PlayableAction


class TopologyChangeAction(PlayableAction):
    """
    This type of :class:`PlayableAction` implements the modifications
    of the grid with "change" topological actions.

    It accepts the key words: "change_line_status" and "change_bus".
    Nothing else is supported and any attempt to use something else
    will have no impact.
    """

    authorized_keys = {"change_line_status", "change_bus"}

    attr_list_vect = ["_change_bus_vect", "_switch_line_status"]
    attr_list_set = set(attr_list_vect)

    def __init__(self):
        super().__init__()
