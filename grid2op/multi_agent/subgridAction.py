# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Action import PlayableAction, ActionSpace, BaseAction
from grid2op.multi_agent.subGridObjects import SubGridObjects


# TODO (later) make that a meta class too
class SubGridActionSpace(SubGridObjects, ActionSpace):
    def __init__(
        self,
        gridobj,
        legal_action,
        agent_name,
        actionClass=BaseAction,  # need to be a base grid2op type (and not a type generated on the fly)
        ):
        SubGridObjects.__init__(self)
        ActionSpace.__init__(self,
                             gridobj=gridobj,
                             legal_action=legal_action,
                             actionClass=actionClass,
                             _extra_name=agent_name)
    
        
         
        
# TODO (later) make that a "metaclass" with argument the ActionType (here playable action)
class SubGridAction(SubGridObjects, PlayableAction):
    def __init__(self):
        SubGridObjects.__init__(self)
        PlayableAction.__init__(self)
    
    def _obj_caract_from_topo_id_others(self, id_):
        obj_id = None
        objt_type = None
        array_subid = None
        for l_id, id_in_topo in enumerate(self.interco_pos_topo_vect):
            if id_in_topo == id_:
                obj_id = l_id
                side_ = "(or)" if self.interco_is_origin[l_id] else "(ex)"
                objt_type = f"interco {side_}"
                array_subid = self.interco_to_subid
        return obj_id, objt_type, array_subid
