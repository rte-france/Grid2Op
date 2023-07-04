# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
from grid2op.Agent import BaseAgent
from grid2op.Observation import BaseObservation
from grid2op.Action import ActionSpace
from grid2op.Exceptions import AgentError


class RecoPowerlinePerArea(BaseAgent):
    """This class acts like the :class:`RecoPowerlineAgent` but it is able
    to reconnect multiple lines at the same steps (one line per area).
    
    The "areas" are defined by a list of list of substation id provided as input.

    Of course the area you provide to the agent should be the same as the areas
    used in the rules of the game. Otherwise, the agent might try to reconnect
    two powerline "in the same area for the environment" which of course will
    lead to an illegal action.

    You can use it like:
    
    .. code-block::
    
        import grid2op
        from grid2op.Agent import RecoPowerlinePerArea

        env_name = "l2rpn_idf_2023" # (or any other env name supporting the feature)
        env = grid2op.make(env_name)
        agent = RecoPowerlinePerArea(env.action_space, env._game_rules.legal_action.substations_id_by_area)
        

    """
    def __init__(self, action_space: ActionSpace, areas_by_sub_id: dict):
        super().__init__(action_space)
        self.lines_to_area_id = np.zeros(type(action_space).n_line, dtype=int) - 1
        for aread_id, (area_nm, sub_this_area) in enumerate(areas_by_sub_id.items()):
            for line_id, subor_id in enumerate(type(action_space).line_or_to_subid):
                if subor_id in sub_this_area:
                    self.lines_to_area_id[line_id] = aread_id
        if np.any(self.lines_to_area_id == -1):
            raise AgentError("some powerline have no area id")
        self.nb_area = len(areas_by_sub_id)
    
    def act(self, observation: BaseObservation, reward: float, done : bool=False):
        line_stat_s = observation.line_status
        cooldown = observation.time_before_cooldown_line
        can_be_reco = ~line_stat_s & (cooldown == 0)
        if not np.any(can_be_reco):
            # no line to reconnect
            return self.action_space()
        area_used = np.full(self.nb_area, fill_value=False, dtype=bool)
        reco_ids = []
        for l_id in np.where(can_be_reco)[0]:
            if not area_used[self.lines_to_area_id[l_id]]:
                reco_ids.append(l_id)
                area_used[self.lines_to_area_id[l_id]] = True
        res = self.action_space({"set_line_status": [(l_id, +1) for l_id in reco_ids]})
        return res
