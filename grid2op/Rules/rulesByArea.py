# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
from itertools import chain
from grid2op.Rules.BaseRules import BaseRules
from grid2op.Rules.LookParam import LookParam
from grid2op.Rules.PreventReconnection import PreventReconnection
from grid2op.Rules.PreventDiscoStorageModif import PreventDiscoStorageModif
from grid2op.Exceptions import (
    IllegalAction, Grid2OpException
)

class RulesByArea(BaseRules):
    """
    This subclass combine :class:`PreventReconnection`, :class: `PreventDiscoStorageModif` to be applied on the whole grid at once,
    while a specifique method look for the legality of simultaneous actions taken on defined areas of a grid.
    An action is declared legal if and only if:

      - It doesn't reconnect more power lines than what is stated in the actual game _parameters
        :class:`grid2op.Parameters`
      - It doesn't attempt to act on more substations and lines within each area that what is stated in the actual game _parameters
        :class:`grid2op.Parameters`
      - It doesn't attempt to modify the power produce by a turned off storage unit
      
    Example
    ---------
    If you want the environment to take into account the rules by area, you can achieve it with:

    .. code-block:
        import grid2op
        from grid2op.Rules.rulesByArea import RulesByArea
        
        # First you set up the areas within the RulesByArea class
        my_gamerules_byarea = RulesByArea([[0,1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])
        # Then you create your environment with it:
        NAME_OF_THE_ENVIRONMENT = "l2rpn_case14_sandbox"
        env = grid2op.make(NAME_OF_THE_ENVIRONMENT,gamerules_class=my_gamerules_byarea)

    """

    def __init__(self, areas_list):
        """
        The initialization of the rule with a list of list of ids of substations composing the aimed areas.
        Parameters
        ----------
        areas_list : list of areas, each placeholder containing the ids of substations of each defined area
        """
        self.substations_id_by_area = {i : sorted(k) for i,k in enumerate(areas_list)}
        
        
    def initialize(self, env):
        """
        This function is used to inform the class instance about the environment specification and check no substation of the grid are left ouside an area. 
        Parameters
        ----------
        env: :class:`grid2op.Environment.Environment`
            An environment instance properly initialized.

        """
        n_sub = env.n_sub
        n_sub_rule = np.sum([len(set(list_ids)) for list_ids in self.substations_id_by_area.values()])
        if n_sub_rule != n_sub: 
            raise Grid2OpException("The number of listed ids of substations in rule initialization does not match the number of substations of the chosen environement. Look for missing ids or doublon")
        else:
            self.lines_id_by_area = {key : sorted(list(chain(*[[item for item in np.where(env.line_or_to_subid == subid)[0]
                                    ] for subid in subid_list]))) for key,subid_list in self.substations_id_by_area.items()}


    def __call__(self, action, env):
        """
        See :func:`BaseRules.__call__` for a definition of the _parameters of this function.
        """
        is_legal, reason = PreventDiscoStorageModif.__call__(self, action, env)
        if not is_legal:
            return False, reason
            
        is_legal, reason = self._lookparam_byarea(action, env)
        if not is_legal:
            return False, reason
            
        return PreventReconnection.__call__(self, action, env)
        

    def can_use_simulate(self, nb_simulate_call_step, nb_simulate_call_episode, param):
        return LookParam.can_use_simulate(
            self, nb_simulate_call_step, nb_simulate_call_episode, param
        )
    
    def _lookparam_byarea(self, action, env):
        """
        See :func:`BaseRules.__call__` for a definition of the parameters of this function.
        """
        # at first iteration, env.current_obs is None...
        powerline_status = env.get_current_line_status()

        aff_lines, aff_subs = action.get_topological_impact(powerline_status)
        if any([(aff_lines[line_ids]).sum() > env._parameters.MAX_LINE_STATUS_CHANGED for line_ids in self.lines_id_by_area.values()]):
            ids = [[k for k in np.where(aff_lines)[0] if k in line_ids] for line_ids in self.lines_id_by_area.values()]
            return False, IllegalAction(
                "More than {} line status affected by the action in one area: {}"
                "".format(env.parameters.MAX_LINE_STATUS_CHANGED, ids)
            )
        if any([(aff_subs[sub_ids]).sum() > env._parameters.MAX_SUB_CHANGED for sub_ids in self.substations_id_by_area.values()]):
            ids = [[k for k in np.where(aff_subs)[0] if k in sub_ids] for sub_ids in self.substations_id_by_area.values()]
            return False, IllegalAction(
                "More than {} substation affected by the action in one area: {}"
                "".format(env.parameters.MAX_SUB_CHANGED, ids)
            )
        return True, None
 