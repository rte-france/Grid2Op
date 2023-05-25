import numpy as np
from itertools import chain
from grid2op.Rules.BaseRules import BaseRules
from grid2op.Rules.LookParam import LookParam
from grid2op.Rules.PreventReconnection import PreventReconnection
from grid2op.Rules.PreventDiscoStorageModif import PreventDiscoStorageModif
from grid2op.Exceptions import (
    IllegalAction,
)

class RulesByArea(BaseRules):
    """
    This subclass combine both :class:`LookParam` and :class:`PreventReconnection` to be applied on defined areas of a grid.
    An action is declared legal if and only if:

      - It doesn't disconnect / reconnect more power lines within each area than  what stated in the actual game _parameters
        :class:`grid2op.Parameters`
      - It doesn't attempt to act on more substations within each area that what is stated in the actual game _parameters
        :class:`grid2op.Parameters`
      - It doesn't attempt to modify the power produce by a turned off storage unit

    """

    def __init__(self, areas_list):
        """
        areas_list : list of areas, each placeholder containing the ids of substations of each defined area
        """
        self.substations_id_by_area = {i : sorted(k) for i,k in enumerate(areas_list)}


    def __call__(self, action, env):
        """
        See :func:`BaseRules.__call__` for a definition of the _parameters of this function.
        """
        self.lines_id_by_area = {key : sorted(list(chain(*[[item for item in np.where(env.line_or_to_subid == subid)[0]
                                   ] for subid in subid_list]))) for key,subid_list in self.substations_id_by_area.items()}

        is_legal, reason = PreventDiscoStorageModif.__call__(self, action, env)
        if not is_legal:
            return False, reason
        
        is_legal, reason = self.LookParamByArea(action, env)
        if not is_legal:
            return False, reason
        
        return PreventReconnection.__call__(self, action, env)
        

    def can_use_simulate(self, nb_simulate_call_step, nb_simulate_call_episode, param):
        return LookParam.can_use_simulate(
            self, nb_simulate_call_step, nb_simulate_call_episode, param
        )
    
    def LookParamByArea(self, action, env):
        """
        See :func:`BaseRules.__call__` for a definition of the parameters of this function.
        """
        # at first iteration, env.current_obs is None...
        powerline_status = env.get_current_line_status()

        aff_lines, aff_subs = action.get_topological_impact(powerline_status)
        if any([np.sum(aff_lines[line_ids]) > env._parameters.MAX_LINE_STATUS_CHANGED for line_ids in self.lines_id_by_area.values()]):
            ids = [[k for k in np.where(aff_lines)[0] if k in line_ids] for line_ids in self.lines_id_by_area.values()]
            print(ids)
            return False, IllegalAction(
                "More than {} line status affected by the action in one area: {}"
                "".format(env.parameters.MAX_LINE_STATUS_CHANGED, ids)
            )
        if any([np.sum(aff_subs[sub_ids]) > env._parameters.MAX_SUB_CHANGED for sub_ids in self.substations_id_by_area.values()]):
            ids = [[k for k in np.where(aff_subs)[0] if k in sub_ids] for sub_ids in self.substations_id_by_area.values()]
            print(ids)
            return False, IllegalAction(
                "More than {} substation affected by the action in one area: {}"
                "".format(env.parameters.MAX_SUB_CHANGED, ids)
            )
        return True, None



 