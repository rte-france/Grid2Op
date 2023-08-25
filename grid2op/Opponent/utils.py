# Copyright (c) 2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from typing import Dict
from grid2op.Opponent.neverAttackBudget import NeverAttackBudget
from grid2op.Opponent.baseOpponent import BaseOpponent
from grid2op.Action import DontAct


def get_kwargs_no_opponent() -> Dict:
    """This dict allows to retrieve a dictionnary you can use as kwargs to disable the opponent.
    
    Examples
    --------
    
    You can use it like
    
    .. code-block::

        import grid2op
        env_name = "l2rpn_case14_sandbox"  # or any other name
        env = grid2op.make(env_name)  # en with possibly an opponent
        
        # if you want to disable the opponent you can do
        kwargs_no_opp = grid2op.Opponent.get_kwargs_no_opponent()
        env_no_opp = grid2op.make(env_name, **kwargs_no_opp)
        # and there the opponent is disabled
        
    """
    res = {
        "opponent_attack_cooldown": 99999999,
        "opponent_attack_duration": 0,
        "opponent_budget_per_ts": 0.,
        "opponent_init_budget": 0.,
        "opponent_action_class": DontAct,
        "opponent_class": BaseOpponent,
        "opponent_budget_class": NeverAttackBudget,
    }
    return res
