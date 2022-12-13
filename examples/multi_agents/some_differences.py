# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
from grid2op.multi_agent import MultiAgentEnv
from grid2op.gym_compat import DiscreteActSpace

if __name__ == "__main__":
    env_name = "l2rpn_case14_sandbox"
    centralized_env = grid2op.make(env_name)
    centralized_discrete = DiscreteActSpace(centralized_env.action_space, attr_to_keep=["set_bus"])
    
    ACTION_DOMAINS = {
        'agent_0' : [0, 1, 2, 3, 4],
        'agent_1' : [5, 6, 7, 8, 9, 10, 11, 12, 13]
    }
    
    ma_env = MultiAgentEnv(centralized_env, ACTION_DOMAINS)
    ma_discrete = {ag_id: DiscreteActSpace(ma_env.action_spaces[ag_id], attr_to_keep=["set_bus"]) for ag_id in ma_env.agents}
    print(f"There are {centralized_discrete.n} total unary actions on the centralized environment, but "
          f"(due to some lacking features) only {sum([el.n for el in ma_discrete.values()])} "
          f"for the decentralized version. "
          f"Basically, you cannot act on the \"interco\" at the moment")
    for sub_id in range(centralized_env.n_sub):
        agent_id = "agent_0"
        if sub_id in ACTION_DOMAINS["agent_1"]:
            agent_id = "agent_1"
        nb_cent = len(centralized_discrete.action_space.get_all_unitary_topologies_set(centralized_discrete.action_space, sub_id))
        
        loc_sub_id = ACTION_DOMAINS[agent_id].index(sub_id)
        nb_ma = len(ma_discrete[agent_id].action_space.get_all_unitary_topologies_set(ma_discrete[agent_id].action_space, loc_sub_id))
        print(f"sub_id  {sub_id}: {nb_cent} vs {nb_ma}")
    