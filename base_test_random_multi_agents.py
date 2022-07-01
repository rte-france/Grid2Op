# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import pdb
import grid2op
from grid2op.Converter import IdToAct
from grid2op.multi_agent import MultiAgentEnv, SubGridObjects
from grid2op.multi_agent.subgridAction import SubGridAction
from grid2op.Agent import RandomAgent
from grid2op.Space import GridObjects
import re


env = grid2op.make("l2rpn_case14_sandbox")

action_domains = {'agent_0' : [0, 1, 2, 3, 4],
                  'agent_1' : [5, 6, 7, 8, 9, 10, 11, 12, 13]}
ma_env = MultiAgentEnv(env, action_domains)

global_random = RandomAgent(env.action_space)
print("GLOBAL RANDOM AGENT MADE")

agents = {}
for agent_nm in action_domains:
    IdToActThis = ma_env.action_spaces[agent_nm].make_local(IdToAct)
    assert IdToActThis.agent_name == agent_nm
    agents[agent_nm] = RandomAgent(ma_env.action_spaces[agent_nm],
                                   action_space_converter=IdToActThis
                                   )
    assert issubclass(ma_env.action_spaces[agent_nm].actionClass, SubGridAction) 
    print(f"created agent {agent_nm}")
    
print("Agents created")
for seed_, ag in enumerate(sorted(agents.keys())):
    agents[ag].seed(seed_)
    
for agent_nm in action_domains:
    print(agents[agent_nm].act(None, None, None))
