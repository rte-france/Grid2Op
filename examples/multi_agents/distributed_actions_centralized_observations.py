# Copyright (c) 2019-2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import grid2op
from grid2op.multi_agent import MultiAgentEnv
import pdb


if __name__ == "__main__":
    env_name = "l2rpn_case14_sandbox"
    cent_env = grid2op.make(env_name)
    
    # agent_name : controlled substation id
    zones = {"agent_0": [0, 1, 2, 3, 4],
             "agent_1": [5, 6, 7, 8, 9, 10, 11, 12, 13]}
    env = MultiAgentEnv(cent_env, action_domains=zones)
    env.seed(0)
    dict_obs = env.reset()
    # dict with: key=agent_name, value=the SubGridObservation
    
    act = {"agent_0": env.action_spaces["agent_0"].sample(),
           "agent_1": env.action_spaces["agent_1"].sample()}
    # dict with key=agent name, value=the SubGridAction (here random)
    
    dict_obs, dict_reward, dict_done, dict_info = env.step(act)
    # all of the above are like in the centralized case, but instead of "normal"
    # things, they are dictionnaries with values being the "normal thing" and the keys
    # the agent names.
    
    env.close()
    cent_env.close()
        