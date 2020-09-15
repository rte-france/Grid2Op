# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import numpy as np
from grid2op.Agent import BaseAgent


class DeltaRedispatchRandomAgent(BaseAgent):
    """
    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\

        Used for test. Prefer using a random agent by selecting only the redispatching action
        that you want.

    This agent will perform some redispatch of a given amount among randomly selected dispatchable
    generators.

    Parameters
    ----------
    action_space: :class:`grid2op.Action.ActionSpace`
         the Grid2Op action space

    n_gens_to_redispatch: `int`
      The maximum number of dispatchable generators to play with

    redispatching_delta: `float`
      The redispatching MW value used in both directions

    """
    def __init__(self, action_space,
                 n_gens_to_redispatch=2,
                 redispatching_delta=1.0):
        super().__init__(action_space)
        self.desired_actions = []

        # Get all generators IDs
        gens_ids = np.arange(self.action_space.n_gen, dtype=int)
        # Filter out non resipatchable IDs
        gens_redisp = gens_ids[self.action_space.gen_redispatchable]
        # Cut if needed
        if len(gens_redisp) > n_gens_to_redispatch:
            gens_redisp = gens_redisp[0:n_gens_to_redispatch]

        # Register do_nothing action
        self.desired_actions.append(self.action_space({}))

        # Register 2 actions per generator
        # (increase or decrease by the delta)
        for gen_id in gens_redisp:
            # Create action redispatch by opposite delta
            act1 = self.action_space({
                "redispatch": [
                    (gen_id, -float(redispatching_delta))
                ]
            })
            
            # Create action redispatch by delta
            act2 = self.action_space({
                "redispatch": [
                    (gen_id, float(redispatching_delta))
                ]
            })

            # Register this generator actions
            self.desired_actions.append(act1)
            self.desired_actions.append(act2)

    def act(self, observation, reward, done=False):
        act = self.space_prng.choice(self.desired_actions)
        return act
