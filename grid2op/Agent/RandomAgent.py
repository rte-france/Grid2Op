# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import pdb

from grid2op.Converter import IdToAct
from grid2op.Agent.AgentWithConverter import AgentWithConverter


class RandomAgent(AgentWithConverter):
    """
    This agent acts randomnly on the powergrid. It uses the :class:`grid2op.Converters.IdToAct` to compute all the
    possible actions available for the environment. And then chooses a random one among all these.

    **NB** Action are taken randomly among unary actions. For example, if a game rules allows to take actions that
    can disconnect a powerline AND modify the topology of a substation an action that do both will not be sampled
    by this class.

    """
    def __init__(self, action_space, action_space_converter=IdToAct, **kwargs_converter):
        AgentWithConverter.__init__(self, action_space, action_space_converter, **kwargs_converter)

    def my_act(self, transformed_observation, reward, done=False):
        return self.action_space.sample()
