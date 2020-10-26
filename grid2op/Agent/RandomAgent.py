# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Converter import IdToAct
from grid2op.Agent.AgentWithConverter import AgentWithConverter


class RandomAgent(AgentWithConverter):
    """
    This agent acts randomly on the powergrid. It uses the :class:`grid2op.Converters.IdToAct` to compute all the
    possible actions available for the environment. And then chooses a random one among all these.

    Notes
    ------
    Actions are taken uniformly at random among unary actions. For example, if a game rules allows to take actions that
    can disconnect a powerline AND modify the topology of a substation an action that do both will not be sampled
    by this class.

    This agent is not equivalent to calling `env.action_space.sample()` because the sampling is not
    done the same manner. This agent sample uniformly among all unary actions whereas
    `env.action_space.sample()` (see :func:`grid2op.Action.SerializableActionSpace.sample` for more
    information about the later).

    """
    def __init__(self, action_space, action_space_converter=IdToAct, **kwargs_converter):
        AgentWithConverter.__init__(self, action_space, action_space_converter, **kwargs_converter)

    def my_act(self, transformed_observation, reward, done=False):
        """
        A random agent will "simply" draw a random number between 0 and the number of action, and return this action.

        This is equivalent to draw uniformly at random a feasible action.

        Notes
        -----
        In order to be working as intended, it is crucial that this method does not rely on any other source
        of "pseudo randomness" than :attr:`grid2op.Space.RandomObject.space_prng`.

        In particular, you must avoid
        to use `np.random.XXXX` or the `random` python module. You can replace any call to `np.random.XXX` by
        `self.space_prng.XXX` (**eg**  `np.random.randint(1,5)` can be replaced by `self.space_prng.randint(1,5)`).

        If you really need other sources of randomness (for example if you use tensorflow or torch) we strongly
        recommend you to overload the :func:`BaseAgent.seed` accordingly.

        """
        my_int = self.space_prng.randint(self.action_space.n)
        return my_int
