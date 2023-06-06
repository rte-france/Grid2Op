# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from abc import ABC, abstractmethod


class BaseRules(ABC):
    """
    This class is a base class that determines whether or not an action is legal in certain environment.
    See the definition of :func:`BaseRules.__call__` for more information.

    Basically, this is an empty class with an overload of the __call__ operator that should return ``True`` or ``False``
    depending on the legality of the action.

    In :class:`grid2op.Environment`, only action of the users are checked for legality.

    """
    
    def initialize(self, env):
        """
        This function is used to inform the class instance about the environment specification. 
        It can be the place to assert the defined rules are suited for the environement.
        Parameters
        ----------
        env: :class:`grid2op.Environment.Environment`
            The environment on which the action is performed. The environement instance is not fully initialized itself.
        """
        pass

    @abstractmethod
    def __call__(self, action, env):
        """
        As opposed to "ambiguous action", "illegal action" are not illegal per se.
        They are legal or not on a certain environment. For example, disconnecting
        a powerline that has been cut off for maintenance is illegal. Saying to action to both disconnect a
        powerline and assign it to bus 2 on it's origin end is ambiguous, and not tolerated in Grid2Op.

        Parameters
        ----------
        action: :class:`grid2op.Action.Action`
            The action of which the legality is tested.
        env: :class:`grid2op.Environment.Environment`
            The environment on which the action is performed.

        Returns
        -------
        is_legal: ``bool``
            Whether the action is legal or not

        reason:
            The cause of the illegal part of the action (should be a grid2op exception)
        """
        pass

    def can_use_simulate(self, nb_simulate_call_step, nb_simulate_call_episode, param):
        """
        This function can be overriden.

        It is expected to return either SimulateUsedTooMuchThisStep or SimulateUsedTooMuchThisEpisode if the number of calls to `obs.simulate`
        is too high in total or for the given step
        """
        return None
