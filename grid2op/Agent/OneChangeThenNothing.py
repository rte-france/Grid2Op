# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from abc import abstractmethod
import pdb

from grid2op.Agent.BaseAgent import BaseAgent


class OneChangeThenNothing(BaseAgent):
    """
    This is a specific kind of BaseAgent. It does an BaseAction (possibly non empty) at the first time step and then does
    nothing.

    This class is an abstract class and cannot be instanciated (ie no object of this class can be created). It must
    be overridden and the method :func:`OneChangeThenNothing._get_dict_act` be defined. Basically, it must know
    what action to do.

    """
    def __init__(self, action_space, action_space_converter=None):
        BaseAgent.__init__(self, action_space)
        self.has_changed = False

    def act(self, observation, reward, done=False):
        if self.has_changed:
            res = self.action_space({})
            self.has_changed = True
        else:
            res = self.action_space(self._get_dict_act())
        return res

    @abstractmethod
    def _get_dict_act(self):
        """
        Function that need to be overridden to indicate which action to perfom.

        Returns
        -------
        res: ``dict``
            A dictionnary that can be converted into a valid :class:`grid2op.BaseAction.BaseAction`. See the help of
            :func:`grid2op.BaseAction.ActionSpace.__call__` for more information.
        """
        pass
