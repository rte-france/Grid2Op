# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
from collections.abc import Iterable

from grid2op.Action import BaseAction
from grid2op.Exceptions import AgentError

from grid2op.Agent.baseAgent import BaseAgent


class FromActionsListAgent(BaseAgent):
    """This type of agent will perform some actions based on a provided list of actions.
    If no action is provided for a given step (for example because it survives for more steps that the
    length of the provided action list, it will do nothing.

    Notes
    -----
    No check are performed to make sure the action types is compatible with the environment. For example, the
    environment might prevent to perform redispatching, but, at the creation of the agent, we do not ensure
    that no actions performing redispatching are performed.
    """

    def __init__(self, action_space, action_list=None):
        BaseAgent.__init__(self, action_space=action_space)
        if action_list is None:
            self._action_list = []
        else:
            if isinstance(action_list, Iterable):
                self._action_list = copy.deepcopy(action_list)
            else:
                raise AgentError(
                    'Impossible to create a "FromActionsListAgent" without providing a valid list of '
                    'actions. Make sure that "action_list" parameters is iterable.'
                )

        # check that everything is valid
        my_dict = copy.deepcopy(type(self.action_space()).cls_to_dict())
        self.__clean_dict_for_compare(my_dict)
        for act_nb, act in enumerate(self._action_list):
            if not isinstance(act, BaseAction):
                raise AgentError(
                    f'Impossible to create a "FromActionsListAgent" with a list that does not '
                    f"contain an action. We found {act} at position {act_nb}, which is NOT a valid "
                    f"grid2op action."
                )
            this_dict = copy.deepcopy(type(act).cls_to_dict())
            self.__clean_dict_for_compare(this_dict)
            if this_dict != my_dict:
                raise AgentError(
                    f'Impossible to create a "FromActionsListAgent" with a list that contains '
                    f"actions from a different environment. Please check action at position {act_nb}."
                )

    def act(self, observation, reward, done=False):
        if observation.current_step < len(self._action_list):
            return self._action_list[observation.current_step]
        else:
            return self.action_space()

    def __clean_dict_for_compare(self, dict_):
        if "glop_version" in dict_:
            del dict_["glop_version"]
        if "_PATH_ENV" in dict_:
            del dict_["_PATH_ENV"]
        if "env_name" in dict_:
            del dict_["env_name"]
