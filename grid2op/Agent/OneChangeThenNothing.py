# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Agent.BaseAgent import BaseAgent


class OneChangeThenNothing(BaseAgent):
    """
    This is a specific kind of BaseAgent. It does an BaseAction (possibly non empty) at the first time step and then does
    nothing.

    This class is an abstract class and cannot be instanciated (ie no object of this class can be created). It must
    be overridden and the method :func:`OneChangeThenNothing._get_dict_act` be defined. Basically, it must know
    what action to do.

    Attributes
    ------------
    my_dict: ``dict`` (class member)
        Representation, as a dictionnary of the only action that this Agent will do at the first time step.

    Examples
    ---------
    We advise to use this class as following

    .. code-block:: python

        import grid2op
        from grid2op.Agent import OneChangeThenNothing
        acts_dict_ = [{}, {"set_line_status": [(0,-1)]}]  # list of dictionaries. Each dictionary
        # represents a valid action

        env = grid2op.make()  # create an environment
        for act_as_dict in zip(acts_dict_):
            # generate the proper class that will perform the first action (encoded by {}) in acts_dict_
            agent_class = OneChangeThenNothing.gen_next(act_as_dict)

            # start a runner with this agent
            runner = Runner(**env.get_params_for_runner(), agentClass=agent_class)
            # run 2 episode with it
            res_2 = runner.run(nb_episode=2)

    """

    my_dict = {}

    def __init__(self, action_space):
        BaseAgent.__init__(self, action_space)
        self.has_changed = False
        self.do_nothing_action = self.action_space({})

    def act(self, observation, reward, done=False):
        if self.has_changed:
            res = self.do_nothing_action
        else:
            res = self.action_space(self._get_dict_act())
            self.has_changed = True
        return res

    def reset(self, obs):
        self.has_changed = False

    def _get_dict_act(self):
        """
        Function that need to be overridden to indicate which action to perform.

        Returns
        -------
        res: ``dict``
            A dictionnary that can be converted into a valid :class:`grid2op.BaseAction.BaseAction`. See the help of
            :func:`grid2op.BaseAction.ActionSpace.__call__` for more information.
        """
        return self.my_dict

    @classmethod
    def gen_next(cls, dict_):
        """
        This function allows to change the dictionnary of the action that the agent will perform.

        See the class level documentation for an example on how to use this.

        Parameters
        ----------
        dict_: ``dict``
            A dictionnary representing an action. This dictionnary is assumed to be convertible into an action.
            No check is performed at this stage.


        """
        cls.my_dict = dict_
        return cls