# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

from grid2op.Reward.BaseReward import BaseReward
from grid2op.Reward.ConstantReward import ConstantReward


class RewardHelper:
    """
    .. warning:: /!\\\\ Internal, do not use unless you know what you are doing /!\\\\
        It is a class internal to the :class:`grid2op.Environment.Environment` do not use outside
        of its purpose and do not attempt to modify it.

    This class aims at making the creation of rewards class more automatic by the :class:`grid2op.Environment`.

    It is not recommended to derived or modified this class. If a different reward need to be used, it is recommended
    to build another object of this class, and change the :attr:`RewardHelper.rewardClass` attribute.

    Attributes
    ----------
    rewardClass: ``type``
        Type of reward that will be use by this helper. Note that the type (and not an instance / object of that type)
        must be given here. It defaults to :class:`ConstantReward`

    template_reward: :class:`BaseReward`
        An object of class :attr:`RewardHelper.rewardClass` used to compute the rewards.

    """
    def __init__(self, rewardClass=ConstantReward):
        self.rewardClass = rewardClass
        self.template_reward = rewardClass()

    def initialize(self, env):
        """
        This function initializes the template_reward with the environment. It is used especially for using
        :func:`RewardHelper.range`.

        Parameters
        ----------
        env: :class:`grid2op.Environment.Environment`
            The current used environment.

        """
        self.template_reward.initialize(env)

    def range(self):
        """
        Provides the range of the rewards.

        Returns
        -------
        res: ``(float, float)``
            The minimum reward per time step (possibly infinity) and the maximum reward per timestep (possibly infinity)
        """
        return self.template_reward.get_range()

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        """
        Gives the reward that follows the execution of the :class:`grid2op.BaseAction.BaseAction` action in the
        :class:`grid2op.Environment.Environment` env;

        Parameters
        ----------
        action: :class:`grid2op.Action.Action`
            The action performed by the BaseAgent.

        env: :class:`grid2op.Environment.Environment`
            The current environment.

        has_error: ``bool``
            Does the action caused an error, such a diverging powerflow for example= (``True``: the action caused
            an error)

        is_done: ``bool``
            Is the game over (``True`` = the game is over)

        is_illegal: ``bool``
            Is the action legal or not (``True`` = the action was illegal). See
            :class:`grid2op.Exceptions.IllegalAction` for more information.

        is_ambiguous: ``bool``
            Is the action ambiguous or not (``True`` = the action was ambiguous). See
            :class:`grid2op.Exceptions.AmbiguousAction` for more information.

        Returns
        -------
        res: ``float``
            The computed reward

        """
        res = self.template_reward(action, env, has_error, is_done, is_illegal, is_ambiguous)
        return res
