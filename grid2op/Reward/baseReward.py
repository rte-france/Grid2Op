# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import logging
from abc import ABC, abstractmethod

import grid2op
from grid2op.dtypes import dt_float
from grid2op.Action import BaseAction


class BaseReward(ABC):
    """
    Base class from which all rewards used in the Grid2Op framework should derived.

    In reinforcement learning, a reward is a signal send by the :class:`grid2op.Environment.Environment` to the
    :class:`grid2op.BaseAgent` indicating how well this agent performs.

    One of the goal of Reinforcement Learning is to maximize the (discounted) sum of (expected) rewards over time.


    You can create all rewards you want in grid2op. The only requirement is that all rewards should inherit this
    BaseReward.

    Attributes
    ----------
    reward_min: ``float``
        The minimum reward an :class:`grid2op.BaseAgent` can get performing the worst possible
        :class:`grid2op.Action.BaseAction` in
        the worst possible scenario.

    reward_max: ``float``
        The maximum reward an :class:`grid2op.Agent.BaseAgent` can get performing the best possible
        :class:`grid2op.Action.BaseAction` in
        the best possible scenario.

    Examples
    ---------
    If you want the environment to compute a reward that is the sum of the flow (this is not a good reward, but
    we use it as an example on how to do it) you can achieve it with:

    .. code-block:: python

        import grid2op
        from grid2op.Reward import BaseReward

        # first you create your reward
        class SumOfFlowReward(BaseReward):
            def __init__(self):
                BaseReward.__init__(self)

            def initialize(self, env):
                # this function is used to inform the class instance about the environment specification
                # you can use `env.n_line` or `env.n_load` or `env.get_thermal_limit()` for example
                # do not forget to initialize "reward_min" and "reward_max"
                self.reward_min = 0.
                self.reward_max = np.sum(env.get_thermal_limit)

                # in this case the maximum reward is obtained when i compute the sum of the maximum flows
                # on each powerline

            def __call__(action, env, has_error, is_done, is_illegal, is_ambiguous):
                # this method is called at the end of 'env.step' to compute the reward
                # in our case we just want to sum the flow on each powerline because... why not...
                if has_error:
                    # see the "Notes" paragraph for more information
                    res = self.reward_min
                else:
                    res = np.sum(env.get_obs().a_or)
                return res

        # then you create your environment with it:
        NAME_OF_THE_ENVIRONMENT = "l2rpn_case14_sandbox"
        env = grid2op.make(NAME_OF_THE_ENVIRONMENT,reward_class=SumOfFlowReward)
        # and do a step with a "do nothing" action
        obs = env.reset()
        obs, reward, done, info = env.step(env.action_space())
        assert np.sum(obs.a_or) == reward
        # the above should be true

    Notes
    ------
    If the flag `has_error` is set to ``True`` this indicates there has been an error in the "env.step" function.
    This might induce some undefined behaviour if using some method of the environment.

    Please make sure to check whether or not this is the case when defining your reward.

    This "new" behaviour has been introduce to "fix" the akward behavior spotted in
    # https://github.com/Grid2Op/grid2op/issues/146

    .. code-block:: python

        def __call__(action, env, has_error, is_done, is_illegal, is_ambiguous):
            if has_error:
                # DO SOMETHING IN THIS CASE
                res = self.reward_min
            else:
                # DO NOT USE `env.get_obs()` (nor any method of the environment `env.XXX` if the flag `has_error`
                # is set to ``True``
                # This might result in undefined behaviour
                res = np.sum(env.get_obs().a_or)
            return res

    """

    def __init__(self, logger: logging.Logger=None):
        """
        Initializes :attr:`BaseReward.reward_min` and :attr:`BaseReward.reward_max`

        """
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(0.0)
        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.disabled = True
        else:
            self.logger: logging.Logger = logger.getChild(f"{type(self).__name__}")
    
    def is_simulated_env(self, env):
        # to prevent cyclical import
        from grid2op.Environment._obsEnv import _ObsEnv
        from grid2op.Environment._forecast_env import _ForecastEnv
        return isinstance(env, (_ObsEnv, _ForecastEnv))
            
    def initialize(self, env: "grid2op.Environment.BaseEnv") -> None:
        """
        If :attr:`BaseReward.reward_min`, :attr:`BaseReward.reward_max` or other custom attributes require to have a
        valid :class:`grid2op.Environment.Environment` to be initialized, this should be done in this method.

        **NB** reward_min and reward_max are used by the environment to compute the maximum and minimum reward and
        cast it in "reward_range" which is part of the openAI gym public interface. If you don't define them, some
        piece of code might not work as expected.

        Parameters
        ----------
        env: :class:`grid2op.Environment.Environment`
            An environment instance properly initialized.

        """
        pass

    def reset(self, env: "grid2op.Environment.BaseEnv") -> None:
        """
        This method is called each time `env` is reset.

        It can be usefull, for example if the reward depends on the length of the current chronics.

        It does nothing by default.

        Parameters
        ----------
        env: :class:`grid2op.Environment.Environment`
            The current environment

        .. danger::
            This function should not modify self.reward_min nor self.reward_max !!!

            It might cause really hard trouble for agent to learn if you do so.

        """
        pass

    @abstractmethod
    def __call__(self,
                 action: BaseAction,
                 env: "grid2op.Environment.BaseEnv",
                 has_error: bool,
                 is_done: bool,
                 is_illegal: bool,
                 is_ambiguous: bool) -> float:
        """
        Method called to compute the reward.

        Parameters
        ----------
        action: :class:`grid2op.Action.Action`
            BaseAction that has been submitted by the :class:`grid2op.BaseAgent`

        env: :class:`grid2op.Environment.Environment`
            An environment instance properly initialized.

        has_error: ``bool``
            Has there been an error, for example a :class:`grid2op.DivergingPowerflow` be thrown when the action has
            been implemented in the environment.

        is_done: ``bool``
            Is the episode over (either because the agent has reached the end, or because there has been a game over)

        is_illegal: ``bool``
            Has the action submitted by the BaseAgent raised an :class:`grid2op.Exceptions.IllegalAction` exception.
            In this case it has been replaced by "do nohting" by the environment. **NB** an illegal action is NOT
            an ambiguous action. See the description of the Action module: :ref:`Illegal-vs-Ambiguous` for more details.

        is_ambiguous: ``bool``
            Has the action submitted by the BaseAgent raised an :class:`grid2op.Exceptions.AmbiguousAction` exception.
            In this case it has been replaced by "do nothing" by the environment. **NB** an illegal action is NOT
            an ambiguous action. See the description of the Action module: :ref:`Illegal-vs-Ambiguous` for more details.

        Returns
        -------
        res: ``float``
            The reward associated to the input parameters.

        Notes
        ------
        All the flags can be used to know on which type of situation the reward is computed.

        For example, if `has_error` is ``True`` it means there was an error during the computation of the powerflow.
        this means there is a "game_over", so ``is_done`` is ``True`` in this case.

        But, if there is ``is_done`` equal to ``True`` but ``has_error`` equal to ``False`` this means that the episode
        is over without any error. In other word, your agent sucessfully managed all the scenario and to get to the
        end of the episode.

        """
        return self.reward_min

    def get_range(self):
        """
        Shorthand to retrieve both the minimum and maximum possible rewards in one command.

        It is not recommended to override this function.

        Returns
        -------
        reward_min: ``float``
            The minimum reward, see :attr:`BaseReward.reward_min`

        reward_max: ``float``
            The maximum reward, see :attr:`BaseReward.reward_max`

        """
        return self.reward_min, self.reward_max

    def set_range(self, reward_min: float, reward_max: float):
        """
        Setter function for the :attr:`BaseReward.reward_min` and :attr:`BaseReward.reward_max`.

        It is not recommended to override this function

        Parameters
        -------
        reward_min: ``float``
            The minimum reward, see :attr:`BaseReward.reward_min`

        reward_max: ``float``
            The maximum reward, see :attr:`BaseReward.reward_max`
        """
        self.reward_min = reward_min
        self.reward_max = reward_max

    def __iter__(self):
        """
        Implements python iterable to get a dict summary using `summary = dict(reward_instance)`
        Can be overloaded by subclass, default implementation gives name, reward_min, reward_max
        """
        yield ("name", self.__class__.__name__)
        yield ("reward_min", float(self.reward_min))
        yield ("reward_max", float(self.reward_max))

    def close(self) -> None:
        """overide this for certain reward that might need specific behaviour"""
        pass

    def is_in_blackout(self, has_error, is_done) -> bool:
        return is_done and has_error
