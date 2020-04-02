"""
This module implements some utilities to get rewards given an :class:`grid2op.BaseAction` an :class:`grid2op.Environment`
and some associated context (like has there been an error etc.)

It is possible to modify the reward to use to better suit a training scheme, or to better take into account
some phenomenon  by simulating the effect of some :class:`grid2op.BaseAction` using :func:`grid2op.BaseObservation.simulate`.
Doing so only requires to derive the :class:`BaseReward`, and most notably the three abstract methods
:func:`BaseReward.__init__`, :func:`BaseReward.initialize` and :func:`BaseReward.__call__`

"""
from abc import ABC, abstractmethod


class BaseReward(ABC):
    """
    Base class from which all rewards used in the Grid2Op framework should derived.

    In reinforcement learning, a reward is a signal send by the :class:`grid2op.Environment` to the
    :class:`grid2op.BaseAgent` indicating how well this agent performs.

    One of the goal of Reinforcement Learning is to maximize the (discounted) sum of (expected) rewards over time.

    Attributes
    ----------
    reward_min: ``float``
        The minimum reward an :class:`grid2op.BaseAgent` can get performing the worst possible :class:`grid2op.BaseAction` in
        the worst possible scenario.

    reward_max: ``float``
        The maximum reward an :class:`grid2op.BaseAgent` can get performing the best possible :class:`grid2op.BaseAction` in
        the best possible scenario.

    """
    @abstractmethod
    def __init__(self):
        """
        Initializes :attr:`BaseReward.reward_min` and :attr:`BaseReward.reward_max`

        """
        self.reward_min = 0
        self.reward_max = 0

    def initialize(self, env):
        """
        If :attr:`BaseReward.reward_min`, :attr:`BaseReward.reward_max` or other custom attributes require to have a
        valid :class:`grid2op.Environement.Environment` to be initialized, this should be done in this method.

        Parameters
        ----------
        env: :class:`grid2op.Environment.Environment`
            An environment instance properly initialized.

        Returns
        -------
        ``None``

        """
        pass

    @abstractmethod
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        """
        Method called to compute the reward.

        Parameters
        ----------
        action: :class:`grid2op.Action.Action`
            BaseAction that has been submitted by the :class:`grid2op.BaseAgent`

        env: :class:`grid2op.Environment.Environment`
            An environment instance properly initialized.

        has_error: ``bool``
            Has there been an error, for example a :class:`grid2op.DivergingPowerFlow` be thrown when the action has
            been implemented in the environment.

        is_done: ``bool``
            Is the episode over (either because the agent has reached the end, or because there has been a game over)

        is_illegal: ``bool``
            Has the action submitted by the BaseAgent raised an :class:`grid2op.Exceptions.IllegalAction` exception.
            In this case it has been
            overidden by "do nohting" by the environment.

        is_ambiguous: ``bool``
            Has the action submitted by the BaseAgent raised an :class:`grid2op.Exceptions.AmbiguousAction` exception.
            In this case it has been
            overidden by "do nothing" by the environment.

        Returns
        -------
        res: ``float``
            The reward associated to the input parameters.

        """
        pass

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
