"""
This module implements some utilities to get rewards given an :class:`grid2op.Action` an :class:`grid2op.Environment`
and some associated context (like has there been an error etc.)

It is possible to modify the reward to use to better suit a training scheme, or to better take into account
some phenomenon  by simulating the effect of some :class:`grid2op.Action` using :func:`grid2op.Observation.simulate`.
Doing so only requires to derive the :class:`Reward`, and most notably the three abstract methods
:func:`Reward.__init__`, :func:`Reward.initialize` and :func:`Reward.__call__`

"""
import numpy as np

from abc import ABC, abstractmethod

try:
    from .Exceptions import Grid2OpException
except (ModuleNotFoundError, ImportError):
    from Exceptions import Grid2OpException


class Reward(ABC):
    """
    Base class from which all rewards used in the Grid2Op framework should derived.

    In reinforcement learning, a reward is a signal send by the :class:`grid2op.Environment` to the
    :class:`grid2op.Agent` indicating how well this agent performs.

    One of the goal of Reinforcement Learning is to maximize the (discounted) sum of (expected) rewards over time.

    Attributes
    ----------
    reward_min: ``float``
        The minimum reward an :class:`grid2op.Agent` can get performing the worst possible :class:`grid2op.Action` in
        the worst possible scenario.

    reward_max: ``float``
        The maximum reward an :class:`grid2op.Agent` can get performing the best possible :class:`grid2op.Action` in
        the best possible scenario.

    """
    @abstractmethod
    def __init__(self):
        """
        Initializes :attr:`Reward.reward_min` and :attr:`Reward.reward_max`

        """
        self.reward_min = 0
        self.reward_max = 0

    def initialize(self, env):
        """
        If :attr:`Reward.reward_min`, :attr:`Reward.reward_max` or other custom attributes require to have a
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
            Action that has been submitted by the :class:`grid2op.Agent`

        env: :class:`grid2op.Environment.Environment`
            An environment instance properly initialized.

        has_error: ``bool``
            Has there been an error, for example a :class:`grid2op.DivergingPowerFlow` be thrown when the action has
            been implemented in the environment.

        is_done: ``bool``
            Is the episode over (either because the agent has reached the end, or because there has been a game over)

        is_illegal: ``bool``
            Has the action submitted by the Agent raised an :class:`grid2op.Exceptions.IllegalAction` exception.
            In this case it has been
            overidden by "do nohting" by the environment.

        is_ambiguous: ``bool``
            Has the action submitted by the Agent raised an :class:`grid2op.Exceptions.AmbiguousAction` exception.
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
            The minimum reward, see :attr:`Reward.reward_min`

        reward_max: ``float``
            The maximum reward, see :attr:`Reward.reward_max`

        """
        return self.reward_min, self.reward_max


class ConstantReward(Reward):
    """
    Most basic implementation of reward: everything has the same values.

    Note that this :class:`Reward` subtype is not usefull at all, whether to train an :attr:`Agent` nor to assess its
    performance of course.

    """
    def __init__(self):
        Reward.__init__(self)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        return 0


class FlatReward(Reward):
    """
    This reward return a fixed number (if there are not error) or 0 if there is an error.

    """
    def __init__(self, per_timestep=1):
        Reward.__init__(self)
        self.per_timestep = per_timestep
        self.total_reward = 0
        self.reward_min = 0
        self.reward_max = per_timestep

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if not has_error:
            res = self.per_timestep
        else:
            res = self.reward_min
        return res


class RedispReward(Reward):
    """
    This reward can be used for environments where redispatching is availble. It assigns a cost to redispatching action
    and penalizes with the losses.
    """
    def __init__(self, alpha_redisph=5.0):
        Reward.__init__(self)
        self.reward_min = None
        self.reward_max = None
        self.max_regret = 0.
        self.alpha_redisph = alpha_redisph

    def initialize(self, env):
        if not env.redispatching_unit_commitment_availble:
            raise Grid2OpException("Impossible to use the RedispReward reward with an environment without generators"
                                   "cost. Please make sure env.redispatching_unit_commitment_availble is available.")
        worst_marginal_cost = np.max(env.gen_cost_per_MW)
        worst_load = np.sum(env.gen_pmax)
        worst_losses = 0.05 * worst_load  # it's not the worst, but definitely an upper bound
        worst_redisp = self.alpha_redisph * np.sum(env.gen_pmax)  # not realistic, but an upper bound
        self.max_regret = (worst_losses + worst_redisp)*worst_marginal_cost
        self.reward_min = -10

        least_loads = (worst_load * 0.5)  # half the capacity of the grid
        least_losses = 0.015 * least_loads  # 1.5% of losses
        least_redisp = 0.  # lower_bound is 0
        base_marginal_cost = np.min(env.gen_cost_per_MW[env.gen_cost_per_MW > 0.])
        min_regret = (least_losses + least_redisp) * base_marginal_cost
        self.reward_max = (self.max_regret - min_regret) / least_loads

    def __call__(self,  action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            res = self.reward_min * 0.5
        else:
            # compute the losses
            gen_p, *_ = env.backend.generators_info()
            load_p, *_ = env.backend.loads_info()
            losses = np.sum(gen_p) - np.sum(load_p)

            # compute the marginal cost
            marginal_cost = np.max(env.gen_cost_per_MW[env.gen_activeprod_t > 0.])

            # redispatching amount
            redisp_cost = self.alpha_redisph * np.sum(np.abs(env.actual_dispatch)) * marginal_cost

            # cost of losses
            losses_cost = losses * marginal_cost

            # total "regret"
            regret = losses_cost + redisp_cost

            # compute reward
            reward = self.max_regret - regret

            # divide it by load, to be less sensitive to load variation
            res = reward / np.sum(load_p)

        return res


class EconomicReward(Reward):
    """
    This reward computes the marginal cost of the powergrid. As RL is about maximising a reward, while we want to
    minimize the cost, this class also ensures that:

    - the reward is positive if there is no game over, no error etc.
    - the reward is inversely proportional to the cost of the grid (the higher the reward, the lower the economic cost).

    """
    def __init__(self):
        Reward.__init__(self)
        self.reward_min = None
        self.reward_max = None
        self.worst_cost = None

    def initialize(self, env):
        if not env.redispatching_unit_commitment_availble:
            raise Grid2OpException("Impossible to use the EconomicReward reward with an environment without generators"
                                   "cost. Please make sure env.redispatching_unit_commitment_availble is available.")
        self.worst_cost = np.sum(env.gen_cost_per_MW *env.gen_pmax)
        self.reward_min = -1.
        self.reward_max = self.worst_cost

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            res = self.reward_min * 0.5
        else:
            # compute the cost of the grid
            res = np.sum(env.current_obs.prod_p * env.gen_cost_per_MW)
            # we want to minimize the cost by maximizing the reward so let's take the opposite
            res *= -1
            # to be sure it's positive, add the highest possible cost
            res += self.worst_cost
        return res


class IncreasingFlatReward(Reward):
    """
    This reward just counts the number of timestep the agent has sucessfully manage to perform.

    It adds a constant reward for each time step sucessfully handled.

    """
    def __init__(self, per_timestep=1):
        Reward.__init__(self)
        self.per_timestep = per_timestep
        self.total_reward = 0
        self.reward_min = 0

    def initialize(self, env):
        if env.chronics_handler.max_timestep() > 0:
            self.reward_max = env.chronics_handler.max_timestep() * self.per_timestep
        else:
            self.reward_max = np.inf

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if not has_error:
            res = env.nb_time_step * self.per_timestep
        else:
            res = self.reward_min
        return res


class L2RPNReward(Reward):
    """
    This is the historical :class:`Reward` used for the Learning To Run a Power Network competition.

    See `L2RPN <https://l2rpn.chalearn.org/>`_ for more information.

    """
    def __init__(self):
        Reward.__init__(self)

    def initialize(self, env):
        self.reward_min = 0.
        self.reward_max = env.backend.n_line

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if not is_done and not has_error:
            line_cap = self.__get_lines_capacity_usage(env)
            res = np.sum(line_cap)
        else:
            # no more data to consider, no powerflow has been run, reward is what it is
            res = self.reward_min
        return res

    @staticmethod
    def __get_lines_capacity_usage(env):
        ampere_flows = np.abs(env.backend.get_line_flow())
        thermal_limits = np.abs(env.backend.get_thermal_limit())
        relative_flow = np.divide(ampere_flows, thermal_limits)

        x = np.minimum(relative_flow, 1)
        lines_capacity_usage_score = np.maximum(1 - x ** 2, 0.)
        return lines_capacity_usage_score


class RewardHelper:
    """
    This class aims at making the creation of rewards class more automatic by the :class:`grid2op.Environment`.

    It is not recommended to derived or modified this class. If a different reward need to be used, it is recommended
    to build another object of this class, and change the :attr:`RewardHelper.rewardClass` attribute.

    Attributes
    ----------
    rewardClass: ``type``
        Type of reward that will be use by this helper. Note that the type (and not an instance / object of that type)
        must be given here. It defaults to :class:`ConstantReward`

    template_reward: :class:`Reward`
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
        Gives the reward that follows the execution of the :class:`grid2op.Action.Action` action in the
        :class:`grid2op.Environment.Environment` env;


        Parameters
        ----------
        action: :class:`grid2op.Action.Action`
            The action performed by the Agent.

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

        """
        if not is_done and not has_error:
            res = self.template_reward(action, env, has_error, is_done, is_illegal, is_ambiguous)
        else:
            # no more data to consider, no powerflow has been run, reward is what it is
            res = self.template_reward.reward_min
        return res