.. currentmodule:: grid2op.Reward

.. _reward-module:

Reward
===================================

This page is organized as follow:

.. contents:: Table of Contents
    :depth: 3

Objectives
-----------
This module implements some utilities to get rewards given an :class:`grid2op.Action` an :class:`grid2op.Environment`
and some associated context (like has there been an error etc.)

It is possible to modify the reward to use to better suit a training scheme, or to better take into account
some phenomenon  by simulating the effect of some :class:`grid2op.Action` using
:func:`grid2op.Observation.BaseObservation.simulate`.

Doing so only requires to derive the :class:`BaseReward`, and most notably the three abstract methods
:func:`BaseReward.__init__`, :func:`BaseReward.initialize` and :func:`BaseReward.__call__`

Customization of the reward
-----------------------------

In grid2op you can customize the reward function / reward kernel used by your agent. By default, when you create an
environment a reward has been specified for you by the creator of the environment and you have nothing to do:

.. code-block:: python

    import grid2op
    env_name = "l2rpn_case14_sandbox"
    
    env = grid2op.make(env_name)

    obs = env.reset()
    an_action = env.action_space()
    obs, reward_value, done, info = env.step(an_action)

The value of the reward function above is computed by a default function that depends on 
the environment you are using. For the example above, the "l2rpn_case14_sandbox" environment is
using the :class:`RedispReward`.

Using a reward function available in grid2op
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to customize your environment by adapting the reward and use a reward available in grid2op
it is rather simple, you need to specify it in the `make` command:


.. code-block:: python

    import grid2op
    from grid2op.Reward import EpisodeDurationReward
    env_name = "l2rpn_case14_sandbox"
    
    env = grid2op.make(env_name, reward_class=EpisodeDurationReward)

    obs = env.reset()
    an_action = env.action_space()
    obs, reward_value, done, info = env.step(an_action)

In this example the `reward_value` is computed using the formula defined in the :class:`EpisodeDurationReward`.

.. note::
    There is no error in the syntax. You need to provide the class and not an object of the class 
    (see next paragraph for more information about that).

At time of writing the available reward functions is :

- :class:`AlarmReward`
- :class:`AlertReward`
- :class:`BridgeReward`
- :class:`CloseToOverflowReward`
- :class:`ConstantReward`
- :class:`DistanceReward`
- :class:`EconomicReward`
- :class:`EpisodeDurationReward`
- :class:`FlatReward`
- :class:`GameplayReward`
- :class:`IncreasingFlatReward`
- :class:`L2RPNReward`
- :class:`LinesCapacityReward`
- :class:`LinesReconnectedReward`
- :class:`N1Reward`
- :class:`RedispReward`

In the provided reward you have also some convenience functions to combine different reward. These are:

- :class:`CombinedReward`
- :class:`CombinedScaledReward`

Basically these two classes allows you to combine (sum) different reward in a single one.

Passing an instance instead of a class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On some occasion, it might be easier to work with instance of classes (object) 
rather than to work with classes (especially if you want to customize the implementation used).
You can do this without any issue:


.. code-block:: python

    import grid2op
    from grid2op.Reward import N1Reward
    env_name = "l2rpn_case14_sandbox"
    
    n1_l1_reward = N1Reward(l_id=1)  # this is an object and not a class.
    env = grid2op.make(env_name, reward_class=n1_l1_reward)

    obs = env.reset()
    an_action = env.action_space()
    obs, reward_value, done, info = env.step(an_action)

In this example `reward_value` is computed as being the maximum flow on all the powerlines after 
the disconnection of powerline `1` (because we specified `l_id=1` at creation). If we
want to know the maximum flows after disconnection of powerline `5` you can call:

.. code-block:: python

    import grid2op
    from grid2op.Reward import N1Reward
    env_name = "l2rpn_case14_sandbox"
    
    n1_l5_reward = N1Reward(l_id=5)  # this is an object and not a class.
    env = grid2op.make(env_name, reward_class=n1_l5_reward)

Customizing the reward for the "simulate"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In grid2op, you have the possibility to `simulate` the impact of an action
on some future steps with the use of `obs.simulate(...)` (see :func:`grid2op.Observation.BaseObservation.simulate`)
or `obs.get_forecast_env()` (see :func:`grid2op.Observation.BaseObservation.get_forecast_env`). 

In these methods you have some computations of rewards. Grid2op lets you allow to customize how these rewards 
are computed. You can change it in multiple fashion:

.. code-block:: python

    import grid2op
    from grid2op.Reward import EpisodeDurationReward
    env_name = "l2rpn_case14_sandbox"
    
    env = grid2op.make(env_name, reward_class=EpisodeDurationReward)
    obs = env.reset()

    an_action = env.action_space()
    sim_obs, sim_reward, sim_d, sim_i = obs.simulate(an_action)

By default `sim_reward` is comupted with the same function as the environment, in this
example :class:`EpisodeDurationReward`.

If for some reason you want to customize the formula used to compute `sim_reward` and cannot (or
does not want to) modify the reward of the environment you can:

.. code-block:: python

    import grid2op
    from grid2op.Reward import EpisodeDurationReward
    env_name = "l2rpn_case14_sandbox"

    env = grid2op.make(env_name)
    obs = env.reset()

    env.observation_space.change_reward(EpisodeDurationReward)
    an_action = env.action_space()

    sim_obs, sim_reward, sim_d, sim_i = obs.simulate(an_action)
    next_obs, reward_value, done, info = env.step(an_action)

In this example, `sim_reward` is computed using the `EpisodeDurationReward` (on forecast data)
and `reward_value` is computed using the default reward of "l2rpn_case14_sandbox" on the
"real" time serie data.

Creating a new reward
~~~~~~~~~~~~~~~~~~~~~~

If you don't find any suitable reward function in grid2op (or in other package) you might
want to implement one yourself.

To that end, you need to implement a class that derives from :class:`BaseReward`, like this:

.. code-block:: python

    import grid2op
    from grid2op.Reward import BaseReward
    from grid2op.Action import BaseAction
    from grid2op.Environment import BaseEnv


    class MyCustomReward(BaseReward):
        def __init__(self, whatever, you, want, logger=None):
            self.whatever = blablabla
            # some code needed
            ...
            super().__init__(logger)

        def __call__(self,
                    action: BaseAction,
                    env: BaseEnv,
                    has_error: bool,
                    is_done: bool,
                    is_illegal: bool,
                    is_ambiguous: bool) -> float:
            # only method really required.
            # called at each step to compute the reward.
            # this is where you need to code the "formula" of your reward
            ...

        def initialize(self, env: BaseEnv):
            # optional
            # called once, the first time the reward is used
            pass

        def reset(self, env: BaseEnv):
            # optional
            # called by the environment each time it is "reset"
            pass
        
        def close(self):
            # optional called once when the environment is deleted
            pass


And then you can use your (custom) reward like any other:

.. code-block:: python

    import grid2op
    from the_above_script import MyCustomReward
    env_name = "l2rpn_case14_sandbox"

    custom_reward = MyCustomReward(whatever=1, you=2, want=42)
    env = grid2op.make(env_name, reward_class=custom_reward)
    obs = env.reset()
    an_action = env.action_space()
    obs, reward_value, done, info = env.step(an_action)

And now `reward_value` is computed using the formula you defined in `__call__`

Training with multiple rewards
-------------------------------
In the standard reinforcement learning framework the reward is unique. In grid2op, we didn't want to modify that.

However powergrid are complex environment with some specific and unsual dynamics. For these reasons it can be
difficult to compress all these signal into one single scalar. To speed up the learning process, to force the
Agent to adopt more resilient strategies etc. it can be usefull to look at different aspect, thus using different
reward. Grid2op allows to do so. At each time step (and also when using the `simulate` function) it is possible
to compute different rewards. This rewards must inherit and be provided at the initialization of the Environment.

This can be done as followed:

.. code-block:: python

    import grid2op
    from grid2op.Reward import GameplayReward, L2RPNReward
    env = grid2op.make("case14_realistic", reward_class=L2RPNReward, other_rewards={"gameplay": GameplayReward})
    obs = env.reset()
    act = env.action_space()  # the do nothing action
    obs, reward, done, info = env.step(act)  # immplement the do nothing action on the environment

On this example, "reward" comes from the :class:`L2RPNReward` and the results of the "reward" computed with the
:class:`GameplayReward` is accessible with the info["rewards"]["gameplay"]. We choose for this example to name the other
rewards, "gameplay" which is related to the name of the reward "GampeplayReward" for convenience. The name
can be absolutely any string you want.


**NB** In the case of L2RPN competitions, the reward can be modified by the competitors, and so is the "other_reward"
key word arguments. The only restriction is that the key "__score" will be use by the organizers to compute the
score the agent. Any attempt to modify it will be erased by the score function used by the organizers without any
warning.

.. _reward-module-reset-focus:

What happens in the "reset"
------------------------------

TODO

Detailed Documentation by class
--------------------------------
.. automodule:: grid2op.Reward
    :members:
    :special-members:
    :autosummary:

.. include:: final.rst
