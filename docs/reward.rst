.. currentmodule:: grid2op.Reward

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

Detailed Documentation by class
--------------------------------
.. automodule:: grid2op.Reward
    :members:
    :special-members:
    :autosummary:

.. include:: final.rst